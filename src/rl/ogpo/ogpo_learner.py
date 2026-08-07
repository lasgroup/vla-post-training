# ruff: noqa: F722
"""OGPO learner: PPO on a Pi05 flow policy with a BC anchor.

This learner inherits the dual-critic + EMA + checkpointing plumbing from
``AdvantageWeightedSFTLearner`` and only swaps the actor train step for
``src/rl/ogpo/update_actor.py``. v1 is strictly on-policy
(``online_ratio == 1.0``) with no success buffer and no offline data path.
"""
import dataclasses
import functools

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding
from src.rl.advantage_weighted_sft.advantage_weighted_sft_learner import (
    AdvantageWeightedSFTLearner,
)
from src.rl.ogpo.update_actor import (
    bc_grad_accumulate,
    loss_and_grad_pg,
    optimizer_tail,
    policy_param_norm,
    sample_and_advantage,
    train_step as ogpo_train_step,
)
from src.rl.prefix_embedding import PREFIX_EMBEDDING_NAME
from src.training.config import OGPOSFTLearnerConfig


# jit-4 (param_norm) cadence, in units of policy updates. 1 = every policy update (default: identical
# to today's every-update emission). Raise to sparsify (exp.py NaN-fills missing keys, EXP:159-163).
_PARAM_NORM_EVERY_N = 1

# Keys the OGPO critic batch never reads (AWR:253-263). "image" is the ~450MB/1024-row lever dropped
# before the critic sample's device_put; the policy sample keeps them (BC + rescoring forward images).
_OGPO_CRITIC_DROP_OBS_KEYS = ("image", "image_mask")


class OGPOAgentLearner(AdvantageWeightedSFTLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self._config.rl, OGPOSFTLearnerConfig), (
            "OGPOAgentLearner requires an OGPOSFTLearnerConfig."
        )
        if self._config.rl.online_ratio != 1.0:
            raise ValueError(
                "OGPO v1 requires online_ratio=1.0 (got "
                f"{self._config.rl.online_ratio}). The PPO ratio is only "
                "valid on on-policy data; the BC anchor uses the same "
                "online batch."
            )
        # Keep the trainable-only EMA on HOST between policy updates so it is OFF-DEVICE during the
        # binding loss/grad (jit-2) and optimizer-tail (jit-3) jits -- the ~11.3 GiB memory lever. It
        # is device_put back only for the two short excursions in update() and (transparently) for
        # collection. pinned_host gives fast H2D; construction crashes here if the backend rejects it
        # (a GPU-only path -- the CPU smokes never instantiate the learner). The Phase-F device-bytes
        # gate on the target GPU confirmed this device_put frees device bytes (bytes_in_use 0 on host,
        # allocated on the excursion, freed on del; nvidia-smi VRAM flat on host placement).
        self._ema_host_sharding = jax.sharding.NamedSharding(
            self._mesh, jax.sharding.PartitionSpec(), memory_kind="pinned_host"
        )
        self._ema = jax.device_put(self._ema, self._ema_host_sharding)
        # Optional success buffer: successful episodes are duplicated here and
        # the BC anchor samples from it once it holds a full batch. Same schema
        # as the online buffer (built from the same dummy data).
        self._success_data_buffer = None
        # task_description -> list of (start, end) ordinal ranges in the
        # success buffer, used by balance_success_buffer_tasks sampling.
        self._success_task_ranges = {}
        if self._config.rl.use_success_buffer:
            import dataclasses as _dc
            success_cfg = _dc.replace(
                self._config,
                rl=_dc.replace(
                    self._config.rl,
                    buffer_capacity=self._config.rl.success_buffer_capacity,
                ),
            )
            orig_config = self._config
            self._config = success_cfg
            self._success_data_buffer = self._get_online_replay_buffer()
            self._config = orig_config
        # Running scale for the EMA-quantile advantage normalizer (host-side
        # Python float, AWR-normalizer semantics: the scale used by an update
        # is the one accumulated BEFORE it). Not checkpointed — re-warms from
        # min_scale within ~1/(1-ema_weight) policy updates after a resume.
        self._adv_scale = float(self._config.rl.normalizer_config.min_scale)
        self._train_step = functools.partial(ogpo_train_step, self._config)
        self._refresh_update_functions()

    def save_episode(self, is_success: bool, env_index: int, task_description: str):
        # Mirror the AWR parent, but additionally copy successful episodes into
        # the success buffer BEFORE the parent consumes/clears episode storage.
        # _save_episode_in_buffer -> _attach_prefix_embeddings_to_episode_data
        # mutates the step dicts in place (unpacks ep["action"] and inserts the
        # prefix keys), so the success-buffer pass must operate on a structural
        # copy — step dicts and their nested obs dicts copied, arrays shared.
        if self._success_data_buffer is not None and is_success:
            episode_copy = [
                {
                    k: (dict(v) if isinstance(v, dict) else v)
                    for k, v in step.items()
                }
                for step in self._episode_storage[env_index]
            ]
            before = self._success_data_buffer.total_inserted
            self._save_episode_in_buffer(
                episode_copy,
                task_description,
                is_success=True,
                target_buffer=self._success_data_buffer,
            )
            after = self._success_data_buffer.total_inserted
            # Record which ordinal range belongs to this task, for
            # task-balanced BC sampling (host-side bookkeeping only).
            if after > before:
                self._success_task_ranges.setdefault(str(task_description), []).append(
                    (before, after)
                )
        super().save_episode(
            is_success=is_success,
            env_index=env_index,
            task_description=task_description,
        )

    def _refresh_update_functions(self):
        # Keep the shared critic jit (AWR:143) and leave the AWR mono policy jit (AWR:161) built but
        # never called (lazy -> never compiled). Siblings (AWR/MPO/FlowGRPO) are unaffected (G3).
        super()._refresh_update_functions()

        # grads sharding that crosses jit-2 -> jit-3; out(jit-2)==in(jit-3) fires the grads->updates
        # donation. filter_state over the sharding tree mirrors the params filter at UA:257.
        self._trainable_params_sharding = nnx.filter_state(
            self._train_state_sharding.params, self._config.trainable_filter
        )

        self._sampler_advantage_jitted = jax.jit(
            functools.partial(sample_and_advantage, self._config),
            in_shardings=(
                self._replicated_sharding,                  # rng (policy_rng)
                self._train_state_sharding,                 # policy_state (r/o)
                self._state_action_critic_state_sharding,   # q_state
                self._value_state_sharding,                 # value_state
                self._data_sharding,                        # policy_observation
                self._data_sharding,                        # critic_prefix (None-passthrough @ WP-A)
                self._ema_sharding,                         # ema (explicit)
            ),
            out_shardings=(
                self._replicated_sharding,                  # x_chain
                self._replicated_sharding,                  # x_next_chain
                self._replicated_sharding,                  # times
                self._replicated_sharding,                  # dt
                self._replicated_sharding,                  # old_lp
                self._replicated_sharding,                  # advantage
                self._replicated_sharding,                  # sampler_aux
            ),
            donate_argnums=(),
        )

        # jit-2a: PPO surrogate + scan grads only. No rng (the PG path reads none),
        # no actions_demo (BC-only), UNDONATED train_state. Emits grads_pg (the scan
        # accumulator), pg_loss, and the 20-key PPO aux — grads_pg then DONATES into
        # jit-2b, exactly like grads donates into jit-3.
        self._loss_grad_pg_jitted = jax.jit(
            functools.partial(loss_and_grad_pg, self._config),
            in_shardings=(
                self._train_state_sharding,                 # policy_state (r/o)
                self._data_sharding,                        # policy_observation
                self._replicated_sharding,                  # x_chain
                self._replicated_sharding,                  # x_next_chain
                self._replicated_sharding,                  # times
                self._replicated_sharding,                  # dt
                self._replicated_sharding,                  # old_lp
                self._replicated_sharding,                  # advantage
            ),
            out_shardings=(
                self._trainable_params_sharding,            # grads_pg
                self._replicated_sharding,                  # pg_loss
                self._replicated_sharding,                  # pg_aux
            ),
            donate_argnums=(),
        )

        # jit-2b: BC anchor. DONATES grads_pg (argnums 0) so the BC grad accumulation
        # (grads_pg + bc_coeff*grads_bc) aliases in-place. Same policy_rng as jit-1/2a
        # for the bc_rng derivation; pre-increment train_state for the fold_in.
        # Returns the same (grads, loss, aux) triple jit-3 / the info dict consume.
        self._bc_grad_accumulate_jitted = jax.jit(
            functools.partial(bc_grad_accumulate, self._config),
            in_shardings=(
                self._trainable_params_sharding,            # grads_pg (DONATED)
                self._replicated_sharding,                  # rng (SAME policy_rng)
                self._train_state_sharding,                 # policy_state (r/o, pre-increment)
                self._data_sharding,                        # policy_observation
                self._data_sharding,                        # actions_demo
                self._replicated_sharding,                  # pg_loss
                self._replicated_sharding,                  # pg_aux
            ),
            out_shardings=(
                self._trainable_params_sharding,            # grads (combined)
                self._replicated_sharding,                  # loss
                self._replicated_sharding,                  # aux
            ),
            donate_argnums=(0,),
        )

        self._optimizer_tail_jitted = jax.jit(
            functools.partial(optimizer_tail, self._config),
            in_shardings=(
                self._train_state_sharding,                 # policy_state (DONATED)
                self._trainable_params_sharding,            # grads (DONATED)
            ),
            out_shardings=self._train_state_sharding,       # new_state
            donate_argnums=(0, 1),
        )

        self._policy_param_norm_jitted = jax.jit(
            functools.partial(policy_param_norm, self._config),
            in_shardings=(self._train_state_sharding.params,),
            out_shardings=self._replicated_sharding,
            donate_argnums=(),
        )

    def critic_digestion_burst(self) -> dict:
        """Critic-only updates after a collection round (no actor, no step count).

        Identical single-step semantics to the update() critic branch — same
        jitted fn, same batch size, same buffer sampling, same rng stream —
        just repeated ``post_collection_critic_steps`` times back-to-back so
        the critic fits the freshly collected distribution before the next
        policy update consumes Q on it. ``training_steps`` is NOT advanced:
        the burst is invisible to every interval schedule (policy cadence,
        logging, collection), it only moves the critic optimizer + rng.
        """
        rl_config = self._config.rl
        assert isinstance(rl_config, OGPOSFTLearnerConfig)
        n = int(rl_config.post_collection_critic_steps)
        critic_batch_size = rl_config.critic.batch_size or self._config.batch_size
        if n <= 0 or self._online_data_buffer.size < critic_batch_size:
            return {}
        last_q_info, last_v_info = {}, {}
        for _ in range(n):
            batch = self._online_data_buffer.sample(
                batch_size=critic_batch_size,
                drop_obs_keys=(
                    _OGPO_CRITIC_DROP_OBS_KEYS
                    if self._config.collect.store_prefix_rep
                    else ()
                ),
            )
            critic_rng, self._rng = jax.random.split(self._rng, 2)
            with sharding.set_mesh(self._mesh):
                q_state, value_state, last_q_info, last_v_info = (
                    self._update_critics_jitted(
                        batch,
                        self._state_action_critic_state,
                        self._value_state,
                        self._train_state,
                        critic_rng,
                    )
                )
            self._state_action_critic_state = q_state
            self._value_state = value_state
        info = (
            {f"burst/q_{k}": v for k, v in last_q_info.items()}
            | {f"burst/value_{k}": v for k, v in last_v_info.items()}
            | {"burst/steps": jnp.asarray(float(n), dtype=jnp.float32)}
        )
        return jax.tree.map(np.asarray, info)

    @at.typecheck
    def update(self) -> dict:
        rl_config = self._config.rl
        assert isinstance(rl_config, OGPOSFTLearnerConfig)

        self._maybe_reset_critic_optimizers()

        self.training_steps += 1
        update_critic = (
            self.training_steps >= rl_config.critic.training_start_step
            and self.training_steps % rl_config.critic.update_interval == 0
        )
        update_policy = (
            self.training_steps >= rl_config.policy.training_start_step
            and self.training_steps % rl_config.policy.update_interval == 0
        )

        if not update_critic and not update_policy:
            return {
                "online_buffer_size": jnp.asarray(
                    float(self._online_data_buffer.size), dtype=jnp.float32
                )
            }

        # online_ratio is pinned to 1.0 (asserted at __init__).
        online_batch_size = self._config.batch_size
        if self._online_data_buffer.size < online_batch_size:
            return {
                "online_buffer_size": jnp.asarray(
                    float(self._online_data_buffer.size), dtype=jnp.float32
                )
            }

        # Critics are small MLPs so a larger batch than the policy is cheap and
        # improves TD stability. Sampled independently so the TD update is not
        # computed on the same data as the PPO update.
        critic_batch_size = rl_config.critic.batch_size or online_batch_size
        # With the stored prefix present the critic reads it directly (AWR:261-263) and never touches
        # images, so drop them off-device. Without it the critic recompute (AWR:265-269) needs
        # data["image"] (MODEL:116), so the gate keeps images when store_prefix_rep is off.
        critic_online_batch = (
            self._online_data_buffer.sample(
                batch_size=critic_batch_size,
                drop_obs_keys=(
                    _OGPO_CRITIC_DROP_OBS_KEYS
                    if self._config.collect.store_prefix_rep
                    else ()
                ),
            )
            if update_critic
            else None
        )
        # Sample the policy batch ONLY on policy-update steps: 9 of 10 steps are
        # critic-only, and this gather moves ~10s of MB of images host->device
        # for nothing (S4, docs/ogpo_speed_memory_analysis.md). Pure overhead
        # removal; the batch content on policy steps is unchanged (the buffer
        # rng is consumed in the same order relative to policy updates as long
        # as critic sampling above stays unconditional-on-its-own-gate).
        online_batch = (
            self._online_data_buffer.sample(batch_size=online_batch_size)
            if update_policy
            else None
        )

        critic_info, actor_info = {}, {}
        if update_critic:
            critic_rng, self._rng = jax.random.split(self._rng, 2)
            with sharding.set_mesh(self._mesh):
                q_state, value_state, q_info, value_info = (
                    self._update_critics_jitted(
                        critic_online_batch,
                        self._state_action_critic_state,
                        self._value_state,
                        self._train_state,
                        critic_rng,
                    )
                )
            self._state_action_critic_state = q_state
            self._value_state = value_state
            critic_info = (
                {f"critic/q_{k}": v for k, v in q_info.items()}
                | {f"critic/value_{k}": v for k, v in value_info.items()}
            )

        if update_policy:
            # Micro-batch gradient accumulation (policy_grad_accum = M): the whole
            # jit-1 -> jit-2a -> jit-2b chain runs M times on M independently
            # sampled batches (fresh success-BC batch each time), grads averaged,
            # optimizer applied ONCE. Peak memory per micro-batch is unchanged;
            # for M > 1 the accumulated fp32 trainable grad tree is additionally
            # co-resident with the next micro-batch's scan accumulator — fine for
            # frozen-backbone runs (~1 GiB), NOT sized for unfrozen ones.
            # PPO correctness: all M micro-batches score against the SAME old
            # policy (the EMA is only advanced after the optimizer apply below).
            num_micro = max(1, int(rl_config.policy_grad_accum))
            grads_acc = None
            loss_list: list = []
            aux_list: list = []
            with sharding.set_mesh(self._mesh):
                for micro_idx in range(num_micro):
                    if micro_idx > 0:
                        online_batch = self._online_data_buffer.sample(
                            batch_size=online_batch_size
                        )
                    # OGPO 3-tuple override (below): critic_prefix is the buffer's stored EMA-computed prefix
                    # rep, threaded past Observation.from_dict. It is None when store_prefix_rep is off, so
                    # jit-1 recomputes under current params (None -> array is a deliberate recompile). The base
                    # 2-tuple (FSL:572) is untouched for siblings (G3).
                    policy_observation, actions_demo, critic_prefix = self._online_batch_to_sft_batch(online_batch)
                    # Success-buffer BC: once the success buffer holds a full batch, the
                    # BC anchor regresses onto successful trajectories instead of the
                    # (mostly failed) online batch. The PPO path is untouched — only the
                    # (observation, actions) pair fed to jit-2b changes.
                    bc_observation, bc_actions = policy_observation, actions_demo
                    if (
                        self._success_data_buffer is not None
                        and self._success_data_buffer.size >= online_batch_size
                    ):
                        ordinals = None
                        if rl_config.balance_success_buffer_tasks:
                            ordinals = self._balanced_success_ordinals(online_batch_size)
                        success_batch = self._success_data_buffer.sample(
                            batch_size=online_batch_size,
                            ordinals=ordinals,
                        )
                        bc_observation, bc_actions, _ = self._online_batch_to_sft_batch(
                            success_batch
                        )
                    policy_rng, self._rng = jax.random.split(self._rng, 2)
                    # G2: a device copy of the host EMA enters ONLY the undonated jit-1, and self._ema is the
                    # donated arg of _ema_update_fn on its final use (below); it is NEVER attached to a donated
                    # policy-jit argument, so the step-900 use-after-donate crash is structurally unreachable --
                    # the former ema_params attach + read-back dance is deleted, not reordered.
                    # Excursion 1: EMA on device for the sampler jit-1 ONLY. device_put does NOT consume
                    # self._ema, so the host copy stays valid.
                    ema_dev = jax.device_put(self._ema, self._ema_sharding)
                    (x_chain, x_next_chain, times, dt, old_lp, advantage,
                     sampler_aux) = self._sampler_advantage_jitted(
                        policy_rng,
                        self._train_state,
                        self._state_action_critic_state,
                        self._value_state,
                        policy_observation,
                        critic_prefix,
                        ema_dev,                         # r/o (jit-1 donate_argnums=())
                    )
                    # LOAD-BEARING: release the 11.3 GiB device EMA BEFORE jit-2 so it is off-device
                    # through the binding jit-2/jit-3. A held handle would pin it (JAX does not offload a
                    # referenced buffer across executables) and the floor stays ~46, not ~34.7 GiB.
                    del ema_dev
                    # --- Advantage post-processing (eager ops on the [B*G] vector at
                    # the jit-1 -> jit-2a boundary; advantage is data to jit-2a, so
                    # no gradient concerns). Normalizer semantics mirror the AWR
                    # normalizer: each update divides by the EMA scale accumulated
                    # BEFORE it, then folds its own (q95-q05) spread into the EMA.
                    if rl_config.normalize_group_advantage:
                        ncfg = rl_config.normalizer_config
                        scale_used = self._adv_scale
                        advantage = advantage / scale_used
                        spread = jnp.maximum(
                            sampler_aux["advantage_q_up"] - sampler_aux["advantage_q_low"],
                            ncfg.min_scale,
                        )
                        self._adv_scale = (
                            ncfg.ema_weight * self._adv_scale
                            + (1.0 - ncfg.ema_weight) * spread
                        )
                        sampler_aux = sampler_aux | {
                            "adv_scale": jnp.asarray(scale_used, dtype=jnp.float32)
                        }
                    if rl_config.adv_clip_sym is not None:
                        clip_c = rl_config.adv_clip_sym
                        sampler_aux = sampler_aux | {
                            "adv_clip_sym_frac": jnp.mean(
                                (jnp.abs(advantage) >= clip_c).astype(jnp.float32)
                            )
                        }
                        advantage = jnp.clip(advantage, -clip_c, clip_c)
                    if rl_config.normalize_group_advantage or rl_config.adv_clip_sym is not None:
                        sampler_aux = sampler_aux | {"advantage_std_final": jnp.std(advantage)}
                    # jit-2a: PPO scan grads (no BC). grads_pg is the donated accumulator
                    # into jit-2b; its scan activations are freed before the BC backward.
                    grads_pg, pg_loss, pg_aux = self._loss_grad_pg_jitted(
                        self._train_state,               # r/o (undonated)
                        policy_observation,
                        x_chain, x_next_chain, times, dt, old_lp, advantage,
                    )
                    # jit-2b: BC anchor, grads accumulated INTO the donated grads_pg. SAME
                    # policy_rng (re-split arity-3, bc_rng=[2]); pre-increment step for the
                    # fold_in (jit-3 increments). grads_pg is consumed here (donated), so it
                    # must not be referenced afterward.
                    grads, loss, loss_aux = self._bc_grad_accumulate_jitted(
                        grads_pg,                        # DONATED accumulator
                        policy_rng,
                        self._train_state,               # pre-increment step for fold_in
                        bc_observation,                  # success-buffer batch when available
                        bc_actions,
                        pg_loss, pg_aux,
                    )
                    if grads_acc is None:
                        grads_acc = grads      # M=1: identical object flow to the pre-accum code
                    else:
                        grads_acc = jax.tree.map(jnp.add, grads_acc, grads)
                    loss_list.append(loss)
                    aux_list.append(loss_aux | sampler_aux)
                if num_micro > 1:
                    grads_acc = jax.tree.map(lambda g: g / num_micro, grads_acc)
                new_state = self._optimizer_tail_jitted(self._train_state, grads_acc)   # DONATES train_state+grads
            if num_micro == 1:
                loss, merged_aux = loss_list[0], aux_list[0]
            else:
                loss = sum(loss_list) / num_micro
                merged_aux = jax.tree.map(
                    lambda *xs: sum(xs) / float(num_micro), *aux_list
                )
            self._train_state = new_state
            # Barrier so jit-3 finishes before excursion 2's H2D allocates: excursion 2's device_put is
            # dispatched right after the jit-3 call returns (async), so without this the eager H2D target
            # buffer coexists with jit-3's peak and inflates the measured floor.
            jax.block_until_ready(new_state)
            # Excursion 2: EMA on device ONLY for its own update, AFTER jit-3. A SECOND device_put (host
            # copy still valid). ema_dev2's FINAL use is the donating _ema_update_fn (donate_argnums=0)
            # -> no read follows -> G2 holds; jit-3 received no EMA. Both operands are trainable-only.
            ema_dev2 = jax.device_put(self._ema, self._ema_sharding)
            new_ema_dev = self._ema_update_fn(
                ema_dev2,
                nnx.filter_state(new_state.params, self._config.trainable_filter),
            )
            self._ema = jax.device_put(new_ema_dev, self._ema_host_sharding)   # D2H
            del new_ema_dev   # release the device EMA so it is not pinned through jit-4 (param_norm)
            # param_norm runs in its OWN jit AFTER jit-3 so the param tree's live range never re-enters the
            # binding tail (that separation is the memory win, not the cadence). N=1 emits every policy
            # update, identical to today. Do NOT gate on log_interval: it lives on the outer experiment
            # config (exp.py), not on the learner's OnlineTrainConfig.
            actor_info = {"loss": loss} | merged_aux
            if self.training_steps % (rl_config.policy.update_interval * _PARAM_NORM_EVERY_N) == 0:
                actor_info["param_norm"] = self._policy_param_norm_jitted(self._train_state.params)
            actor_info = {f"actor/{k}": v for k, v in actor_info.items()}

        info = (
            actor_info
            | critic_info
            | {
                "online_buffer_size": jnp.asarray(
                    float(self._online_data_buffer.size), dtype=jnp.float32
                )
            }
        )
        if self._success_data_buffer is not None:
            info["success_buffer_size"] = jnp.asarray(
                float(self._success_data_buffer.size), dtype=jnp.float32
            )
        # Return DEVICE arrays (S3, docs/ogpo_speed_memory_analysis.md): the old
        # jax.tree.map(np.asarray, ...) forced a host sync on ~50 scalars EVERY
        # step, stalling async dispatch of the next step's jits. exp.py already
        # normalizes + jax.device_get()s at log_interval, so materializing here
        # is pure overhead. Values are identical, just fetched lazily.
        return info

    def _balanced_success_ordinals(self, batch_size: int) -> "np.ndarray | None":
        """Ordinals for a task-balanced sample of the success buffer.

        Splits the batch equally across tasks that have live (non-evicted)
        success transitions, sampling uniformly within each task's ordinal
        ranges. Falls back to None (uniform sampling) if fewer than 2 tasks
        have live data. Remainder slots go to the largest task pools.
        """
        buf = self._success_data_buffer
        live = {}
        for task, ranges in self._success_task_ranges.items():
            ords = []
            for lo, hi in ranges:
                lo2 = max(lo, buf.valid_start)
                if hi > lo2:
                    ords.append((lo2, hi))
            n = sum(hi - lo for lo, hi in ords)
            if n > 0:
                live[task] = (ords, n)
        if len(live) < 2:
            return None
        tasks = sorted(live, key=lambda t: -live[t][1])
        per = batch_size // len(tasks)
        counts = {t: per for t in tasks}
        for i in range(batch_size - per * len(tasks)):
            counts[tasks[i % len(tasks)]] += 1
        out = []
        rng = buf._rng
        for t in tasks:
            ords, n = live[t]
            flat = rng.integers(0, n, size=counts[t])
            # map flat indices into the task's (possibly multiple) ranges
            spans = np.array([hi - lo for lo, hi in ords])
            starts = np.array([lo for lo, _ in ords])
            cum = np.cumsum(spans)
            seg = np.searchsorted(cum, flat, side="right")
            offset = flat - np.where(seg > 0, cum[seg - 1], 0)
            out.append(starts[seg] + offset)
        return np.concatenate(out)

    def _online_batch_to_sft_batch(
        self, online_batch: dict
    ) -> tuple[_model.Observation, _model.Actions, at.Float[at.Array, "b embed"] | None]:
        # The base (FSL:572-578) returns only (Observation, actions); Observation.from_dict
        # (MODEL:110-129) drops the stored "prefix_embedding" key. OGPO threads that stored EMA-computed
        # rep as a third sidecar element so jit-1 can skip the current-params PaliGemma recompute. The
        # base 2-tuple stays intact for the siblings (G3); only OGL.update() (OGPO-owned) reads this
        # 3-tuple. next_observation prefix is NOT needed: the actor path scores Q/V on the CURRENT state
        # only (next-obs prefix is a critic-path concern, AWR:263).
        observation = _model.Observation.from_dict(online_batch["observation"])
        actions = online_batch["actions"]
        stored_prefix = online_batch["observation"].get(PREFIX_EMBEDDING_NAME)
        return observation, actions, stored_prefix
