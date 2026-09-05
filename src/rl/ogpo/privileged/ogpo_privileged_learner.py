# ruff: noqa: F722
"""OGPO with a privileged, per-task critic — the critic-capacity upper bound.

Motivation. The multi-task OGPO critic sees a mean-pooled PaliGemma prefix plus
the robot's proprioceptive state, and it is ONE critic shared across every
training task. Both choices bound how well it can rank actions, and the
advantage the actor consumes is nothing but that ranking. This learner removes
both bounds, using information a real robot would not have, to measure what they
cost:

  1. State — the simulator's own state (robot proprioception + object poses),
             taken straight from the LIBERO observation, instead of the pooled
             VLM prefix. See ``src/rl/privileged_state.py``.
  2. Heads — one independent critic per training task, selected per sample, so
             no capacity is spent separating tasks.

The TD target is the baseline's: Q backs up through V(s'), V is fit to
Q(s, a_buffer), via the SHARED ``advantage_weighted_sft/update_critic`` steps —
not a copy of them. ``privileged_backup="next_action_q"`` swaps in a Q(s', a')
backup on the action the policy took at s' (see ``update_critic.py`` here), but
that is an opt-in ablation, not the default: the point of the default arm is
that only the critic's INPUTS and PARAMETERIZATION differ from the baseline.

The ACTOR is untouched throughout: same PPO surrogate, same BC anchor, same
group-relative advantage, same sampling. So a paired run against
``scripts/ogpo_multitask_4task.sh`` attributes any difference to the critic.

Everything here is an override of ``OGPOAgentLearner``; nothing in the default
stack changes.
"""
import dataclasses
import functools
import logging
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
from src.rl.advantage_weighted_sft.advantage_weighted_sft_learner import CriticSpec
from src.rl.networks.rl_networks import ObsType
from src.rl.networks.task_ensemble_critic import (
    TASK_ID_KEY,
    make_task_ensemble_defs,
    per_task_optimizer,
)
from src.rl.ogpo.ogpo_learner import OGPOAgentLearner, _OGPO_CRITIC_DROP_OBS_KEYS
from src.rl.ogpo.privileged.update_critic import (
    train_q_step as privileged_train_q_step,
    train_value_step as privileged_train_value_step,
)
from src.rl.privileged_state import (
    PRIVILEGED_STATE_NAME,
    PRIVILEGED_TASK_ID_NAME,
    privileged_task_ids,
)
from src.training.config import OGPOPrivilegedLearnerConfig


class OGPOPrivilegedLearner(OGPOAgentLearner):
    # The critic's two observation columns are read by the CRITIC, not the
    # policy, so the repack transform in _save_episode_in_buffer would drop
    # them; route them around it.
    _buffer_obs_passthrough_keys = (PRIVILEGED_STATE_NAME, PRIVILEGED_TASK_ID_NAME)

    @property
    def _uses_next_action_backup(self) -> bool:
        """True iff the Q target bootstraps on the policy's action at s'.

        Gates three things together — they must agree or the buffer schema and
        the critic batch drift apart: the `next_actions` buffer column, its
        per-episode write, and the 7-element critic batch the privileged train
        steps unpack.
        """
        return self._config.rl.privileged_backup == "next_action_q"

    def _build_critic_spec(self, config) -> CriticSpec:
        """Per-task BroNet ensembles over the privileged state.

        Runs before ``super().__init__``, so it reads ``config`` only — and
        unlike the base hook it never instantiates the pi0 model, because the
        privileged critic's input width does not depend on the prefix.
        """
        rl = config.rl
        if not isinstance(rl, OGPOPrivilegedLearnerConfig):
            raise TypeError(
                "OGPOPrivilegedLearner requires an OGPOPrivilegedLearnerConfig, got "
                f"{type(rl).__name__}."
            )
        if not config.collect.store_privileged_state:
            raise ValueError(
                "OGPOPrivilegedLearner needs collect.store_privileged_state=True — "
                "without it the buffer carries no privileged state for the critic to "
                "read. Set --collect.store_privileged_state."
            )
        if config.collect.domain != "libero":
            raise NotImplementedError(
                "Privileged state extraction is implemented for the libero domain "
                f"only (got {config.collect.domain!r})."
            )
        if config.collect.store_prefix_rep:
            raise ValueError(
                "collect.store_prefix_rep must be off for the privileged critic: it "
                "reads no prefix rep, so the buffer allocates no prefix column and "
                "the episode insert would fail its schema check. Drop "
                "--collect.store_prefix_rep (it also saves the per-step prefix "
                "forward during collection)."
            )
        if not rl.critic.use_bronet:
            raise ValueError(
                "The privileged critic is built from BroNet blocks; set "
                "--rl.critic.use_bronet (the pi0-backbone critic defs take the "
                "policy observation, which this critic does not use)."
            )
        if rl.n_samples > 1:
            # Best-of-N collection scores candidate actions with the critic from
            # inside sample_actions, which assembles a prefix-embedding critic
            # observation the privileged critic cannot read.
            raise ValueError(
                "Best-of-N collection (rl.n_samples > 1) is not supported with the "
                "privileged critic: the collection-time scoring path builds a "
                "prefix-embedding critic observation. Set --rl.n_samples 1."
            )

        # Set here, not in __init__: _make_buffer_dummy_data reads them from
        # inside FilteredSFTLearner.__init__, which this hook precedes.
        self._privileged_state_dim = int(config.collect.privileged_state_dim)
        self._privileged_tasks = privileged_task_ids(config.collect.tasks)
        state_dim = self._privileged_state_dim
        num_tasks = len(self._privileged_tasks)
        dummy_obs = {
            "state": jnp.zeros((1, state_dim), dtype=jnp.float32),
            TASK_ID_KEY: jnp.zeros((1, 1), dtype=jnp.float32),
        }
        state_action_critic_def, state_value_def = make_task_ensemble_defs(
            hidden_dim=rl.critic.bronet_hidden_dim,
            depth=rl.critic.bronet_depth,
            num_qs=rl.critic.num_qs,
            num_vs=rl.critic.num_vs,
            num_tasks=num_tasks,
            num_bins=rl.critic.num_value_bins,
        )
        return CriticSpec(
            dummy_obs=dummy_obs,
            dummy_act=config.model.fake_act(batch_size=1),
            state_action_critic_def=state_action_critic_def,
            state_value_def=state_value_def,
            # The privileged critic never reads a prefix rep, so the buffer gets
            # no prefix column even if collect.store_prefix_rep is on.
            prefix_embed_dim=None,
            transition_state_dim=int(
                config.model.fake_obs(batch_size=1).state.shape[-1]
            ),
            # Per-task gradient clip: the T critics share one parameter tree, so
            # the config's single global clip would let one task's gradient
            # magnitude throttle every other task's update — coupling the heads
            # back together through the optimizer after the forward went to the
            # trouble of keeping them disjoint.
            critic_tx=per_task_optimizer(
                rl.critic.optimizer, rl.critic.lr_schedule, num_tasks
            ),
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = self._config
        assert isinstance(config.rl, OGPOPrivilegedLearnerConfig)
        logging.info(
            "Privileged critic: %d per-task heads (%s), state_dim=%d, backup=%s",
            len(self._privileged_tasks),
            ", ".join(self._privileged_tasks),
            self._privileged_state_dim,
            config.rl.privileged_backup,
        )

        if self._uses_next_action_backup:
            # Swap in the 7-element-batch critic steps, then rebuild the critic
            # jit around them (jax.jit is lazy, so nothing was compiled against
            # the inherited steps). On the "value" default the inherited AWR
            # steps stay in place — the privileged arm then runs the SAME
            # numeric critic code as the baseline, on a different observation.
            self._q_train_step = functools.partial(privileged_train_q_step, config)
            self._value_train_step = functools.partial(
                privileged_train_value_step, config
            )
            self._refresh_update_functions()

    # ---------------------------------------------------------------- buffer

    def _make_buffer_dummy_data(self) -> dict:
        dummy = super()._make_buffer_dummy_data()
        dummy["observations"][PRIVILEGED_STATE_NAME] = np.zeros(
            (1, self._privileged_state_dim), dtype=np.float32
        )
        dummy["observations"][PRIVILEGED_TASK_ID_NAME] = np.zeros(
            (1, 1), dtype=np.float32
        )
        if self._uses_next_action_backup:
            # The action the policy took at the NEXT observation, for that
            # backup only: one action chunk per transition, ~640 MB at the
            # 500k-transition capacity, so the "value" default does not pay it.
            dummy["next_actions"] = np.zeros_like(dummy["actions"])
        return dummy

    def _extra_transition_fields(
        self, *, actions_out: np.ndarray, n_windows: int, act_h: int
    ) -> dict[str, np.ndarray]:
        """``next_actions[i]`` = the transformed action window at obs ``i + act_h``.

        A transition's next observation is ``obs_index + act_h`` by construction
        (filtered_sft_learner), and ``actions_out`` is the transformed action
        tensor over ``n_windows + act_h`` windows — the tail being the last real
        window repeated — so this slice is always in range. Windows whose next
        observation runs past the episode fall back to that repeated tail; they
        are terminal or near-terminal, and terminal transitions carry
        ``discount == 0``, which multiplies the bootstrap away entirely.
        """
        if not self._uses_next_action_backup:
            return {}
        next_actions = np.asarray(actions_out)[act_h : act_h + n_windows]
        return {"next_actions": next_actions.astype(np.float32)}

    # ---------------------------------------------------------------- critic

    def _critic_drop_obs_keys(self) -> tuple[str, ...]:
        # The privileged critic reads neither images nor a prefix rep, so the
        # image tensors are dropped unconditionally (the base gate keeps them
        # when store_prefix_rep is off, for the prefix recompute this critic
        # never performs).
        return _OGPO_CRITIC_DROP_OBS_KEYS

    def _privileged_critic_obs(self, observation: dict[str, Any]) -> ObsType:
        """Map buffer observation columns onto the critic's own obs schema."""
        return {
            "state": jnp.asarray(observation[PRIVILEGED_STATE_NAME], dtype=jnp.float32),
            TASK_ID_KEY: jnp.asarray(
                observation[PRIVILEGED_TASK_ID_NAME], dtype=jnp.float32
            ),
        }

    def _online_batch_to_critic_batch(self, online_batch: dict[str, Any], policy_state):
        """Critic batch over the privileged observation.

        On the "value" default this is the SHARED 6-element ``CriticBatch``, so
        the inherited AWR train steps consume it unchanged — the privileged arm
        and the baseline arm run the same critic code. The "next_action_q"
        ablation appends the stored next action, giving the 7-element batch the
        privileged train steps unpack.
        """
        # No prefix recompute: the critic's state comes from the buffer, so the
        # policy state is unused here (the base signature keeps it for the
        # shared _update_critics call site).
        del policy_state
        observation = self._privileged_critic_obs(online_batch["observation"])
        next_observation = self._privileged_critic_obs(online_batch["next_observation"])
        tail = (
            online_batch["reward"],
            online_batch["discount"],
            online_batch["mc_return"],
        )
        if self._uses_next_action_backup:
            return (
                observation,
                online_batch["actions"],
                next_observation,
                online_batch["next_actions"],
                *tail,
            )
        return (observation, online_batch["actions"], next_observation, *tail)

    def _burst_critic_update_fn(self):
        """Burst jit built from the PRIVILEGED critic steps.

        Only reached under the ``next_action_q`` ablation. The base
        implementation's MC-target variant builds a SECOND jit from the AWR
        steps, which would crash on the 7-element batch; on the "value" default
        the batch is the shared 6-element one and the base is already correct.
        """
        if not self._uses_next_action_backup:
            return super()._burst_critic_update_fn()
        rl_config = self._config.rl
        if not rl_config.burst_use_mc_targets:
            return self._update_critics_jitted
        if getattr(self, "_burst_mc_critics_jitted", None) is None:
            from src.training.config import StepSchedule

            mc_cfg = dataclasses.replace(
                self._config,
                rl=dataclasses.replace(
                    rl_config,
                    critic=dataclasses.replace(
                        rl_config.critic,
                        td_weight_schedule=StepSchedule(
                            init_value=0.0, end_value=0.0, switch_step=1
                        ),
                    ),
                ),
            )
            _q_step = functools.partial(privileged_train_q_step, mc_cfg)
            _v_step = functools.partial(privileged_train_value_step, mc_cfg)

            def _mc_wrapper(batch, q_state, value_state, policy_state, rng):
                batch = self._online_batch_to_critic_batch(batch, policy_state)
                q_rng, v_rng, rng = jax.random.split(rng, 3)
                q_state, q_info = _q_step(q_rng, q_state, value_state, batch)
                value_state, value_info = _v_step(v_rng, value_state, q_state, batch)
                return q_state, value_state, q_info, value_info

            self._burst_mc_critics_jitted = jax.jit(
                _mc_wrapper,
                in_shardings=(
                    self._data_sharding,
                    self._state_action_critic_state_sharding,
                    self._value_state_sharding,
                    self._train_state_sharding,
                    self._replicated_sharding,
                ),
                out_shardings=(
                    self._state_action_critic_state_sharding,
                    self._value_state_sharding,
                    self._replicated_sharding,
                    self._replicated_sharding,
                ),
                donate_argnums=(1, 2),
            )
        return self._burst_mc_critics_jitted

    # ----------------------------------------------------------------- actor

    def _online_batch_to_sft_batch(
        self, online_batch: dict
    ) -> tuple[_model.Observation, _model.Actions, ObsType]:
        """Third element is a COMPLETE critic observation, not a prefix vector.

        ``sample_and_advantage`` uses a dict sidecar verbatim as the critic
        observation (update_actor.py), which is what the privileged critic
        needs: its state is the simulator's, so nothing about it can be derived
        from the policy observation.
        """
        return (
            _model.Observation.from_dict(online_batch["observation"]),
            online_batch["actions"],
            self._privileged_critic_obs(online_batch["observation"]),
        )
