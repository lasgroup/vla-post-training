"""FlowAWR learner.

AWR critic + exp(A/beta/scale) advantage weighting on buffer actions,
+ per-flow-step PPO surrogate over the buffer-stored noise trajectory. 
"""
import gc
import functools
from typing import Any, Dict

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.transforms as _transforms

from src.rl.flow_awr.update_actor import train_step as flow_awr_train_step
from src.rl.mpo_weighted_sft.mpo_weighted_sft_learner import MPOWeightedSFTLearner
from src.rl.prefix_embedding import PREFIX_EMBEDDING_NAME
from src.rl.replay_buffer import ShardedReplayBuffer
from src.rl.types import StepData
from src.training.config import FlowAWRSFTLearnerConfig


class FlowAWRLearner(MPOWeightedSFTLearner):
    def _policy_mc_return_sharding(self):
        return self._replicated_sharding

    def __init__(self, *args, **kwargs):
        self._latest_rollout_info: Dict[str, np.ndarray] | None = None
        self._latest_action_chunk: np.ndarray | None = None
        # `_collecting` toggles between collection-time noise_level (rl_config
        # value) and eval-time noise_level=0. Mirrors MPOLearner's pattern.
        self._collecting: bool = False

        super().__init__(*args, **kwargs)

        rl_config = self._config.rl
        assert isinstance(rl_config, FlowAWRSFTLearnerConfig)
        if rl_config.online_ratio < 1.0:
            raise ValueError(
                "FlowAWRLearner requires online_ratio=1.0 because offline data "
                "does not carry rollout_info (the noise trajectory is only "
                "produced by the policy at collection time)."
            )

        self._policy._sample_kwargs["num_steps"] = rl_config.num_steps
        self._collection_noise_level = rl_config.noise_level

        # Drop AWR's policy JIT and rebuild with the FlowAWR train_step
        # (whose batch is a 4-tuple including rollout_info).
        del self._update_policy_jitted
        gc.collect()

        self._train_step = functools.partial(flow_awr_train_step, self._config)

        def _policy_wrapper(batch, policy_state, q_state, value_state, rng, mc_return, scale):
            return self._update_policy(
                batch=batch,
                policy_state=policy_state,
                q_state=q_state,
                value_state=value_state,
                rng=rng,
                mc_return=mc_return,
                scale=scale,
            )

        self._update_policy_jitted = jax.jit(
            _policy_wrapper,
            in_shardings=(
                self._data_sharding,                         # batch (4-tuple)
                self._train_state_sharding,                  # policy_state
                self._state_action_critic_state_sharding,    # q_state
                self._value_state_sharding,                  # value_state
                self._replicated_sharding,                   # rng
                self._replicated_sharding,                   # mc_return (None)
                self._replicated_sharding,                   # scale
            ),
            out_shardings=(
                self._train_state_sharding,
                self._replicated_sharding,
            ),
            donate_argnums=(1,),
        )

    # ---------- action substitution policy ----------

    def _replace_buffer_actions_with_policy_actions(self, critic_update: bool = True) -> bool:
        return False

    # ---------- collection-side: capture rollout_info ----------

    @at.typecheck
    def _sample_action(
        self,
        observations: Dict,
        rng: jax.random.PRNGKey,
        train_state: training_utils.TrainState,
        return_prefix_rep: bool = False,
    ):
        """Sample actions and stash rollout_info + the env-space full chunk
        for the next add_data call."""
        rl_config = self._config.rl
        assert isinstance(rl_config, FlowAWRSFTLearnerConfig)

        # Respect use_ema_for_sampling. Falls back to current params if EMA
        # isn't being kept, matching how MPOLearner / flow_mpo handle this.
        use_ema = (
            rl_config.use_ema_for_sampling and train_state.ema_params is not None
        )
        params = train_state.ema_params if use_ema else train_state.params
        model = nnx.merge(train_state.model_def, params)
        model.eval()
        first_obs = next(iter(observations.values()), None)
        if first_obs is None:
            raise ValueError("Observation dictionary is empty.")
        first_obs = np.asarray(first_obs)
        batch_size = first_obs.shape[0] if first_obs.ndim > 1 else 1
        noise = jax.random.normal(
            rng, (batch_size, self._policy.action_horizon, self._policy.action_dim)
        )
        num_devices = len(jax.devices())
        sharding_spec = (
            self._policy_sharding_spec if batch_size % num_devices == 0 else None
        )

        nl = self._collection_noise_level if self._collecting else 0.0
        result = self._policy.infer_with_model(
            model=model,
            obs=observations,
            noise=noise,
            noise_level=nl,
            return_info_dict=True,
            return_prefix_rep=return_prefix_rep,
            sharding_spec=sharding_spec,
        )
        actions = result["actions"]
        rollout_info = result.get("rollout_info")
        prefix_rep = result.get("prefix_rep")

        if batch_size == 1 and actions.ndim == 2:
            actions = actions[np.newaxis, ...]

        self._latest_rollout_info = rollout_info
        self._latest_action_chunk = (
            np.asarray(actions, dtype=np.float32) if self._collecting else None
        )

        if return_prefix_rep:
            if prefix_rep is None:
                raise ValueError(
                    "infer_with_model did not return a prefix_rep despite "
                    "return_prefix_rep=True."
                )
            if batch_size == 1 and prefix_rep.ndim < 3:
                prefix_rep = prefix_rep[np.newaxis, ...]
            return (actions, prefix_rep)
        return actions

    def sample_actions(self, observations, **kwargs):
        # Set _collecting=True so _sample_action uses the collection
        # noise_level and stashes the env-space chunk for add_data.
        self._collecting = True
        try:
            return super().sample_actions(observations, **kwargs)
        finally:
            self._collecting = False

    def eval_actions(self, observations, **kwargs):
        # Eval-time samples should be deterministic and don't need rollout_info.
        self._collecting = False
        return super().eval_actions(observations, **kwargs)

    def add_data(self, step_data: StepData):
        super().add_data(step_data)
        rollout_info = self._latest_rollout_info
        action_chunk = self._latest_action_chunk
        if rollout_info is None:
            return
        env_num = self._config.collect.env_num
        for i in range(env_num):
            # rollout_info leaves: (num_steps, env_num, ...) for x/x_next/log_prob
            # and (num_steps, env_num) for time. Slice axis 1 -> (num_steps, ...).
            ep_rollout = jax.tree_util.tree_map(
                lambda x: np.asarray(x[:, i]), rollout_info
            )
            self._episode_storage[i][-1]["rollout_info"] = ep_rollout
            if action_chunk is not None:
                # action_chunk shape: (env_num, action_horizon, action_dim).
                # Stash per-env so _save_episode_in_buffer can use the FULL
                # sampled chunk (env-space) instead of the trimmed and
                # potentially-mixed-origin step_data["action"].
                self._episode_storage[i][-1]["full_action_chunk"] = np.asarray(
                    action_chunk[i], dtype=np.float32
                )

    def _get_online_replay_buffer(self) -> ShardedReplayBuffer:
        # Mirror parent's transform setup so _save_episode_in_buffer can run.
        data_config = self._data_loader.data_config()
        tt_types = (_transforms.TokenizePrompt, _transforms.TokenizeFASTInputs)
        token_transforms = [
            t for t in data_config.model_transforms.inputs if isinstance(t, tt_types)
        ]
        non_token_transforms = [
            t
            for t in data_config.model_transforms.inputs
            if not isinstance(t, tt_types)
        ]
        assert len(token_transforms) == 1, (
            f"Expected exactly one token transform, found {len(token_transforms)}"
        )
        self._token_transform = token_transforms[0]
        self._pre_token_transform = _transforms.compose(
            [
                *data_config.repack_transforms.inputs,
                *data_config.data_transforms.inputs,
                _transforms.Normalize(
                    data_config.norm_stats, use_quantiles=data_config.use_quantile_norm
                ),
                *non_token_transforms,
            ]
        )
        self._token_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        rl_config = self._config.rl
        assert isinstance(rl_config, FlowAWRSFTLearnerConfig)

        obs_spec, act_spec = self._config.model.inputs_spec(batch_size=1)
        obs_spec_dict = obs_spec.to_dict()
        dummy_obs_dict = jax.tree.map(
            lambda spec: np.zeros(spec.shape, dtype=spec.dtype), obs_spec_dict
        )
        dummy_obs_dict = {k: v for k, v in dummy_obs_dict.items() if v is not None}
        if "image" in dummy_obs_dict:
            dummy_obs_dict["image"] = jax.tree.map(
                lambda v: v.astype(np.uint8), dummy_obs_dict["image"]
            )

        num_steps = rl_config.num_steps
        action_horizon = self._config.model.action_horizon
        action_dim = self._config.model.action_dim

        dummy_data = {
            "observation": dummy_obs_dict,
            "actions": np.zeros(act_spec.shape, dtype=act_spec.dtype),
            "next_observation": dummy_obs_dict,
            "reward": np.zeros((1,), dtype=np.float32),
            "mc_return": np.zeros((1,), dtype=np.float32),
            "discount": np.zeros((1,), dtype=np.float32),
            # Per-row noise trajectory. Leading "1" is the batch dim that
            # ShardedReplayBuffer uses for pre-allocation.
            "rollout_info": {
                "x":        np.zeros((1, num_steps, action_horizon, action_dim), dtype=np.float32),
                "x_next":   np.zeros((1, num_steps, action_horizon, action_dim), dtype=np.float32),
                "time":     np.zeros((1, num_steps), dtype=np.float32),
                "log_prob": np.zeros((1, num_steps, action_horizon), dtype=np.float32),
            },
        }

        return ShardedReplayBuffer(
            dummy_data=dummy_data,
            max_capacity=self._config.rl.buffer_capacity,
            data_sharding=self._data_sharding,
            seed=self._config.seed,
            preprocess_fn=None,
            postprocess_fn=None,
            freeze_dict=False,
        )

    # ---------- per-sample windowing on save ----------

    def _save_episode_in_buffer(self, episode_data, task_description):
        rl_config = self._config.rl
        assert isinstance(rl_config, FlowAWRSFTLearnerConfig)

        if not episode_data:
            return
        for ep in episode_data:
            if "rollout_info" not in ep:
                # Should never happen given add_data populates it, but guard
                # to avoid silently dropping bad data.
                return

        if self._config.collect.store_prefix_rep:
            self._attach_prefix_embeddings_to_episode_data(
                episode_data, task_description=task_description
            )

        replan_steps = self._config.collect.replan_steps
        discount = self._config.rl.discount

        # Concatenate per-step sequences across chunks to find episode end.
        rewards = np.concatenate(
            [np.asarray(ep["reward"]) for ep in episode_data], axis=0
        )
        terminate = np.concatenate(
            [np.asarray(ep["terminate"]) for ep in episode_data], axis=0
        )
        truncate = np.concatenate(
            [np.asarray(ep["truncate"]) for ep in episode_data], axis=0
        )
        done = np.logical_or(terminate, truncate)
        if not np.any(done):
            return
        end_idx = int(np.where(done)[0][0])
        n_steps = end_idx + 1

        last_chunk_idx = end_idx // replan_steps
        n_used_chunks = last_chunk_idx + 1
        if n_used_chunks > len(episode_data):
            n_used_chunks = len(episode_data)

        chunk_rewards = np.zeros(n_used_chunks, dtype=np.float32)
        chunk_discounts = np.zeros(n_used_chunks, dtype=np.float32)
        for i in range(n_used_chunks):
            env_start = i * replan_steps
            chunk_done = False
            steps_in_chunk = 0
            for j in range(replan_steps):
                env_idx = env_start + j
                if env_idx >= n_steps:
                    break
                chunk_rewards[i] += float(discount ** j) * float(rewards[env_idx])
                steps_in_chunk += 1
                if done[env_idx]:
                    chunk_done = True
                    break
            chunk_discounts[i] = (
                0.0 if chunk_done else float(discount ** steps_in_chunk)
            )

    
        all_gammas = np.array(
            [discount ** t for t in range(n_steps)], dtype=np.float32
        )
        cum_discounted = (all_gammas * rewards[:n_steps])[::-1].cumsum()[::-1]
        chunk_starts = np.minimum(
            np.arange(n_used_chunks) * replan_steps, n_steps - 1
        )
        chunk_mc_returns = (cum_discounted[chunk_starts]
                            / np.maximum(all_gammas[chunk_starts], 1e-12))


        if any("full_action_chunk" not in ep for ep in episode_data[:n_used_chunks]):
            return  # collection went through a path that didn't stash chunks
        chunk_actions = np.stack(
            [
                np.asarray(ep["full_action_chunk"], dtype=np.float32)
                for ep in episode_data[:n_used_chunks]
            ]
        )
        chunk_actions = self.post_step_action_filter(chunk_actions)

        # Per-chunk rollout_info: stack the per-chunk dicts along axis 0.
        chunk_rollout_info = jax.tree_util.tree_map(
            lambda *xs: np.stack(xs, axis=0).astype(np.float32),
            *(episode_data[i]["rollout_info"] for i in range(n_used_chunks)),
        )

        # Per-chunk obs (first obs of chunk's replan window) and next_obs (last
        # next_obs in chunk's replan window).
        def _take_first(x):
            return x[0]

        def _take_last(x):
            return x[-1]

        chunk_obs = jax.tree_util.tree_map(
            lambda *xs: np.stack(xs, axis=0),
            *(
                jax.tree_util.tree_map(_take_first, episode_data[i]["observation"])
                for i in range(n_used_chunks)
            ),
        )
        chunk_next_obs = jax.tree_util.tree_map(
            lambda *xs: np.stack(xs, axis=0),
            *(
                jax.tree_util.tree_map(_take_last, episode_data[i]["next_observation"])
                for i in range(n_used_chunks)
            ),
        )
        # Strip "observation/" prefix that the env wrappers leave on keys.
        chunk_obs = {
            k[len("observation/") :] if k.startswith("observation/") else k: v
            for k, v in chunk_obs.items()
        }
        chunk_next_obs = {
            k[len("observation/") :] if k.startswith("observation/") else k: v
            for k, v in chunk_next_obs.items()
        }

        n_rows = n_used_chunks

        def transform(obs, act, prompt):
            obs.update({"actions": act, "prompt": prompt})
            obs = self._pre_token_transform(obs)
            obs["image_mask"] = {
                k: np.full((n_rows,), bool(v)) for k, v in obs["image_mask"].items()
            }
            if isinstance(self._token_transform, _transforms.TokenizePrompt):
                if prompt not in self._token_cache:
                    tok = self._token_transform({"prompt": prompt})
                    self._token_cache[prompt] = (
                        tok["tokenized_prompt"],
                        tok["tokenized_prompt_mask"],
                    )
                tokens, token_masks = self._token_cache[prompt]
                obs["tokenized_prompt"] = np.broadcast_to(
                    tokens, (n_rows,) + tokens.shape
                ).copy()
                obs["tokenized_prompt_mask"] = np.broadcast_to(
                    token_masks, (n_rows,) + token_masks.shape
                ).copy()
            else:
                raise TypeError(
                    f"Unsupported token transform: {type(self._token_transform)}"
                )
            actions_out = obs.pop("actions")
            obs.pop("prompt")
            return obs, actions_out

        chunk_next_obs, _ = transform(chunk_next_obs, chunk_actions, str(task_description))
        chunk_obs, chunk_actions = transform(chunk_obs, chunk_actions, str(task_description))

        self._online_data_buffer.insert(
            {
                "observation": chunk_obs,
                "actions": chunk_actions.astype(np.float32),
                "next_observation": chunk_next_obs,
                "reward": chunk_rewards.astype(np.float32),
                "mc_return": chunk_mc_returns.astype(np.float32),
                "discount": chunk_discounts.astype(np.float32),
                "rollout_info": chunk_rollout_info,
            }
        )
        self._collection_success_episodes += 1


    def _online_batch_to_sft_batch(
        self, online_batch: Dict[str, Any]
    ) -> tuple[_model.Observation, _model.Actions, Dict[str, Any]]:
        return (
            _model.Observation.from_dict(online_batch["observation"]),
            online_batch["actions"],
            online_batch["rollout_info"],
        )

    def _sft_batch_to_actor_batch(
        self,
        sft_batch: tuple[_model.Observation, _model.Actions, Dict[str, Any]],
        policy_state: training_utils.TrainState,
    ) -> tuple[_model.Observation, Any, _model.Actions, Dict[str, Any]]:
        policy_observation, actions, rollout_info = sft_batch
        policy_obs_dict = policy_observation.to_dict()

        critic_observation: dict[str, Any] = {
            "state": policy_obs_dict["state"],
        }
        prefix_embedding = self._recompute_prefix_embedding(
            observation=policy_obs_dict,
            policy_state=policy_state,
        )
        critic_observation[PREFIX_EMBEDDING_NAME] = prefix_embedding

        return policy_observation, critic_observation, actions, rollout_info

    def _update_policy(
        self,
        batch,
        policy_state: training_utils.TrainState,
        q_state: training_utils.TrainState,
        value_state: training_utils.TrainState,
        rng: at.KeyArrayLike,
        mc_return: at.Array | None = None,
        scale: at.Array | float = 1.0,
    ):
        batch = self._sft_batch_to_actor_batch(batch, policy_state=policy_state)
        policy_state, info = self._train_step(
            rng,
            policy_state,
            q_state,
            value_state,
            batch,
            mc_return=mc_return,
            scale=scale,
        )
        return policy_state, info


    @at.typecheck
    def update(self) -> dict:
        rl_config = self._config.rl
        assert isinstance(rl_config, FlowAWRSFTLearnerConfig)
        normalizer_config = rl_config.normalizer_config

        if rl_config.critic_pre_training_steps == self.training_steps:
            # Mirror AWR's optimizer reset at the end of critic warmup.
            self._state_action_critic_state = self._reset_optimizer_with_ema(
                self._state_action_critic_state
            )
            self._value_state = self._reset_optimizer_with_ema(self._value_state)

        self.training_steps += 1
        update_critic = (
            self.training_steps >= rl_config.critic_training_start_step
            and self.training_steps % rl_config.critic_update_interval == 0
        )
        update_policy = (
            self.training_steps >= rl_config.policy_training_start_step
            and self.training_steps % rl_config.policy_update_interval == 0
        )
        if not update_critic and not update_policy:
            return {
                "online_buffer_size": jnp.asarray(
                    float(self._online_data_buffer.size), dtype=jnp.float32
                )
            }

        online_batch_size = int(
            self._config.batch_size * min(1.0, rl_config.online_ratio)
        )
        use_online = self._online_data_buffer.size >= online_batch_size
        if not use_online:
            return {
                "online_buffer_size": jnp.asarray(
                    float(self._online_data_buffer.size), dtype=jnp.float32
                )
            }

        online_batch = self._online_data_buffer.sample(batch_size=online_batch_size)
        critic_info, actor_info = {}, {}

        if update_critic:
            critic_rng, self._rng = jax.random.split(self._rng, 2)
            with sharding.set_mesh(self._mesh):
                q_state, value_state, q_info, value_info = (
                    self._update_critics_jitted(
                        online_batch,
                        self._state_action_critic_state,
                        self._value_state,
                        self._train_state,
                        critic_rng,
                    )
                )
            self._state_action_critic_state = q_state
            self._value_state = value_state
            critic_info = {
                f"critic/q_{key}": value for key, value in q_info.items()
            } | {f"critic/value_{key}": value for key, value in value_info.items()}

        if update_policy:
            sft_batch = self._online_batch_to_sft_batch(online_batch)
            sft_batch = jax.device_put(sft_batch, self._data_sharding)
            policy_rng, self._rng = jax.random.split(self._rng, 2)
            scale = self._normalizer_state.scale
            with sharding.set_mesh(self._mesh):
                policy_state, actor_info = self._update_policy_jitted(
                    sft_batch,
                    self._train_state,
                    self._state_action_critic_state,
                    self._value_state,
                    policy_rng,
                    None,  # mc_return — FlowAWR uses Q-V on buffer actions
                    scale,
                )
            self._train_state = policy_state
            self._maybe_restore_policy_ema_after_resume()

            # Update host-side normalizer the same way AWR does.
            scale_update, bias_update = 1.0, 0.0
            if normalizer_config.method is not None:
                if normalizer_config.method == "quantile":
                    q_up = actor_info["advantage_q_up"]
                    q_low = actor_info["advantage_q_low"]
                    scale_update = q_up - q_low
                    bias_update = q_low
                elif normalizer_config.method == "standard_normal":
                    scale_update = actor_info["advantage_std"]
                    bias_update = actor_info["advantage_mean"]
                elif normalizer_config.method == "min_max":
                    scale_update = actor_info["advantage_max"] - actor_info["advantage_min"]
                    bias_update = actor_info["advantage_min"]
                else:
                    raise NotImplementedError(
                        f"Unknown normalizer method: {normalizer_config.method}"
                    )
                scale_update = jnp.clip(scale_update, min=normalizer_config.min_scale)
            self._normalizer_state, normalizer_info = self._update_normalizer(
                normalizer_state=self._normalizer_state,
                bias=bias_update,
                scale=scale_update,
            )
            actor_info = actor_info | normalizer_info
            actor_info = {f"actor/{key}": value for key, value in actor_info.items()}

        info = (
            actor_info
            | critic_info
            | {
                "online_buffer_size": jnp.asarray(
                    float(self._online_data_buffer.size), dtype=jnp.float32
                )
            }
        )
        info = jax.tree.map(np.asarray, info)
        return info

    @staticmethod
    def _reset_optimizer_with_ema(state: training_utils.TrainState) -> training_utils.TrainState:
        import dataclasses
        opt_state = state.tx.init(nnx.filter_state(state.params, nnx.Param))
        new_ema = jax.tree.map(jnp.copy, state.params)
        return dataclasses.replace(state, opt_state=opt_state, ema_params=new_ema)
