# ruff: noqa: F722
import dataclasses
import functools
import logging
from typing import Any, Dict, Tuple
import gc

import etils.epath as epath
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.transforms as _transforms
from src.rl.value_distribution import get_value_bounds, make_value_distribution
from src.rl.advantage_weighted_sft.update_actor import (
    train_step as train_actor_step,
)
from src.rl.advantage_weighted_sft.update_critic import (
    init_state_action_critic_train_state,
    init_state_value_train_state,
    train_q_step,
    train_value_step,
)
from src.rl.best_of_n.update_critic import _build_pi0_backbone_critic_defs
from src.rl.networks.rl_networks import ObsType
from src.rl.filtered_sft_agent.filtered_sft_learner import (
    FilteredSFTLearner,
    _copy_nnx_state,
)
from src.rl.advantage_weighted_sft.memory_logging import log_memory_debug
from src.rl.prefix_embedding import PREFIX_EMBEDDING_NAME
from src.training.config import AdvantageWeightedSFTLearnerConfig, Normalizer, NormalizerState


class AdvantageWeightedSFTLearner(FilteredSFTLearner):
    def __init__(self, config):

        assert isinstance(self._config.rl, AdvantageWeightedSFTLearnerConfig), (
            "Only AdvantageWeightedSFTLearnerConfig should be passed to the AWR agent"
        )

        model = config.model.create(jax.random.key(config.seed))
        fake_obs = config.model.fake_obs(batch_size=1)
        prefix_rep = model.get_prefix_rep(fake_obs)[0]
        del model
        assert prefix_rep.ndim == 3, f"Expected prefix_rep to have shape (batch, seq_len, embed_dim), but got {prefix_rep.shape}"
        prefix_embedding_shape = tuple(prefix_rep.shape[2:])
        dummy_obs = {
            "state": fake_obs.state,
            PREFIX_EMBEDDING_NAME: jnp.zeros((1, *prefix_embedding_shape), dtype=jnp.float32)
        }
        dummy_act = config.model.fake_act(batch_size=1)
        state_action_critic_def, state_value_def = _build_pi0_backbone_critic_defs(config)

        self._prefix_embed_dim = None
        if config.collect.store_prefix_rep and PREFIX_EMBEDDING_NAME in dummy_obs:
            self._prefix_embed_dim = int(np.asarray(dummy_obs[PREFIX_EMBEDDING_NAME]).shape[-1])

        super().__init__(config)

        data_config = self._data_loader.data_config()
        normalizer = _transforms.Normalize(
            data_config.norm_stats,
            use_quantiles=data_config.use_quantile_norm,
        )
        self._state_normalize = normalizer
        self._action_normalize = normalizer
        self._transition_state_dim = int(dummy_obs["state"].shape[-1])

        q_init_rng, v_init_rng, self._rng = jax.random.split(self._rng, 3)
        self._state_action_critic_state, self._state_action_critic_state_sharding = (
            init_state_action_critic_train_state(
                self._config,
                q_init_rng,
                self._mesh,
                critic_def=state_action_critic_def,
                dummy_obs=dummy_obs,
                dummy_act=dummy_act,
            )
        )

        self._value_state, self._value_state_sharding = init_state_value_train_state(
            self._config,
            v_init_rng,
            self._mesh,
            critic_def=state_value_def,
            dummy_obs=dummy_obs,
        )
        self._rl_state_checkpointer = ocp.StandardCheckpointer()
        if self._resuming:
            self._restore_rl_checkpoint(step=int(self.training_steps))
        jax.block_until_ready(self._state_action_critic_state)
        jax.block_until_ready(self._value_state)

        del self._train_step
        gc.collect()

        self._normalizer = Normalizer(
            ema_weight=self._config.rl.normalizer_config.ema_weight,
        )
        self._normalizer_state = self._normalizer.init()
        # 1. Un-JIT the inner steps (JAX will compile these as part of the outer methods)
        self._q_train_step = functools.partial(train_q_step, self._config)
        self._value_train_step = functools.partial(train_value_step, self._config)
        self._train_step = functools.partial(train_actor_step, self._config)
        self._refresh_update_functions()

    def _policy_mc_return_sharding(self):
        return self._data_sharding

    def _refresh_update_functions(self):
        self._train_state_sharding = sharding.fsdp_sharding(
            self._train_state, self._mesh, log=False
        )

        def _critics_wrapper(batch, q_state, value_state, policy_state, rng):
            return self._update_critics(
                batch=batch,
                q_state=q_state,
                value_state=value_state,
                policy_state=policy_state,
                rng=rng,
            )

        def _policy_wrapper(batch, policy_state, q_state, value_state, rng, mc_return, is_success, scale):
            return self._update_policy(
                batch=batch,
                policy_state=policy_state,
                q_state=q_state,
                value_state=value_state,
                rng=rng,
                mc_return=mc_return,
                is_success=is_success,
                scale=scale,
            )

        self._update_critics_jitted = jax.jit(
            _critics_wrapper,
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

        self._update_policy_jitted = jax.jit(
            _policy_wrapper,
            in_shardings=(
                self._data_sharding,
                self._train_state_sharding,
                self._state_action_critic_state_sharding,
                self._value_state_sharding,
                self._replicated_sharding,
                self._policy_mc_return_sharding(),
                self._data_sharding,
                self._replicated_sharding,
            ),
            out_shardings=(
                self._train_state_sharding,
                self._replicated_sharding,
            ),
            donate_argnums=(1,),
        )

    def _maybe_restore_policy_ema_after_resume(self):
        if not self._resume_restore_ema:
            return
        self._train_state = dataclasses.replace(
            self._train_state,
            ema_decay=self._resume_ema_decay,
            ema_params=_copy_nnx_state(self._train_state.params),
        )
        self._resume_restore_ema = False
        self._resume_ema_decay = None
        self._refresh_update_functions()

    def _make_buffer_dummy_data(self) -> dict:
        dummy = super()._make_buffer_dummy_data()
        if self._prefix_embed_dim is not None:
            zeros = np.zeros((1, self._prefix_embed_dim), dtype=np.float32)
            dummy["observation"][PREFIX_EMBEDDING_NAME] = zeros
            dummy["next_observation"][PREFIX_EMBEDDING_NAME] = zeros
        return dummy

    def _rl_checkpoint_state(self) -> dict[str, training_utils.TrainState]:
        return {
            "state_action_critic_state": self._state_action_critic_state,
            "value_state": self._value_state,
            "normalizer_state": self._normalizer_state
        }

    def _rl_checkpoint_dir(self) -> epath.Path:
        return epath.Path(self._config.checkpoint_dir) / "rl_state"

    def _rl_checkpoint_path(self, step: int) -> epath.Path:
        return self._rl_checkpoint_dir() / str(int(step))

    def _restore_rl_checkpoint(self, *, step: int) -> None:
        path = self._rl_checkpoint_path(step)
        if not path.exists():
            logging.warning(
                "No RL critic checkpoint found at %s; starting critics from scratch.",
                path,
            )
            return
        restored = self._rl_state_checkpointer.restore(
            path,
            self._rl_checkpoint_state(),
        )
        self._state_action_critic_state = restored["state_action_critic_state"]
        self._value_state = restored["value_state"]
        self._normalizer_state = restored["normalizer_state"]

    def save_checkpoint(self, step: int | None = None):
        if step is None:
            step = self.training_steps
        super().save_checkpoint(step=step)
        path = self._rl_checkpoint_path(step)
        if path.exists():
            return
        self._rl_checkpoint_dir().mkdir(parents=True, exist_ok=True)
        self._rl_state_checkpointer.save(
            path,
            self._rl_checkpoint_state(),
        )

    def _recompute_prefix_embedding(
        self,
        *,
        observation: dict[str, Any],
        policy_state: training_utils.TrainState,
    ) -> at.Float[at.Array, "batch embed"] | None:
        model = self._get_policy_model(policy_state)
        # Both SFT-loader Observations and online-buffer dicts are already
        # fully transformed (repack, LiberoInputs, Normalize, tokenize, etc.)
        # by the data pipeline / _preprocess_insert
        obs = _model.Observation.from_dict(observation)
        prefix = self._policy._get_prefix_rep_with_model(model, observation=obs)
        prefix = prefix.reshape((prefix.shape[0], -1, prefix.shape[-1]))
        prefix = jnp.mean(prefix, axis=1)
        return prefix

    @staticmethod
    def _get_policy_model(policy_state: training_utils.TrainState) -> _model.BaseModel:
        """Merge policy params into a model. Call once per update() to avoid duplicates."""
        params = (
            policy_state.ema_params
            if policy_state.ema_params is not None
            else policy_state.params
        )
        model = nnx.merge(policy_state.model_def, params)
        model.eval()
        return model

    @at.typecheck
    def _online_batch_to_critic_batch(
        self,
        online_batch: dict[str, Any],
        policy_state: training_utils.TrainState,
    ) -> tuple[
        ObsType,
        _model.Actions,
        ObsType,
        at.Float[at.Array, " b"],
        at.Float[at.Array, " b"],
        at.Float[at.Array, " b"],
    ]:
        online_observation = online_batch["observation"]
        observation_dict: dict[str, Any] = {
            "state": online_observation["state"],
        }

        next_observation = online_batch["next_observation"]
        next_observation_dict: dict[str, Any] = {"state": next_observation["state"]}

        if PREFIX_EMBEDDING_NAME in online_observation and PREFIX_EMBEDDING_NAME in next_observation:
            observation_dict[PREFIX_EMBEDDING_NAME] = online_observation[PREFIX_EMBEDDING_NAME]
            next_observation_dict[PREFIX_EMBEDDING_NAME] = next_observation[PREFIX_EMBEDDING_NAME]
        else:
            observation_dict[PREFIX_EMBEDDING_NAME] = self._recompute_prefix_embedding(
                observation=online_observation, policy_state=policy_state,
            )
            next_observation_dict[PREFIX_EMBEDDING_NAME] = self._recompute_prefix_embedding(
                observation=next_observation, policy_state=policy_state,
            )

        return (
            observation_dict,
            online_batch["actions"],
            next_observation_dict,
            online_batch["reward"],
            online_batch["discount"],
            online_batch["mc_return"],
        )

    def _sft_batch_to_actor_batch(
        self,
        sft_batch: tuple[_model.Observation, _model.Actions],
        policy_state: training_utils.TrainState,
    ) -> tuple[_model.Observation, ObsType, _model.Actions]:
        policy_observation, actions = sft_batch
        policy_obs_dict = policy_observation.to_dict()

        critic_observation: dict[str, Any] = {
            "state": policy_obs_dict["state"],
        }

        prefix_embedding = self._recompute_prefix_embedding(
            observation=policy_obs_dict,
            policy_state=policy_state,
        )

        critic_observation[PREFIX_EMBEDDING_NAME] = prefix_embedding

        return policy_observation, critic_observation, actions

    def sample_actions(self, observations, **kwargs):
        """Best-of-N collection: sample n_samples candidates per env and keep the highest-Q one.

        Falls through to the base class (single sample) when n_samples <= 1 or before
        critic_inference_start_step so the critic has time to warm up first.
        """
        assert isinstance(self._config.rl, AdvantageWeightedSFTLearnerConfig)
        n_samples = self._config.rl.n_samples
        if n_samples <= 1 or self.training_steps < self._config.rl.critic_inference_start_step:
            return super().sample_actions(observations, **kwargs)

        rng, self._rng = jax.random.split(self._rng)
        task_description = kwargs.get("task_description")

        if task_description is None or isinstance(task_description, str):
            task_description = [task_description] * next(
                np.asarray(v).shape[0] for v in observations.values()
            )
        # Group envs by task so the prompt embedding is computed once per unique
        # task rather than once per env, which matters when many envs share a task.
        task_to_indices: dict[str, list[int]] = {}
        for i, task in enumerate(task_description):
            task_to_indices.setdefault(str(task), []).append(i)

        env_num = len(task_description)
        return_prefix_rep = self._config.collect.store_prefix_rep
        all_best_actions = None
        all_best_prefix = None

        q_params = (
            self._state_action_critic_state.ema_params
            if self._state_action_critic_state.ema_params is not None
            else self._state_action_critic_state.params
        )
        q_model = nnx.merge(self._state_action_critic_state.model_def, q_params)
        q_model.eval()

        params = (
            self._train_state.ema_params
            if self._train_state.ema_params is not None
            else self._train_state.params
        )
        policy_model = nnx.merge(self._train_state.model_def, params)
        policy_model.eval()

        for task, indices in task_to_indices.items():
            group_obs = jax.tree.map(lambda x: np.asarray(x)[indices], observations)
            processed_obs = self._process_obs_for_pi0(group_obs, task_description=task)
            group_env_num = len(indices)

            tiled_obs = {
                k: (v if k == "prompt" else np.repeat(np.asarray(v), n_samples, axis=0))
                for k, v in processed_obs.items()
            }
            group_actions = self._sample_action(tiled_obs, rng, self._train_state)

            raw_state = np.asarray(processed_obs["observation/state"])
            state = np.asarray(self._state_normalize({"state": raw_state})["state"])
            if state.shape[-1] < self._transition_state_dim:
                pad_width = [(0, 0)] * state.ndim
                pad_width[-1] = (0, self._transition_state_dim - state.shape[-1])
                state = np.pad(state, pad_width, mode="constant", constant_values=0.0)
            state = jnp.repeat(jnp.asarray(state, dtype=jnp.float32), n_samples, axis=0)
            critic_obs: dict = {"state": state}

            per_env_inputs = [
                self._policy._input_transform(
                    {
                        k: (v if k == "prompt" else np.asarray(v)[i])
                        for k, v in processed_obs.items()
                    }
                )
                for i in range(group_env_num)
            ]

            def _stack_prefix_inputs(*values):
                first = values[0]
                if first is None:
                    return None
                return jnp.stack([jnp.asarray(v) for v in values], axis=0)

            inputs = jax.tree.map(_stack_prefix_inputs, *per_env_inputs)
            batch_size = group_env_num

            def _as_batched_array(value):
                if value is None:
                    return None
                value = jnp.asarray(value)
                if value.ndim > 0 and value.shape[0] == batch_size:
                    return value
                return jnp.broadcast_to(value[jnp.newaxis, ...], (batch_size,) + value.shape)

            inputs = {
                k: (
                    jax.tree.map(lambda x: None if x is None else jnp.asarray(x), v)
                    if k in ("image", "state")
                    else jax.tree.map(_as_batched_array, v)
                )
                for k, v in inputs.items()
            }
            obs_for_prefix = _model.Observation.from_dict(inputs)
            prefix = self._get_prefix_rep_with_model(m=policy_model, observation=obs_for_prefix)
            prefix = np.asarray(prefix)
            if prefix.ndim == 3:
                prefix = prefix.reshape(prefix.shape[0], -1, prefix.shape[-1]).mean(axis=1)
            critic_obs[PREFIX_EMBEDDING_NAME] = jnp.repeat(
                jnp.asarray(prefix), n_samples, axis=0
            )

            actions_norm = np.asarray(
                self._action_normalize({"actions": np.asarray(group_actions)})["actions"]
            )
            model_act_dim = self._config.model.action_dim
            if actions_norm.shape[-1] < model_act_dim:
                pad_width = [(0, 0)] * actions_norm.ndim
                pad_width[-1] = (0, model_act_dim - actions_norm.shape[-1])
                actions_norm = np.pad(actions_norm, pad_width, mode="constant", constant_values=0.0)
            flat_actions = jnp.asarray(actions_norm.reshape(group_env_num * n_samples, -1))
            q_logits = q_model(critic_obs, flat_actions)

            rl_config = self._config.rl
            _lower, _upper = get_value_bounds(self._config)
            q_dist = make_value_distribution(
                q_logits, rl_config.critic.num_value_bins, _lower, _upper
            )
            scores = np.asarray(q_dist.mean())
            if scores.ndim > 1:
                scores = scores.min(axis=0)
            scores = scores.reshape(group_env_num, n_samples)
            best_idx = scores.argmax(axis=1)

            group_actions = np.asarray(group_actions).reshape(
                group_env_num, n_samples, *np.asarray(group_actions).shape[1:]
            )
            best = group_actions[np.arange(group_env_num), best_idx]

            if all_best_actions is None:
                all_best_actions = np.zeros((env_num, *best.shape[1:]), dtype=np.float32)
            all_best_actions[indices] = np.asarray(best, dtype=np.float32)

            if return_prefix_rep:
                if all_best_prefix is None:
                    all_best_prefix = np.zeros((env_num, prefix.shape[-1]), dtype=np.float32)
                all_best_prefix[indices] = np.asarray(prefix, dtype=np.float32)

        return (all_best_actions, all_best_prefix) if return_prefix_rep else all_best_actions

    def save_episode(self, is_success: bool, env_index: int, task_description: str):
        assert env_index in range(
            len(self._episode_storage)
        ), f"env_index must be between 0 and {len(self._episode_storage) - 1}, but got {env_index}."
        # extract episode data from storage and empty it
        episode_data = self._episode_storage[env_index]
        self._episode_storage[env_index] = []
        if self._config.rl.store_success_episodes_only and not is_success:
            return
        self._save_episode_in_buffer(episode_data, task_description, is_success=is_success)

    def _update_normalizer(self, normalizer_state, bias, scale) -> Tuple[NormalizerState, dict[str, at.Array]]:
        normalizer_state = self._normalizer.update(
            normalizer_state=normalizer_state,
            bias=bias,
            scale=scale,
        )
        return normalizer_state, {
            'normalizer_bias': jnp.mean(normalizer_state.bias),
            'normalizer_scale': jnp.mean(normalizer_state.scale),
        }

    @at.typecheck
    def _update_critics(
        self,
        batch: Dict[str, Any],
        q_state: training_utils.TrainState,
        value_state: training_utils.TrainState,
        policy_state: training_utils.TrainState,
        rng: at.KeyArrayLike,
    ) -> Tuple[
        training_utils.TrainState,
        training_utils.TrainState,
        dict[str, at.Array],
        dict[str, at.Array],
    ]:
        # Add prefix representation to the batch for the critic
        batch = self._online_batch_to_critic_batch(
            batch,
            policy_state,
        )

        # Update the state action critic state
        num_updates = max(self._config.rl.critic.num_updates_per_batch, 1)
        for _ in range(num_updates):
            q_rng, v_rng, rng = jax.random.split(rng, 3)
            q_state, q_info = self._q_train_step(
                q_rng,
                q_state,
                value_state,
                batch,
            )
            # Update the value state
            value_state, value_info = self._value_train_step(
                v_rng,
                value_state,
                q_state,
                batch,
            )

        return q_state, value_state, q_info, value_info

    def _update_policy(
        self,
        batch: tuple[_model.Observation, _model.Actions],
        policy_state: training_utils.TrainState,
        q_state: training_utils.TrainState,
        value_state: training_utils.TrainState,
        rng: at.KeyArrayLike,
        mc_return: at.Array | None = None,
        is_success: at.Float[at.Array, " b"] | None = None,
        scale: at.Array | float = 1.0,
    ):
        # Add prefix representation to the batch
        batch = self._sft_batch_to_actor_batch(
            batch,
            policy_state=policy_state,
        )
        # Update the policy state
        policy_state, info = self._train_step(
            rng,
            policy_state,
            q_state,
            value_state,
            batch,
            mc_return=mc_return,
            is_success=is_success,
            scale=scale,
        )

        return policy_state, info

    @at.typecheck
    def update(self) -> dict:
        if self._config.rl.critic.pre_training_steps == self.training_steps:
            # Reset optimizer state of the value and q function
            q_opt_state = self._state_action_critic_state.tx.init(
                nnx.filter_state(self._state_action_critic_state.params, nnx.Param)
            )
            new_ema_state_action_critic_params = jax.tree.map(
                jnp.copy, self._state_action_critic_state.params
            )
            self._state_action_critic_state = dataclasses.replace(
                self._state_action_critic_state,
                opt_state=q_opt_state,
                ema_params=new_ema_state_action_critic_params,
            )
            del new_ema_state_action_critic_params, q_opt_state

            v_opt_state = self._value_state.tx.init(
                nnx.filter_state(self._value_state.params, nnx.Param)
            )
            new_ema_value_params = jax.tree.map(jnp.copy, self._value_state.params)
            self._value_state = dataclasses.replace(
                self._value_state,
                opt_state=v_opt_state,
                ema_params=new_ema_value_params,
            )
            del new_ema_value_params, v_opt_state

        self.training_steps += 1
        update_critic = (
            self.training_steps >= self._config.rl.critic.training_start_step
            and self.training_steps % self._config.rl.critic.update_interval == 0
        )
        update_policy = (
            self.training_steps >= self._config.rl.policy.training_start_step
            and self.training_steps % self._config.rl.policy.update_interval == 0
        )
        if not update_critic and not update_policy:
            return {
                "online_buffer_size": jnp.asarray(
                    float(self._online_data_buffer.size), dtype=jnp.float32
                )
            }

        policy_batch_size = int(
            self._config.batch_size * min(1.0, self._config.rl.online_ratio)
        )
        # Critics are small MLPs so a larger batch than the policy is cheap and
        # improves TD stability. Falls back to policy_batch_size when not set.
        critic_batch_size = self._config.rl.critic.batch_size or policy_batch_size
        use_online = self._online_data_buffer.size >= policy_batch_size
        if self._config.rl.use_mc_returns and not use_online:
            update_policy = False

        critic_info, actor_info = {}, {}
        mc_return = None
        is_success = None
        if use_online:
            # Two independent samples: critics may use a larger batch than the policy.
            critic_online_batch = self._online_data_buffer.sample(batch_size=critic_batch_size) if update_critic else None
            online_batch = self._online_data_buffer.sample(batch_size=policy_batch_size)
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

                critic_info = {
                    f"critic/q_{key}": value for key, value in q_info.items()
                } | {f"critic/value_{key}": value for key, value in value_info.items()}
            if self._config.rl.use_mc_returns:
                mc_return = online_batch["mc_return"]
                assert self._config.rl.online_ratio >= 1.0, (
                    "use_mc_returns requires online_ratio >= 1.0 "
                    "(MC returns are not available for offline data)"
                )
            online_is_success = jnp.asarray(online_batch["is_success"], dtype=jnp.float32)
            online_batch = self._online_batch_to_sft_batch(online_batch)
            online_ratio = self._config.rl.online_ratio
            if online_ratio >= 1.0:
                batch = online_batch
                is_success = online_is_success
            elif online_ratio > 0:
                # Mix online and offline into a fixed-size batch instead of
                # concatenating (which would double the batch and OOM).
                batch = next(self._data_iter)
                first_leaf = jax.tree.leaves(batch)[0]
                batch_size = first_leaf.shape[0]
                n_online = min(
                    int(batch_size * online_ratio),
                    jax.tree.leaves(online_batch)[0].shape[0],
                )
                n_offline = batch_size - n_online
                batch = jax.tree.map(
                    lambda x, y: jnp.concatenate([x[:n_offline], y[:n_online]], axis=0),
                    batch,
                    online_batch,
                )
                # The online batch may be replicated (PartitionSpec()) while
                # the SFT batch is sharded. Re-shard the mixed result to
                # match the data sharding expected by _train_step.
                batch = jax.device_put(batch, self._data_sharding)
                # Assumes all offline data is successful demos
                is_success = jnp.concatenate([
                    jnp.ones(n_offline, dtype=jnp.float32),
                    online_is_success[:n_online],
                ])
                del online_batch
                gc.collect()
            else:
                # online_ratio == 0: pure offline batch; no success signal available.
                batch = next(self._data_iter)
                first_leaf = jax.tree.leaves(batch)[0]
                # Assumes all offline data is successful demos
                is_success = jnp.ones(first_leaf.shape[0], dtype=jnp.float32)
        else:
            if self._data_iter is None:
                return {
                    "online_buffer_size": jnp.asarray(
                        float(self._online_data_buffer.size), dtype=jnp.float32
                    )
                }
            # Buffer not yet populated; fall back to offline data with zero is_success.
            batch = next(self._data_iter)
            first_leaf = jax.tree.leaves(batch)[0]
            # Assumes all offline data is successful demos
            is_success = jnp.ones(first_leaf.shape[0], dtype=jnp.float32)
        if update_policy:
            policy_rng, self._rng = jax.random.split(self._rng, 2)
            scale = self._normalizer_state.scale if self._config.rl.normalize_advantages else 1.0
            with sharding.set_mesh(self._mesh):
                policy_state, actor_info = self._update_policy_jitted(
                    batch,
                    self._train_state,
                    self._state_action_critic_state,
                    self._value_state,
                    policy_rng,
                    mc_return,
                    is_success,
                    scale,
                )

            self._train_state = policy_state
            self._maybe_restore_policy_ema_after_resume()
            scale, bias = 1.0, 0.0
            normalizer_config = self._config.rl.normalizer_config
            if normalizer_config.method is not None:
                if normalizer_config.method == 'quantile':
                    q_up, q_low = actor_info['advantage_q_up'], actor_info['advantage_q_low']
                    scale = q_up - q_low
                    bias = q_low
                elif normalizer_config.method == 'standard_normal':
                    scale = actor_info['advantage_std']
                    bias = actor_info['advantage_mean']
                elif normalizer_config.method == 'min_max':
                    q_max, q_min = actor_info['advantage_max'], actor_info['advantage_min']
                    scale = q_max - q_min
                    bias = q_min
                else:
                    raise NotImplementedError
                scale = jnp.clip(scale, min=normalizer_config.min_scale)
            self._normalizer_state, normalizer_info = self._update_normalizer(
                normalizer_state=self._normalizer_state,
                bias=bias,
                scale=scale)
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
