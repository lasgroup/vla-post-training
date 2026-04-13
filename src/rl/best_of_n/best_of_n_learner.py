# ruff: noqa: F722
import dataclasses
import functools
from typing import Any, Dict, Tuple
import gc

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.transforms as _transforms
from src.rl.value_distribution import get_value_bounds, make_value_distribution
from src.rl.best_of_n.update_critic import (
    init_state_action_critic_train_state,
    init_state_value_train_state,
    train_q_step,
    train_value_step,
    StateActionCriticDef,
    StateValueDef,
)
from src.rl.networks.rl_networks import ObsType, ActionType
from src.rl.filtered_sft_agent.filtered_sft_learner import FilteredSFTLearner
from src.rl.advantage_weighted_sft.memory_logging import log_memory_debug
from src.rl.prefix_embedding import PREFIX_EMBEDDING_NAME
from src.training.config import BestofNLearnerConfig, OnlineTrainConfig


class BestofNLearner(FilteredSFTLearner):
    def __init__(
            self,
            config: OnlineTrainConfig,
            dummy_obs: ObsType,
            dummy_act: ActionType,
            state_action_critic_def: StateActionCriticDef,
            state_value_def: StateValueDef,
            task_description: str,
            debug: bool = False,
    ):
        self.task_description = task_description
        self.debug = debug

        super().__init__(config)

        # Initialize normalization and dimension attributes for critic inference
        data_config = self._data_loader.data_config()
        _norm = _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm)
        self._state_normalize = _norm
        self._action_normalize = _norm
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
        jax.block_until_ready(self._state_action_critic_state)
        jax.block_until_ready(self._value_state)

        if self.debug:
            log_memory_debug(
                "init",
                train_state=self._train_state,
                state_action_critic_state=self._state_action_critic_state,
                value_state=self._value_state,
            )

        del self._train_step
        gc.collect()

        # 1. Un-JIT the inner steps (JAX will compile these as part of the outer methods)
        self._q_train_step = functools.partial(train_q_step, self._config)
        self._value_train_step = functools.partial(train_value_step, self._config)
        # self._train_step = functools.partial(train_actor_step, self._config)

        # 2. Create closures to drop 'self' from the JIT signature
        def _critics_wrapper(batch, q_state, value_state, policy_state, rng):
            return self._update_critics(
                batch=batch,
                q_state=q_state,
                value_state=value_state,
                policy_state=policy_state,
                rng=rng,
            )

        # 3. JIT the wrappers with your distributed shardings
        self._update_critics_jitted = jax.jit(
            _critics_wrapper,
            in_shardings=(
                self._data_sharding,  # batch
                self._state_action_critic_state_sharding,  # q_state
                self._value_state_sharding,  # value_state
                self._train_state_sharding,  # policy_state
                self._replicated_sharding,  # rng
            ),
            out_shardings=(
                self._state_action_critic_state_sharding,  # q_state
                self._value_state_sharding,  # value_state
                self._replicated_sharding,  # q_info
                self._replicated_sharding,  # value_info
            ),
            donate_argnums=(1, 2),  # Donates q_state (arg 1) and value_state (arg 2)
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

        if self._config.rl.critic_encoder_type == "pi0_prefix":
            curr_prefix_embedding = self._recompute_prefix_embedding(
                observation=online_observation,
                policy_state=policy_state,
            )
            observation_dict[PREFIX_EMBEDDING_NAME] = curr_prefix_embedding

            next_prefix_embedding = self._recompute_prefix_embedding(
                observation=next_observation, policy_state=policy_state
            )
            next_observation_dict[PREFIX_EMBEDDING_NAME] = next_prefix_embedding
        else:  # resnet
            # Buffer stores images under observation["image"]["base_0_rgb"] / ["left_wrist_0_rgb"]
            # after the pi0 LiberoInputs transform (uint8, HWC).
            observation_dict["image"] = online_observation["image"]["base_0_rgb"]
            observation_dict["wrist_image"] = online_observation["image"]["left_wrist_0_rgb"]
            next_observation_dict["image"] = next_observation["image"]["base_0_rgb"]
            next_observation_dict["wrist_image"] = next_observation["image"]["left_wrist_0_rgb"]

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

        if self._config.rl.critic_encoder_type == "pi0_prefix":
            prefix_embedding = self._recompute_prefix_embedding(
                observation=policy_obs_dict,
                policy_state=policy_state,
            )
            critic_observation[PREFIX_EMBEDDING_NAME] = prefix_embedding
        else:  # resnet
            # policy_obs_dict["image"] is a dict of float32 images in [-1, 1]; convert to uint8.
            def _to_uint8(img):
                return jnp.clip((jnp.asarray(img) + 1.0) * 127.5, 0, 255).astype(jnp.uint8)
            critic_observation["image"] = _to_uint8(policy_obs_dict["image"]["base_0_rgb"])
            critic_observation["wrist_image"] = _to_uint8(policy_obs_dict["image"]["left_wrist_0_rgb"])

        return policy_observation, critic_observation, actions

    def save_episode(self, is_success: bool, env_index: int, task_description: str):
        assert env_index in range(len(self._episode_storage)), \
            f"env_index must be between 0 and {len(self._episode_storage) - 1}, but got {env_index}."
        # extract episode data from storage and empty it
        episode_data = self._episode_storage[env_index]
        self._episode_storage[env_index] = []
        # filtered SFT keeps only successful episodes.
        self._save_episode_in_buffer(episode_data, task_description)

    def _infer_policy_batch_size(self, observations: Dict) -> int:
        first = next(v for k, v in observations.items() if k != "prompt")
        return np.asarray(first).shape[0]

    def sample_actions(self, observations, **kwargs):
        if self.training_steps < self._config.rl.critic_inference_start_step:
            return super().sample_actions(observations, **kwargs)
        n_samples = self._config.rl.n_samples
        rng, self._rng = jax.random.split(self._rng)
        task_description = kwargs.get("task_description")

        # Group envs by task so each _sample_action call gets a single string prompt
        task_to_indices: dict[str, list[int]] = {}
        for i, task in enumerate(task_description):
            task_to_indices.setdefault(task, []).append(i)

        env_num = len(task_description)
        all_best_actions = None

        # Build q-model once, shared across task groups
        q_params = (
            self._state_action_critic_state.ema_params
            if self._state_action_critic_state.ema_params is not None
            else self._state_action_critic_state.params
        )
        q_model = nnx.merge(self._state_action_critic_state.model_def, q_params)
        q_model.eval()

        # Build policy model once for prefix embedding
        params = (
            self._train_state.ema_params
            if self._train_state.ema_params is not None
            else self._train_state.params
        )
        policy_model = nnx.merge(self._train_state.model_def, params)
        policy_model.eval()

        for task, indices in task_to_indices.items():
            group_obs = jax.tree.map(lambda x: x[indices], observations)
            processed_obs = self._process_obs_for_pi0(group_obs, task_description=task)
            group_env_num = len(indices)

            # 1. Tile obs along batch dim and sample all candidates in one pass
            tiled_obs = {
                k: (v if k == "prompt" else np.repeat(np.asarray(v), n_samples, axis=0))
                for k, v in processed_obs.items()
            }
            group_actions = self._sample_action(tiled_obs, rng, self._train_state)
            # group_actions: [group_env_num * n_samples, horizon, dim]

            # 2. Build critic observation (normalize + pad state to match buffer preprocessing)
            raw_state = np.asarray(processed_obs["observation/state"])
            state = np.asarray(self._state_normalize({"state": raw_state})["state"])
            if state.shape[-1] < self._transition_state_dim:
                pad_width = [(0, 0)] * state.ndim
                pad_width[-1] = (0, self._transition_state_dim - state.shape[-1])
                state = np.pad(state, pad_width, mode="constant", constant_values=0.0)
            state = jnp.repeat(jnp.asarray(state, dtype=jnp.float32), n_samples, axis=0)

            if self._config.rl.critic_encoder_type == "pi0_prefix":
                # Compute prefix embedding (on non-tiled obs, then tile)
                inputs = self._policy._input_transform(processed_obs)
                for _mask_key in ("image_mask", "image_masks"):
                    if _mask_key in inputs:
                        inputs[_mask_key] = {k: np.full((group_env_num,), bool(v), dtype=bool) for k, v in inputs[_mask_key].items()}
                for _tok_key in ("tokenized_prompt", "tokenized_prompt_mask", "token_ar_mask", "token_loss_mask"):
                    if _tok_key in inputs and inputs[_tok_key] is not None:
                        arr = np.asarray(inputs[_tok_key])
                        if arr.ndim == 1:
                            inputs[_tok_key] = np.repeat(arr[np.newaxis], group_env_num, axis=0)
                obs_for_prefix = _model.Observation.from_dict(inputs)
                prefix = self._get_prefix_rep_with_model(m=policy_model, observation=obs_for_prefix)
                prefix = np.asarray(prefix)
                if prefix.ndim == 3:
                    prefix = prefix.reshape(prefix.shape[0], -1, prefix.shape[-1]).mean(axis=1)
                # prefix: [group_env_num, embed_dim]
                prefix_tiled = jnp.repeat(jnp.asarray(prefix), n_samples, axis=0)
                critic_obs = {"state": state, PREFIX_EMBEDDING_NAME: prefix_tiled}
            else:  # resnet
                # Use raw images directly; no prefix computation needed
                image = np.repeat(np.asarray(processed_obs["observation/image"], dtype=np.uint8), n_samples, axis=0)
                wrist_image = np.repeat(np.asarray(processed_obs["observation/wrist_image"], dtype=np.uint8), n_samples, axis=0)
                critic_obs = {"state": state, "image": jnp.asarray(image), "wrist_image": jnp.asarray(wrist_image)}

            # 4. Score all candidates with Q-critic
            # Normalize robot-space actions (7-dim for LIBERO), then zero-pad to model_act_dim.
            # This matches the buffer format: _pre_token_transform normalizes first, then
            # LiberoInputs (in non_token_transforms) pads 7 → model_act_dim with zeros.
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
            # q_logits: [num_qs, batch] for Gaussian or [num_qs, batch, K] for Categorical

            # 5. Reduce ensemble, select best per env
            rl_config = self._config.rl
            _lower, _upper = get_value_bounds(self._config)
            q_dist = make_value_distribution(q_logits, rl_config.num_value_bins, _lower, _upper)
            scores = np.asarray(q_dist.mean())  # [num_qs, batch] or [batch]
            if scores.ndim > 1:
                scores = scores.min(axis=0)
            scores = scores.reshape(group_env_num, n_samples)
            best_idx = scores.argmax(axis=1)

            group_actions = np.asarray(group_actions).reshape(group_env_num, n_samples, *np.asarray(group_actions).shape[1:])
            best = group_actions[np.arange(group_env_num), best_idx]

            if all_best_actions is None:
                all_best_actions = np.zeros((env_num, *best.shape[1:]), dtype=np.float32)
            all_best_actions[indices] = np.asarray(best, dtype=np.float32)

        return all_best_actions

    @at.typecheck
    def _get_on_policy_action(
            self,
            online_observation: _model.Observation,
            policy_state: training_utils.TrainState,
            rng: at.KeyArrayLike,
    ) -> _model.Actions:
        model = self._get_policy_model(policy_state)
        sampled_actions = model.sample_actions(
            observation=online_observation,
            rng=rng,
            return_info_dict=False,
            return_prefix_rep=False,
        )
        # model.sample_actions returns normalized model-space actions (batch, horizon, model_act_dim).
        # This matches the buffer format: _pre_token_transform normalizes robot actions then
        # zero-pads to model_act_dim via LiberoInputs (in non_token_transforms).
        return sampled_actions
        
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
        assert isinstance(self._config.rl, BestofNLearnerConfig), "Expected BestofNLearnerConfig for BestofNLearner"
        if self._config.rl.train_on_policy_value_function:
            # We replace the action from the batch with the on policy action
            # This ensures that we train an on policy critic.
            policy_sample_rng, rng = jax.random.split(rng, 2)
            value_actions = self._get_on_policy_action(
                online_observation=_model.Observation.from_dict(batch["observation"]),
                policy_state=policy_state,
                rng=policy_sample_rng,
            )
        else:
            value_actions = batch["actions"]
        # Add prefix representation to the batch for the critic
        batch = self._online_batch_to_critic_batch(
            batch,
            policy_state,
        )

        value_batch = (batch[0], value_actions, batch[2], batch[3], batch[4], batch[5])
        
        # Update the state action critic state
        num_updates = max(self._config.rl.num_critic_updates_per_batch, 1)

        for _ in range(num_updates):
            q_rng, v_rng, rng = jax.random.split(rng, 3)
            # Update the state action critic state
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
                value_batch,
            )

        return q_state, value_state, q_info, value_info

    def pretrain_with_offline_data(self):
        self.warm_start_training_steps += 1
        assert isinstance(self._config.rl, BestofNLearnerConfig), (
            "Only BestofN config should be passed to the best-of-N agent"
        )
        update_critic = False
        critic_info = {}

        if self._offline_data_buffer is None or self._offline_data_buffer.size == 0:
            raise ValueError(
                "Cannot pretrain agent: offline buffer is empty. "
                "Set offline_buffer_load_paths in the config."
            )

        batch = self._offline_data_buffer.sample(
            batch_size=self._config.batch_size
        )
        if update_critic:
            critic_rng, self._rng = jax.random.split(self._rng, 2)
            with sharding.set_mesh(self._mesh):
                q_state, value_state, q_info, value_info = (
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
            critic_info = {f"pretrain/q/{k}": v for k, v in q_info.items()
                            } | {f"pretrain/value/{k}": v for k, v in value_info.items()}
        info = jax.tree.map(np.asarray, critic_info)
        return info

    def _prepare_critic_state_after_pretraining(self):
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

    @at.typecheck
    def update(self) -> dict:
        assert isinstance(self._config.rl, BestofNLearnerConfig), (
            "Only BestofN config should be passed to the best-of-N agent"
        )
        rl_config = self._config.rl

        if rl_config.critic_pre_training_steps == self.training_steps:
            # Reset optimizer state of the value and q function
            q_opt_state = self._state_action_critic_state.tx.init(
                nnx.filter_state(self._state_action_critic_state.params, nnx.Param)
            )
            new_ema_state_action_critic_params = jax.tree.map(jnp.copy, self._state_action_critic_state.params)
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
        if self.debug:
            log_memory_debug("step_start", training_steps=self.training_steps)

        self.training_steps += 1
        update_critic = (
                self.training_steps >= rl_config.critic_training_start_step
                and self.training_steps % rl_config.critic_update_interval == 0
        )

        if not update_critic:
            return {
                "online_buffer_size": jnp.asarray(
                    float(self._online_data_buffer.size), dtype=jnp.float32
                )
            }

        batch_size = self._config.batch_size
        online_batch_size = int(batch_size * min(1.0, self._config.rl.online_ratio))
        use_online = (
                online_batch_size > 0 and self._online_data_buffer.size >= online_batch_size
        )
        use_online = self._online_data_buffer.size >= online_batch_size
        has_offline = self._offline_data_buffer is not None and self._offline_data_buffer.size > 0

        critic_info = {}

        # --- Determine the critic batch (replay-buffer format) ---
        offline_batch = None
        if has_offline:
            offline_batch = self._offline_data_buffer.sample(
                batch_size=batch_size,
            )
        if use_online:
            critic_source_batch = self._online_data_buffer.sample(batch_size=online_batch_size)

            if offline_batch is not None and rl_config.online_ratio < 1.0:
                n_online = min(
                    int(batch_size * rl_config.online_ratio),
                    jax.tree.leaves(critic_source_batch)[0].shape[0],
                )
                n_offline = batch_size - n_online
                critic_source_batch = jax.tree.map(
                    lambda x, y: jnp.concatenate([x[:n_offline], y[:n_online]], axis=0),
                    offline_batch,
                    critic_source_batch,
                )
                # The online batch may be replicated (PartitionSpec()) while
                # the offline batch is sharded. Re-shard the mixed result to
                # match the data sharding expected by _train_step.
                critic_source_batch = jax.device_put(critic_source_batch, self._data_sharding)
                del offline_batch
                gc.collect()
        else:
            critic_source_batch = None
            if offline_batch is not None:
                critic_source_batch = offline_batch
                del offline_batch
                gc.collect()
            
        if critic_source_batch is None:
            return {
                "online_buffer_size": jnp.asarray(
                    float(self._online_data_buffer.size), dtype=jnp.float32
                )
            }

        if update_critic:
            if self.debug:
                log_memory_debug(
                    "before_critics", train_state=self._train_state, batch=critic_source_batch
                )
            critic_rng, self._rng = jax.random.split(self._rng, 2)
            with sharding.set_mesh(self._mesh):
                q_state, value_state, q_info, value_info = (
                    self._update_critics_jitted(
                        critic_source_batch,
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
            if self.debug:
                log_memory_debug("after_update_critics")
        info = (
                critic_info
                | {
                    "online_buffer_size": jnp.asarray(
                        float(self._online_data_buffer.size), dtype=jnp.float32
                    )
                }
        )
        info = jax.tree.map(np.asarray, info)
        return info
