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

        curr_prefix_embedding = self._recompute_prefix_embedding(
            observation=online_observation,
            policy_state=policy_state,
        )

        observation_dict[PREFIX_EMBEDDING_NAME] = curr_prefix_embedding

        next_observation = online_batch["next_observation"]
        next_observation_dict: dict[str, Any] = {"state": next_observation["state"]}

        next_prefix_embedding = self._recompute_prefix_embedding(
            observation=next_observation, policy_state=policy_state
        )

        next_observation_dict[PREFIX_EMBEDDING_NAME] = next_prefix_embedding

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

    def save_episode(self, is_success: bool, env_index: int, task_description: str):
        assert env_index in range(len(self._episode_storage)), \
            f"env_index must be between 0 and {len(self._episode_storage) - 1}, but got {env_index}."
        # extract episode data from storage and empty it
        episode_data = self._episode_storage[env_index]
        self._episode_storage[env_index] = []
        # filtered SFT keeps only successful episodes.
        self._save_episode_in_buffer(episode_data, task_description)

    def sample_actions(self, observations, **kwargs):
        if self.training_steps < self._config.rl.critic_inference_start_step:
            return super().sample_actions(observations, **kwargs)
        n_samples = self._config.rl.n_samples
        rng, self._rng = jax.random.split(self._rng)
        task_description = kwargs.get("task_description")
        processed_obs = self._process_obs_for_pi0(
            observations, task_description=task_description
        )
        env_num = self._infer_policy_batch_size(processed_obs)

        # 1. Tile obs along batch dim and sample all candidates in one pass
        # "prompt" is a string/scalar and must not be tiled.
        tiled_obs = {
            k: (v if k == "prompt" else np.repeat(np.asarray(v), n_samples, axis=0))
            for k, v in processed_obs.items()
        }
        all_actions = self._sample_action(
            tiled_obs, rng, self._train_state, batch_actions=False
        )
        # all_actions: [env_num * n_samples, horizon, dim]

        # 2. Compute prefix embedding (on non-tiled obs, then tile)
        params = (
            self._train_state.ema_params
            if self._train_state.ema_params is not None
            else self._train_state.params
        )
        model = nnx.merge(self._train_state.model_def, params)
        model.eval()
        prefix = self._compute_prefix_rep_with_model(model=model, observations=processed_obs)
        # Pool from [env_num, tokens, embed_dim] -> [env_num, embed_dim]
        prefix = np.asarray(prefix)
        if prefix.ndim == 3:
            prefix = prefix.reshape(prefix.shape[0], -1, prefix.shape[-1]).mean(axis=1)
        # prefix: [env_num, embed_dim]

        # 3. Build critic observation (normalize + pad state to match buffer preprocessing)
        raw_state = np.asarray(processed_obs["observation/state"])
        state = np.asarray(self._state_normalize({"state": raw_state})["state"])
        if state.shape[-1] < self._transition_state_dim:
            pad_width = [(0, 0)] * state.ndim
            pad_width[-1] = (0, self._transition_state_dim - state.shape[-1])
            state = np.pad(state, pad_width, mode="constant", constant_values=0.0)
        state = jnp.repeat(jnp.asarray(state, dtype=jnp.float32), n_samples, axis=0)
        prefix_tiled = jnp.repeat(jnp.asarray(prefix), n_samples, axis=0)
        critic_obs = {"state": state, PREFIX_EMBEDDING_NAME: prefix_tiled}

        # 4. Score all candidates with Q-critic
        q_params = (
            self._state_action_critic_state.ema_params
            if self._state_action_critic_state.ema_params is not None
            else self._state_action_critic_state.params
        )
        q_model = nnx.merge(self._state_action_critic_state.model_def, q_params)
        q_model.eval()
        # Normalize and pad actions to match buffer preprocessing (policy outputs are unnormalized, 7-dim)
        actions_norm = np.asarray(
            self._action_normalize({"actions": np.asarray(all_actions)})["actions"]
        )
        actions_norm = np.pad(
            actions_norm,
            [(0, 0)] * (actions_norm.ndim - 1) + [(0, max(0, self._act_dim - actions_norm.shape[-1]))],
            mode="constant",
        )
        flat_actions = jnp.asarray(actions_norm).reshape(env_num * n_samples, -1)
        q_values = np.asarray(q_model(critic_obs, flat_actions))
        # q_values: [num_qs, env_num * n_samples]

        # 5. Reduce ensemble, select best per env
        if q_values.ndim > 1:
            q_values = q_values.min(axis=0)          # [env_num * n_samples]
        q_values = q_values.reshape(env_num, n_samples)
        best_idx = q_values.argmax(axis=1)            # [env_num]

        all_actions = np.asarray(all_actions).reshape(env_num, n_samples, *all_actions.shape[1:])
        best_actions = all_actions[np.arange(env_num), best_idx]  # [env_num, horizon, dim]
        return np.asarray(best_actions, dtype=np.float32)

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
        
        # Update the state action critic state
        num_updates = max(self._config.rl.num_critic_updates_per_batch, 1)

        value_batch = (batch[0], value_actions, batch[2], batch[3], batch[4], batch[5])

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

    @at.typecheck
    def update(self) -> dict:
        assert isinstance(self._config.rl, BestofNLearnerConfig), "Only BestofN config should " \
                                                                                "be passed to the best-of-N agent"
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

        online_batch_size = int(self._config.batch_size * min(1.0, self._config.rl.online_ratio))
        use_online = (
                self._online_data_buffer.size >= online_batch_size
        )

        critic_info = {}
        if use_online:
            online_batch = self._online_data_buffer.sample(batch_size=online_batch_size)
            if update_critic:
                if self.debug:
                    log_memory_debug(
                        "before_critics", train_state=self._train_state, batch=online_batch
                    )
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
