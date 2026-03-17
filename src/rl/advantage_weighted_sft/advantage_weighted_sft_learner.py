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
from src.rl.advantage_weighted_sft.update_actor import (
    train_step as train_actor_step,
)
from src.rl.advantage_weighted_sft.update_critic import (
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
from src.training.config import OnlineTrainConfig, AdvantageWeightedSFTLearnerConfig


class AdvantageWeightedSFTLearner(FilteredSFTLearner):
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
        self._train_step = functools.partial(train_actor_step, self._config)

        # 2. Create closures to drop 'self' from the JIT signature
        def _critics_wrapper(batch, q_state, value_state, policy_state, rng):
            return self._update_critics(
                batch=batch,
                q_state=q_state,
                value_state=value_state,
                policy_state=policy_state,
                rng=rng,
            )

        def _policy_wrapper(batch, policy_state, q_state, value_state, rng, mc_return):
            return self._update_policy(
                batch=batch,
                policy_state=policy_state,
                q_state=q_state,
                value_state=value_state,
                rng=rng,
                mc_return=mc_return,
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

        self._update_policy_jitted = jax.jit(
            _policy_wrapper,
            in_shardings=(
                self._data_sharding,  # batch
                self._train_state_sharding,  # policy_state
                self._state_action_critic_state_sharding,  # q_state
                self._value_state_sharding,  # value_state
                self._replicated_sharding,  # rng
                self._data_sharding,  # mc_return
            ),
            out_shardings=(
                self._train_state_sharding,  # policy_state
                self._replicated_sharding,  # info
            ),
            donate_argnums=(1,),  # Donates policy_state (arg 1)
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
        assert env_index in range(
            len(self._episode_storage)
        ), f"env_index must be between 0 and {len(self._episode_storage) - 1}, but got {env_index}."
        # extract episode data from storage and empty it
        episode_data = self._episode_storage[env_index]
        self._episode_storage[env_index] = []
        # filtered SFT keeps only successful episodes.
        self._save_episode_in_buffer(episode_data, task_description)

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
        assert isinstance(self._config.rl, AdvantageWeightedSFTLearnerConfig)
        num_updates = max(self._config.rl.num_critic_updates_per_batch, 1)
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
        )

        return policy_state, info

    @at.typecheck
    def update(self) -> dict:
        assert isinstance(self._config.rl, AdvantageWeightedSFTLearnerConfig), (
            "Only Advantage SFT config should " "be passed to the filtered SFT agent"
        )
        rl_config = self._config.rl
        if self.debug:
            log_memory_debug("step_start", training_steps=self.training_steps)

        if rl_config.critic_pre_training_steps == self.training_steps:
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
            self._config.batch_size * min(1.0, self._config.rl.online_ratio)
        )
        use_online = self._online_data_buffer.size >= online_batch_size

        critic_info, actor_info = {}, {}
        mc_return = None
        if use_online:
            online_batch = self._online_data_buffer.sample(batch_size=online_batch_size)
            if update_critic:
                if self.debug:
                    log_memory_debug(
                        "before_critics",
                        train_state=self._train_state,
                        batch=online_batch,
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
            if rl_config.use_mc_returns:
                mc_return = online_batch["mc_return"]
                assert rl_config.online_ratio >= 1.0, (
                    "use_mc_returns requires online_ratio >= 1.0 "
                    "(MC returns are not available for offline data)"
                )
            online_batch = self._online_batch_to_sft_batch(online_batch)
            online_ratio = rl_config.online_ratio
            if online_ratio >= 1.0:
                batch = online_batch
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
                del online_batch
                gc.collect()
            else:
                batch = next(self._data_iter)
        else:
            batch = next(self._data_iter)
        if update_policy:
            if self.debug:
                log_memory_debug("before_update_policy")
            policy_rng, self._rng = jax.random.split(self._rng, 2)
            with sharding.set_mesh(self._mesh):
                policy_state, actor_info = self._update_policy_jitted(
                    batch,
                    self._train_state,
                    self._state_action_critic_state,
                    self._value_state,
                    policy_rng,
                    mc_return,
                )

            self._train_state = policy_state
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
