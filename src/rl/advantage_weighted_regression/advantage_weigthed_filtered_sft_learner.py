# ruff: noqa: F722
import functools
from typing import Any
import gc

import flax.nnx as nnx
import jax
import jax.numpy as jnp

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding
from src.rl.advantage_weighted_regression.update_actor import (
    train_step as train_actor_step,
)
from src.rl.advantage_weighted_regression.update_critic import (
    init_state_action_critic_train_state,
    init_state_value_train_state,
    train_q_step,
    train_value_step,
    StateActionCriticDef,
    StateValueDef,
    CriticBatch,
)
from src.rl.networks.rl_networks import ObsType, ActionType
from src.rl.filtered_sft_agent.filtered_sft_learner import FilteredSFTLearner
from src.rl.prefix_embedding import PREFIX_EMBEDDING_NAME
from src.training.config import OnlineTrainConfig


class AdvantageWeightedFilteredSFTLearner(FilteredSFTLearner):
    def __init__(
        self,
        config: OnlineTrainConfig,
        dummy_obs: ObsType,
        dummy_act: ActionType,
        state_action_critic_def: StateActionCriticDef,
        state_value_def: StateValueDef,
    ):
        super().__init__(config)
        self._critic_update_frequency = self._get_critic_update_frequency()
        self._policy_update_frequency = self._get_policy_update_frequency()

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

        self._q_train_step = jax.jit(
            functools.partial(train_q_step, self._config),
            in_shardings=(
                self._replicated_sharding,
                self._state_action_critic_state_sharding,
                self._value_state_sharding,
                self._data_sharding,
            ),
            out_shardings=(
                self._state_action_critic_state_sharding,
                self._replicated_sharding,
            ),
            donate_argnums=(1,),
        )
        self._value_train_step = jax.jit(
            functools.partial(train_value_step, self._config),
            in_shardings=(
                self._replicated_sharding,
                self._value_state_sharding,
                self._state_action_critic_state_sharding,
                self._data_sharding,
            ),
            out_shardings=(self._value_state_sharding, self._replicated_sharding),
            donate_argnums=(1,),
        )
        del self._train_step
        gc.collect()
        self._train_step = jax.jit(
            functools.partial(train_actor_step, self._config),
            in_shardings=(
                self._replicated_sharding,
                self._train_state_sharding,
                self._state_action_critic_state_sharding,
                self._value_state_sharding,
                self._data_sharding,
            ),
            out_shardings=(
                self._train_state_sharding,
                self._replicated_sharding,
            ),
            donate_argnums=(1,),
        )

    def _get_critic_update_frequency(self) -> int:
        rl_config = getattr(self._config, "rl", None)
        updates = int(getattr(rl_config, "critic_update_frequency", 1))
        return max(1, updates)

    def _get_policy_update_frequency(self) -> int:
        rl_config = getattr(self._config, "rl", None)
        updates = int(getattr(rl_config, "policy_update_frequency", 1))
        return max(1, updates)

    def _build_model_observation(
        self, observation: dict[str, Any]
    ) -> _model.Observation | None:
        if not isinstance(observation, dict):
            return None
        if "image" not in observation or "image_mask" not in observation:
            return None
        if "state" not in observation:
            return None

        model_observation: dict[str, Any] = {
            "image": dict(observation["image"]),
            "image_mask": dict(observation["image_mask"]),
            "state": observation["state"],
        }
        for key in (
            "tokenized_prompt",
            "tokenized_prompt_mask",
            "token_ar_mask",
            "token_loss_mask",
        ):
            if key in observation:
                model_observation[key] = observation[key]

        return _model.Observation.from_dict(model_observation)

    def _recompute_prefix_embedding(
        self,
        *,
        model: _model.BaseModel,
        observation: dict[str, Any] | None,
    ) -> jax.Array | None:
        if observation is None:
            return None
        model_observation = self._build_model_observation(observation)
        if model_observation is None:
            return None
        prefix = self._get_prefix_rep_with_model(
            m=model,
            observation=model_observation,
        )
        prefix = jnp.asarray(prefix, dtype=jnp.float32)
        if prefix.ndim == 1:
            return prefix[jnp.newaxis, :]
        if prefix.ndim == 2:
            return prefix
        # Prefix reps are usually [B, S, E]; pool token axis to [B, E].
        prefix = prefix.reshape((prefix.shape[0], -1, prefix.shape[-1]))
        return jnp.mean(prefix, axis=1)

    @at.typecheck
    def _online_batch_to_critic_batch(
        self,
        online_batch: dict[str, Any],
        *,
        policy_model: _model.BaseModel,
    ) -> tuple[
        ObsType,
        _model.Actions,
        ObsType,
        at.Float[at.Array, " b"],
        at.Float[at.Array, " b"],
    ]:
        online_observation = online_batch["observation"]
        observation_dict: dict[str, Any] = {
            "state": online_observation["state"],
        }
        curr_prefix_embedding = self._recompute_prefix_embedding(
            model=policy_model,
            observation=online_observation,
        )
        if curr_prefix_embedding is not None:
            observation_dict[PREFIX_EMBEDDING_NAME] = curr_prefix_embedding

        next_observation = online_batch["next_observation"]
        next_observation_dict: dict[str, Any] = {"state": next_observation["state"]}
        next_prefix_embedding = self._recompute_prefix_embedding(
            model=policy_model,
            observation=(
                next_observation if isinstance(next_observation, dict) else None
            ),
        )
        if next_prefix_embedding is not None:
            next_observation_dict[PREFIX_EMBEDDING_NAME] = next_prefix_embedding

        return (
            observation_dict,
            online_batch["actions"],
            next_observation_dict,
            online_batch["reward"],
            online_batch["discount"],
        )

    def _sft_batch_to_actor_batch(
        self,
        sft_batch: tuple[_model.Observation, _model.Actions],
        *,
        policy_model: _model.BaseModel,
    ) -> tuple[_model.Observation, ObsType, _model.Actions]:
        policy_observation, actions = sft_batch
        if isinstance(policy_observation, _model.Observation):
            pass
        elif hasattr(policy_observation, "to_dict"):
            policy_observation = _model.Observation.from_dict(
                dict(policy_observation.to_dict())
            )
        elif isinstance(policy_observation, dict):
            policy_observation = _model.Observation.from_dict(dict(policy_observation))
        else:
            raise TypeError(
                "Unsupported observation type for actor update: "
                f"{type(policy_observation)}."
            )
        policy_obs_dict = policy_observation.to_dict()

        critic_observation: dict[str, Any] = {
            "state": policy_obs_dict["state"],
        }
        prefix_embedding = self._recompute_prefix_embedding(
            model=policy_model,
            observation=policy_obs_dict,
        )
        if prefix_embedding is not None:
            critic_observation[PREFIX_EMBEDDING_NAME] = prefix_embedding

        return policy_observation, critic_observation, actions

    @at.typecheck
    def _update_critics(self, batch: CriticBatch) -> dict[str, at.Array]:
        q_rng, v_rng, self._rng = jax.random.split(self._rng, 3)
        with sharding.set_mesh(self._mesh):
            q_state, q_info = self._q_train_step(
                q_rng,
                self._state_action_critic_state,
                self._value_state,
                batch,
            )
            value_state, value_info = self._value_train_step(
                v_rng,
                self._value_state,
                q_state,
                batch,
            )
        self._state_action_critic_state = q_state
        self._value_state = value_state
        current_info = {
                f"critic/q_{key}": value for key, value in q_info.items()
            } | {f"critic/value_{key}": value for key, value in value_info.items()}
        return current_info

    def _update_policy(self, batch: tuple[_model.Observation, ObsType, _model.Actions]):
        train_rng, self._rng = jax.random.split(self._rng)
        with sharding.set_mesh(self._mesh):
            train_state, actor_info = self._train_step(
                train_rng,
                self._train_state,
                self._state_action_critic_state,
                self._value_state,
                batch,
            )
        self._train_state = train_state
        info = {f"actor/{key}": value for key, value in actor_info.items()}
        return info

    @at.typecheck
    def update(self) -> dict[str, at.Array]:
        self.training_steps += 1
        update_critic = self._critic_update_frequency % self.training_steps == 0
        update_policy = self._policy_update_frequency % self.training_steps == 0
        if not update_critic and not update_policy:
            return {'online_buffer_size': jnp.asarray(
                    float(self._online_data_buffer.size), dtype=jnp.float32)}

        batch = next(self._data_iter)
        use_online = (
            self._online_data_buffer.size >= self._online_data_buffer.batch_size
        )
        params = (
            self._train_state.ema_params
            if self._train_state.ema_params is not None
            else self._train_state.params
        )
        policy_model = nnx.merge(self._train_state.model_def, params)
        critic_info, actor_info = {}, {}
        if use_online:
            online_batch_raw = self._online_data_buffer.sample()
            if update_critic:
                critic_batch = self._online_batch_to_critic_batch(
                    online_batch_raw,
                    policy_model=policy_model,
                )
                critic_info = self._update_critics(critic_batch)
            online_batch = self._online_batch_to_sft_batch(online_batch_raw)
            online_ratio = float(getattr(self._config.collect, "online_ratio", 0.5))
            if online_ratio >= 1.0:
                batch = online_batch
            elif online_ratio > 0:
                batch = jax.tree.map(
                    lambda x, y: jnp.concatenate([x, y], axis=0),
                    batch,
                    online_batch,
                )
        if update_policy:
            actor_batch = self._sft_batch_to_actor_batch(
                batch,
                policy_model=policy_model,
            )
            actor_info = self._update_policy(actor_batch)
        return actor_info | critic_info | {'online_buffer_size': jnp.asarray(
                    float(self._online_data_buffer.size), dtype=jnp.float32
                )}
