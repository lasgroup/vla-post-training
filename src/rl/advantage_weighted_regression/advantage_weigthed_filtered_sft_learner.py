# ruff: noqa: F722
import functools
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding
from src.rl.advantage_weighted_regression.update_critic import (
    init_state_action_critic_train_state,
    init_state_value_train_state,
    train_q_step,
    train_value_step,
    StateActionCriticDef,
    StateValueDef
)
from src.rl.networks.rl_networks import ObsType, ActionType
from src.rl.filtered_sft_agent.filtered_sft_learner import FilteredSFTLearner
from src.rl.prefix_embedding import PREFIX_EMBEDDING_NAME
from src.training.config import OnlineTrainConfig


class AdvantageWeightedFilteredSFTLearner(FilteredSFTLearner):
    def __init__(self,
                 config: OnlineTrainConfig,
                 dummy_obs: ObsType,
                 dummy_act: ActionType,
                 state_action_critic_def: StateActionCriticDef,
                 state_value_def: StateValueDef,
                 ):
        super().__init__(config)
        self._critic_updates_per_step = self._get_critic_updates_per_step()

        q_init_rng, v_init_rng, self._rng = jax.random.split(self._rng, 3)
        self._state_action_critic_state, self._state_action_critic_state_sharding = (
            init_state_action_critic_train_state(
                self._config,
                q_init_rng,
                self._mesh,
                critic_def=state_action_critic_def,
                dummy_obs=dummy_obs,
                dummy_act=dummy_act
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

    def _get_critic_updates_per_step(self) -> int:
        rl_config = getattr(self._config, "rl", None)
        updates = int(getattr(rl_config, "critic_updates_per_step", 1))
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
            observation=next_observation if isinstance(next_observation, dict) else None,
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

    @at.typecheck
    def _update_critics(self) -> dict[str, at.Array]:
        critic_info: dict[str, at.Array] = {}
        params = (
            self._train_state.ema_params
            if self._train_state.ema_params is not None
            else self._train_state.params
        )
        policy_model = nnx.merge(self._train_state.model_def, params)
        for _ in range(self._critic_updates_per_step):
            transition_batch = self.sample_online_transitions()
            critic_batch = self._online_batch_to_critic_batch(
                transition_batch,
                policy_model=policy_model,
            )
            q_rng, v_rng, self._rng = jax.random.split(self._rng, 3)

            with sharding.set_mesh(self._mesh):
                q_state, q_info = self._q_train_step(
                    q_rng,
                    self._state_action_critic_state,
                    self._value_state,
                    critic_batch,
                )
                value_state, value_info = self._value_train_step(
                    v_rng,
                    self._value_state,
                    q_state,
                    critic_batch,
                )
            self._state_action_critic_state = q_state
            self._value_state = value_state

            current_info = {
                f"critic/q_{key}": value for key, value in q_info.items()
            } | {f"critic/value_{key}": value for key, value in value_info.items()}
            if not critic_info:
                critic_info = current_info
            else:
                critic_info = {
                    key: critic_info[key] + current_info[key] for key in critic_info
                }

        if self._critic_updates_per_step > 1:
            critic_info = {
                key: value / self._critic_updates_per_step
                for key, value in critic_info.items()
            }

        critic_info["critic/online_buffer_size"] = jnp.asarray(
            float(self._online_data_buffer.size), dtype=jnp.float32
        )
        return critic_info

    @at.typecheck
    def update(self) -> dict[str, at.Array]:
        info = super().update()
        if self._online_data_buffer.size < self._online_data_buffer.batch_size:
            return info | {
                "critic/online_buffer_size": jnp.asarray(
                    float(self._online_data_buffer.size), dtype=jnp.float32
                )
            }

        critic_info = self._update_critics()
        return info | critic_info
