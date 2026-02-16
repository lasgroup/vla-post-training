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
    StateActionValueCritic,
    StateValueCritic,
    critic_hidden_dims,
    init_critic_train_state,
    train_q_step,
    train_value_step,
)
from src.rl.legacy_filtered_sft_agent.legacy_filtered_sft_learner import (
    LegacyFilteredSFTLearner,
)
from src.training.config import OnlineTrainConfig


class AdvantageWeightedFilteredSFTLearner(LegacyFilteredSFTLearner):
    def __init__(self, config: OnlineTrainConfig):
        super().__init__(config)
        self._critic_updates_per_step = self._get_critic_updates_per_step()

        obs_spec, action_spec = self._config.model.inputs_spec(batch_size=1)
        state_dim = int(obs_spec.state.shape[-1])
        action_horizon = int(action_spec.shape[-2])
        action_dim = int(action_spec.shape[-1])
        hidden_dims = critic_hidden_dims(self._config)

        q_init_rng, v_init_rng, self._rng = jax.random.split(self._rng, 3)
        self._state_action_critic_state, self._state_action_critic_state_sharding = (
            init_critic_train_state(
                self._config,
                q_init_rng,
                self._mesh,
                critic_factory=lambda rng: StateActionValueCritic(
                    state_dim=state_dim,
                    action_horizon=action_horizon,
                    action_dim=action_dim,
                    hidden_dims=hidden_dims,
                    rngs=nnx.Rngs(rng),
                ),
            )
        )
        self._value_state, self._value_state_sharding = init_critic_train_state(
            self._config,
            v_init_rng,
            self._mesh,
            critic_factory=lambda rng: StateValueCritic(
                state_dim=state_dim,
                hidden_dims=hidden_dims,
                rngs=nnx.Rngs(rng),
            ),
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

    @at.typecheck
    def _online_batch_to_critic_batch(self, online_batch: dict[str, Any]) -> tuple[
        _model.Observation,
        _model.Actions,
        at.Float[at.Array, "b s"],
        at.Float[at.Array, " b"],
        at.Float[at.Array, " b"],
    ]:
        online_observation = online_batch["observation"]
        observation_dict: dict[str, Any] = {
            "image": dict(online_observation["image"]),
            "image_mask": dict(online_observation["image_mask"]),
            "state": online_observation["state"],
        }
        for key in (
            "tokenized_prompt",
            "tokenized_prompt_mask",
            "token_ar_mask",
            "token_loss_mask",
        ):
            if key in online_observation:
                observation_dict[key] = online_observation[key]

        return (
            _model.Observation.from_dict(observation_dict),
            online_batch["actions"],
            online_batch["next_observation"]["state"],
            online_batch["reward"],
            online_batch["discount"],
        )

    @at.typecheck
    def _update_critics(self) -> dict[str, at.Array]:
        critic_info: dict[str, at.Array] = {}
        for _ in range(self._critic_updates_per_step):
            transition_batch = self.sample_online_transitions()
            critic_batch = self._online_batch_to_critic_batch(transition_batch)
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
