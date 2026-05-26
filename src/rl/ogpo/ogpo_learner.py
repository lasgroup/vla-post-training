# ruff: noqa: F722
"""OGPO learner: PPO on a Pi05 flow policy with a BC anchor.

This learner inherits the dual-critic + EMA + checkpointing plumbing from
``AdvantageWeightedSFTLearner`` and only swaps the actor train step for
``src/rl/ogpo/update_actor.py``. v1 is strictly on-policy
(``online_ratio == 1.0``) with no success buffer and no offline data path.
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np

import openpi.shared.array_typing as at
import openpi.training.sharding as sharding
from src.rl.advantage_weighted_sft.advantage_weighted_sft_learner import (
    AdvantageWeightedSFTLearner,
)
from src.rl.ogpo.update_actor import train_step as ogpo_train_step
from src.training.config import OGPOSFTLearnerConfig


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
        self._train_step = functools.partial(ogpo_train_step, self._config)
        self._refresh_update_functions()

    @at.typecheck
    def update(self) -> dict:
        rl_config = self._config.rl
        assert isinstance(rl_config, OGPOSFTLearnerConfig)

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

        # online_ratio is pinned to 1.0 (asserted at __init__).
        online_batch_size = self._config.batch_size
        if self._online_data_buffer.size < online_batch_size:
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
            critic_info = (
                {f"critic/q_{k}": v for k, v in q_info.items()}
                | {f"critic/value_{k}": v for k, v in value_info.items()}
            )

        if update_policy:
            policy_batch = self._online_batch_to_sft_batch(online_batch)
            policy_rng, self._rng = jax.random.split(self._rng, 2)
            with sharding.set_mesh(self._mesh):
                policy_state, actor_info = self._update_policy_jitted(
                    policy_batch,
                    self._train_state,
                    self._state_action_critic_state,
                    self._value_state,
                    policy_rng,
                    None,        # mc_return: unused by OGPO v1
                    1.0,         # scale: unused by OGPO v1
                )
            self._train_state = policy_state
            self._maybe_restore_policy_ema_after_resume()
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
        return jax.tree.map(np.asarray, info)

    def save_episode(self, is_success: bool, env_index: int, task_description: str):
        """Store every rollout, regardless of success. v1 has no success filter."""
        assert env_index in range(len(self._episode_storage)), (
            f"env_index must be between 0 and {len(self._episode_storage) - 1}, "
            f"but got {env_index}."
        )
        episode_data = self._episode_storage[env_index]
        self._episode_storage[env_index] = []
        self._save_episode_in_buffer(episode_data, task_description)
