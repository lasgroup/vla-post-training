import gc
import functools

import jax
import jax.numpy as jnp
import numpy as np
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding

from src.rl.flow_grpo.update_actor import train_step as flow_grpo_train_step
from src.rl.mpo_weighted_sft.mpo_weighted_sft_learner import MPOWeightedSFTLearner
from src.training.config import FlowGRPOSFTLearnerConfig


class FlowGRPOLearner(MPOWeightedSFTLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_step = functools.partial(flow_grpo_train_step, self._config)
        self._refresh_update_functions()

    @at.typecheck
    def update(self) -> dict:
        assert isinstance(self._config.rl, FlowGRPOSFTLearnerConfig)
        rl_config = self._config.rl

        self.training_steps += 1
        update_critic = (
            self.training_steps >= rl_config.critic.training_start_step
            and self.training_steps % rl_config.critic.update_interval == 0
        )
        update_policy = (
            self.training_steps >= rl_config.policy.training_start_step
            and self.training_steps % rl_config.policy.update_interval == 0
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
        if use_online:
            online_batch = self._online_data_buffer.sample(batch_size=online_batch_size)
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
            online_batch = self._online_batch_to_sft_batch(online_batch)
            online_ratio = rl_config.online_ratio
            if online_ratio >= 1.0:
                batch = online_batch
            elif online_ratio > 0:
                batch = next(self._data_iter)
                first_leaf = jax.tree.leaves(batch)[0]
                batch_size = first_leaf.shape[0]
                n_online = min(
                    int(batch_size * online_ratio),
                    jax.tree.leaves(online_batch)[0].shape[0],
                )
                n_offline = batch_size - n_online
                policy_batch = jax.tree.map(
                    lambda x, y: jnp.concatenate([x[:n_offline], y[:n_online]], axis=0),
                    batch,
                    online_batch,
                )
                del online_batch
                gc.collect()
            else:
                batch = next(self._data_iter)
        if update_policy:
            # When group_size > 1, the flow GRPO train_step internally repeats
            # each sample group_size times. Randomly subsample the batch so that
            # reduced_batch_size * group_size == original_batch_size.
            if rl_config.group_size > 1 and not rl_config.use_mpo_advantage_weight:
                subsample_rng, self._rng = jax.random.split(self._rng, 2)
                batch_size = self._config.batch_size
                reduced_size = batch_size // rl_config.group_size
                indices = jax.random.permutation(subsample_rng, batch_size)[
                    :reduced_size
                ]
                policy_batch = jax.tree.map(lambda x: x[indices], batch)
                policy_batch = jax.device_put(policy_batch, self._data_sharding)
            else:
                policy_batch = jax.device_put(batch, self._data_sharding)

            policy_rng, self._rng = jax.random.split(self._rng, 2)
            with sharding.set_mesh(self._mesh):
                policy_state, actor_info = self._update_policy_jitted(
                    policy_batch,
                    self._train_state,
                    self._state_action_critic_state,
                    self._value_state,
                    policy_rng,
                    None,
                )
            self._train_state = policy_state
            self._maybe_restore_policy_ema_after_resume()
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
