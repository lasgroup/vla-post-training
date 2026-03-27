import dataclasses
import functools
import gc

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
from src.rl.grouped_mpo.update_actor import train_step as grouped_mpo_train_step
from src.rl.mpo_weighted_sft.mpo_weighted_sft_learner import MPOWeightedSFTLearner
from src.training.config import GroupedMPOWeightedSFTLearnerConfig


class GroupedMPOWeightedSFTLearner(MPOWeightedSFTLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self._update_policy_jitted
        gc.collect()

        self._train_step = functools.partial(grouped_mpo_train_step, self._config)

        def _policy_wrapper(batch, policy_state, q_state, value_state, rng):
            return self._update_policy(
                batch=batch,
                policy_state=policy_state,
                q_state=q_state,
                value_state=value_state,
                rng=rng,
            )

        self._update_policy_jitted = jax.jit(
            _policy_wrapper,
            in_shardings=(
                self._data_sharding,
                self._train_state_sharding,
                self._state_action_critic_state_sharding,
                self._value_state_sharding,
                self._replicated_sharding,
            ),
            out_shardings=(
                self._train_state_sharding,
                self._replicated_sharding,
            ),
            donate_argnums=(1,),
        )

    def _update_policy(
        self,
        batch,
        policy_state,
        q_state,
        value_state,
        rng,
        **kwargs,
    ):
        # Skip MPO's _get_on_policy_action — grouped MPO train_step samples
        # its own actions internally, so the MPO on-policy action is wasted.
        batch = self._sft_batch_to_actor_batch(
            batch,
            policy_state=policy_state,
        )
        policy_state, info = self._train_step(
            rng,
            policy_state,
            q_state,
            value_state,
            batch,
        )
        return policy_state, info

    @at.typecheck
    def _get_on_policy_action(
        self,
        online_observation: _model.Observation,
        policy_state: training_utils.TrainState,
        rng: at.KeyArrayLike,
    ) -> _model.Actions:
        model = self._get_policy_model(policy_state)
        rl_config = self._config.rl
        assert isinstance(rl_config, GroupedMPOWeightedSFTLearnerConfig)
        if rl_config.align_critic_sampling:
            sample_rng, noise_rng = jax.random.split(rng)
            batch_size = online_observation.state.shape[0]
            noise = jax.random.normal(
                noise_rng,
                (batch_size, model.action_horizon, model.action_dim),
            )
            return model.sample_actions(
                rng=sample_rng,
                observation=online_observation,
                noise=noise,
                num_steps=rl_config.num_steps,
                noise_level=rl_config.noise_level,
                return_info_dict=False,
                return_prefix_rep=False,
            )
        return model.sample_actions(
            observation=online_observation,
            rng=rng,
            return_info_dict=False,
            return_prefix_rep=False,
        )

    @at.typecheck
    def update(self) -> dict:
        assert isinstance(self._config.rl, GroupedMPOWeightedSFTLearnerConfig)
        rl_config = self._config.rl

        if rl_config.critic_pre_training_steps == self.training_steps:
            q_opt_state = self._state_action_critic_state.tx.init(
                nnx.filter_state(self._state_action_critic_state.params, nnx.Param)
            )
            new_ema_q_params = jax.tree.map(
                jnp.copy, self._state_action_critic_state.params
            )
            self._state_action_critic_state = dataclasses.replace(
                self._state_action_critic_state,
                opt_state=q_opt_state,
                ema_params=new_ema_q_params,
            )
            del new_ema_q_params, q_opt_state

            v_opt_state = self._value_state.tx.init(
                nnx.filter_state(self._value_state.params, nnx.Param)
            )
            new_ema_v_params = jax.tree.map(jnp.copy, self._value_state.params)
            self._value_state = dataclasses.replace(
                self._value_state,
                opt_state=v_opt_state,
                ema_params=new_ema_v_params,
            )
            del new_ema_v_params, v_opt_state

        self.training_steps += 1
        critic_frozen = (
            rl_config.freeze_critic_at_step is not None
            and self.training_steps >= rl_config.freeze_critic_at_step
        )
        update_critic = (
            not critic_frozen
            and self.training_steps >= rl_config.critic_training_start_step
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

        batch = next(self._data_iter)
        online_batch_size = int(self._config.batch_size * min(1.0, rl_config.online_ratio))
        use_online = online_batch_size > 0 and self._online_data_buffer.size >= online_batch_size

        critic_info, actor_info = {}, {}
        if use_online:
            online_batch = self._online_data_buffer.sample(batch_size=online_batch_size)
            if update_critic:
                critic_rng, self._rng = jax.random.split(self._rng, 2)
                with sharding.set_mesh(self._mesh):
                    q_state, value_state, q_info, value_info = self._update_critics_jitted(
                        online_batch,
                        self._state_action_critic_state,
                        self._value_state,
                        self._train_state,
                        critic_rng,
                    )
                self._state_action_critic_state = q_state
                self._value_state = value_state
                critic_info = {
                    f"critic/q_{key}": value for key, value in q_info.items()
                } | {f"critic/value_{key}": value for key, value in value_info.items()}
            # Filter to successful episodes only for the policy update if configured.
            if rl_config.policy_only_successful:
                success_mask = online_batch["is_success"] > 0.5
                n_success = int(jnp.sum(success_mask))
                if n_success > 0:
                    success_indices = jnp.where(success_mask, size=n_success)[0]
                    online_batch = jax.tree.map(lambda x: x[success_indices], online_batch)
            online_batch = self._online_batch_to_sft_batch(online_batch)
            online_ratio = rl_config.online_ratio
            if online_ratio >= 1.0:
                batch = online_batch
            elif online_ratio > 0:
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
                batch = jax.device_put(batch, self._data_sharding)
                del online_batch
                gc.collect()

        if update_policy:
            group_size = max(rl_config.group_size, 1)
            if group_size > 1:
                subsample_rng, self._rng = jax.random.split(self._rng, 2)
                batch_size = int(jax.tree.leaves(batch)[0].shape[0])
                if batch_size % group_size != 0:
                    raise ValueError(
                        "Grouped MPO actor batch size must be divisible by the "
                        f"group size: {batch_size} vs {group_size}."
                    )
                reduced_size = batch_size // group_size
                indices = jax.random.permutation(subsample_rng, batch_size)[:reduced_size]
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
                )
            self._train_state = policy_state
            actor_info = {f"actor/{key}": value for key, value in actor_info.items()}

        info = (
            actor_info
            | critic_info
            | {
                "online_buffer_size": jnp.asarray(
                    float(self._online_data_buffer.size), dtype=jnp.float32
                ),
                "critic_frozen": jnp.asarray(float(critic_frozen), dtype=jnp.float32),
            }
        )
        info = jax.tree.map(np.asarray, info)
        return info
