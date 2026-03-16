import gc
import functools
import dataclasses
from src.rl.mpo_weighted_sft.mpo_weighted_sft_learner import MPOWeightedSFTLearner
from src.rl.flow_grpo.update_actor import train_step as flow_grpo_train_step
from src.training.config import FlowGRPOSFTLearnerConfig
import flax.nnx as nnx
import openpi.training.sharding as sharding
import jax
import jax.numpy as jnp
import numpy as np
import openpi.shared.array_typing as at
import openpi.training.utils as training_utils
import openpi.models.model as _model


class FlowGRPOLearner(MPOWeightedSFTLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Delete only the policy JIT method; keep _update_critics_jitted from parent
        del self._update_policy_jitted
        gc.collect()

        # Override _train_step with the flow GRPO actor update
        self._train_step = functools.partial(flow_grpo_train_step, self._config)

        # Re-create policy JIT wrapper
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
                self._data_sharding,  # batch
                self._train_state_sharding,  # policy_state
                self._state_action_critic_state_sharding,  # q_state
                self._value_state_sharding,  # value_state
                self._replicated_sharding,  # rng
            ),
            out_shardings=(
                self._train_state_sharding,  # policy_state
                self._replicated_sharding,  # info
            ),
            donate_argnums=(1,),
        )

    @staticmethod
    def _get_policy_model(policy_state: training_utils.TrainState) -> _model.BaseModel:
        model = nnx.merge(policy_state.model_def, policy_state.params)
        model.eval()
        return model

    def _use_ema_for_data_collection(self) -> bool:
        return False

    def _use_ema_for_evaluation(self) -> bool:
        return self._train_state.ema_params is not None

    @at.typecheck
    def update(self) -> dict:
        assert isinstance(self._config.rl, FlowGRPOSFTLearnerConfig)
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

        batch = next(self._data_iter)
        online_batch_size = int(self._config.batch_size * min(1.0, rl_config.online_ratio))
        use_online = online_batch_size > 0 and self._online_data_buffer.size >= online_batch_size

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
            effective_group_size = rl_config.group_size
            # When group_size > 1, the flow GRPO train_step internally repeats each sample group_size times. Randomly subsample the batch so that reduced_batch_size * group_size == original_batch_size.
            if effective_group_size > 1:
                subsample_rng, self._rng = jax.random.split(self._rng, 2)
                batch_size = int(jax.tree.leaves(batch)[0].shape[0])
                if batch_size % effective_group_size != 0:
                    raise ValueError(
                        "Flow-GRPO actor batch size must be divisible by the "
                        f"effective group size: {batch_size} vs {effective_group_size}."
                    )
                reduced_size = batch_size // effective_group_size
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
