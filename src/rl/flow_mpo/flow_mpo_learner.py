import gc
import functools
from src.rl.mpo_weighted_sft.mpo_weighted_sft_learner import MPOWeightedSFTLearner
from src.rl.flow_mpo.update_actor import train_step as flow_mpo_train_step
from src.training.config import FlowMPOSFTLearnerConfig, Normalizer, NormalizerState
import openpi.training.sharding as sharding
import jax
import jax.numpy as jnp
import numpy as np
import openpi.shared.array_typing as at


class FlowMPOLearner(MPOWeightedSFTLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Delete only the policy JIT method; keep _update_critics_jitted from parent
        del self._update_policy_jitted
        gc.collect()

        # Override _train_step with the flow mpo actor update
        self._train_step = functools.partial(flow_mpo_train_step, self._config)

        # NormalizerState-based EMA for advantage scale (DreamerV3-style).
        # Initialize scale to config.rl.advantage_scale instead of 1.0 so the
        # adaptive path does not have a cold-start: at step 0, exp(A / scale / beta)
        normalizer_config = self._config.rl.normalizer_config
        self._normalizer = Normalizer(ema_weight=normalizer_config.ema_weight)
        self._normalizer_state = NormalizerState(
            bias=jnp.asarray(0.0, dtype=jnp.float32),
            scale=jnp.asarray(self._config.rl.advantage_scale, dtype=jnp.float32),
            ema_weight=normalizer_config.ema_weight,
        )

        # Re-create policy JIT wrapper with an extra normalizer_state slot
        def _policy_wrapper(batch, policy_state, q_state, value_state, rng, mc_return, normalizer_state):
            return self._update_policy(
                batch=batch,
                policy_state=policy_state,
                q_state=q_state,
                value_state=value_state,
                rng=rng,
                mc_return=mc_return,
                normalizer_state=normalizer_state,
            )

        self._update_policy_jitted = jax.jit(
            _policy_wrapper,
            in_shardings=(
                self._data_sharding,  # batch
                self._train_state_sharding,  # policy_state
                self._state_action_critic_state_sharding,  # q_state
                self._value_state_sharding,  # value_state
                self._replicated_sharding,  # rng
                self._replicated_sharding,  # mc_return
                self._replicated_sharding,  # normalizer_state
            ),
            out_shardings=(
                self._train_state_sharding,  # policy_state
                self._replicated_sharding,  # info
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
        mc_return=None,
        normalizer_state=None,
    ):
        assert isinstance(self._config.rl, FlowMPOSFTLearnerConfig)
        if self._replace_buffer_actions_with_policy_actions(critic_update=False):
            policy_sample_rng, rng = jax.random.split(rng, 2)
            on_policy_action = self._get_on_policy_action(
                online_observation=batch[0],
                policy_state=policy_state,
                rng=policy_sample_rng,
            )
            batch = (batch[0], on_policy_action)
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
            mc_return=mc_return,
            normalizer_state=normalizer_state,
        )
        return policy_state, info

    def _replace_buffer_actions_with_policy_actions(self, critic_update: bool = True) -> bool:
        if critic_update:
            return super()._replace_buffer_actions_with_policy_actions(critic_update=critic_update)
        else:
            return False

    @at.typecheck
    def update(self) -> dict:
        assert isinstance(self._config.rl, FlowMPOSFTLearnerConfig)
        rl_config = self._config.rl

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
                batch = jax.tree.map(
                    lambda x, y: jnp.concatenate([x[:n_offline], y[:n_online]], axis=0),
                    batch,
                    online_batch,
                )
                del online_batch
                gc.collect()
            else:
                batch = next(self._data_iter)
        else:
            # Online buffer not ready; fall back to offline data
            batch = next(self._data_iter)
        if update_policy:
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
                    self._normalizer_state,
                )
            self._train_state = policy_state

            # Update NormalizerState on host using batch advantage stats.
            # The updated state will be consumed by the next train_step.
            if rl_config.use_adaptive_advantage_scale:
                normalizer_config = rl_config.normalizer_config
                if normalizer_config.method == "quantile":
                    scale = actor_info["advantage_q_up"] - actor_info["advantage_q_low"]
                    bias = actor_info["advantage_q_low"]
                elif normalizer_config.method == "standard_normal":
                    scale = actor_info["advantage_std"]
                    bias = actor_info["advantage_mean"]
                elif normalizer_config.method == "min_max":
                    scale = actor_info["advantage_max"] - actor_info["advantage_min"]
                    bias = actor_info["advantage_min"]
                else:
                    raise NotImplementedError(
                        f"Unknown normalizer method: {normalizer_config.method}"
                    )
                scale = jnp.clip(scale, min=normalizer_config.min_scale)
                self._normalizer_state = self._normalizer.update(
                    self._normalizer_state, bias=bias, scale=scale
                )

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
