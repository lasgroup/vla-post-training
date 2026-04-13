import gc
import functools
from typing import Dict, Any
from src.rl.mpo_weighted_sft.mpo_weighted_sft_learner import MPOWeightedSFTLearner
from src.rl.flow_grpo.update_actor import train_step as flow_grpo_train_step
from src.training.config import FlowGRPOSFTLearnerConfig
import jax


class FlowGRPOLearner(MPOWeightedSFTLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Delete only the policy JIT method; keep _update_critics_jitted from parent
        del self._update_policy_jitted
        gc.collect()

        # Override _train_step with the flow GRPO actor update
        self._train_step = functools.partial(flow_grpo_train_step, self._config)

        # Re-create policy JIT wrapper
        def _policy_wrapper(batch, policy_state, q_state, value_state, rng, mc_return):
            return self._update_policy(
                batch=batch,
                policy_state=policy_state,
                q_state=q_state,
                value_state=value_state,
                rng=rng,
                mc_return=mc_return,
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
            donate_argnums=(1,),
        )

    def _online_batch_to_sft_batch(self,
                                   online_batch: Dict[str, Any]):
        assert isinstance(self._config.rl, FlowGRPOSFTLearnerConfig)
        rl_config = self._config.rl
        batch_size = self._config.batch_size
        sft_batch = super()._online_batch_to_sft_batch(online_batch)
        # When group_size > 1, the flow GRPO train_step internally repeats
        # each sample group_size times. Randomly subsample the batch so that
        # reduced_batch_size * group_size == original_batch_size.
        if rl_config.group_size > 1:
            subsample_rng, self._rng = jax.random.split(self._rng, 2)
            reduced_size = batch_size // rl_config.group_size
            indices = jax.random.permutation(subsample_rng, batch_size)[:reduced_size]
            sft_batch = jax.tree.map(lambda x: x[indices], sft_batch)
            sft_batch = jax.device_put(sft_batch, self._data_sharding)
        return sft_batch

    def _replace_buffer_actions_with_policy_actions(self, critic_update: bool = True) -> bool:
        if critic_update:
            return super()._replace_buffer_actions_with_policy_actions(critic_update=critic_update)
        else:
            return False



