import gc
from typing import Dict, Any, Tuple
from src.rl.advantage_weighted_sft.advantage_weighted_sft_learner import (
    AdvantageWeightedSFTLearner,
)
import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.training.utils as training_utils
import jax
from src.training.config import MPOWeightedSFTLearnerConfig


class MPOWeightedSFTLearner(AdvantageWeightedSFTLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Delete super class methods
        del self._update_critics_jitted
        del self._update_policy_jitted
        gc.collect()

        # 2. Create closures to drop 'self' from the JIT signature
        def _critics_wrapper(batch, q_state, value_state, policy_state, rng):
            return self._update_critics(
                batch=batch,
                q_state=q_state,
                value_state=value_state,
                policy_state=policy_state,
                rng=rng,
            )

        def _policy_wrapper(batch, policy_state, q_state, value_state, rng):
            return self._update_policy(
                batch=batch,
                policy_state=policy_state,
                q_state=q_state,
                value_state=value_state,
                rng=rng,
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
            ),
            out_shardings=(
                self._train_state_sharding,  # policy_state
                self._replicated_sharding,  # info
            ),
            donate_argnums=(1,),  # Donates policy_state (arg 1)
        )

    @at.typecheck
    def _get_on_policy_action(
        self,
        online_observation: _model.Observation,
        policy_state: training_utils.TrainState,
        rng: at.KeyArrayLike,
    ) -> _model.Actions:
        model = self._get_policy_model(policy_state)
        sampled_actions = model.sample_actions(
            observation=online_observation,
            rng=rng,
            return_info_dict=False,
            return_prefix_rep=False,
        )
        return sampled_actions

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
        policy_sample_rng, rng = jax.random.split(rng, 2)
        on_policy_action = self._get_on_policy_action(
            online_observation=batch["observation"],
            policy_state=policy_state,
            rng=policy_sample_rng,
        )

        batch = self._online_batch_to_critic_batch(
            batch,
            policy_state,
        )
        q_rng, v_rng = jax.random.split(rng, 2)
        # Update the state action critic state
        q_state, q_info = self._q_train_step(
            q_rng,
            q_state,
            value_state,
            batch,
        )
        # Update the value state
        # We replace the action from the batch with the on policy action
        # This ensures that we train an on policy critic.
        batch = (batch[0], on_policy_action, batch[2], batch[3], batch[4])
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
    ):
        assert isinstance(self._config.rl, MPOWeightedSFTLearnerConfig)
        if not self._config.rl.store_buffer_actions_in_batch:
            policy_sample_rng, rng = jax.random.split(rng, 2)
            on_policy_action = self._get_on_policy_action(
                online_observation=batch[0],
                policy_state=policy_state,
                rng=policy_sample_rng,
            )
            batch = (batch[0], on_policy_action)
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
        )

        return policy_state, info
