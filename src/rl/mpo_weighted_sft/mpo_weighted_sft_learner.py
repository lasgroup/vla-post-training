from typing import Any, Dict, Tuple

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.training.utils as training_utils
import jax

from src.rl.advantage_weighted_sft.advantage_weighted_sft_learner import (
    AdvantageWeightedSFTLearner,
)
from src.training.config import MPOWeightedSFTLearnerConfig


class MPOWeightedSFTLearner(AdvantageWeightedSFTLearner):
    def _policy_mc_return_sharding(self):
        return self._replicated_sharding

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
        assert isinstance(self._config.rl, MPOWeightedSFTLearnerConfig)
        if self._config.rl.store_buffer_actions_in_batch:
            value_action = batch["actions"]
        else:
            value_action = self._get_on_policy_action(
                online_observation=_model.Observation.from_dict(batch["observation"]),
                policy_state=policy_state,
                rng=policy_sample_rng,
            )

        batch = self._online_batch_to_critic_batch(
            batch,
            policy_state,
        )
        # We replace the action from the batch with the on policy action
        # This ensures that we train an on policy critic.
        value_batch = (
            batch[0],
            value_action,
            batch[2],
            batch[3],
            batch[4],
            batch[5],
        )
        # Update the state action critic state
        assert isinstance(self._config.rl, MPOWeightedSFTLearnerConfig)
        num_updates = max(self._config.rl.critic.num_updates_per_batch, 1)
        for _ in range(num_updates):
            q_rng, v_rng, rng = jax.random.split(rng, 3)
            q_state, q_info = self._q_train_step(
                q_rng,
                q_state,
                value_state,
                batch,
            )
            # Update the value state
            value_state, value_info = self._value_train_step(
                v_rng,
                value_state,
                q_state,
                value_batch,
            )

        return q_state, value_state, q_info, value_info

    def _update_policy(
        self,
        batch: tuple[_model.Observation, _model.Actions],
        policy_state: training_utils.TrainState,
        q_state: training_utils.TrainState,
        value_state: training_utils.TrainState,
        rng: at.KeyArrayLike,
        mc_return: at.Array | None = None,
        is_success: at.Float[at.Array, " b"] | None = None,
        scale: at.Array | float = 1.0,
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
            mc_return=mc_return,
            is_success=is_success,
            scale=scale,
        )

        return policy_state, info
