from typing import Dict

import numpy as np
from flax.training import checkpoints
from flax.training.train_state import TrainState
import jax

from src.rl_training.common import (
    eval_actions_jit,
    eval_log_prob_jit,
    eval_mse_jit,
    eval_reward_function_jit,
    sample_actions_jit,
    DatasetDict,
)

from collections import namedtuple
import openpi.training.utils as training_utils

StepData = namedtuple(
    "StepData", ["obs", "action", "next_obs", "reward", "terminate", "truncate"]
)


def get_batch_stats(actor):
    if hasattr(actor, "batch_stats"):
        return actor.batch_stats
    else:
        return None


class Agent(object):
    _actor: TrainState | training_utils.TrainState
    _critic: TrainState
    _rng: jax.random.PRNGKey
    training_steps: int = 0
    env_steps: int = 0
    episodes: int = 0
    base_dir: str = "/agent"

    def eval_actions(self, observations: np.ndarray | Dict) -> np.ndarray:
        if isinstance(self._actor, training_utils.TrainState):
            raise NotImplementedError
        else:
            actions = eval_actions_jit(
                self._actor.apply_fn,
                self._actor.params,
                observations,
                get_batch_stats(self._actor),
            )
            return np.asarray(actions)

    def eval_log_probs(self, batch: DatasetDict | Dict) -> float:
        if isinstance(self._actor, training_utils.TrainState):
            raise NotImplementedError
        else:
            return eval_log_prob_jit(
                self._actor.apply_fn,
                self._actor.params,
                get_batch_stats(self._actor),
                batch,
            )

    def eval_mse(self, batch: DatasetDict) -> float:
        if isinstance(self._actor, training_utils.TrainState):
            raise NotImplementedError
        else:
            return eval_mse_jit(
                self._actor.apply_fn,
                self._actor.params,
                get_batch_stats(self._actor),
                batch,
            )

    def eval_reward_function(self, batch: DatasetDict) -> float:
        if isinstance(self._actor, training_utils.TrainState):
            raise NotImplementedError
        else:
            return eval_reward_function_jit(
                self._actor.apply_fn, self._actor.params, self._actor.batch_stats, batch
            )

    def sample_actions(
        self, observations: np.ndarray | Dict, batch_actions: bool = True
    ) -> np.ndarray:
        if isinstance(self._actor, training_utils.TrainState):
            raise NotImplementedError
        else:
            rng, actions = sample_actions_jit(
                self._rng,
                self._actor.apply_fn,
                self._actor.params,
                observations,
                get_batch_stats(self._actor),
            )

            self._rng = rng
            return np.asarray(actions)

    @property
    def _save_dict(self):
        return None

    def save_checkpoint(self, step, keep_every_n_steps: int | None = None):
        checkpoints.save_checkpoint(
            self.base_dir,
            self._save_dict,
            step,
            prefix="checkpoint",
            overwrite=False,
            keep_every_n_steps=keep_every_n_steps,
        )

    def restore_checkpoint(self, dir):
        raise NotImplementedError

    def add_data(self, step_data: StepData):
        raise NotImplementedError

    def save_episode(self, is_success: bool = False, env_index: int = 0):
        self.episodes += 1
        return True

    def start_data_collection(self):
        return True

    def update(self):
        raise NotImplementedError
