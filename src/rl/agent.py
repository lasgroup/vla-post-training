from typing import Dict, Callable

import numpy as np
from flax.training.train_state import TrainState
import jax
from abc import abstractmethod


from src.rl.types import StepData
import openpi.training.utils as training_utils
from gymnasium import Env

EnvFn = Callable[[int], Env]


def get_batch_stats(actor):
    if hasattr(actor, "batch_stats"):
        return actor.batch_stats
    else:
        return None


class Agent(object):
    _actor: TrainState | training_utils.TrainState
    _rng: jax.random.PRNGKey
    training_steps: int = 0
    env_steps: int = 0
    episodes: int = 0
    base_dir: str = "/agent"

    @abstractmethod
    def eval_actions(self, observations: np.ndarray | Dict, **kwargs) -> np.ndarray:
        """Deterministic action decoding."""
        raise NotImplementedError

    @abstractmethod
    def sample_actions(
        self, observations: np.ndarray | Dict, **kwargs
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Stochastic action decoding."""
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, step: int):
        raise NotImplementedError

    @abstractmethod
    def add_data(self, step_data: StepData):
        raise NotImplementedError

    @abstractmethod
    def save_episode(self, is_success: bool = False, env_index: int = 0, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def start_data_collection(self, step: int | None = None):
        raise NotImplementedError

    @abstractmethod
    def end_data_collection(self, step: int | None = None) -> int:
        raise NotImplementedError

    @abstractmethod
    def update(self):
        raise NotImplementedError
