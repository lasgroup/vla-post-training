from typing import Dict

import numpy as np
import jax
from abc import abstractmethod


class Agent(object):
    _rng: jax.random.PRNGKey
    training_steps: int = 0
    total_collected_episodes: int = 0

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
    def add_data(self, step_data):
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
