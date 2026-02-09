from typing import Any, Dict, Union
import openpi.training.config as _config
import flax
import numpy as np
import dataclasses

DataType = Union[np.ndarray, Dict[str, "DataType"]]
PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]


@dataclasses.dataclass(frozen=True)
class EnvKwargs:
    env: str = "libero"
    task_description: str = ""
    resize_image: int = 224
    add_states: bool = True
    num_steps_wait_upon_reset: int = 0
    warm_up_action: Any | None = None


@dataclasses.dataclass(frozen=True)
class OnlineLearningConfig:
    episode_update_frequency: int | None = 1
    env_steps_update_frequency: int | None = None
    start_step: int = 0
    start_episode: int = 0
    num_train_steps_per_update: int = 100
    num_train_steps: int = 30_000
    fm_noise_level: float = 0.0
    num_envs: int = 1
    # Optional env/variant context (used by filtered SFT data collection).
    variant: Any | None = None
    replan_steps: int = 1
    discount: float = 0.99
    default_prompt: str | None = None
    # Online buffer + mixing controls (optional, used by filtered SFT).
    online_batch_size: int | None = None
    online_max_samples: int = 100_000
    online_ratio: float = 0.5
    online_shuffle: bool = True
    # Diffusion sampling steps for pi0; None uses model default.
    pi0_num_steps: int | None = None
    # Time limit used to restrict max episode length in the environment.
    time_limit: int = 1_000


@dataclasses.dataclass(frozen=True)
class OnlineTrainingConfig:
    base_policy_config: _config.TrainConfig
    online_learning_config: OnlineLearningConfig
    obs_prefix_key: str = "pi0/"
