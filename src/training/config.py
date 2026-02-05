import dataclasses
import difflib
import tyro
from openpi.training.config import _CONFIGS, TrainConfig, DataConfig, pi0_config, LeRobotLiberoDataConfig
from typing import Sequence

import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders


@dataclasses.dataclass(frozen=True)
class CollectionConfig:
    collect_interval: int = 100
    env_num: int = 4
    env_resolution: int = 256
    num_rollouts: int = 50
    tasks: list[str] = dataclasses.field(default_factory=lambda: ["libero_90_59"])
    replan_steps: int = 5
    num_steps_wait: int = 10


@dataclasses.dataclass(frozen=True)
class OnlineDataConfig(DataConfig):
    # additional LeRobot repo paths to include (keeps repos separate but concatenates them for training)
    additional_repo_paths: Sequence[str] = ()


@dataclasses.dataclass(frozen=True)
class OnlineTrainConfig(TrainConfig):
    # additional configs for online training
    collect: CollectionConfig = CollectionConfig()


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS.extend([
    #
    # Online training configs.
    #
    # These train configs define the hyperparameters for online data collection and fine-tuning.
    OnlineTrainConfig(
        name="pi05_libero_online",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=OnlineDataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=100,  # override default warmup steps
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_libero/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=10_000,
        num_workers=4,  # override default num_workers
    ),
])

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
