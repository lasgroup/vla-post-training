import dataclasses
import difflib
import tyro
from openpi.training.config import (
    _CONFIGS,
    TrainConfig,
    DataConfig,
    pi0_config,
    LeRobotLiberoDataConfig,
)
from typing import Literal, Sequence

import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders


@dataclasses.dataclass(frozen=True)
class CollectionConfig:
    collect_interval: int = 200
    env_num: int = 4
    env_resolution: int = 256
    resize_image: int = 224
    num_rollouts: int = 50
    tasks: list[str] = dataclasses.field(default_factory=lambda: ["libero_90_59"])
    replan_steps: int = 5
    num_steps_wait: int = 10


@dataclasses.dataclass(frozen=True)
class MolmoConfig:
    benchmark_dir: str = ""
    eval_config_cls: str = (
        "molmo_spaces.evaluation.configs.evaluation_configs:PiPolicyEvalConfig"
    )
    episode_sampling: Literal["sequential", "random"] = "sequential"
    task_horizon_steps: int | None = None
    # If None, the environment task description from benchmark metadata is used.
    task_description: str | None = None

    # Observation mapping from Molmo observations to OpenPI input keys.
    exo_camera_key: str = "exo_camera_1"
    wrist_camera_key: str = "wrist_camera"
    gripper_obs_norm: float = 0.824033

    # Action mapping from OpenPI output to Molmo env actions.
    grasping_type: Literal["continuous", "binary"] = "binary"
    gripper_threshold: float = 0.5
    gripper_scale: float = 255.0


@dataclasses.dataclass(frozen=True)
class OnlineDataConfig(DataConfig):
    # additional LeRobot repo paths to include (keeps repos separate but concatenates them for training)
    additional_repo_paths: Sequence[str] = ()


@dataclasses.dataclass(frozen=True)
class OnlineTrainConfig(TrainConfig):
    # additional configs for online training
    collect: CollectionConfig = CollectionConfig()
    domain: Literal["libero", "molmo"] = "libero"
    molmo: MolmoConfig = MolmoConfig()
    online_ratio: float = 0.5  # ratio of online vs offline data in each training batch
    online_buffer_size: int = 1024  # capacity of the online replay buffer
    discount: float = 0.99
    buffer_save_path: str | None = None  # if set, save each episode to this directory
    buffer_load_paths: Sequence[str] = ()  # directories to load episodes from on init


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS.extend(
    [
        #
        # Online training configs.
        #
        # These train configs define the hyperparameters for online data collection and fine-tuning.
        OnlineTrainConfig(
            name="pi05_libero_online",
            model=pi0_config.Pi0Config(
                pi05=True, action_horizon=10, discrete_state_input=False
            ),
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
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_libero/params"
            ),
            pytorch_weight_path="/path/to/your/pytorch_weight_path",
            num_train_steps=10_000,
            num_workers=4,  # override default num_workers
        ),
    ]
)

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli(
        {k: (k, v) for k, v in _CONFIGS_DICT.items()}
    )


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(
            config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0
        )
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
