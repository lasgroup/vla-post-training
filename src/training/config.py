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
from typing import Sequence
import re

import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders


@dataclasses.dataclass(frozen=True)
class CollectionConfig:
    collect_interval: int = 200
    env_num: int = 4 # Currently not used as we create one env per task
    env_resolution: int = 256
    resize_image: int = 224
    num_rollouts: int = 50
    tasks: list[str] = dataclasses.field(
        default_factory=lambda: ["libero_90_59", "libero_90_60", "libero_90_61", "libero_90_62"],
        metadata={
            "help": (
                "List of tasks to collect. Supports individual task names (e.g., 'libero_90_34'), "
                "ranges (e.g., 'libero_90_22-56'), and optional multipliers (e.g., 'libero_90_59x4' "
                "or 'libero_90_22-56x4'). The total number of expanded tasks must be divisible by 4."
            )
        },
    )
    replan_steps: int = 5
    num_steps_wait: int = 10

    def __post_init__(self):
        # Expand task ranges and handle multipliers
        expanded_tasks = []
        for task in self.tasks:
            # 1. Extract optional multiplier (e.g., "x4")
            multiplier = 1
            base_task = task
            mult_match = re.search(r"x(\d+)$", task)
            if mult_match:
                multiplier = int(mult_match.group(1))
                base_task = task[:mult_match.start()]

            # 2. Check if the base task is a range
            range_match = re.match(r"(.+)_(\d+)-(\d+)$", base_task)
            sub_tasks = []
            if range_match:
                prefix = range_match.group(1)
                start = int(range_match.group(2))
                end = int(range_match.group(3))
                for i in range(start, end + 1):
                    sub_tasks.append(f"{prefix}_{i}")
            else:
                sub_tasks.append(base_task)
            
            # 3. Add to expanded list, repeating by the multiplier
            for sub_task in sub_tasks:
                expanded_tasks.extend([sub_task] * multiplier)
        
        object.__setattr__(self, 'tasks', expanded_tasks)

        # Check divisibility by 4
        num_tasks = len(self.tasks)
        if num_tasks % 4 != 0:
            raise ValueError(
                f"Invalid number of tasks: {num_tasks}. "
                f"The task count must be a multiple of 4 (e.g., {((num_tasks // 4) + 1) * 4}) "
                "because the current sharding implementation does not support "
                "non-uniform task distributions across devices yet."
            )

        # Count occurrences of each base task
        counts = {}
        for task in expanded_tasks:
            counts[task] = counts.get(task, 0) + 1

        # Generate task_names
        task_name_counters = {}
        task_names = []
        for task in expanded_tasks:
            if counts[task] == 1:
                # Unique task → keep original
                task_names.append(task)
            else:
                # Multiple instances → add index
                idx = task_name_counters.get(task, 0)
                task_names.append(f"{task}_{idx}")
                task_name_counters[task] = idx + 1

        object.__setattr__(self, 'task_names', task_names)


@dataclasses.dataclass(frozen=True)
class OnlineDataConfig(DataConfig):
    # additional LeRobot repo paths to include (keeps repos separate but concatenates them for training)
    additional_repo_paths: Sequence[str] = ()


@dataclasses.dataclass(frozen=True)
class OnlineTrainConfig(TrainConfig):
    # additional configs for online training
    collect: CollectionConfig = CollectionConfig()
    domain: str = "libero"
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
