import dataclasses
import difflib
import tyro
from openpi.training.config import (
    _CONFIGS,
    TrainConfig,
    DataConfig,
    pi0_config,
    SimpleDataConfig,
    AssetsConfig,
    ModelType,
    LeRobotLiberoDataConfig,
)
from typing import Literal, Sequence
import re

import openpi.training.optimizer as _optimizer
import openpi.policies.droid_policy as _droid_policy
import openpi.transforms as _openpi_transforms
import optax
import openpi.training.weight_loaders as weight_loaders
import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class NormalizerState:
    bias: jax.Array
    scale: jax.Array
    ema_weight: float


class Normalizer:
    def __init__(self, ema_weight: float = 0.99):
        self._ema_weight = ema_weight

    def init(self) -> NormalizerState:
        return NormalizerState(
            bias=jnp.array(0.0),
            scale=jnp.array(1.0),
            ema_weight=self._ema_weight,
        )

    @staticmethod
    @jax.jit
    def update(normalizer_state: NormalizerState, bias: jax.Array, scale: jax.Array) -> NormalizerState:
        prev_bias = normalizer_state.bias
        prev_scale = normalizer_state.scale
        ema_weight = normalizer_state.ema_weight
        new_bias = (1.0 - ema_weight) * bias + ema_weight * prev_bias
        new_scale = (1.0 - ema_weight) * scale + ema_weight * prev_scale
        return normalizer_state.replace(bias=new_bias, scale=new_scale)


@dataclasses.dataclass(frozen=True)
class ConstantSchedule(_optimizer.LRScheduleConfig):
    """Constant learning rate schedule."""

    value: float = 5e-5

    def create(self) -> optax.Schedule:
        return optax.constant_schedule(self.value)


@dataclasses.dataclass(frozen=True)
class LinearSchedule(_optimizer.LRScheduleConfig):
    """Linear schedule that starts at init_value and ends at end_value"""

    init_value: float = 0.0
    end_value: float = 1.0
    transition_steps: int = 1000

    def create(self) -> optax.Schedule:
        return optax.linear_schedule(
            self.init_value, self.end_value, self.transition_steps
        )


@dataclasses.dataclass(frozen=True)
class StepSchedule(_optimizer.LRScheduleConfig):
    """Step schedule: returns init_value for steps < switch_step, then end_value."""

    init_value: float = 0.0
    end_value: float = 1.0
    switch_step: int = 1000

    def create(self) -> optax.Schedule:
        return optax.join_schedules(
            schedules=[
                optax.constant_schedule(self.init_value),
                optax.constant_schedule(self.end_value),
            ],
            boundaries=[self.switch_step],
        )


@dataclasses.dataclass(frozen=True)
class NormalizerConfig:
    q_up: float = 0.95
    q_low: float = 0.05
    method: str = 'quantile'
    min_scale: float = 1.0
    ema_weight: float = 0.99


@dataclasses.dataclass(frozen=True)
class RLAlgorithmConfig:
    discount: float = 0.99
    buffer_capacity: int = 1024
    # Shared by both policy and critic sampling: fraction of each batch drawn from
    # the online replay buffer (rest comes from the offline SFT dataset).
    online_ratio: float = 0.5


@dataclasses.dataclass(frozen=True)
class PolicyTrainingConfig:
    update_interval: int = 1
    training_start_step: int = 0


@dataclasses.dataclass(frozen=True)
class CriticTrainingConfig:
    update_interval: int = 1
    training_start_step: int = 0
    use_ema: bool = True
    ema_decay: float = 0.995
    reduction: str = "min"
    encoder_hidden_dims: Sequence[int] = (512, 512)
    decoder_hidden_dims: Sequence[int] = (256, 256)
    num_qs: int = 2
    num_vs: int = 2
    num_updates_per_batch: int = 1
    td_weight_schedule: StepSchedule = StepSchedule(init_value=0.0, end_value=1.0, switch_step=1_000)
    pre_training_steps: int = 1_000
    num_value_bins: int = 1  # 1: Gaussian (MSE-equivalent), >1 = Categorical over bins
    value_lower_bound: float | None = None  # If None: auto-computed from reward type and discount
    value_upper_bound: float | None = None
    value_target_type: str = "two_hot"  # "one_hot" | "two_hot"
    use_distributional_critic: bool = False
    distributional_target_reduction: str = "min"
    inference_start_step: int = 100
    # Critics are lightweight (MLP-only); a larger batch than the policy often
    # stabilises TD learning without a meaningful memory cost.
    batch_size: int | None = None  # If None: use the global config.batch_size
    # Class-level attributes (not dataclass fields) so subclasses can override the default.
    lr_schedule = ConstantSchedule(value=1e-4)
    optimizer = _optimizer.AdamW(clip_gradient_norm=1.0)
    # BRONet critic (alternative to the MLP backbone)
    use_bronet: bool = False
    bronet_hidden_dim: int = 512
    bronet_depth: int = 2


# Define hyperparameter structures for your algorithms
@dataclasses.dataclass(frozen=True)
class FilteredSFTLearnerConfig(RLAlgorithmConfig):
    policy: PolicyTrainingConfig = PolicyTrainingConfig()


@dataclasses.dataclass(frozen=True)
class BestofNLearnerConfig(FilteredSFTLearnerConfig):
    online_ratio: float = 1.0
    critic: CriticTrainingConfig = CriticTrainingConfig()
    n_samples: int = 32
    discount: float = 0.995
    train_on_policy_value_function: bool = False


@dataclasses.dataclass(frozen=True)
class CollectionConfig:
    collect_interval: int = 200
    env_num: int = 4
    env_resolution: int = 256
    resize_image_h: int = 224
    resize_image_w: int = 224
    num_rollouts: int = 50
    num_initial_rollouts: int | None = None
    domain: Literal["libero", "molmo"] = "libero"
    tasks: list[str] | str = dataclasses.field(
        default_factory=lambda: ["libero_90_59"],
        metadata={
            "help": (
                "Task(s) to collect. Can be a single string (e.g., 'libero_90_59') or a list. "
                "Supports ranges (e.g., 'libero_90_22-56') and optional multipliers (e.g., 'libero_90_59x4' "
                "or 'libero_90_22-56x4')."
            )
        },
    )
    eval_tasks: list[str] | str = dataclasses.field(
        default_factory=lambda: ["libero_90_59"],
        metadata={
            "help": (
                "Task(s) to evaluate. Can be a single string (e.g., 'libero_90_59') or a list. "
                "Supports ranges (e.g., 'libero_90_22-56') and optional multipliers (e.g., 'libero_90_59x4' "
                "or 'libero_90_22-56x4')."
            )
        },
    )
    replan_steps: int = 5
    num_steps_wait: int = 10
    use_time_to_success_as_reward: bool = True
    fix_mc_returns: bool = True
    store_prefix_rep: bool = False
    eval_env_num: int = 4
    eval_interval: int = 300
    num_eval_rollouts: int = 32
    max_episode_steps: int = 400  # used to auto-compute value bounds

    def expand_tasks(self, tasks: str) -> list[str]:
        # Expand task ranges and handle multipliers
        expanded_tasks = []
        for task in tasks:
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
        return expanded_tasks

    def __post_init__(self):
        tasks = [self.tasks] if isinstance(self.tasks, str) else self.tasks
        expanded_tasks = self.expand_tasks(tasks)
        object.__setattr__(self, 'tasks', expanded_tasks)

        eval_tasks = [self.eval_tasks] if isinstance(self.eval_tasks, str) else self.eval_tasks
        expanded_eval_tasks = self.expand_tasks(eval_tasks)
        object.__setattr__(self, 'eval_tasks', expanded_eval_tasks)


@dataclasses.dataclass(frozen=True)
class OnlineDataConfig(DataConfig):
    # additional LeRobot repo paths to include (keeps repos separate but concatenates them for training)
    additional_repo_paths: Sequence[str] = ()


@dataclasses.dataclass(frozen=True)
class OnlineTrainConfig(TrainConfig):
    # additional configs for online training
    group_name: str = "online_training"
    collect: CollectionConfig = CollectionConfig()
    rl: RLAlgorithmConfig = FilteredSFTLearnerConfig()
    max_runtime: int = 60 * 60 * 24
    requeue_before_eval: bool = False
    free_buffer_before_eval: bool = False
    default_prompt: str | None = None
    
    def __post_init__(self):
        super().__post_init__()

        if isinstance(self.rl, (BestofNLearnerConfig)):
            if self.rl.critic.use_distributional_critic:
                # The C51 distributional backup is only implemented for best-of-N.
                # Other critic-based algos (AWR and its subclasses MPO/FlowGRPO/OGPO)
                # share CriticTrainingConfig, so the flag is settable there but would
                # be silently ignored — reject it explicitly to avoid that footgun.
                assert isinstance(self.rl, BestofNLearnerConfig), (
                    "use_distributional_critic=True is only supported for "
                    "BestofNLearnerConfig; it is not implemented for "
                    f"{type(self.rl).__name__}."
                )
                assert self.rl.critic.num_value_bins > 1, (
                    "use_distributional_critic=True requires num_value_bins > 1; "
                    "set rl.critic.num_value_bins (e.g. 51) in the config."
                )
            # Value bounds are resolved lazily in get_value_bounds(), not cached
            # here: caching into value_lower/upper_bound would suppress
            # re-resolution after tyro/YAML overrides.


def make_base_libero_config(
    name: str, rl_config: RLAlgorithmConfig, **kwargs
) -> OnlineTrainConfig:
    """
    Factory function to generate a base OnlineTrainConfig.
    Injects the specific RL algorithm config to keep the _CONFIGS list DRY.

    Extra kwargs are forwarded to OnlineTrainConfig (e.g. freeze_filter,
    ema_decay, num_train_steps overrides).
    """
    defaults = dict(
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            assets=AssetsConfig(
                # load norm_stats from pretrained checkpoint
                assets_dir="gs://openpi-assets/checkpoints/pi05_libero/assets",
            ),
            base_config=OnlineDataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=ConstantSchedule(value=5e-5),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_libero/params"
        ),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=10_000,
        num_workers=4,
    )
    defaults.update(kwargs)
    return OnlineTrainConfig(name=name, rl=rl_config, **defaults)


def make_base_molmo_config(
        name: str, rl_config: RLAlgorithmConfig
) -> OnlineTrainConfig:
    """
    Factory function to generate a base OnlineTrainConfig for Molmo.
    Injects the specific RL algorithm config to keep the _CONFIGS list DRY.
    """
    return OnlineTrainConfig(
        name=name,
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=15, discrete_state_input=False
        ),
        data=SimpleDataConfig(
            repo_id=None,
            assets=AssetsConfig(
                # load norm_stats from pretrained checkpoint
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid_jointpos/assets",
                asset_id="droid",
            ),
            data_transforms=lambda model: _openpi_transforms.Group(
                inputs=[_droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[
                    _openpi_transforms.AbsoluteActions(
                        _openpi_transforms.make_bool_mask(7, -1)
                    ),
                    _droid_policy.DroidOutputs(),
                ],
            ),
            base_config=OnlineDataConfig(prompt_from_task=True),
        ),
        batch_size=256,
        lr_schedule=ConstantSchedule(value=5e-5),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_droid_jointpos/params"
        ),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=10_000,
        num_workers=4,  # override default num_workers
        rl=rl_config,
        collect=CollectionConfig(
            domain="molmo",
            max_episode_steps=450,
            resize_image_h=224,
            resize_image_w=224,
        )
    )


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS.extend(
    [
        #
        # Online training configs.
        #
        # These train configs define the hyperparameters for online data collection and fine-tuning.
        # 1. Filtered SFT
        make_base_libero_config(
            name="pi05_libero_online_filtered_sft",
            rl_config=FilteredSFTLearnerConfig(),
        ),
        make_base_molmo_config(
            name="pi05_molmo_online_filtered_sft",
            rl_config=FilteredSFTLearnerConfig(online_ratio=1.0),
        ),
        # 2. Best of N
        make_base_libero_config(
            name="pi05_libero_online_best_of_n",
            rl_config=BestofNLearnerConfig(),
        ),
        make_base_molmo_config(
            name="pi05_molmo_online_best_of_n",
            rl_config=BestofNLearnerConfig(online_ratio=1.0),
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
