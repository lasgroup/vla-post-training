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
import re

import openpi.training.optimizer as _optimizer
import optax
import openpi.training.weight_loaders as weight_loaders


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
class RLAlgorithmConfig:
    discount: float = 0.99
    buffer_capacity: int = 1024
    buffer_save_path: str | None = None  # if set, save each episode to this directory
    buffer_load_paths: Sequence[str] = ()  # directories to load episodes from on init
    offline_buffer_load_paths: Sequence[str] = ()
    num_offline_pretraining_steps: int = 0


# Define hyperparameter structures for your algorithms
@dataclasses.dataclass(frozen=True)
class BestofNLearnerConfig(RLAlgorithmConfig):
    n_samples: int = 32
    online_ratio: float = 1.0
    critic_update_interval: int = 1
    critic_training_start_step: int = 0
    use_ema_critic: bool = True
    critic_ema_decay: float = 0.995
    critic_reduction: str = "min"
    critic_lr_schedule = ConstantSchedule(value=3e-4)
    critic_optimizer = _optimizer.AdamW(clip_gradient_norm=1.0)
    critic_encoder_type: str = "pi0_prefix"     # "pi0_prefix" | "resnet" | "pi0_prefix_resnet"
    critic_encoder_hidden_dims: Sequence[int] = (512, 512)
    critic_action_compress_dim: int | None = None  # None = no compression; e.g. 32 compresses flat action chunk via MLP
    critic_decoder_hidden_dims: Sequence[int] = (256, 256)
    critic_num_qs: int = 2
    critic_num_vs: int = 2
    num_critic_updates_per_batch: int = 1
    critic_inference_start_step: int = 900
    td_weight_schedule: StepSchedule = StepSchedule(
        init_value=0.0, end_value=1.0, switch_step=1_000
    )
    train_on_policy_value_function: bool = False
    critic_pre_training_steps: int = 1_000
    warm_start_critic_update_interval: int | None = None
    offline_buffer_capacity: int = 250_000
    num_value_bins: int = 1                     # 1 = Gaussian (MSE-equivalent), >1 = Categorical over bins
    value_lower_bound: float | None = None      # None -> auto-compute from reward type and discount
    value_upper_bound: float | None = None
    value_target_type: str = "one_hot"          # "one_hot" | "two_hot"
    train_pi0_prefix_encoder: bool = False      # finetune the Pi0 VLM backbone jointly with the critics
    pi0_encoder_lr_schedule: _optimizer.LRScheduleConfig = ConstantSchedule(value=1e-5)


@dataclasses.dataclass(frozen=True)
class FilteredSFTLearnerConfig(RLAlgorithmConfig):
    policy_update_interval: int = 1
    warm_start_policy_update_interval: int | None = None
    policy_training_start_step: int = 0
    online_ratio: float = 0.5
    reset_policy_params_to_ema_period: int | None = None
    offline_buffer_capacity: int = 100_000


@dataclasses.dataclass(frozen=True)
class AdvantageWeightedSFTLearnerConfig(FilteredSFTLearnerConfig):
    critic_update_interval: int = 1
    critic_training_start_step: int = 0
    use_ema_critic: bool = True
    critic_ema_decay: float = 0.995
    beta: float = 0.05
    weight_clip: float = 20.0
    advantage_scale: float = 10.0
    critic_reduction: str = "min"
    critic_lr_schedule = ConstantSchedule(value=1e-4)
    critic_optimizer = _optimizer.AdamW(clip_gradient_norm=1.0)
    critic_encoder_hidden_dims: Sequence[int] = (512, 512)
    critic_decoder_hidden_dims: Sequence[int] = (256, 256)
    td_weight_schedule: StepSchedule = StepSchedule(
        init_value=0.0, end_value=1.0, switch_step=1_000
    )
    critic_pre_training_steps: int = 1_000
    warm_start_critic_update_interval: int | None = None
    critic_num_qs: int = 2
    critic_num_vs: int = 2
    num_critic_updates_per_batch: int = 1
    use_mc_returns: bool = False
    store_success_episodes_only: bool = False
    num_value_bins: int = 1
    value_lower_bound: float | None = None
    value_upper_bound: float | None = None
    value_target_type: str = "one_hot"


@dataclasses.dataclass(frozen=True)
class MPOWeightedSFTLearnerConfig(AdvantageWeightedSFTLearnerConfig):
    store_buffer_actions_in_batch: bool = False


@dataclasses.dataclass(frozen=True)
class FlowGRPOSFTLearnerConfig(MPOWeightedSFTLearnerConfig):
    group_size: int = 8
    num_steps: int = 10
    noise_level: float = 0.3
    normalize_adv: bool = True

@dataclasses.dataclass(frozen=True)
class FlowMPOSFTLearnerConfig(MPOWeightedSFTLearnerConfig):
    group_size: int = 1
    num_steps: int = 10
    noise_level: float = 0.3
    normalize_adv: bool = True

@dataclasses.dataclass(frozen=True)
class FlowPGSFTLearnerConfig(MPOWeightedSFTLearnerConfig):
    num_steps: int = 10
    noise_level: float = 0.3
    kl_coef: float = 0.01
    use_ema_for_sampling: bool = True

@dataclasses.dataclass(frozen=True)
class DSRLLearnerConfig(RLAlgorithmConfig):
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    # Network architecture (kept explicit for parity across scripts/experiments).
    critic_decoder_hidden_dims: tuple[int, ...] = (128, 128, 128)
    policy_decoder_hidden_dims: tuple[int, ...] = (128, 128, 128)
    critic_num_qs: int = 10
    critic_reduction: str = "mean"
    backup_entropy: bool = False
    critic_update_frequency: int = 1
    actor_update_frequency: int = 1
    critic_ema_decay: float | None = 0.995
    encoder_type: str = "small"
    encoder_norm: str = "group"
    use_spatial_softmax: bool = True
    softmax_temperature: float = 1.0
    image_latent_dim: int = 50
    use_image_bottleneck: bool = True
    use_state_branch: bool = True
    autotune_alpha: bool = True
    init_alpha: float = 1.0
    target_entropy: str | float = "auto"
    policy_distribution: str = "tanh_normal"
    sac_image_size: int = 64
    random_crop_padding: int = 4
    warmup_gaussian_noise: bool = True


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
class CollectionConfig:
    collect_interval: int = 200
    env_num: int = 4
    env_resolution: int = 256
    resize_image: int = 224
    num_rollouts: int = 50
    num_initial_rollouts: int | None = None
    domain: Literal["libero", "molmo"] = "libero"
    molmo: MolmoConfig = MolmoConfig()
    tasks: list[str] | str = dataclasses.field(
        default_factory=lambda: ["libero_90_59"],
        metadata={
            "help": (
                "Task(s) to collect. Can be a single string (e.g., 'libero_90_59') or a list. "
                "Supports ranges (e.g., 'libero_90_22-56') and optional multipliers (e.g., 'libero_90_59x4' "
                "or 'libero_90_22-56x4'). After expansion, the number of tasks must divide env_num evenly; "
                "tasks are then repeated to fill all env_num environments."
            )
        },
    )
    eval_tasks: list[str] | str = dataclasses.field(
        default_factory=lambda: ["libero_90_59"],
        metadata={
            "help": (
                "Task(s) to evaluate. Can be a single string (e.g., 'libero_90_59') or a list. "
                "Supports ranges (e.g., 'libero_90_22-56') and optional multipliers (e.g., 'libero_90_59x4' "
                "or 'libero_90_22-56x4'). After expansion, the number of eval tasks must divide eval_env_num "
                "evenly; tasks are then repeated to fill all eval_env_num environments."
            )
        },
    )
    replan_steps: int = 5
    num_steps_wait: int = 10
    max_episode_steps: int = 400  # Value for LIBERO-90, used to compute value bounds. 
    use_time_to_success_as_reward: bool = False
    store_prefix_rep: bool = False
    eval_env_num: int = 4
    eval_interval: int = 300
    num_eval_rollouts: int = 32

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
        assert self.env_num % len(expanded_tasks) == 0, (
            f"env_num ({self.env_num}) must be divisible by the number of expanded tasks ({len(expanded_tasks)})."
        )
        object.__setattr__(self, 'tasks', expanded_tasks * (self.env_num // len(expanded_tasks)))

        eval_tasks = [self.eval_tasks] if isinstance(self.eval_tasks, str) else self.eval_tasks
        expanded_eval_tasks = self.expand_tasks(eval_tasks)
        assert self.eval_env_num % len(expanded_eval_tasks) == 0, (
            f"eval_env_num ({self.eval_env_num}) must be divisible by the number of expanded eval tasks ({len(expanded_eval_tasks)})."
        )
        object.__setattr__(self, 'eval_tasks', expanded_eval_tasks * (self.eval_env_num // len(expanded_eval_tasks)))


@dataclasses.dataclass(frozen=True)
class OnlineDataConfig(DataConfig):
    # additional LeRobot repo paths to include (keeps repos separate but concatenates them for training)
    additional_repo_paths: Sequence[str] = ()


@dataclasses.dataclass(frozen=True)
class OnlineTrainConfig(TrainConfig):
    # additional configs for online training
    collect: CollectionConfig = CollectionConfig()
    rl: RLAlgorithmConfig = FilteredSFTLearnerConfig()
    default_prompt: str | None = None
    algorithm: str = ""
    tags: list[str] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.rl, (BestofNLearnerConfig, AdvantageWeightedSFTLearnerConfig)):
            if self.rl.value_lower_bound is None or self.rl.value_upper_bound is None:
                discount = float(self.rl.discount)
                T = int(self.collect.max_episode_steps)
                if self.collect.use_time_to_success_as_reward:
                    lower = -(1.0 - discount**T) / (1.0 - discount) if discount < 1.0 else -float(T)
                    upper = 0.0
                else:
                    lower = 0.0
                    upper = 1.0
                # Add half-bin-width padding on each side so that the actual extreme
                # return values (e.g. mc_return=0 at terminal step) never land exactly
                # on a bin center, which would degenerate two_hot into one_hot.
                num_bins = self.rl.num_value_bins
                if num_bins > 1:
                    half_bw = (upper - lower) / (2 * (num_bins - 1))
                    lower -= half_bw
                    upper += half_bw
                object.__setattr__(self, "rl", dataclasses.replace(
                    self.rl, value_lower_bound=lower, value_upper_bound=upper
                ))


def make_base_online_config(
    name: str, rl_config: RLAlgorithmConfig
) -> OnlineTrainConfig:
    """
    Factory function to generate a base OnlineTrainConfig.
    Injects the specific RL algorithm config to keep the _CONFIGS list DRY.
    """
    return OnlineTrainConfig(
        name=name,
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
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
        num_workers=4,  # override default num_workers
        rl=rl_config,
    )


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS.extend(
    [
        #
        # Online training configs.
        #
        # These train configs define the hyperparameters for online data collection and fine-tuning.
        # 1. Filtered SFT
        make_base_online_config(
            name="pi05_libero_online_filtered_sft",
            rl_config=FilteredSFTLearnerConfig(),
        ),
        # 2. Advantage Weighted SFT (AWSFT)
        make_base_online_config(
            name="pi05_libero_online_aw_sft",
            rl_config=AdvantageWeightedSFTLearnerConfig(
                policy_update_interval=20,
                policy_training_start_step=100,
            ),
        ),
        # 3. MPO Weighted SFT
        make_base_online_config(
            name="pi05_libero_online_mpo_sft",
            rl_config=MPOWeightedSFTLearnerConfig(
                store_buffer_actions_in_batch=False,
                policy_update_interval=20,
                policy_training_start_step=100,
            ),
        ),
        make_base_online_config(
            name="pi05_libero_online_flow_grpo_sft",
            rl_config=FlowGRPOSFTLearnerConfig(
                store_buffer_actions_in_batch=True,
                policy_update_interval=20,
                policy_training_start_step=100,
            ),
        ),
        # 4. Best of N
        make_base_online_config(
            name="pi05_libero_online_best_of_n",
            rl_config=BestofNLearnerConfig(),
        ),
        make_base_online_config(
            name="pi05_libero_online_dsrl",
            rl_config=DSRLLearnerConfig(),
        )
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
