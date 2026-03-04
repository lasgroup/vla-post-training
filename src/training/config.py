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
        return optax.linear_schedule(self.init_value, self.end_value, self.transition_steps)


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
    buffer_capacity: int = 256 * 8


# Define hyperparameter structures for your algorithms
@dataclasses.dataclass(frozen=True)
class BestofNLearnerConfig(RLAlgorithmConfig):
    n_samples: int = 8
    online_ratio: float = 0.0
    critic_update_interval: int = 1
    critic_training_start_step: int = 0
    use_ema_critic: bool = True
    critic_ema_decay: float = 0.995
    critic_reduction: str = "min"
    critic_lr_schedule = ConstantSchedule(value=1e-4)
    critic_optimizer = _optimizer.AdamW(clip_gradient_norm=1.0)
    critic_encoder_hidden_dims: Sequence[int] = (512, 512)
    critic_decoder_hidden_dims: Sequence[int] = (256, 256)
    critic_num_qs: int = 2
    critic_num_vs: int = 2
    num_critic_updates_per_batch: int = 1
    critic_inference_start_step: int = 100
    td_weight_schedule = StepSchedule(init_value=0.0, end_value=1.0, switch_step=1_000)
    train_on_policy_value_function: bool = False

@dataclasses.dataclass(frozen=True)
class FilteredSFTLearnerConfig(RLAlgorithmConfig):
    policy_update_interval: int = 1
    policy_training_start_step: int = 0
    online_ratio: float = 0.5


@dataclasses.dataclass(frozen=True)
class AdvantageWeightedSFTLearnerConfig(FilteredSFTLearnerConfig):
    critic_update_interval: int = 1
    critic_training_start_step: int = 0
    use_ema_critic: bool = True
    critic_ema_decay: float = 0.995
    beta: float = 0.05
    weight_clip: float = 20.0
    critic_reduction: str = "min"
    critic_lr_schedule = ConstantSchedule(value=1e-4)
    critic_optimizer = _optimizer.AdamW(clip_gradient_norm=1.0)
    critic_encoder_hidden_dims: Sequence[int] = (512, 512)
    critic_decoder_hidden_dims: Sequence[int] = (256, 256)
    td_weight_schedule = StepSchedule(init_value=0.0, end_value=1.0, switch_step=1_000)
    critic_num_qs: int = 2
    critic_num_vs: int = 2
    num_critic_updates_per_batch: int = 1


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
class CollectionConfig:
    collect_interval: int = 200
    env_num: int = 4
    env_resolution: int = 256
    resize_image: int = 224
    add_states: bool = True
    num_rollouts: int = 50
    tasks: list[str] = dataclasses.field(default_factory=lambda: ["libero_90_59"])
    replan_steps: int = 5
    num_steps_wait: int = 10
    add_per_step_data: bool = True
    use_time_to_success_as_reward: bool = False
    seed: int = 42
    obs_prefix_key: str = "pi0"
    store_prefix_rep: bool = False


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
            rl_config=FilteredSFTLearnerConfig(
                policy_update_interval=1,
                policy_training_start_step=0,
            ),
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
