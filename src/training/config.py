import dataclasses
import tyro
from openpi.training.config import (
    TrainConfig,
    DataConfig,
    pi0_config,
    SimpleDataConfig,
    AssetsConfig,
    ModelType,
    LeRobotLiberoDataConfig,
)
from typing import Literal, Sequence

import openpi.training.optimizer as _optimizer
import openpi.policies.droid_policy as _droid_policy
import openpi.transforms as _openpi_transforms
import jax.numpy as jnp
import optax
import openpi.training.weight_loaders as weight_loaders


@dataclasses.dataclass(frozen=True)
class ConstantSchedule(_optimizer.LRScheduleConfig):
    """Constant learning rate schedule."""

    value: float = 5e-5

    def create(self) -> optax.Schedule:
        return optax.constant_schedule(self.value)


@dataclasses.dataclass(frozen=True)
class StepSchedule:
    """Returns init_value for steps < switch_step, then end_value."""

    init_value: float = 0.0
    end_value: float = 1.0
    switch_step: int = 1000

    def __call__(self, step):
        return jnp.where(step < self.switch_step, self.init_value, self.end_value)


@dataclasses.dataclass(frozen=True)
class RLAlgorithmConfig:
    discount: float = 0.99
    buffer_capacity: int = 1024


@dataclasses.dataclass(frozen=True)
class PolicyTrainingConfig:
    update_interval: int = 1


@dataclasses.dataclass(frozen=True)
class CriticTrainingConfig:
    update_interval: int = 1
    ema_decay: float = 0.995
    encoder_hidden_dims: Sequence[int] = (512, 512)
    decoder_hidden_dims: Sequence[int] = (256, 256)
    num_qs: int = 2
    num_vs: int = 2
    td_weight_schedule: StepSchedule = StepSchedule(init_value=1.0, end_value=1.0, switch_step=1_000)  # pure TD
    num_value_bins: int = 1  # 1: Gaussian (MSE-equivalent), >1 = Categorical over bins
    value_target_type: str = "two_hot"  # "one_hot" | "two_hot"
    use_distributional_critic: bool = False
    distributional_target_reduction: str = "min"
    inference_start_step: int = 1  # step 0 collects with the plain policy (faster)
    # Critics are lightweight (MLP-only); a larger batch than the policy often
    # stabilises TD learning without a meaningful memory cost.
    batch_size: int = 1024
    # Class-level attributes (not dataclass fields) so subclasses can override the default.
    lr_schedule = ConstantSchedule(value=1e-4)
    optimizer = _optimizer.AdamW()
    # BRONet critic (alternative to the MLP backbone)
    use_bronet: bool = True
    bronet_hidden_dim: int = 1024
    bronet_depth: int = 2


# Define hyperparameter structures for your algorithms
@dataclasses.dataclass(frozen=True)
class FilteredSFTLearnerConfig(RLAlgorithmConfig):
    policy: PolicyTrainingConfig = PolicyTrainingConfig()


@dataclasses.dataclass(frozen=True)
class BestofNLearnerConfig(FilteredSFTLearnerConfig):
    critic: CriticTrainingConfig = CriticTrainingConfig()
    n_samples: int = 32
    discount: float = 0.995


@dataclasses.dataclass(frozen=True)
class CollectionConfig:
    collect_interval: int = 200
    env_num: int = 4
    env_resolution: int = 256
    resize_image_h: int = 224
    resize_image_w: int = 224
    num_rollouts: int = 20
    num_initial_rollouts: int | None = None
    domain: Literal["libero", "molmo"] = "libero"
    tasks: list[str] = dataclasses.field(default_factory=lambda: ["libero_90_59"])
    eval_tasks: list[str] = dataclasses.field(default_factory=lambda: ["libero_90_59"])
    replan_steps: int = 5
    num_steps_wait: int = 10
    eval_env_num: int = 4
    eval_interval: int = 300
    num_eval_rollouts: int = 32
    max_episode_steps: int = 400  # used to auto-compute value bounds


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


def _make_config(name: str, rl_config: RLAlgorithmConfig, **domain_kwargs) -> OnlineTrainConfig:
    return OnlineTrainConfig(
        name=name,
        rl=rl_config,
        batch_size=256,
        lr_schedule=ConstantSchedule(value=2.5e-5),
        ema_decay=0.999,
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=10_000,
        seed=0,
        **domain_kwargs,
    )


def make_base_libero_config(name: str, rl_config: RLAlgorithmConfig) -> OnlineTrainConfig:
    return _make_config(
        name,
        rl_config,
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            assets=AssetsConfig(
                # load norm_stats from pretrained checkpoint
                assets_dir="gs://openpi-assets/checkpoints/pi05_libero/assets",
            ),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_libero/params"
        ),
    )


def make_base_molmo_config(name: str, rl_config: RLAlgorithmConfig) -> OnlineTrainConfig:
    return _make_config(
        name,
        rl_config,
        model=pi0_config.Pi0Config(pi05=True, action_horizon=15, discrete_state_input=False),
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
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_droid_jointpos/params"
        ),
        collect=CollectionConfig(domain="molmo", max_episode_steps=450),
    )


_CONFIGS = [
    make_base_libero_config("pi05_libero_online_filtered_sft", FilteredSFTLearnerConfig()),
    make_base_molmo_config("pi05_molmo_online_filtered_sft", FilteredSFTLearnerConfig()),
    make_base_libero_config("pi05_libero_online_best_of_n", BestofNLearnerConfig()),
    make_base_molmo_config("pi05_molmo_online_best_of_n", BestofNLearnerConfig()),
]
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli(
        {k: (k, v) for k, v in _CONFIGS_DICT.items()}
    )
