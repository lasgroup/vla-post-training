import dataclasses
from typing import Any, Optional, Sequence

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
    obs_prefix_key: str = "pi0"
    online_ratio: float = 1.0


@dataclasses.dataclass(frozen=True)
class SACModelConfig:
    obs_dim: int = 1024  # Update this to match actual prefix_rep flattened dim
    action_horizon: int = 10
    action_dim: int = 7
    hidden_dim: int = 256

    def create(self, rng):
        import flax.nnx as nnx
        from src.rl.dsrl_agent.dsrl_learner import SACModel
        return SACModel(
            self.obs_dim, 
            self.action_horizon * self.action_dim, 
            self.hidden_dim, 
            rngs=nnx.Rngs(rng)
        )


@dataclasses.dataclass(frozen=True)
class DSRLTrainConfig:
    # additional configs for online training
    collect: CollectionConfig = CollectionConfig()

    batch_size: int = 256
    seed: int = 42
    discount: float = 0.99
    fsdp_devices: int = 1
    
    model: SACModelConfig = SACModelConfig()

    lr: float = 3e-4
    warmup_steps: int = 100
    decay_steps: int = 1_000_000
    clip_gradient_norm: float = 1.0

    trainable_filter: str = ".*"
    freeze_filter: str = "none"
    ema_decay: float = 0.995

    resume: bool = False
    checkpoint_dir: str = "/tmp/dsrl_checkpoints"
    checkpoint_base_dir: str = "/tmp/dsrl_checkpoints"
    keep_period: int = 1000
    overwrite: bool = True
    weight_loader: Optional[Any] = None
    
    # Generic train_online.py loop params
    num_train_steps: int = 30_000
    log_interval: int = 100
    save_interval: int = 1000
    wandb_enabled: bool = False
    exp_name: str = "dsrl_experiment"
    project_name: str = "dsrl"
    name: str = "dsrl_run"
    
    # SAC specific Configs
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    tau: float = 0.005
    alpha: float = 0.2
    target_entropy: float = -7.0  # -dim
