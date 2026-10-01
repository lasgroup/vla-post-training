# Codebase Overview

This repository implements online RL fine-tuning of the **Pi0.5** vision-language-action (VLA) foundation model. Multiple actor-critic and filtered-SFT algorithms are implemented. All share a common infrastructure for data collection, environment interaction, checkpointing, and W&B logging.

The **openpi/** directory is a vendored copy of the OpenPI library that provides the foundation model, training utilities (optimizer, sharding, checkpointing), and shared utilities used throughout.

---

## Repository Layout

```
vla-post-training/
├── scripts/                    # Per-algorithm entry points and SLURM launchers
│   ├── best_of_n_agent/
│   │   ├── exp.py              # Training entry point
│   │   └── launcher.py         # Hyperparameter sweep + SLURM submission
│   ├── awr_agent/              # Advantage Weighted SFT
│   ├── filtered_sft_agent/
│   ├── mpo_agent/
│   ├── flow_grpo_agent/
│   ├── dsrl_agent/
│   └── launcher_util.py        # Shared sbatch/srun utilities (generate_run_commands, etc.)
├── src/
│   ├── training/               # Shared training infrastructure
│   │   ├── config.py           # All config dataclasses (OnlineTrainConfig, BestofNLearnerConfig, …)
│   │   ├── collect.py          # collect_data(), evaluate_policy()
│   │   ├── data_loader.py      # Offline dataset loading and mixing
│   │   └── runtime_state.py    # Checkpoint save/restore helpers
│   └── rl/
│       ├── agent.py            # Abstract Agent base class
│       ├── replay_buffer.py    # Online replay buffer
│       ├── types.py            # StepData namedtuple
│       ├── value_distribution.py  # Distributional value function support
│       ├── best_of_n/
│       ├── filtered_sft_agent/
│       ├── advantage_weighted_sft/
│       ├── mpo_weighted_sft/
│       ├── flow_grpo/
│       ├── dsrl/
│       └── networks/           # Shared critic / policy network building blocks
└── openpi/                     # Vendored OpenPI dependency
    └── src/openpi/
        ├── models/             # Pi0.5 model (compute_loss, sample_actions)
        ├── training/           # TrainState, optimizer, sharding, checkpoints
        └── shared/             # array_typing (@typecheck), nnx_utils
```

---

## Algorithm Hierarchy

All algorithms extend `Agent` (in `src/rl/agent.py`), which defines the interface: `eval_actions`, `sample_actions`, `add_data`, `update`, `save_checkpoint`.

```
Agent (abstract)
└── FilteredSFTLearner            # Online SFT on filtered (successful) trajectories
    ├── BestofNLearner            # Adds dual Q/V critics for value-guided filtering
    └── AdvantageWeightedSFTLearner  # Adds advantage-weighted policy loss
        └── MPOWeightedSFTLearner    # On-policy critic variant
            └── FlowGRPOLearner      # Diffusion-based flow-guided refinement

DSRLLearner                       # Separate SAC-style implementation (extends Agent directly)
```

---

## Best-of-N Agent (Focus)

**Concept**: Collect N trajectories, train Q(s,a) and V(s) critics to score them, then fine-tune the policy via SFT weighted by critic-estimated advantages.

### Key files

| File | Role |
|---|---|
| `scripts/best_of_n_agent/exp.py` | Entry point: builds critic defs, instantiates `BestofNLearner`, runs training loop |
| `scripts/best_of_n_agent/launcher.py` | SLURM launcher; defines hyperparameter grid in `applicable_configs` |
| `src/rl/best_of_n/best_of_n_learner.py` | `BestofNLearner` class; holds critic train states, dispatches updates |
| `src/rl/best_of_n/update_critic.py` | `train_q_step`, `train_value_step`, critic init helpers |
| `src/rl/value_distribution.py` | Distributional value support (Gaussian / Categorical) |

### Critic architecture

Both critics share an encoder. The encoder is built inside `exp.py` via `_build_pi0_backbone_critic_defs`:

```
Observation (state vector + optional Pi0.5 prefix embedding)
  └── MLPEncoder   (src/rl/networks/encoders/encoders.py)
        ├── StateActionEnsembleDecoder  → Q(s,a)   (ensemble of num_qs heads)
        └── StateValueEnsembleDecoder  → V(s)      (ensemble of num_vs heads)
```

Decoder output shape:
- `num_value_bins=1` → scalar per ensemble member → Gaussian distribution (MSE-equivalent)
- `num_value_bins>1` → K logits per ensemble member → Categorical distribution

The ensemble reduction (min or mean across heads) is done in `summarize_critic_values` inside `update_critic.py`.

### Distributional critics (`src/rl/value_distribution.py`)

- `GaussianValueDistribution`: wraps scalar logits; `log_prob = -square(logit - target)`; `mean() = logit`
- `CategoricalValueDistribution`: K uniformly-spaced bins over [lower, upper]; supports `one_hot` (hard) and `two_hot` (linear interpolation) target encoding; `mean() = Σ softmax(logits) * bin_centers`
- `make_value_distribution(logits, num_bins, lower, upper, target_type)`: factory
- `get_value_bounds(config)`: reads `config.rl.value_lower_bound / value_upper_bound`

Value bounds can be auto-computed in `OnlineTrainConfig.__post_init__` from `discount`, `collect.max_episode_steps`, and `collect.use_time_to_success_as_reward`.

### Config (`BestofNLearnerConfig` in `src/training/config.py`)

Key fields beyond the shared base:
```python
num_value_bins: int = 1           # 1 = Gaussian (MSE), >1 = categorical
value_lower_bound: float | None   # auto-filled if None
value_upper_bound: float | None   # auto-filled if None
value_target_type: str = "one_hot"

critic_encoder_hidden_dims: tuple
critic_decoder_hidden_dims: tuple
critic_num_qs: int                # Q-ensemble size
critic_num_vs: int                # V-ensemble size
critic_lr_schedule: LRScheduleConfig
critic_optimizer: OptimizerConfig
num_critic_updates_per_batch: int
use_ema_critic: bool
critic_ema_decay: float | None
td_weight_schedule: LRScheduleConfig  # interpolates TD vs MC loss
```

### Training step flow

1. `collect_data()` runs rollouts, stores `StepData` in replay buffer
2. `agent.update()` in `BestofNLearner`:
   - Samples a batch from the buffer
   - Calls `train_q_step(config, rng, q_state, value_state, batch)` — updates Q-network
   - Calls `train_value_step(config, rng, value_state, q_state, batch)` — updates V-network
   - Calls parent `FilteredSFTLearner.update()` for the policy SFT step

### Prefix embeddings

`exp.py` optionally extracts the Pi0.5 transformer prefix representation and passes it as an extra `"pi0_prefix"` key in the critic observation dict. Shape is inferred at startup via `_infer_prefix_embedding_shape`.

---

## Other Algorithms

### FilteredSFT (`src/rl/filtered_sft_agent/`)
Simplest baseline. Collects online rollouts, keeps only successful episodes, fine-tunes via SFT. No critic. Config: `FilteredSFTLearnerConfig`.

### Advantage Weighted SFT (`src/rl/advantage_weighted_sft/`)
Same dual Q/V critic setup as Best-of-N. Policy loss is SFT weighted by `exp(A/β)` where `A = Q(s,a) - V(s)`. `update_actor.py` computes advantages and the weighted loss. Config: `AdvantageWeightedSFTLearnerConfig`.

### MPO Weighted SFT (`src/rl/mpo_weighted_sft/`)
Extends AWR. When `store_buffer_actions_in_batch=False`, samples fresh on-policy actions from the current policy for critic targets instead of replay buffer actions. Config: `MPOWeightedSFTLearnerConfig`.

### Flow GRPO (`src/rl/flow_grpo/`)
Extends MPO. Groups trajectories by prompt (`group_size`), uses within-group advantage normalization, and applies flow-matching-style diffusion refinement at a configurable noise level. Config: `FlowGRPOSFTLearnerConfig`.

### DSRL (`src/rl/dsrl/`)
Standalone SAC-style implementation. Does not extend FilteredSFT. Uses an entropy coefficient α with auto-tuning (`update_alpha.py`), a Tanh-Normal policy, and an ensemble of 10 Q-networks. Supports spatial softmax image encoders. Config: `DSRLLearnerConfig`.

---

## Shared Networks (`src/rl/networks/`)

```
networks/
├── rl_networks.py          # StateActionCritic, StateValue, Policy (top-level modules)
├── mlp.py                  # MLP (hidden_dims, layer_norm, dropout, activate_final)
├── constants.py            # default_init() — kernel initializer
├── encoders/
│   ├── encoders.py         # MLPEncoder, ImageEncoder
│   └── resnet.py           # ResNet / ImpalaNet CNN backbones
└── decoders/
    └── values/
        ├── state_action_value.py   # StateActionValueDecoder, StateActionEnsembleDecoder
        └── state_value.py          # StateValueDecoder, StateValueEnsembleDecoder
```

`StateActionCritic` and `StateValue` (in `rl_networks.py`) are the top-level critic modules used by both Best-of-N and AWR. They compose an encoder with a decoder.

---

## Configuration System (`src/training/config.py`)

`OnlineTrainConfig` wraps `TrainConfig` (from openpi) plus:
- `rl: RLAlgorithmConfig` — one of the algorithm configs above
- `collect: CollectionConfig` — env setup, tasks, rollout counts, reward shaping

Config instances are created by name from a registry (`_CONFIGS` dict), then extended via CLI overrides parsed by `tyro`. The pattern in every `exp.py`:
```python
if __name__ == "__main__":
    main(_config.cli())
```

Pre-defined config names follow the pattern `pi05_libero_online_{algorithm}`, e.g.:
- `pi05_libero_online_best_of_n`
- `pi05_libero_online_aw_sft`
- `pi05_libero_online_filtered_sft`

---

## Launcher Utilities (`scripts/launcher_util.py`)

- `generate_srun_command(script, config_name, flags)` — builds the `srun uv run ...` command string
- `generate_run_commands(command_list, ...)` — submits via `sbatch` (swiss-ai mode) or runs locally; writes `.sbatch.sh` scripts under `logs/sbatch/`; in `--dry` mode prints the sbatch command and the inner srun command
- `dict_permutations(d)` — cartesian product over hyperparameter grid (supports tuple keys for grouped sweeps)
- `apply_requeue_flags(flags, enabled)` — enforces `resume=True, overwrite=False, requeue=True`

---

## Data Flow Summary

```
Environment (libero / molmo)
  └─ VectorEnv wrappers (src/envs/)
        └─ collect_data() / evaluate_policy()  (src/training/collect.py)
              └─ agent.sample_actions() / agent.add_data()
                    └─ ReplayBuffer  (src/rl/replay_buffer.py)
                          └─ agent.update()
                                ├─ train_q_step / train_value_step  (update_critic.py)
                                └─ policy SFT step  (update.py / update_actor.py)
```

Metrics are logged to W&B at every `log_interval` steps. Checkpoints (policy + critic states) are saved via Orbax at every `collect_interval`.
