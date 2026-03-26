# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-task post-training framework for Vision-Language-Action (VLA) models. Implements online RL algorithms (Filtered-SFT, Advantage-Weighted SFT, MPO-Weighted SFT, Flow-GRPO, Grouped-MPO, DSRL, Best-of-N) to fine-tune foundation models (pi0, pi0.5) from the `openpi` submodule on robotic tasks (LIBERO, MOLMO-Spaces).

JAX-based training with distributed sharding. Uses `uv` as the package manager.

## Common Commands

```bash
# Run an experiment locally
uv run scripts/flow_grpo_agent/exp.py pi05_libero_online_flow_grpo_sft

# Launch SLURM sweep (dry run to preview)
./scripts/flow_grpo_agent/launcher.py --project_name my_project --dry

# Launch locally
./scripts/flow_grpo_agent/launcher.py --project_name my_project --mode local

# Tests
uv run pytest tests/
uv run pytest tests/nnx_networks/ -v          # specific test dir
uv run pytest src/rl/filtered_sft_agent/ -v    # in-source tests

# Linting
uv run ruff check --fix src/
uv run ruff format src/
```

## Architecture

### Agent Hierarchy (`src/rl/`)

All learners inherit from a base `Agent` class (`src/rl/agent.py`) with this interface: `sample_actions`, `eval_actions`, `add_data(StepData)`, `save_episode`, `update`, `save_checkpoint`.

```
Agent
  FilteredSFTLearner              # SFT on (successful) episodes
    AdvantageWeightedSFTLearner   # + critic, advantage-weighted loss
      MPOWeightedSFTLearner       # + on-policy critic updates
        GroupedMPOLearner          # + group sampling
        FlowGRPOLearner           # + flow-based GRPO, KL penalty
  DSRLLearner                     # SAC-based, separate hierarchy
  BestofNLearner                  # Simple baseline
```

Each learner overrides `_update_policy()` / `_update_critics()` / `_train_step()` and uses JIT-compiled closures (no `self` in JAX-compiled code).

### Config System (`src/training/config.py`)

Configs are frozen dataclasses resolved via `tyro`. Each algorithm has a corresponding `*Config` class mirroring the learner hierarchy. Named configs are registered in `_CONFIGS` at the bottom of the file.

CLI override pattern: `--rl.policy_training_start_step 900`, `--rl.td_weight_schedule.switch_step 1500`

Task notation supports ranges and multipliers: `libero_90_22-56x4` expands to tasks 22-56 each repeated 4 times. Total expanded tasks must match `env_num` / `eval_env_num`.

### Training Loop (`src/training/collect.py`)

Standard collect-then-update loop: agent samples action chunks -> env steps through chunk -> shifted window observation alignment -> `add_data(StepData)` -> replay buffer -> `agent.update()` (JIT-compiled) -> periodic `evaluate_policy()`.

### Network Architecture (`src/rl/networks/`)

- `StateActionCritic`: encoder -> embedding -> Q-decoder (ensemble)
- `StateValue`: encoder -> embedding -> V-decoder (ensemble)
- Policy/value decoders in `networks/decoders/`; observation encoders in `networks/encoders/`

### Environment Layer (`src/envs/`)

Wrappers chain: `QueryFrequencyWrapper` (action chunking) -> `Pi0ObservationWrapper` (format for pi0) -> `TimeToSuccessAsRewardWrapper` (optional). Vectorized via `SubprocVectorEnv` or `DummyVectorEnv`.

### Launcher System (`scripts/launcher_util.py`, `scripts/*/launcher.py`)

Each algorithm has its own `launcher.py` that defines hyperparameter grids and calls `generate_run_commands()` to submit SLURM jobs or run locally. Uses `dict_permutations()` for grid sweeps.

### Key Named Configs

| Config Name | Algorithm |
|---|---|
| `pi05_libero_online_filtered_sft` | Filtered SFT |
| `pi05_libero_online_aw_sft` | Advantage-Weighted SFT |
| `pi05_libero_online_mpo_sft` | MPO-Weighted SFT |
| `pi05_libero_online_grouped_mpo_sft` | Grouped MPO |
| `pi05_libero_online_flow_grpo_sft` | Flow-GRPO |
| `pi05_libero_online_best_of_n` | Best-of-N |
| `pi05_libero_online_dsrl` | DSRL |

## Key Dependencies

- **JAX/Flax/Optax**: Core ML stack (JIT, autodiff, neural nets, optimizers)
- **openpi** (submodule): Foundation VLA models (pi0, pi0.5), checkpoint loading, base training configs
- **tyro**: CLI argument parsing with dataclass support
- **wandb**: Experiment tracking
- **orbax-checkpoint**: Model checkpointing
- **jaxtyping + beartype**: Runtime shape checking on arrays
