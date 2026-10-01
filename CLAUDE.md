# Claude Context

Read `AGENTS.md` first for the overall architecture. This file covers recent implementation work, gotchas, and current in-progress changes.

---

## Current Branch

Working on `main`. There are uncommitted changes implementing **offline critic pretraining** for Best-of-N. Run `git diff --stat HEAD` to see which files are modified.

---

## Recently Merged PRs

- **PR #43** (`ralf/prefix_storing_integration`): Pi0 prefix embedding caching for Best-of-N
- **PR #42** (`ralf/distributional_critic_integration`): Distributional critic losses for Best-of-N

---

## In-Progress: Offline Critic Pretraining

**Goal**: Load a pre-built offline episode buffer at startup and pretrain the critics on it before online collection begins.

### Files changed (not yet committed/PR'd)

| File | Change |
|---|---|
| `src/rl/replay_buffer.py` | Added `load_paths`, `save_path` params; `load_episodes()`, `save_episode()` methods |
| `src/training/config.py` | Added `offline_buffer_load_paths`, `num_offline_pretraining_steps`, `offline_buffer_capacity` to `FilteredSFTLearnerConfig`; `BestofNLearnerConfig` overrides `offline_buffer_capacity=250_000` |
| `src/rl/filtered_sft_agent/filtered_sft_learner.py` | Added `_get_offline_data_buffer()` + `self._offline_data_buffer` in `__init__` |
| `src/rl/best_of_n/best_of_n_learner.py` | Added `self.warm_start_training_steps`, `pretrain_with_offline_data()` |
| `scripts/best_of_n_agent/exp.py` | Added offline pretraining loop before the main training loop |
| `scripts/best_of_n_agent/launcher.py` | Added `--num_offline_pretraining_steps`, `--offline_buffer_load_paths` flags; `rl.num_offline_pretraining_steps` in `applicable_configs`; auto-sets offline buffer path |

### Design decisions for extensibility

- `offline_buffer_load_paths` and `num_offline_pretraining_steps` live on `FilteredSFTLearnerConfig` (not the base `RLAlgorithmConfig`) because the offline buffer infrastructure lives in `FilteredSFTLearner`. AWR inherits `FilteredSFTLearnerConfig` so it gets these fields for free.
- `_get_offline_data_buffer()` is in `FilteredSFTLearner` (base class) — any subclass (AWR, BestofN) gets `self._offline_data_buffer` automatically.
- `pretrain_with_offline_data()` is BestofN-specific (trains critics). AWR would add its own version (trains policy).

### How pretraining works

`exp.py` runs the pretraining loop before the main loop:
```python
for pretrain_step in range(warmup_start, config.rl.num_offline_pretraining_steps):
    info = agent.pretrain_with_offline_data()
    # logs to W&B under "pretrain/*" metric group
```

`pretrain_with_offline_data()` samples from `self._offline_data_buffer`, calls `_update_critics_jitted`. Since offline episodes have no cached prefix embeddings, `_online_batch_to_critic_batch` falls back to `_recompute_prefix_embedding` automatically (same path as when `store_prefix_rep=False`).

### Building the offline buffer

The offline buffer is built by running `scripts/convert_hf_libero_to_online_buffer.py` (ported from `origin/value_learning_pretraining_on_offline_data`). It downloads `physical-intelligence/libero` from HuggingFace, applies pi0 transforms, and writes `episode_XXXXXX.h5df` files.

The dataset is already cached at `/capstor/scratch/cscs/ralfroemer/huggingface/lerobot/physical-intelligence/libero/`.

Run command:
```bash
HF_LEROBOT_HOME=/capstor/scratch/cscs/ralfroemer/huggingface/lerobot \
uv run scripts/convert_hf_libero_to_online_buffer.py \
    --output-dir /capstor/scratch/cscs/ralfroemer/libero_offline_buffer
```

Once built, update the auto-set path in `launcher.py`:
```python
flags["rl.offline_buffer_load_paths"] = ["/capstor/scratch/cscs/ralfroemer/libero_offline_buffer"]
```

---

## Prefix Embedding Caching (`store_prefix_rep`)

**What it does**: At collection time, `BestofNLearner.sample_actions()` returns `(actions, prefix)` instead of `actions` when `config.collect.store_prefix_rep=True`. The prefix (pi0 transformer prefix representation, averaged over sequence) is stored alongside the transition in the replay buffer. During critic updates, cached prefixes are used instead of recomputing.

**Key implementation details**:

- `collect.py`: `action_chunk = agent.sample_actions(...)` returns either an array or a `(array, prefix)` tuple. Both `collect_data` and `evaluate_policy` extract `env_action_chunk = action_chunk[0] if isinstance(action_chunk, tuple) else action_chunk` before passing to `env.step`.
- `FilteredSFTLearner._sample_action()`: When `return_prefix_rep=True`, bypasses `infer_with_model` (which crashes on tuple output) and calls `_sample_actions_with_model` + `_output_transform` separately.
- `BestofNLearner._make_buffer_dummy_data()`: Injects prefix embedding into buffer schema so the buffer allocates storage for it. `_prefix_embed_dim` must be set BEFORE `super().__init__()` because the base class calls `_make_buffer_dummy_data` during init.
- `_online_batch_to_critic_batch()`: Uses cached prefix if present in batch, otherwise calls `_recompute_prefix_embedding`.
- Libero key convention: env keys have `"observation/"` prefix. After `obs_key_process_fn`, it's stripped. Prefix is stored as `f"observation/{PREFIX_EMBEDDING_NAME}"` in episodes → becomes `PREFIX_EMBEDDING_NAME` in the buffer.

---

## `ShardedReplayBuffer` API (`src/rl/replay_buffer.py`)

```python
ShardedReplayBuffer(
    dummy_data,
    max_capacity,
    ...,
    load_paths=None,    # list[str]: load episode files on init
    save_path=None,     # str: directory to save episodes when insert(save_episode=True)
)

buffer.insert(data, save_episode=False)   # save_episode=True writes to save_path
buffer.load_episodes(paths)               # loads episode_*.h5df files from dirs
buffer.save_episode(data, save_dir)       # writes one episode_XXXXXX.h5df
```

Episode files are `h5df` format (HDF5). The existing `save_shard`/`restore_shards` methods are for resumable training checkpointing and are separate from episode-level save/load.

---

## Launcher Pattern (`scripts/best_of_n_agent/launcher.py`)

- `applicable_configs` dict: each key is a CLI flag (e.g. `"rl.n_samples"`), each value is a list of options. `dict_permutations` takes the cartesian product. Tuple keys (e.g. `("collect.tasks", "collect.eval_tasks")`) group flags that must vary together.
- `flags` dict sets defaults; `flags.update(combo)` applies the grid values (grid overrides defaults).
- `--print_commands`: prints raw `uv run ...` commands without submitting to SLURM.
- `--dry`: submits sbatch jobs but the inner srun doesn't execute.
- Auto-set logic after `flags.update(combo)` handles derived flags (e.g. offline buffer path when pretraining steps > 0).

---

## Known Gotchas

- **tyro + frozen dataclasses**: Config fields must be annotated with a type to be CLI-overridable. Fields like `critic_lr_schedule = ConstantSchedule(...)` without annotation are NOT overridable via CLI.
- **`infer_with_model` with `return_prefix_rep=True`**: crashes because `_jax_inference` applies `_output_transform` to the `(actions, prefix)` tuple. Must bypass by calling `_sample_actions_with_model` then `_output_transform` separately.
- **`SubprocVectorEnv.step`** does `action[i]` for each worker — integer indexing on a tuple returns the i-th element, not the i-th env's action. Always extract pure action array before `env.step`.
- **`load_episodes` in `ShardedReplayBuffer`** must call `insert(data, save_episode=False)` to avoid re-saving loaded episodes.
- **W&B tags**: Not supported via tyro CLI (no `tags` field in `TrainConfig`). Group field (`--group`) is supported.

---

## Cluster / Environment Notes

- Cluster: Clariden (Swiss AI / CSCS). Jobs run via `sbatch` + `srun` with `--environment=vla-post-training`.
- Interactive node: `clariden-ln004`. Run `uv run` directly here for testing.
- HuggingFace cache: `/capstor/scratch/cscs/ralfroemer/huggingface/`. Use `HF_LEROBOT_HOME=.../lerobot` (not `LEROBOT_HOME` — that's deprecated in lerobot 0.3.3).
- Checkpoints: `/capstor/scratch/cscs/ralfroemer/checkpoints/`.
- SLURM account: `a143`.
