# Training on Euler (ETH Zürich)

## One-time environment setup

Euler does not use containers, so dependencies are managed with conda + uv. The conda env provides Python 3.12 and build tools; `uv` then manages the actual project dependencies.

```bash
conda create -n vla-post-training python=3.12 cmake uv -c conda-forge -y
conda activate vla-post-training
```

Clone with submodules:

```bash
git clone --recurse-submodules <repo-url>
cd vla-post-training
# or, if already cloned:
git submodule update --init --recursive
```

Install all dependencies:

```bash
uv sync
uv pip install -e openpi/ -e molmospaces/
```

All scripts are run via `uv run` (e.g. `uv run scripts/exp.py ...`), which automatically uses the project's `.venv` without needing to activate it. The launcher already does this; if calling scripts directly, prefix with `uv run`.

## GPU requirements

Pi0.5 full fine-tuning requires at least **2× A100-80GB**. The train state (params + Adam optimizer + EMA) is ~50 GiB and must be sharded across GPUs via FSDP — it does not fit on a single GPU. EMA is also FSDP-sharded (element-wise update, no cross-device communication needed).

| GPUs | fsdp_devices | Resident/GPU | Headroom | Batch size |
|------|-------------|-------------|---------|------------|
| 1 | 1 | ~50 GiB | OOM | — |
| 2 | 2 | ~25 GiB | ~55 GiB | 256 (128/GPU) — *untested* |
| 4 | 4 | ~12.5 GiB | ~67 GiB | 256 (64/GPU) — **confirmed working** |

The confirmed setup is 4 GPUs (half a standard Euler node) with batch_size=256. With EMA sharding, 2 GPUs have ~55 GiB headroom so batch_size=256 (128/GPU) should also work — use `fsft_multitask_libero_euler_test_2gpu.yaml` to validate. If confirmed, the full sweep drops from 56 → 28 GPUs.

## Submitting jobs

Use `--mode euler`. The launcher checks that all required model checkpoints, tokenizer, and LIBERO scene assets are cached before submitting, and downloads anything missing from the login node (which has internet access).

```bash
# Dry run — preview sbatch commands without downloading or submitting
./scripts/launcher.py --config scripts/configs/tuning/fsft_multitask_libero_v0.yaml \
    --mode euler --num_gpus 4 --dry

# Submit
./scripts/launcher.py --config scripts/configs/tuning/fsft_multitask_libero_v0.yaml \
    --mode euler --num_gpus 4
```

Results and checkpoints land in `/cluster/scratch/$USER/results/<project>_<group>/`.

### Euler-specific launcher flags

| Flag | Default | Description |
|------|---------|-------------|
| `--mode euler` | — | Euler submission mode (generates sbatch scripts without `srun --environment`) |
| `--gpu_type` | `a100_80gb` | GPU type for the `--gpus=<type>:<n>` sbatch directive |
| `--num_gpus` | `1` | GPUs per job; use `4` for the confirmed setup, possibly `2` for lighter runs |
| `--mem` | `8G` | Memory per CPU; total RAM = `mem × (8 × num_gpus)` |
| `--duration` | `11:59:00` | SLURM time limit |
| `--partition` | _(none)_ | SLURM partition. If unset, Euler's scheduler picks a default. Run `sinfo` on the login node to see available partitions. `gpupr.4h` (priority queue) is useful for short test runs. |
| `--force` | off | Skip the "proceed?" confirmation prompt |
| `--skip_requeue` | off | Disable auto-requeue on time-limit (requeue is on by default) |

### FSDP sharding

`openpi`'s `fsdp_devices` config field defaults to `1`, which replicates the full train state on every GPU instead of sharding it. The launcher automatically injects `--fsdp_devices <num_gpus>` for any multi-GPU job unless the sweep YAML sets it explicitly. This shards params + optimizer state across GPUs.

When calling `exp.py` directly (without the launcher) on multiple GPUs, pass `--fsdp_devices N` yourself.

### Note on sweep size

`fsft_multitask_libero_v0.yaml` runs 7 task conditions × 2 seeds = 14 jobs. At 4 GPUs each, this is 56 GPUs in parallel; at 2 GPUs (once validated) this drops to 28 GPUs with the same batch_size=256. If cluster quota is a concern, reduce `seed` to `[0]` first or use 2 GPUs.

## Asset management

Model checkpoints and LIBERO scene assets are cached in `/cluster/scratch/$USER/openpi_cache`. The launcher downloads them automatically on first submission. Compute nodes have no internet access and will fail at startup if assets are missing.

To re-download manually (e.g. after reinstalling the venv, which wipes the LIBERO scenes from the hf-libero package directory):

```bash
uv run scripts/download_assets.py --cache_dir /cluster/scratch/$USER/openpi_cache
```

## Config sweeps

The launcher reads a YAML sweep config and submits one job per parameter combination (Cartesian product of `params`):

```yaml
script: scripts/exp.py
config_name: pi05_libero_online_filtered_sft
project_name: vla-post-training
group_name: my_sweep
params:
  seed: [0, 1]
  rl.beta: [0.1, 0.5]
```

This submits 4 jobs. Task IDs support range syntax (`libero_90_22-56` → tasks 22–56) and multipliers (`libero_90_59x4` → task repeated 4 times).

## Troubleshooting

**OOM on all GPUs with identical allocation sizes** — the model is being replicated, not sharded. Check that the launcher emitted `--fsdp_devices N` in the dry-run output. If running without the launcher, pass `--fsdp_devices 4` directly.

**Assets missing on compute node** — run `download_assets.py` on the login node (see above). The LIBERO scene assets live inside the `hf-libero` package directory and are wiped by `uv sync`.

**JIT compilation takes a long time** — compilation of pi0.5 takes ~10–15 minutes on first run. This is normal; subsequent runs load a compiled cache.
