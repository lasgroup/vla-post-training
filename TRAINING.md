# Training on Euler (ETH Zürich)

## One-time environment setup

Clone with submodules, then run the setup script from the repo root:

```bash
git clone --recurse-submodules <repo-url>
cd vla-post-training
# or, if already cloned:
git submodule update --init --recursive
```

```bash
./euler-install.sh
```

This creates the `vla-post-training` conda env (Python 3.12 + cmake + uv), installs all dependencies into `.venv`, and registers the `openpi` and `molmospaces` submodules as editable installs. 

After the initial installation setup, you only need to activate the conda env with 

```bash
conda activate vla-post-training
```

All scripts must be run via `uv run python` (e.g. `uv run python scripts/exp.py ...`), which automatically uses the project's `.venv` without needing to activate it. This applies to the launcher too — `uv run python scripts/launcher.py ...`.

## GPU requirements

Pi0.5 full fine-tuning requires **4× A100-80GB** (half a standard Euler node). The train state (params + Adam optimizer + EMA) is ~50 GiB and must be sharded across GPUs via FSDP. EMA is FSDP-sharded (element-wise update, no cross-device communication needed).

| GPUs | fsdp_devices | Resident/GPU | Headroom | Batch size |
|------|-------------|-------------|---------|------------|
| 1 | 1 | ~50 GiB | OOM | — |
| 4 | 4 | ~12.5 GiB | ~67 GiB | 256 (64/GPU) — **confirmed working** |

## Submitting jobs

Use `--mode euler`. The launcher checks that all required model checkpoints, tokenizer, and LIBERO scene assets are cached before submitting, and downloads anything missing from the login node (which has internet access).

```bash
# Dry run — preview sbatch commands without downloading or submitting
uv run python scripts/launcher.py --config scripts/configs/tuning/fsft_multitask_libero_v0.yaml \
    --mode euler --num_gpus 4 --dry

# Submit
uv run python scripts/launcher.py --config scripts/configs/tuning/fsft_multitask_libero_v0.yaml \
    --mode euler --num_gpus 4
```

Results and checkpoints land in `/cluster/scratch/$USER/results/<project>_<group>/`.

### Euler-specific launcher flags

| Flag | Default | Description |
|------|---------|-------------|
| `--mode euler` | — | Euler submission mode (generates sbatch scripts without `srun --environment`) |
| `--gpu_type` | `a100_80gb` | GPU type for the `--gpus=<type>:<n>` sbatch directive |
| `--num_gpus` | `1` | GPUs per job; use `4` for the confirmed setup |
| `--mem` | `8G` | Memory per CPU; total RAM = `mem × (8 × num_gpus)` |
| `--duration` | `11:59:00` | SLURM time limit |
| `--partition` | _(none)_ | SLURM partition. If unset, Euler's scheduler picks a default. Run `sinfo` on the login node to see available partitions. `gpupr.4h` (priority queue) is useful for short test runs. |
| `--force` | off | Skip the "proceed?" confirmation prompt |
| `--skip_requeue` | off | Disable auto-requeue on time-limit (requeue is on by default) |

### FSDP sharding

`openpi`'s `fsdp_devices` config field defaults to `1`, which replicates the full train state on every GPU instead of sharding it. The launcher automatically injects `--fsdp_devices <num_gpus>` for any multi-GPU job unless the sweep YAML sets it explicitly. This shards params + optimizer state across GPUs.

When calling `exp.py` directly (without the launcher) on multiple GPUs, pass `--fsdp_devices N` yourself.

### Note on sweep size

`fsft_multitask_libero_v0.yaml` runs 7 task conditions × 2 seeds = 14 jobs. At 4 GPUs each, this is 56 GPUs in parallel. If cluster quota is a concern, reduce `seed` to `[0]` first.

## Asset management

Model checkpoints and LIBERO scene assets are cached in `/cluster/scratch/$USER/openpi_cache`. The launcher downloads them automatically on first submission. Compute nodes have no internet access and will fail at startup if assets are missing.

To re-download manually (e.g. after reinstalling the venv, which wipes the LIBERO scenes from the hf-libero package directory):

```bash
uv run python scripts/download_assets.py --cache_dir /cluster/scratch/$USER/openpi_cache
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
