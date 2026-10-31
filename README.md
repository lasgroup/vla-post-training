# OGPO

**Off-Policy Generative Policy Optimization**

## Install

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (CUDA 13)
uv sync --extra cuda13 --extra robomimic

# Or CUDA 12
uv sync --extra cuda12 --extra robomimic
```

Drop `--extra robomimic` if not running Robomimic environments (square, transport, tool_hang).

## Running Locally

```bash
# OGPO on square
bash scripts/ogpo/square.sh

# Override any parameter
bash scripts/ogpo/square.sh --seed=42 --online_steps=500000

# Other algorithms
bash scripts/ogpo/transport.sh
bash scripts/bptt/square.sh
bash scripts/qc/square.sh
bash scripts/dsrl/square.sh
bash scripts/expo/square.sh
bash scripts/fql/square.sh
```

## Running with SLURM

Every script doubles as an sbatch job (SBATCH headers are built in):

```bash
sbatch scripts/ogpo/square.sh
sbatch scripts/ogpo/square.sh --seed=42
sbatch scripts/bptt/transport.sh
sbatch scripts/fql/toolhang.sh
```

## pi0.5 examples on MolmoSpaces

Four recipes fine-tune the pi0.5 `pi05_droid_jointpos` checkpoint on MolmoSpaces pick tasks (one seed each, sized for 4 x 96 GB GPUs).
The hyperparameters are those of the multi-task-144 OGPO runs; only the task set and what scales with it differ.

| | one task (box 138) | 16 tasks |
|---|---|---|
| OGPO | `scripts/ogpo_pi05/molmo_box138.sh` | `scripts/ogpo_pi05/molmo_mt16.sh` |
| paper AWR (not yet run on MolmoSpaces) | `scripts/awr_pi05/molmo_box138.sh` | `scripts/awr_pi05/molmo_mt16.sh` |

```bash
# 1. once: the molmo Python env (.venv-molmo). openpi-client is a package inside the openpi repository.
bash scripts/ogpo/setup_molmo_venv.sh --molmospaces=<lasgroup/molmospaces checkout @ 2a828d5> \
                                      --openpi_client=<openpi checkout>/packages/openpi-client

# 2. once: pi05_droid_jointpos, the PaliGemma tokenizer and the MolmoSpaces assets (needs gsutil)
export OPENPI_CACHE=... MLSPACES_ASSETS_DIR=... MLSPACES_CACHE_DIR=... NLTK_DATA=...   # directories to fill
mkdir -p logs && sbatch scripts/ogpo/download_molmo_assets.sbatch    # or run it with bash on a compute node

# 3. train, with the same four variables exported (the script exits if one is unset; its header lists what it needs)
SAVE_DIR=... bash scripts/ogpo_pi05/molmo_mt16.sh                    # or sbatch it (mkdir -p logs first)
```

Details for running and tuning pi0.5 (flags, tuning knobs, memory, outputs, resume): [pi05_tuning.md](pi05_tuning.md).

The recipes log to Weights & Biases: run `wandb login` first, or `export WANDB_MODE=offline` to keep the run files local (`wandb sync`
them later). `--log=false` turns W&B off but also stops the CSV logs and `checkpoint_latest.pkl` from being written.

Any `--variable=value` after the script name overrides the variable of the same name in its config block (an unknown name is silently
ignored). The mt16 recipes keep the replay images in host RAM: see the script header, or pass `--buffer_storage_root=<node-local dir>`.

## Citation

```bibtex
@article{ogpo2026,
  title={OGPO: Off-Policy Generative Policy Optimization},
  author={TBD},
  year={2026}
}
```
