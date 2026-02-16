#!/usr/bin/env bash
#SBATCH --job-name=test
#SBATCH --account=a143
#SBATCH --time=03:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

EXP_NAME="${EXP_NAME:-test}"
COLLECT_SEED="${COLLECT_SEED:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-openpi}"

# Added --account=a143 to srun as well to ensure the container inherits it.
srun --account=a143 --environment=vla-post-training \
  uv run scripts/train_online_with_agent.py pi05_libero_online \
  --overwrite \
  --checkpoint_base_dir /capstor/scratch/cscs/${USER}/checkpoints \
  --weight-loader.params-path gs://openpi-assets/checkpoints/pi05_libero/params \
  --exp_name "${EXP_NAME}" \
  --project_name "${WANDB_PROJECT}" \
  --seed "${COLLECT_SEED}" \
  --collect.seed "${COLLECT_SEED}" \
  --batch_size 256
