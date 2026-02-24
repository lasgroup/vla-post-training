#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=a143
#SBATCH --time=03:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

EXP_NAME="${EXP_NAME:-test}"
WANDB_PROJECT="${WANDB_PROJECT:-openpi}"
TRAIN_SEED="${TRAIN_SEED:-0}"
TRAIN_CONFIG="${TRAIN_CONFIG:-pi05_libero_online_aw_sft}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-/capstor/scratch/cscs/${USER}/checkpoints}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
# Added --account=a143 to srun as well to ensure the container inherits it.
srun --account=a143 --environment=vla-post-training \
uv run scripts/train_online_with_awr_agent.py "${TRAIN_CONFIG}" \
--overwrite \
--checkpoint_base_dir "${CHECKPOINT_BASE_DIR}" \
--exp_name "${EXP_NAME}" \
--project_name "${WANDB_PROJECT}" \
--seed "${TRAIN_SEED}" \
--collect.seed "${TRAIN_SEED}" \
--log_interval "${LOG_INTERVAL}"
