#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=a143
#SBATCH --time=03:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

EXP_NAME="${EXP_NAME:-test}"
COLLECT_SEED="${COLLECT_SEED:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-openpi}"
COLLECT_INTERVAL="${COLLECT_INTERVAL:-200}"
CRITIC_UPDATE_INTERVAL="${CRITIC_UPDATE_INTERVAL:-1}"
POLICY_UPDATE_INTERVAL="${POLICY_UPDATE_INTERVAL:-50}"
CRITIC_TRAINING_START_STEP="${CRITIC_TRAINING_START_STEP:-1}"
POLICY_TRAINING_START_STEP="${POLICY_TRAINING_START_STEP:-400}"

# Added --account=a143 to srun as well to ensure the container inherits it.
srun --account=a143 --environment=vla-post-training \
uv run scripts/train_online_with_awr_agent.py pi05_libero_online \
--overwrite \
--checkpoint_base_dir /capstor/scratch/cscs/${USER}/checkpoints \
--weight-loader.params-path gs://openpi-assets/checkpoints/pi05_libero/params \
--exp_name "${EXP_NAME}" \
--project_name "${WANDB_PROJECT}" \
--collect.seed "${COLLECT_SEED}" \
--collect.collect_interval "${COLLECT_INTERVAL}" \
--critic_update_interval "${CRITIC_UPDATE_INTERVAL}" \
--policy_update_interval "${POLICY_UPDATE_INTERVAL}" \
--critic_training_start_step "${CRITIC_TRAINING_START_STEP}" \
--policy_training_start_step "${POLICY_TRAINING_START_STEP}" \
--batch_size 256
