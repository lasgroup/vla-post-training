#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/submit_train.sh"
WANDB_PROJECT="${1:-openpi}"
NUM_SEEDS=5

for ((seed = 0; seed < NUM_SEEDS; seed++)); do
  exp_name="test_${seed}"
  echo "Submitting ${TRAIN_SCRIPT} with --collect.seed=${seed}, --exp_name=${exp_name}, --project_name=${WANDB_PROJECT}"
  sbatch \
    --job-name="${exp_name}" \
    --export=ALL,EXP_NAME="${exp_name}",COLLECT_SEED="${seed}",WANDB_PROJECT="${WANDB_PROJECT}" \
    "${TRAIN_SCRIPT}"
done
