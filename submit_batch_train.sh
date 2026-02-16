#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <wandb_project_name> [exp_prefix]"
  exit 1
fi

WANDB_PROJECT="$1"
EXP_PREFIX="${2:-agent_online}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/submit_train.sh"

for seed in {0..4}; do
  exp_name="${EXP_PREFIX}_${seed}"
  echo "Submitting ${TRAIN_SCRIPT} with --collect.seed=${seed}, --exp_name=${exp_name}, --project_name=${WANDB_PROJECT}"
  sbatch \
    --job-name="${exp_name}" \
    --export=ALL,EXP_NAME="${exp_name}",COLLECT_SEED="${seed}",WANDB_PROJECT="${WANDB_PROJECT}" \
    "${TRAIN_SCRIPT}"
done
