#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/submit_train.sh"

for seed in {0..5}; do
  exp_name="test_${seed}"
  echo "Submitting ${TRAIN_SCRIPT} with --collect.seed=${seed}, --exp_name=${exp_name}"
  sbatch \
    --job-name="${exp_name}" \
    --export=ALL,EXP_NAME="${exp_name}",COLLECT_SEED="${seed}" \
    "${TRAIN_SCRIPT}"
done
