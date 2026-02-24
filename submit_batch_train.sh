#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/submit_train.sh"
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <wandb_project> [time_limit]" >&2
  echo "Example: $0 my_project 05:30:00" >&2
  exit 1
fi

WANDB_PROJECT="$1"
LOG_INTERVAL="${LOG_INTERVAL:-25}"
NUM_SEEDS="${NUM_SEEDS:-5}"
TRAIN_CONFIG="${TRAIN_CONFIG:-pi05_libero_online_aw_sft}"
# Takes the 2nd argument if provided, otherwise defaults to 03:30:00
JOB_TIME="${2:-03:30:00}"

if [[ -n "${SEEDS:-}" ]]; then
  read -r -a seed_values <<<"${SEEDS}"
else
  seed_values=()
  for ((seed = 0; seed < NUM_SEEDS; seed++)); do
    seed_values+=("${seed}")
  done
fi

if ((${#seed_values[@]} != NUM_SEEDS)); then
  echo "Expected ${NUM_SEEDS} seeds, got ${#seed_values[@]} (SEEDS='${SEEDS:-}')." >&2
  exit 1
fi

total_jobs=0
for seed in "${seed_values[@]}"; do
  exp_name="${WANDB_PROJECT}_seed${seed}"
  echo "Submitting ${exp_name} (project=${WANDB_PROJECT}, seed=${seed}, config=${TRAIN_CONFIG}, log_interval=${LOG_INTERVAL})"
  sbatch \
    --job-name="${exp_name}" \
    --time="${JOB_TIME}" \
    --export=ALL,EXP_NAME="${exp_name}",LOG_INTERVAL="${LOG_INTERVAL}",TRAIN_SEED="${seed}",WANDB_PROJECT="${WANDB_PROJECT}",TRAIN_CONFIG="${TRAIN_CONFIG}" \
    "${TRAIN_SCRIPT}"
  ((total_jobs += 1))
done

echo "Submitted ${total_jobs} jobs to W&B project '${WANDB_PROJECT}'."
