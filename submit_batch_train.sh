#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/submit_train.sh"
WANDB_PROJECT="${1:-openpi}"
NUM_SEEDS="${NUM_SEEDS:-5}"
COLLECT_INTERVAL="${COLLECT_INTERVAL:-200}"
CRITIC_UPDATE_INTERVAL="${CRITIC_UPDATE_INTERVAL:-1}"
POLICY_UPDATE_INTERVAL="${POLICY_UPDATE_INTERVAL:-4}"
CRITIC_TRAINING_START_STEP="${CRITIC_TRAINING_START_STEP:-1}"
POLICY_TRAINING_START_STEPS="${POLICY_TRAINING_START_STEPS:-400 1000 2000}"
EXP_PREFIX="${EXP_PREFIX:-awr_grid}"

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
for policy_start_step in ${POLICY_TRAINING_START_STEPS}; do
  for seed in "${seed_values[@]}"; do
    exp_name="${EXP_PREFIX}_pstart${policy_start_step}_seed${seed}"
    echo "Submitting ${exp_name} (project=${WANDB_PROJECT}, seed=${seed}, policy_start=${policy_start_step})"
    sbatch \
      --job-name="${exp_name}" \
      --export=ALL,EXP_NAME="${exp_name}",COLLECT_SEED="${seed}",WANDB_PROJECT="${WANDB_PROJECT}",COLLECT_INTERVAL="${COLLECT_INTERVAL}",CRITIC_UPDATE_INTERVAL="${CRITIC_UPDATE_INTERVAL}",POLICY_UPDATE_INTERVAL="${POLICY_UPDATE_INTERVAL}",CRITIC_TRAINING_START_STEP="${CRITIC_TRAINING_START_STEP}",POLICY_TRAINING_START_STEP="${policy_start_step}" \
      "${TRAIN_SCRIPT}"
    ((total_jobs += 1))
  done
done

echo "Submitted ${total_jobs} jobs to W&B project '${WANDB_PROJECT}'."
