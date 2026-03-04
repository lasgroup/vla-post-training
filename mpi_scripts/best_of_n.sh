#!/bin/bash
# ────────────────────────────────────────────────────────────────────
# Best-of-N value learning training — called by best_of_n.sub
# SEED is passed via HTCondor environment.
# ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run.sh"

SEED="${SEED:-0}"

python3 scripts/best_of_n_agent/exp.py pi05_libero_online_best_of_n \
    --seed "${SEED}" \
    --collect.seed "${SEED}" \
    --overwrite \
    --project_name vla-post-training \
    --exp_name "best_of_n_seed${SEED}" \
    --checkpoint_base_dir "${CHECKPOINT_BASE_DIR}" \
    --log_interval 25 \
    --batch_size 32
