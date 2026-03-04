#!/bin/bash
# ────────────────────────────────────────────────────────────────────
# Filtered SFT training — called by filtered_sft.sub
# SEED is passed via HTCondor environment.
# ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run.sh"

SEED="${SEED:-0}"

python3 scripts/filtered_sft_agent/exp.py pi05_libero_online_filtered_sft \
    --seed "${SEED}" \
    --collect.seed "${SEED}" \
    --overwrite \
    --project_name vla-post-training \
    --exp_name "filtered_sft_seed${SEED}" \
    --checkpoint_base_dir "${CHECKPOINT_BASE_DIR}" \
    --log_interval 25 \
    --batch_size 32
