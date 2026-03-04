#!/bin/bash
# ────────────────────────────────────────────────────────────────────
# MPO (Maximum a-Posteriori) Weighted SFT training — called by mpo.sub
# SEED is passed via HTCondor environment.
# ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run.sh"

SEED="${SEED:-0}"

python3 scripts/mpo_agent/exp.py pi05_libero_online_mpo_sft \
    --seed "${SEED}" \
    --collect.seed "${SEED}" \
    --overwrite \
    --project_name vla-post-training \
    --exp_name "mpo_seed${SEED}" \
    --checkpoint_base_dir "${CHECKPOINT_BASE_DIR}" \
    --log_interval 25 \
    --batch_size 32
