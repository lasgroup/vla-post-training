#!/bin/bash
# ────────────────────────────────────────────────────────────────────
# Flow-GRPO training — called by flow_grpo.sub
# SEED is passed via HTCondor environment.
# ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run.sh"

SEED="${SEED:-0}"

python3 scripts/flow_grpo_agent/exp.py pi05_libero_online_flow_grpo_sft \
    --seed "${SEED}" \
    --collect.seed "${SEED}" \
    --overwrite \
    --project_name vla-post-training \
    --exp_name "flow_grpo_seed${SEED}" \
    --checkpoint_base_dir "${CHECKPOINT_BASE_DIR}" \
    --log_interval 25 \
    --batch_size 32
