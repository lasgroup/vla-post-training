#!/bin/bash
# ---------------------------------------------------------------------------
# Two-stage pipeline for arms (iii)/(iv): succBC-finetuned-backbone critic.
#
# Stage A (steps 0-20k): BC-ONLY warmstart with the PaliGemma LLM backbone
#   UNFROZEN (SigLIP stays frozen). PG fully muted (pg_start_step > horizon).
#   Pure SFT on the success buffer — the backbone adapts to the task.
#   Saves a checkpoint at 20k, then exits (num_train_steps 20000).
#
# Stage B (steps 0-80k fresh run): the normal FROZEN recipe, but weights
#   initialized from stage A's checkpoint. The policy's flow head continues
#   from its BC state; the backbone is frozen at its BC-adapted weights; the
#   critic's prefix embeddings (store_prefix_rep) are computed by THIS
#   backbone — i.e. "critic takes the PaliGemma after succBC fine-tuning".
#   PPO ramps in from step 0 of stage B (the policy is already warm).
#
# Env vars: ARM (suffix), GPU, SEED (default 0),
#           CANC=1 to add normalizer+clip in stage B (arm iv).
# ---------------------------------------------------------------------------
set -euo pipefail
: "${ARM:?set ARM}"; : "${GPU:?set GPU}"
SEED="${SEED:-0}"

export PATH="$HOME/.local/bin:$PATH"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
STORE_ROOT="$PROJECT_DIR/run_store"
CKPT_BASE="$STORE_ROOT/checkpoints/stability_study"
STAGE_A_EXP="stab_${ARM}_bcA"
STAGE_B_EXP="stab_${ARM}"

# --- Stage A: unfrozen-backbone BC-only warmstart, 20k steps -------------
if [ ! -d "$CKPT_BASE/pi05_libero_online_ogpo_sft_unfrozen_backbone/$STAGE_A_EXP/20000" ]; then
  echo "[pipeline] stage A: unfrozen BC warmstart -> 20k"
  env GPU="$GPU" ARM="${ARM}_bcA" SEED="$SEED" \
      ENTRY=scripts/exp_ogpo_unfrozen_backbone.py \
      CONFIG_NAME=pi05_libero_online_ogpo_sft_unfrozen_backbone \
      N_STEPS=20000 SAVE_INT=20000 PG_START=99999 BURST=1000 \
      XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
      bash "$PROJECT_DIR/scripts/stability_study.sh"
  echo "[pipeline] stage A done"
fi

# --- Stage B: frozen recipe from the BC-adapted checkpoint ----------------
PARAMS="$CKPT_BASE/pi05_libero_online_ogpo_sft_unfrozen_backbone/$STAGE_A_EXP/20000/params"
[ -d "$PARAMS" ] || { echo "[pipeline] FATAL: stage A params missing at $PARAMS"; exit 1; }
echo "[pipeline] stage B: frozen PPO from BC-adapted backbone"
EXTRA=""
[ "${CANC:-0}" = "1" ] && EXTRA="NORM=1 CLIP_SYM=4.0"
env GPU="$GPU" ARM="$ARM" SEED="$SEED" \
    WEIGHT_LOADER="$PARAMS" \
    N_STEPS=80000 PG_START=0 PG_RAMP=5000 BURST=1000 CONS=1 $EXTRA \
    bash "$PROJECT_DIR/scripts/stability_study.sh"
