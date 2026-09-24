#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --job-name=vnext_probe
#SBATCH --gres=gpu:1
#SBATCH --constraint="VRAM_80GB|VRAM_96GB|VRAM_141GB"
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/home/pchellap/logs/%x_%j.out
#SBATCH --error=/home/pchellap/logs/%x_%j.out
# ---------------------------------------------------------------------------
# Does V separate the next-states the Best-of-N candidates lead to?
#
# With td_weight=0.95 the critic learns Q from r + gamma*V(s'), and in LIBERO
# s' is near-deterministic given (s,a) -- so that target is the low-noise
# channel through which any action signal reaches Q. If V barely separates the
# N next-states, Q cannot rank the N candidates however the Q head is trained.
#
# READ-ONLY: resumes the checkpoint, takes no train step, never saves.
# ON `preempt`: the probe flushes JSON after every state, so a kill costs the
# state in flight. TRAINING recipes must not run here.
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_DIR=/home/pchellap/Projects/OGPO-VLA/vla-post-training
cd "$PROJECT_DIR"

ARM="${ARM:-ref}"
SEED="${SEED:-0}"
OUT=/home/pchellap/logs/vnext_${SLURM_JOB_ID:-manual}.json

echo "[vnext] node=$(hostname) job=${SLURM_JOB_ID:-?} arm=$ARM seed=$SEED"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

REAL_BASE=/data/group_data/maxlab/common_datasets/pchellap/vla-post-training/checkpoints/ogpo_multitask_4task
EXP_NAME="mt4_${ARM}_s${SEED}"
SRC="$REAL_BASE/pi05_libero_online_ogpo_ref/$EXP_NAME"
# timeout: a wedged /data mount otherwise blocks ls in D state for the whole
# walltime with the GPU idle (job 10193108, 75 min).
timeout 120 ls -d "$SRC" || { echo "FATAL: run dir not visible or mount hung: $SRC"; exit 1; }

# The recipe owns the env preamble AND the config flag block; ENTRY only swaps
# the python entry point, so the probe sees byte-identical config to the run
# that produced the checkpoint. --resume restores instead of overwriting.
export ENTRY=scripts/probe_value_next_spread.py
export CKPT_MODE_FLAG=--resume   # explicit: the recipe already defaults to --resume
                                 # (FRESH=1 would wipe), so this is belt-and-braces
                                 # for a read-only probe. NOT both flags -- exclusive.
export GPU=0
export ARM SEED

exec bash scripts/ogpo_multitask_4task_ref.sh \
  --probe.out "$OUT" \
  --probe.states "${STATES:-64}" \
  --probe.max-walk-in "${MAX_WALK_IN:-12}" \
  --probe.seed "${PROBE_SEED:-777}"
