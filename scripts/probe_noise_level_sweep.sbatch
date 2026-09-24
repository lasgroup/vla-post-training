#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --job-name=noise_sweep
#SBATCH --gres=gpu:1
#SBATCH --constraint="VRAM_80GB|VRAM_96GB|VRAM_141GB"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/pchellap/logs/%x_%j.out
#SBATCH --error=/home/pchellap/logs/%x_%j.out
# ---------------------------------------------------------------------------
# Noise-level sweep: re-run the Tier-B battery across a ladder of sampling
# noise_level values, to see whether a higher collection noise lifts the Q
# spread Best-of-N ranks by above the critic's own error bar -- and what it
# costs in displacement off the policy's mode.
# READ-ONLY: no simulator, no training, writes only a JSON.
#
# ON `preempt`: fine HERE, unlike the training recipes (save_interval=100000
# means a kill at hour 10 loses everything). This probe flushes its JSON after
# every noise level, so a preemption costs at most the level in flight.
# `general` has no 80GB cards available to this account.
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_DIR=/home/pchellap/Projects/OGPO-VLA/vla-post-training
cd "$PROJECT_DIR"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/openpi/packages/openpi-client/src:$PROJECT_DIR/openpi/src:$PROJECT_DIR/openpi/packages/openpi-client:$PROJECT_DIR/molmospaces"

CKPT_ROOT=/data/group_data/maxlab/common_datasets/pchellap/vla-post-training/checkpoints/ogpo_multitask_4task/pi05_libero_online_ogpo_ref
OUT=/home/pchellap/logs/noise_sweep_${SLURM_JOB_ID:-manual}.json

echo "[sweep] node=$(hostname) job=${SLURM_JOB_ID:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# timeout: a wedged /data mount otherwise parks this in uninterruptible I/O
# for the whole walltime with the GPU idle (job 10193108, 75 min on ls).
timeout 120 ls -d "$CKPT_ROOT" || { echo "FATAL: checkpoint root not visible or mount hung"; exit 1; }

# Ladder brackets both our recipe value (0.02) and the reference-equivalent
# (0.0697; OGPO_public tapered sigma_base=0.05 at 10 flow steps).
exec uv run python scripts/probe_noise_level_sweep.py \
  --ckpt-root "$CKPT_ROOT" \
  --runs mt4_ref_s0 mt4_ref_nopg_s0 \
  --step 100000 \
  --num-states 128 \
  --group 8 \
  --chunk 4 \
  --noise-levels 0.0 0.01 0.02 0.05 0.07 0.1 0.2 0.3 0.5 \
  --out "$OUT"
