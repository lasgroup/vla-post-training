#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --job-name=cand_probe
#SBATCH --gres=gpu:1
#SBATCH --constraint="VRAM_80GB|VRAM_96GB|VRAM_141GB"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/home/pchellap/logs/%x_%j.out
#SBATCH --error=/home/pchellap/logs/%x_%j.out
# ---------------------------------------------------------------------------
# Tier-B probe: replay pi0.5 on saved buffer states, sample 8 candidate action
# chunks per state exactly as collection does, and measure how far apart they
# are -- both in action space (the "equivalent eps" on the Tier-A ladder) and in
# Q. READ-ONLY: no simulator, no training, writes only a JSON.
#
# ON `preempt`: fine HERE, unlike the training recipes. Those must not run on
# preempt because save_interval=100000 means a kill at hour 10 loses the whole
# run. This probe is ~30 min, read-only, and flushes its JSON after every arm,
# so the worst a preemption costs is the arm in flight. `general` has no 80GB
# cards available to this account, which is why we are here.
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_DIR=/home/pchellap/Projects/OGPO-VLA/vla-post-training
cd "$PROJECT_DIR"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/openpi/packages/openpi-client/src:$PROJECT_DIR/openpi/src:$PROJECT_DIR/openpi/packages/openpi-client:$PROJECT_DIR/molmospaces"

CKPT_ROOT=/data/group_data/maxlab/common_datasets/pchellap/vla-post-training/checkpoints/ogpo_multitask_4task/pi05_libero_online_ogpo_ref
OUT=/home/pchellap/logs/cand_probe_${SLURM_JOB_ID:-manual}.json

echo "[tierB] node=$(hostname) job=${SLURM_JOB_ID:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
timeout 120 ls -d "$CKPT_ROOT" || { echo "FATAL: checkpoint root not visible or mount hung"; exit 1; }

exec uv run python scripts/probe_policy_candidate_spread.py \
  --ckpt-root "$CKPT_ROOT" \
  --runs mt4_ref_s0 mt4_ref_nopg_s0 mt4_ref_nc_s0 \
  --step 100000 \
  --num-states 192 \
  --group 8 \
  --chunk 4 \
  --out "$OUT"
