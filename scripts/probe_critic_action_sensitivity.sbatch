#!/bin/bash
#SBATCH --partition=general
#SBATCH --job-name=critic_probe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/pchellap/logs/%x_%j.out
#SBATCH --error=/home/pchellap/logs/%x_%j.out
# ---------------------------------------------------------------------------
# Tier-A critic action-sensitivity probe. READ-ONLY: loads rl_state/100000 and
# the replay shard saved beside it, trains nothing, writes only a JSON.
#
# Runs on `general` (not maxlab) because it needs one GPU for a few minutes and
# should not sit behind the maxlab queue. /data/group_data/maxlab is not mounted
# on the login node, which is the only reason this is a job at all.
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_DIR=/home/pchellap/Projects/OGPO-VLA/vla-post-training
cd "$PROJECT_DIR"

CKPT_ROOT=/data/group_data/maxlab/common_datasets/pchellap/vla-post-training/checkpoints/ogpo_multitask_4task/pi05_libero_online_ogpo_ref
OUT=/home/pchellap/logs/critic_probe_${SLURM_JOB_ID:-manual}.json

echo "[probe] node=$(hostname) job=${SLURM_JOB_ID:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
ls -d "$CKPT_ROOT" || { echo "FATAL: checkpoint root not visible from this node"; exit 1; }

exec uv run python scripts/probe_critic_action_sensitivity.py \
  --ckpt-root "$CKPT_ROOT" \
  --runs mt4_ref_s0 mt4_ref_nc_s0 mt4_ref_nopg_s0 \
  --step 100000 \
  --num-states 1024 \
  --num-swap 256 \
  --out "$OUT"
