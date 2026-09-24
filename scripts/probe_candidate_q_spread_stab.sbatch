#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --job-name=q_spread_stab
#SBATCH --gres=gpu:1
#SBATCH --constraint="VRAM_80GB|VRAM_96GB|VRAM_141GB"
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/home/pchellap/logs/%x_%j.out
#SBATCH --error=/home/pchellap/logs/%x_%j.out
# ---------------------------------------------------------------------------
# Candidate Q-spread probe (scripts/probe_candidate_q_spread.py) against a
# SINGLE-TASK STABILITY-STUDY checkpoint -- the frozen-backbone stab_<ARM>
# runs under the sibling vla_single_task store root (config
# pi05_libero_online_ogpo_sft, 2 Q heads reduced by min, collected with plain
# sampling, no success bonus, episode multiplier 1). The mt4/ref checkpoints
# use scripts/probe_candidate_q_spread.sbatch instead.
#
# READ-ONLY wrt the checkpoint (--resume, no agent.update()/save). One GPU /
# FSDP=1, which is what these runs trained with (stability_study.sh emits
# --fsdp_devices 1); an FSDP mismatch would fail in restore, see the mt4
# wrapper's header.
#
# ENTRY-swap through scripts/stability_study.sh. Only CKPT_BASE_DIR is
# redirected to the sibling root -- the way the campaign itself was launched;
# every cache (openpi assets, LIBERO config, wandb) stays in this repo's
# run_store. Redirecting STORE_ROOT wholesale (as stage3_rollouts.sbatch does,
# which only worked because its submitting shell exported the cache paths)
# moves the openpi asset cache to a root that has none, and the policy-twin
# load dies on a missing _METADATA (job 10352338). ARM reconstructs the exp
# name stab_${ARM}, TASK selects the task, the NCB arm's three flags are
# re-exported so the rendered config is the run's own (they only affect
# update(), inert here), and M (--rl.n_samples) rides on the recipe's
# trailing-argument passthrough (docs/changes/2026-09-07-stability-recipe-task-knob/).
#
# --mem=200G: the checkpoint's own training run (job 10210760) peaked at
# 138.6 GiB MaxRSS holding the same 100k-step buffer plus optimizer state;
# the stage-3 probe on this root (job 10274728) hit exactly its 128G cap, a
# censored number. 200G leaves ~44% over the training run.
#
# Read-only wrt the sibling root: only the checkpoint dir under it is read.
#
#   sbatch scripts/probe_candidate_q_spread_stab.sbatch                    # BoN on
#   QSPREAD_BON=0 sbatch scripts/probe_candidate_q_spread_stab.sbatch      # BoN off
#   ARM=NCB_libero_90_82 TASK=libero_90_82 sbatch scripts/probe_candidate_q_spread_stab.sbatch
#   QSPREAD_EPISODES=8 QSPREAD_PASSES=1 sbatch ...                          # 8 episodes, M=8
#
# MEASURED (job 10352521, stab_NCB_libero_90_38 @100k, 1x RTX PRO 6000):
# 24 min total for 16 episodes at M = 4 x 8 -- ~11 min startup incl. the
# 250k-buffer restore, then ~8 min per 8-episode wave (82 chunks); MaxRSS
# 108 GB. --time/--mem above are generous; ~50 waves would fit in 8 h. The
# JSON is flushed per wave, so a time-out keeps the finished waves.
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_DIR=/home/pchellap/Projects/OGPO-VLA/vla-post-training
cd "$PROJECT_DIR"

export QSPREAD_BON="${QSPREAD_BON:-1}"
OUT_DIR="/home/pchellap/logs/q_spread_stab_${SLURM_JOB_ID:-manual}_bon${QSPREAD_BON}"
mkdir -p "$OUT_DIR"

echo "[qspread-stab] node=$(hostname) job=${SLURM_JOB_ID:-?} out=$OUT_DIR bon=$QSPREAD_BON"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

export ARM="${ARM:-NCB_libero_90_38}"
export TASK="${TASK:-libero_90_38}"
export SEED="${SEED:-0}"
# The finished run's checkpoint root (only visible from compute nodes). NOT
# STORE_ROOT -- see the header.
export CKPT_BASE_DIR=/data/group_data/maxlab/common_datasets/pchellap/vla_single_task/checkpoints/stability_study
SRC="$CKPT_BASE_DIR/pi05_libero_online_ogpo_sft/stab_${ARM}"
# timeout guards a wedged /data mount (job 10193108 sat in D state for its walltime).
timeout 120 ls -d "$SRC" || { echo "FATAL: run dir not visible or mount hung: $SRC"; exit 1; }

export ENTRY=scripts/probe_candidate_q_spread.py
export CKPT_MODE_FLAG=--resume   # explicit, read-only; the recipe defaults to it anyway
export GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
# The NCB arm: normalizer + symmetric clip + critic burst (its log's banner:
# --rl.normalize_group_advantage --rl.adv_clip_sym 4.0 --rl.post_collection_critic_steps 1000).
export NORM="${NORM:-1}"
export CLIP_SYM="${CLIP_SYM:-4.0}"
export BURST="${BURST:-1000}"
# M = QSPREAD_PASSES x BON_N candidates per query. BON_N is rl.n_samples, the
# per-call batch (the observation is tiled BON_N times before the VLM prefix
# pass, so it sets the peak memory); QSPREAD_PASSES repeats the call and the
# probe takes the argmax over the union -- exact, since the candidates are iid
# draws and the critic's score is deterministic per (state, candidate).
# Default 4 x 8 = 32 at the collection-validated per-call batch of 8.
BON_N="${BON_N:-8}"
export QSPREAD_PASSES="${QSPREAD_PASSES:-4}"

export QSPREAD_OUT_DIR="${QSPREAD_OUT_DIR:-$OUT_DIR}"
export QSPREAD_EPISODES="${QSPREAD_EPISODES:-16}"   # 2 waves of collect.env_num=8
export QSPREAD_SEED="${QSPREAD_SEED:-0}"

exec bash scripts/stability_study.sh --rl.n_samples "$BON_N"
