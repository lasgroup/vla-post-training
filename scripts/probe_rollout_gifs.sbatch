#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --job-name=rollout_gif
#SBATCH --gres=gpu:1
#SBATCH --constraint="VRAM_80GB|VRAM_96GB|VRAM_141GB"
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=04:00:00
#SBATCH --output=/home/pchellap/logs/%x_%j.out
#SBATCH --error=/home/pchellap/logs/%x_%j.out
# ---------------------------------------------------------------------------
# Qualitative rollout probe: GIFs from mt4_mt2_b128_ep2_s0's final checkpoint
# (config pi05_libero_online_ogpo_ref) on its own two tasks, episode length
# doubled exactly as that run trained (EP_MULT=2).
#
# READ-ONLY wrt the checkpoint (--resume, no agent.update()/save -- see
# scripts/probe_rollout_gifs.py). Single GPU / FSDP=1 for scheduling
# convenience even though the run trained on FSDP=2 / 2 GPUs -- orbax
# restores by the checkpoint's global array shape and the CURRENT process's
# target sharding, not by the saving-time mesh, so this changes only how the
# weights are sharded across devices, not which weights are loaded. See
# docs/changes/2026-09-05-rollout-gif-probe/BLAST-RADIUS.md.
#
# --mem=300G matches the training run's own allocation, not this probe's own
# footprint: OGPOAgentLearner(cfg) unconditionally restores the full online
# replay buffer + success buffer on --resume (filtered_sft_learner.py,
# ogpo_learner.py._restore_extra_resume_state) regardless of whether this
# read-only probe ever touches them, and by the run's final step those hold
# substantially more than the 83k/48k rows measured at step 40k (see
# [[mt2-batch128-run]]).
#
# Submit:  sbatch scripts/probe_rollout_gifs.sbatch
# Override episode count / seed / gif stride via --export=ALL,ROLLOUT_EPISODES=...
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_DIR=/home/pchellap/Projects/OGPO-VLA/vla-post-training
cd "$PROJECT_DIR"

OUT_DIR="/home/pchellap/logs/rollout_gif_${SLURM_JOB_ID:-manual}"
mkdir -p "$OUT_DIR"

echo "[rollout] node=$(hostname) job=${SLURM_JOB_ID:-?} out=$OUT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

ARM="${ARM:-mt2_b128_ep2}"
SEED="${SEED:-0}"
EXP_NAME="mt4_${ARM}_s${SEED}"

# timeout: a wedged /data mount otherwise blocks ls in D state for the whole
# walltime with the GPU idle (same failure mode as job 10193108 on Tier C).
REAL_BASE=/data/group_data/maxlab/common_datasets/pchellap/vla-post-training/checkpoints/ogpo_multitask_4task
SRC="$REAL_BASE/pi05_libero_online_ogpo_ref/$EXP_NAME"
timeout 120 ls -d "$SRC" || { echo "FATAL: run dir not visible or mount hung: $SRC"; exit 1; }

# The recipe owns the env preamble AND the config flag block; ENTRY only swaps
# the python entry point, so the knobs that matter for restoring and driving
# this checkpoint -- CONFIG_NAME, ARM, SEED, TASKS, EP_MULT, FSDP -- render
# identically to the run that produced it (a few inert-at-eval ones, e.g.
# num_train_steps/max_runtime, don't -- this contract is about the flag BLOCK
# not being cloned a second time to drift, not byte-identical CLI text; see
# docs/changes/2026-09-05-rollout-gif-probe/VERIFICATION.md finding 3).
# --resume restores it instead of overwriting.
export ENTRY=scripts/probe_rollout_gifs.py
export CKPT_MODE_FLAG=--resume   # explicit: the recipe already defaults to --resume
                                 # (FRESH=1 would wipe), so this is belt-and-braces
                                 # for a read-only probe. NOT both flags -- exclusive.
export GPU=0
export ARM SEED
export TASKS="${TASKS:-libero_90_38 libero_90_82}"
export EP_MULT="${EP_MULT:-2}"
export BATCH="${BATCH:-128}"   # matches the training run's batch_size; inert at eval

export ROLLOUT_OUT_DIR="${ROLLOUT_OUT_DIR:-$OUT_DIR}"
export ROLLOUT_EPISODES="${ROLLOUT_EPISODES:-8}"
export ROLLOUT_SEED="${ROLLOUT_SEED:-0}"
export ROLLOUT_GIF_STRIDE="${ROLLOUT_GIF_STRIDE:-2}"

exec bash scripts/ogpo_multitask_4task_ref.sh
