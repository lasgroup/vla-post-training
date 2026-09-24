#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --job-name=q_spread
#SBATCH --gres=gpu:2
#SBATCH --constraint="VRAM_80GB|VRAM_96GB|VRAM_141GB"
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=04:00:00
#SBATCH --output=/home/pchellap/logs/%x_%j.out
#SBATCH --error=/home/pchellap/logs/%x_%j.out
# ---------------------------------------------------------------------------
# Candidate Q-spread probe: N rollouts of TASKS from a trained checkpoint; at
# every policy query the M=BON_N sampled candidates are scored by the critic
# and the variance of the M reduced Q-scores is recorded. QSPREAD_BON=1
# executes the argmax-Q candidate (production best-of-N), QSPREAD_BON=0
# executes candidate 0 (plain-policy control). One plot per rollout, one mean
# trace per task, one JSON with the raw scores. scripts/probe_candidate_q_spread.py.
#
# READ-ONLY wrt the checkpoint (--resume, no agent.update()/save).
#
# TWO GPUs / FSDP=2, matching the run that wrote the checkpoint. This is NOT
# optional: openpi's restore_state passes the jax.eval_shape train state with
# NO target shardings, so orbax falls back to the SAVED sharding and maps its
# device ids onto jax.devices() by id (orbax .../metadata/sharding.py:119-131).
# An FSDP=2 checkpoint on a 1-GPU job therefore dies in restore with
# "sharding passed to deserialization should be specified ... Got None"
# (rollout_gif job 10324589, 2026-09-05). Device KIND is not compared, only
# the id set, so any two GPUs work. Set FSDP to the checkpoint's fsdp_devices.
#
# --mem=300G: OGPOAgentLearner(cfg) unconditionally restores the full replay
# buffer + success buffer on --resume regardless of whether a read-only probe
# touches them. MEASURED (job 10349712, mt4_mt2_b128_ep2_s0 at step 100k,
# 2x RTX PRO 6000): MaxRSS 215 GB, so 300G is not slack. Wall clock: ~24 min
# restore + startup, then ~28 min per 8-episode wave at EP_MULT=2 (the wave
# runs to its LONGEST episode, 160 chunks, at the full 8x8 candidate batch).
# Budget QSPREAD_EPISODES against --time accordingly: ~7 waves fit in 4 h.
#
# Checkpoint and task are env vars, overridable at submit time:
#   sbatch scripts/probe_candidate_q_spread.sbatch                     # BoN on
#   sbatch --export=ALL,QSPREAD_BON=0 scripts/probe_candidate_q_spread.sbatch
#   sbatch --export=ALL,ARM=<arm>,SEED=<s>,TASKS=libero_90_31,BON_N=16,QSPREAD_EPISODES=16 ...
# ARM/SEED/CONFIG_NAME name the checkpoint directory; the step is whatever that
# directory holds (openpi keeps one). TASKS is the task(s) to roll out; for a
# PER_TASK_CRITIC=1 checkpoint it must equal the training set (slot count).
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_DIR=/home/pchellap/Projects/OGPO-VLA/vla-post-training
cd "$PROJECT_DIR"

export QSPREAD_BON="${QSPREAD_BON:-1}"
OUT_DIR="/home/pchellap/logs/q_spread_${SLURM_JOB_ID:-manual}_bon${QSPREAD_BON}"
mkdir -p "$OUT_DIR"

echo "[qspread] node=$(hostname) job=${SLURM_JOB_ID:-?} out=$OUT_DIR bon=$QSPREAD_BON"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

ARM="${ARM:-mt2_b128_ep2}"
SEED="${SEED:-0}"
CONFIG_NAME="${CONFIG_NAME:-pi05_libero_online_ogpo_ref}"
EXP_NAME="mt4_${ARM}_s${SEED}"

# timeout: a wedged /data mount otherwise blocks ls in D state for the whole
# walltime with the GPU idle (same failure mode as job 10193108 on Tier C).
REAL_BASE=/data/group_data/maxlab/common_datasets/pchellap/vla-post-training/checkpoints/ogpo_multitask_4task
SRC="$REAL_BASE/$CONFIG_NAME/$EXP_NAME"
timeout 120 ls -d "$SRC" || { echo "FATAL: run dir not visible or mount hung: $SRC"; exit 1; }

# The recipe owns the env preamble AND the config flag block; ENTRY only swaps
# the python entry point, so the knobs that matter for restoring and driving
# this checkpoint -- CONFIG_NAME, ARM, SEED, TASKS, EP_MULT, FSDP, BON_N --
# render from the same flag block as the run that produced it (a few
# inert-at-eval ones don't, see docs/changes/2026-09-05-rollout-gif-probe/
# VERIFICATION.md finding 3). --resume restores it instead of overwriting.
export ENTRY=scripts/probe_candidate_q_spread.py
export CKPT_MODE_FLAG=--resume   # explicit: the recipe already defaults to --resume
                                 # (FRESH=1 would wipe); belt-and-braces for a
                                 # read-only probe. NOT both flags -- exclusive.
# Take the allocation SLURM exposes (relative ids under cgroups) so both
# GPUs of --gres=gpu:2 are visible to JAX, as ogpo_multitask_4task_ref_maxlab.sbatch does.
export GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0,1}}"
export FSDP="${FSDP:-2}"     # = the checkpoint's fsdp_devices (see header)
export ARM SEED CONFIG_NAME
export TASKS="${TASKS:-libero_90_38}"
export EP_MULT="${EP_MULT:-2}"
export BATCH="${BATCH:-128}"   # matches the training run's batch_size; inert at eval
export BON_N="${BON_N:-8}"     # per-call candidates (--rl.n_samples); M = QSPREAD_PASSES x BON_N
export QSPREAD_PASSES="${QSPREAD_PASSES:-1}"   # >1 splits M over calls at constant peak memory

export QSPREAD_OUT_DIR="${QSPREAD_OUT_DIR:-$OUT_DIR}"
export QSPREAD_EPISODES="${QSPREAD_EPISODES:-8}"
export QSPREAD_SEED="${QSPREAD_SEED:-0}"

exec bash scripts/ogpo_multitask_4task_ref.sh
