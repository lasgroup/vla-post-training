#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --job-name=tierc_probe
#SBATCH --gres=gpu:1
#SBATCH --constraint="VRAM_80GB|VRAM_96GB|VRAM_141GB"
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/home/pchellap/logs/%x_%j.out
#SBATCH --error=/home/pchellap/logs/%x_%j.out
# ---------------------------------------------------------------------------
# Tier-C probe: counterfactual rollouts. Restores the deployed policy+critic,
# drives 8 envs to the SAME state by deterministic replay, executes the 8
# best-of-N candidates one per env, and continues under pi to termination --
# so the critic's ranking can be checked against realised return.
#
# READ-ONLY with respect to the training run: it resumes the checkpoint but
# never writes one back (no train step is taken, save_interval is never hit).
# It does run the simulator, unlike Tier A/B.
#
# PHASE is the knob. Run `calibrate` FIRST -- it measures the continuation noise
# floor and prints how many probe states the real measurement needs. `full`
# without that number is a guess about how long this has to run.
#
# ON `preempt`: acceptable because the probe flushes its JSON after every probe
# state, so a kill costs the state in flight. The TRAINING recipes must not run
# here (save_interval=100000 => a kill at hour 10 loses the run).
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_DIR=/home/pchellap/Projects/OGPO-VLA/vla-post-training
cd "$PROJECT_DIR"

PHASE="${PHASE:-calibrate}"
ARM="${ARM:-ref}"
SEED="${SEED:-0}"
OUT=/home/pchellap/logs/tierc_${PHASE}_${SLURM_JOB_ID:-manual}.json

echo "[tierC] node=$(hostname) job=${SLURM_JOB_ID:-?} phase=$PHASE arm=$ARM seed=$SEED"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# NO CHECKPOINT SHADOW -- verified unnecessary, and it was expensive.
# The concern was that constructing the learner mutates the run directory two
# ways. Both are settled:
#   1. initialize_checkpoint_dir calls checkpoint_dir.rmtree() when
#      overwrite=True (openpi checkpoints.py:26-29), and overwrite = not resume.
#      BLOCKED IN CODE: the probe raises before OGPOAgentLearner(cfg) if
#      cfg.resume is False, so that branch is unreachable.
#   2. orbax max_to_keep garbage collection. It does NOT run at construction:
#      _get_old_steps_to_remove() is called only from save()
#      (orbax checkpoint_manager.py:1421, inside def save at :1207), and
#      cleanup_tmp_directories defaults False (:337) and is never set here.
#      The probe never saves.
# A `cp -al` shadow was tried and ran 1h50m without finishing (job 10193599):
# hardlinks are metadata ops, and orbax writes thousands of small ocdbt chunks
# onto a networked filesystem. Not free, and not needed.
REAL_BASE=/data/group_data/maxlab/common_datasets/pchellap/vla-post-training/checkpoints/ogpo_multitask_4task
EXP_NAME="mt4_${ARM}_s${SEED}"
SRC="$REAL_BASE/pi05_libero_online_ogpo_ref/$EXP_NAME"
# timeout: a wedged /data mount otherwise blocks ls in D state for the whole
# walltime with the GPU idle (job 10193108, 75 min).
timeout 120 ls -d "$SRC" || { echo "FATAL: run dir not visible or mount hung: $SRC"; exit 1; }

# The recipe owns the env preamble AND the config flag block; ENTRY only swaps
# the python entry point, so the probe sees byte-identical config to the run
# that produced the checkpoint. --resume restores it instead of overwriting.
export ENTRY=scripts/probe_counterfactual_rollouts.py
export CKPT_MODE_FLAG=--resume   # explicit: the recipe already defaults to --resume
                                 # (FRESH=1 would wipe), so this is belt-and-braces
                                 # for a read-only probe. NOT both flags -- exclusive.
export GPU=0
export ARM SEED

exec bash scripts/ogpo_multitask_4task_ref.sh \
  --probe.phase "$PHASE" \
  --probe.out "$OUT" \
  --probe.calib-states "${CALIB_STATES:-10}" \
  --probe.trunks "${TRUNKS:-8}" \
  --probe.per-episode "${PER_EPISODE:-3}" \
  --probe.max-probes "${MAX_PROBES:-48}" \
  --probe.repeats "${REPEATS:-1}" \
  --probe.seed "${PROBE_SEED:-12345}"
