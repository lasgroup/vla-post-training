#!/bin/bash
#SBATCH --partition=maxlab
#SBATCH --qos=maxlab_qos
#SBATCH --job-name=stab_ref
#SBATCH --gres=gpu:1
#SBATCH --constraint=VRAM_96GB
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=48:00:00
#SBATCH --requeue
# A requeue reuses %j, and sbatch's default open mode truncates, so without this
# each attempt erases the log of the attempt it is recovering from
# (scripts/launcher.py:219-220 pairs the two flags for the same reason).
#SBATCH --open-mode=append
#SBATCH --output=/home/pchellap/logs/%x_%j.out
#SBATCH --error=/home/pchellap/logs/%x_%j.out
# ---------------------------------------------------------------------------
# maxlab wrapper for scripts/stability_study_ref.sh (single-task,
# reference-aligned stack). Mirrors ogpo_multitask_4task_ref_maxlab.sbatch's
# resources and exit-42 requeue handling; same VRAM_96GB / 16 cpu / 150G
# request as the multitask ref smoke (job 10178533: peak GPU 75.8/95.6 GiB,
# peak host 93.7/150 GB) -- BUT that measurement was taken at
# BATCH=32 x group_num_samples=8 = 256 chains. It does NOT justify this
# request at any other BATCH: chain count scales linearly with BATCH, and
# 256 chains already used 79% of a 96GB card. Not independently remeasured
# for single-task at any BATCH, so treat 256-chain headroom as an estimate,
# and anything above 256 chains as UNMEASURED -- OOM risk, not a sized request.
#
# NOTE ON PARTITION CHOICE: do NOT move this to `preempt` for a full run --
# preemption SIGKILLs mid-step, so exp.py never reaches the exit-42 handoff
# below and the run loses everything since the last 10k-step collect/eval
# boundary. The resubmit itself is non-destructive (--resume by default), but
# maxlab's queue wait is still the cheaper risk (same reasoning as
# ogpo_multitask_4task_ref_maxlab.sbatch).
#
# Required: ARM, TASK (fail fast in stability_study.sh / stability_study_ref.sh
# if unset -- ARM must encode the task per stability_study.sh's own contract,
# e.g. ARM=REF_libero_90_31 TASK=libero_90_31).
#
# Submit:  sbatch --export=ALL,ARM=REF_libero_90_31,TASK=libero_90_31 \
#              scripts/stability_study_ref_maxlab.sbatch
# All env knobs of scripts/stability_study_ref.sh / scripts/stability_study.sh
# pass through via --export.
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_DIR=/home/pchellap/Projects/OGPO-VLA/vla-post-training
cd "$PROJECT_DIR"

: "${ARM:?set ARM via --export (e.g. ARM=REF_libero_90_31)}"
: "${TASK:?set TASK via --export (e.g. TASK=libero_90_31)}"

# stability_study.sh's own N_STEPS default is 100000, so range(start, 100000)
# stops at step 99999 and the eval_interval=10000 condition never fires at
# 100k (nine evals, not ten) -- same reasoning as
# ogpo_multitask_4task_ref_maxlab.sbatch's NUM_STEPS=100001.
export N_STEPS="${N_STEPS:-100001}"

# SLURM renumbers the allocated GPUs inside the cgroup; take the whole list so
# a --gres=gpu:2 allocation is actually visible to JAX.
export GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"

# Checkpoints (rl_state + runtime_state/replay+success shards) default to
# GROUP STORAGE, not $HOME -- home is a 100G-quota NFS mount and a 100k-step
# run's replay buffer + optimizer state does not fit two arms of it
# comfortably (2026-09-10: home had 37G free after one arm reached step
# 10000). Same convention ogpo_multitask_4task.sh uses for its CKPT_BASE_DIR
# (stability_study.sh's own CKPT_BASE_DIR override, added for the
# candidate-q-spread stab probe, is what makes this a one-line default rather
# than a script change). Caches (openpi weights, HF, torch, wandb) stay under
# $STORE_ROOT/cache on $HOME -- they're a bounded, one-time, shared-across-runs
# cost, not what grows unbounded over a run, so moving them doesn't address
# the space problem and isn't done here.
export CKPT_BASE_DIR="${CKPT_BASE_DIR:-/data/group_data/maxlab/common_datasets/${USER:-pchellap}/vla-post-training/checkpoints/stability_study}"

echo "[stab-ref] node=$(hostname) job=${SLURM_JOB_ID:-?} arm=$ARM task=$TASK"
echo "[stab-ref] gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo n/a)"

# exp.py exits 42 when it has saved a resumable epoch and wants the wall clock
# back (scripts/exp.py:5, :191-193); requeue then continues from that checkpoint.
# `exec` cannot be used here -- the exit status has to come back to this shell.
# MAX_RUNTIME needs no override: the recipe's 47 h default is under this
# wrapper's 48 h wall.
child_status=0
bash scripts/stability_study_ref.sh || child_status=$?

if [[ "$child_status" -eq 42 ]]; then
  echo "[$(date --iso-8601=seconds)] Job ${SLURM_JOB_ID} requested requeue." >&2
  if scontrol requeue "${SLURM_JOB_ID}"; then
    echo "[$(date --iso-8601=seconds)] Requeue submitted for job ${SLURM_JOB_ID}." >&2
    exit 0
  fi
  echo "[$(date --iso-8601=seconds)] Failed to requeue job ${SLURM_JOB_ID}." >&2
  exit "$child_status"
fi

exit "$child_status"
