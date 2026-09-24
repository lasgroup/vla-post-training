#!/bin/bash
#SBATCH --partition=maxlab
#SBATCH --qos=maxlab_qos
#SBATCH --job-name=mt4_ref
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
# maxlab wrapper for scripts/ogpo_multitask_4task_ref.sh (the reference-aligned
# stack). Mirrors ogpo_multitask_4task_maxlab.sbatch exactly -- same partition,
# qos, gres, constraint, cpus and memory as the ten completed arms -- so the new
# runs are resource-comparable with them.
#
# Measured headroom from the smoke (job 10178533, same RTX_PRO_6000 / 96GB):
#   peak GPU 75.8 / 95.6 GiB, peak host 93.7 / 150 GB.
#
# NOTE ON PARTITION CHOICE: do NOT move this to `preempt` for a full run.
# (Not for the reason this comment used to give: save_interval is never read by
# scripts/exp.py -- it is an openpi TrainConfig field. Our loop checkpoints at
# every collect/eval boundary, i.e. every 10k steps, exp.py:185-190.) The real
# exposure is that preemption SIGKILLs mid-step, so exp.py never reaches the
# exit-42 handoff below and the run loses everything since the last 10k
# boundary. The resubmit itself is no longer destructive -- the recipe defaults
# to --resume -- but maxlab's queue wait is still the cheaper risk.
#
# Submit:  sbatch --export=ALL,ARM=ref,NUM_STEPS=100001 \
#              scripts/ogpo_multitask_4task_ref_maxlab.sbatch
# Arms:    see scripts/submit_mt4_ref_arms.sh
# All env knobs of ogpo_multitask_4task_ref.sh pass through via --export.
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_DIR=/home/pchellap/Projects/OGPO-VLA/vla-post-training
cd "$PROJECT_DIR"

# SLURM renumbers the allocated GPUs inside the cgroup; take the whole list so
# a --gres=gpu:2 allocation is actually visible to JAX (FSDP=2 needs both).
export GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
export NUM_STEPS="${NUM_STEPS:-100001}"

echo "[mt4-ref] node=$(hostname) job=${SLURM_JOB_ID:-?} arm=${ARM:-ref} steps=$NUM_STEPS"
echo "[mt4-ref] gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo n/a)"

# exp.py exits 42 when it has saved a resumable epoch and wants the wall clock
# back (scripts/exp.py:5, :191-193); requeue then continues from that checkpoint.
# Same shape as scripts/launcher.py:65-82. `exec` cannot be used -- the status
# has to come back here. MAX_RUNTIME needs no override here: the recipe's 47 h
# default is already under this wrapper's 48 h wall.
child_status=0
bash scripts/ogpo_multitask_4task_ref.sh || child_status=$?

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
