#!/bin/bash
#SBATCH --partition=maxlab
#SBATCH --qos=maxlab_qos
#SBATCH --job-name=mt4_ogpo
#SBATCH --gres=gpu:1
#SBATCH --constraint=VRAM_96GB
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=25:00:00
#SBATCH --requeue
# A requeue reuses %j, and sbatch's default open mode truncates, so without this
# each attempt erases the log of the attempt it is recovering from
# (scripts/launcher.py:219-220 pairs the two flags for the same reason).
#SBATCH --open-mode=append
# 2026-08-23: babel-m9-16 has a faulting RDMA NIC -- dmesg shows repeated
# `mlx5_0/1: QP <n> error: local protection error` on new queue pairs all day.
# It is NOT share-specific: on 08-22 it wedged the nas6 mount (mount.nfs stuck
# in D state for hours); on 08-23 it killed two runs on *nas5* paths --
# job 10194773 at step 99999 with `Failed to fsync file [OS error 512]` while
# writing the final checkpoint, and job 10198072 at step 10000 with
# `os.getcwd() -> FileNotFoundError` while spawning env workers. Moving
# CKPT_BASE_DIR to nas5 does NOT make the node safe. Add
#   #SBATCH --exclude=babel-m9-16
# for anything long-running until the admins have drained/rebooted it.
#SBATCH --output=/home/pchellap/logs/%x_%j.out
#SBATCH --error=/home/pchellap/logs/%x_%j.out
# ---------------------------------------------------------------------------
# maxlab-partition wrapper for scripts/ogpo_multitask_4task.sh (1 GPU).
# Submit:   sbatch scripts/ogpo_multitask_4task_maxlab.sbatch
# Arms:     sbatch --export=ALL,ARM=noburst,BURST=0 scripts/ogpo_multitask_4task_maxlab.sbatch
# All env knobs of ogpo_multitask_4task.sh pass through via --export.
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_DIR=/home/pchellap/Projects/OGPO-VLA/vla-post-training
cd "$PROJECT_DIR"

# SLURM renumbers the allocated GPUs inside the cgroup; pass the list through.
# Batch 32 is global under FSDP, so more GPUs never changes the science —
# to shard for speed, submit with: sbatch --export=ALL,FSDP=2 --gres=gpu:2 ...
export GPU="${CUDA_VISIBLE_DEVICES:-0}"
export FSDP="${FSDP:-1}"

# 23.5 h, under the 25 h wall above. The recipe's default is 47 h (sized for the
# 48 h ref wrapper), so without this override SLURM SIGKILLs the job ~22 h before
# exp.py's runtime check can save and exit 42 -- the requeue below was dead code.
export MAX_RUNTIME="${MAX_RUNTIME:-84600}"

# exp.py exits 42 when it has saved a resumable epoch and wants the wall clock
# back (scripts/exp.py:5, :191-193); requeue then continues from that checkpoint.
# Same shape as scripts/launcher.py:65-82. `exec` cannot be used -- the status
# has to come back here.
child_status=0
bash scripts/ogpo_multitask_4task.sh || child_status=$?

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
