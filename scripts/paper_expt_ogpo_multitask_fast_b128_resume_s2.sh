#!/bin/bash
#SBATCH --partition=maxlab
#SBATCH --qos=maxlab_qos
#SBATCH --job-name=paper_ogpo_mt4_fast_b128_s2
#SBATCH --gres=gpu:2
#SBATCH --constraint=VRAM_96GB
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/pchellap/logs/%x_%j.out
#SBATCH --error=/home/pchellap/logs/%x_%j.out
# ---------------------------------------------------------------------------
# RESUME of the multitask "fast" seed-2 run onto 2 GPUs (2026-09-21).
#
# Continues paper_expt_ogpo_multitask_fast_s2.sh (job 10514867, batch 32 /
# grad_accum 8 / FSDP=1, 1 GPU) from ITS OWN checkpoint: same ARM and SEED, so
# the same exp dir mt4_paper_expt_ogpo_multitask_fast_s2, and --resume is the
# recipe default. Only three things differ from that script:
#     fsdp_devices  1 -> 2      (2 GPUs; cross-topology resume, see
#                                docs/changes/2026-09-21-fsdp-topology-resume/)
#     batch_size    32 -> 128
#     policy_grad_accum 8 -> 2  (effective batch stays 256 states/step)
# Measured 2026-09-21: 33.4 s per 100 steps vs 47.2 s (~1.41x faster).
#
# DO NOT SUBMIT while the batch-32 job is queued or running: two writers on one
# checkpoint dir. The guard below refuses to start in that case (bypassed under
# DRY=1, or with ALLOW_OVERLAP=1 for a deliberate brief overlap: `sbatch
# --export=ALL,ALLOW_OVERLAP=1 ...`; both jobs then append to metrics.jsonl, which
# duplicates the overlapped step range -- harmless, filter by last-wins). Cancel the old job only AFTER its step-50000 checkpoint is finalized;
# the resumed job redoes that step's collect/eval round, exactly like the
# single-task restarts did.
#
# The 300G / 32 CPUs (vs 150G / 16 for the batch-32 job) match the 2-GPU probe
# that ran clean; host RAM has been the failure mode on long multitask runs.
# The expected run length (~45 h) is under MAX_RUNTIME (47 h) so no requeue is
# expected; if it does hit it, --requeue resumes from the last checkpoint.
#
# Submit: sbatch scripts/paper_expt_ogpo_multitask_fast_b128_resume_s2.sh
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_DIR=/home/pchellap/Projects/OGPO-VLA/vla-post-training
cd "$PROJECT_DIR"

if [ "${DRY:-0}" != "1" ] && [ "${ALLOW_OVERLAP:-0}" != "1" ] && squeue -u "$USER" -h -n paper_ogpo_mt4_fast_s2 -t RUNNING,PENDING | grep -q .; then
  echo "[b128-resume-s2] REFUSING to start: batch-32 job paper_ogpo_mt4_fast_s2 is still queued/running and would race this job on the same checkpoint dir. Cancel it after its step-50000 checkpoint is finalized." >&2
  exit 1
fi

export ARM="paper_expt_ogpo_multitask_fast"
export SEED="2"
export NUM_STEPS="500001"
export BON_N="1"
export NUM_QS="2"
export CRITIC_RED="min"
export COLLECT_INT="50000"
export PG_START="0"
export BATCH="128"
export FSDP="2"

export GPU="${GPU:-0,1}"

echo "[b128-resume-s2] node=$(hostname) job=${SLURM_JOB_ID:-?} arm=$ARM seed=$SEED batch=$BATCH fsdp=$FSDP gpu=$GPU"
echo "[b128-resume-s2] gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | paste -sd';' || echo n/a)"

child_status=0
bash scripts/ogpo_multitask_4task_ref.sh \
  --project_name vla-post-training \
  --group_name policy_extraction_tasks4_ogpo_fast_v0 \
  --rl.policy.update_interval 100 \
  --rl.policy_grad_accum 2 \
  --rl.buffer_capacity 250000 \
  --rl.noise_level 0.3 \
  --collect.eval_interval 100000 \
  || child_status=$?

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
