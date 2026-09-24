#!/bin/bash
#SBATCH --partition=maxlab
#SBATCH --qos=maxlab_qos
#SBATCH --job-name=paper_ogpo_mt4_s2
#SBATCH --gres=gpu:1
#SBATCH --constraint=VRAM_96GB
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/pchellap/logs/%x_%j.out
#SBATCH --error=/home/pchellap/logs/%x_%j.out
# ---------------------------------------------------------------------------
# Paper experiment: OGPO, reference-aligned (pi05_libero_online_ogpo_ref),
# 4-task LIBERO (libero_90_79/31/82/38 -- same set as
# pi05_libero_online_ogpo_sft_pertask), evaluated IN-DISTRIBUTION (the same 4
# tasks only). Seed 2 of 3 (paper_expt_ogpo_multitask_s{0,1,2}.sh). Multitask
# counterpart of paper_expt_ogpo_single_task_s{0,1,2}.sh.
#
# See paper_expt_ogpo_multitask_s0.sh for the full decision log (source diff,
# what matches, what deliberately differs, and why).
#
# Submit:  sbatch scripts/paper_expt_ogpo_multitask_s2.sh
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_DIR=/home/pchellap/Projects/OGPO-VLA/vla-post-training
cd "$PROJECT_DIR"

export ARM="paper_expt_ogpo_multitask"
export SEED="2"
export NUM_STEPS="500001"
export BON_N="32"
export PG_START="0"

# SLURM renumbers the allocated GPUs inside the cgroup; take the whole list
# so a --gres=gpu:2 allocation is actually visible to JAX.
export GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"

echo "[paper-expt-mt4] node=$(hostname) job=${SLURM_JOB_ID:-?} arm=$ARM seed=$SEED"
echo "[paper-expt-mt4] gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo n/a)"

# exp.py exits 42 when it has saved a resumable epoch and wants the wall
# clock back (scripts/exp.py:5, :191-193); requeue then continues from that
# checkpoint. Plain `bash`, not `exec` -- the exit status has to come back
# to this shell.
child_status=0
bash scripts/ogpo_multitask_4task_ref.sh \
  --project_name vla-post-training \
  --group_name policy_extraction_tasks4_ogpo_ref_v0 \
  --rl.buffer_capacity 250000 \
  --rl.noise_level 0.3 \
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
