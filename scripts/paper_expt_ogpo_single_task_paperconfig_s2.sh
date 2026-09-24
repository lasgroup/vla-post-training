#!/bin/bash
#SBATCH --partition=maxlab
#SBATCH --qos=maxlab_qos
#SBATCH --job-name=paper_ogpo_pc_s2
#SBATCH --gres=gpu:1
#SBATCH --constraint=VRAM_96GB
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/pchellap/logs/%x_%j.out
#SBATCH --error=/home/pchellap/logs/%x_%j.out
# ---------------------------------------------------------------------------
# Paper experiment: OGPO, single train task libero_90_82, matching the
# hyperparameter budget described in the paper excerpt the maintainer pasted
# 2026-09-19 (M=10 policy-improvement iterations, K=20 episodes/task/epoch,
# 2 Q/V critic heads aggregated by min) crossed with marco/parl's own
# reference factory defaults (make_base_libero_config: batch_size=256,
# num_train_steps=10_000) for the fields the paper excerpt didn't state.
# Seed 2 of 3 (paper_expt_ogpo_single_task_paperconfig_s{0,1,2}.sh).
#
# See paper_expt_ogpo_single_task_paperconfig_s0.sh for the full decision
# log (what changed from the established single-task recipe, what didn't,
# and why).
#
# Submit:  sbatch scripts/paper_expt_ogpo_single_task_paperconfig_s2.sh
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_DIR=/home/pchellap/Projects/OGPO-VLA/vla-post-training
cd "$PROJECT_DIR"

export ARM="paper_expt_ogpo_single_task_paperconfig_s2"
export TASK="libero_90_82"
export SEED="2"
export N_STEPS="100001"
export QS="2"

export GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
export CKPT_BASE_DIR="${CKPT_BASE_DIR:-/data/group_data/maxlab/common_datasets/${USER:-pchellap}/vla-post-training/checkpoints/stability_study}"

echo "[paper-expt-pc] node=$(hostname) job=${SLURM_JOB_ID:-?} arm=$ARM task=$TASK seed=$SEED"
echo "[paper-expt-pc] gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo n/a)"

child_status=0
bash scripts/stability_study_ref.sh \
  --project_name vla-post-training \
  --group_name policy_extraction_tasks1_ogpo_paperconfig_v0 \
  --rl.n_samples 32 \
  --rl.noise_level 0.3 \
  --rl.critic.reduction min \
  --rl.policy_grad_accum 8 \
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
