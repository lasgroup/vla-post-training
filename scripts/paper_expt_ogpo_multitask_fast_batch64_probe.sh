#!/bin/bash
#SBATCH --partition=maxlab
#SBATCH --qos=maxlab_qos
#SBATCH --job-name=paper_ogpo_mt4_fast_b64_probe
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
# THROWAWAY PROBE (2026-09-21): can batch_size=64 fit on a single RTX PRO 6000
# (96GB, maxlab's VRAM_96GB) for the multitask "fast" recipe, at half the
# grad-accum (4 instead of 8) so the effective per-microbatch memory stays in
# the same ballpark as the batch=32/accum=8 fast runs currently in flight
# (paper_expt_ogpo_multitask_fast_s{0,1,2}.sh, jobs 10514865/66/67)?
#
# Reasoning for trying this: batch=32 x group_num_samples=8 = 256 local SDE
# chains was already validated safe on a 96GB card at the ORIGINAL
# single-task baseline (75.8/95.6 GiB peak, 10-head critic, job 10178533).
# The fast recipe's 2-head critic is lighter, so there should be headroom --
# batch=64 doubles that to 512 chains, which is untested. Also raised here:
# XLA_PYTHON_CLIENT_MEM_FRACTION defaults to 0.75 (eager preallocation), which
# may understate true fit -- bumped to 0.92 for this probe so a real OOM
# reflects actual tensor need, not a self-imposed ceiling.
#
# Same knob set as paper_expt_ogpo_multitask_fast_s0.sh otherwise (single
# GPU, no FSDP sharding; update_interval=100, n_samples=1, 2-head min-critic,
# buffer_capacity=250000, noise_level=0.3, collect_interval=50000,
# eval_interval=100000, num_train_steps=500001) so memory/throughput are
# directly comparable to the batch=32/accum=8 arms.
#
# Distinct ARM/exp_name so this does NOT touch the batch=32 fast checkpoints.
# Purely a feasibility + throughput probe -- no monitor attached, maintainer
# will check on it manually and decide whether to keep, kill, or promote to
# a real run.
#
# Submit: sbatch scripts/paper_expt_ogpo_multitask_fast_batch64_probe.sh
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_DIR=/home/pchellap/Projects/OGPO-VLA/vla-post-training
cd "$PROJECT_DIR"

export ARM="paper_expt_ogpo_multitask_fast_batch64_probe"
export SEED="0"
export NUM_STEPS="500001"
export BON_N="1"
export NUM_QS="2"
export CRITIC_RED="min"
export COLLECT_INT="50000"
export PG_START="0"
export BATCH="64"

export XLA_PYTHON_CLIENT_MEM_FRACTION="0.92"

export GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"

echo "[paper-mt4-fast-b64-probe] node=$(hostname) job=${SLURM_JOB_ID:-?} arm=$ARM seed=$SEED batch=$BATCH xla_mem_fraction=$XLA_PYTHON_CLIENT_MEM_FRACTION"
echo "[paper-mt4-fast-b64-probe] gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo n/a)"

child_status=0
bash scripts/ogpo_multitask_4task_ref.sh \
  --project_name vla-post-training \
  --group_name policy_extraction_tasks4_ogpo_fast_batch64_probe_v0 \
  --rl.policy.update_interval 100 \
  --rl.policy_grad_accum 4 \
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
