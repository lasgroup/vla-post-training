#!/bin/bash
#SBATCH --partition=maxlab
#SBATCH --qos=maxlab_qos
#SBATCH --job-name=paper_ogpo_mt4_fast_b128_2gpu_probe
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
# THROWAWAY PROBE (2026-09-21): companion to paper_expt_ogpo_multitask_fast_
# batch64_probe.sh (job 10520254) -- same 512-chains-per-GPU footprint and
# same 256-effective-states-per-step target, reached the other way: 2 GPUs
# (fsdp_devices=2) instead of 1, batch=128 instead of 64, grad_accum=2
# instead of 4.
#
# Chain-count math (chains/GPU = (batch_size / fsdp_devices) x
# group_num_samples=8), matching the derivation convention used throughout
# this probe series:
#   batch64_probe:  (64  / 1) x 8 = 512 chains/GPU, accum=4 -> effective
#                   batch = 64*4  = 256 states/step, 1 GPU.
#   this probe:     (128 / 2) x 8 = 512 chains/GPU, accum=2 -> effective
#                   batch = 128*2 = 256 states/step, 2 GPUs.
# Same peak per-GPU memory target, same effective batch size -- the only
# difference is whether the second half of the work is done by grad_accum
# (sequential, 1 GPU) or by FSDP sharding (parallel, 2 GPUs, cross-GPU sync
# overhead). This mirrors the single-task fast derivation's Option
# A-vs-B/D comparison (paper_expt_ogpo_single_task_fast_s0.sh:22-43), which
# found FSDP sync overhead usually loses to grad_accum's extra sequential
# passes -- untested here at this chain count/task count, hence the probe.
#
# XLA_PYTHON_CLIENT_MEM_FRACTION left at the script default (0.75, NOT
# bumped to 0.92 like the batch64 probe) -- 512 chains/GPU here is split
# across 2 independent XLA clients (one per GPU) rather than a single
# client's full 512-chain call, so per-client pressure should be lower even
# at the same nominal chain count; bump later if this OOMs.
#
# --mem=300G / --cpus-per-task=32 (double the single-GPU probe's 150G/16):
# two GPUs means two XLA host processes plus NCCL coordination; also matches
# the host-RAM-OOM fix noted for other multitask jobs earlier in this
# campaign (150G was insufficient under sustained load).
#
# Distinct ARM/exp_name -- does not touch the batch=32 fast checkpoints or
# the batch64_probe checkpoint. No monitor attached; maintainer will check
# on it manually and compare throughput/peak-memory against batch64_probe to
# pick a winner.
#
# Submit: sbatch scripts/paper_expt_ogpo_multitask_fast_batch128_2gpu_probe.sh
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_DIR=/home/pchellap/Projects/OGPO-VLA/vla-post-training
cd "$PROJECT_DIR"

export ARM="paper_expt_ogpo_multitask_fast_batch128_2gpu_probe"
export SEED="0"
export NUM_STEPS="500001"
export BON_N="1"
export NUM_QS="2"
export CRITIC_RED="min"
export COLLECT_INT="50000"
export PG_START="0"
export BATCH="128"
export FSDP="2"

export GPU="${GPU:-0,1}"

echo "[paper-mt4-fast-b128-2gpu-probe] node=$(hostname) job=${SLURM_JOB_ID:-?} arm=$ARM seed=$SEED batch=$BATCH fsdp=$FSDP gpu=$GPU"
echo "[paper-mt4-fast-b128-2gpu-probe] gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo n/a)"

child_status=0
bash scripts/ogpo_multitask_4task_ref.sh \
  --project_name vla-post-training \
  --group_name policy_extraction_tasks4_ogpo_fast_batch128_2gpu_probe_v0 \
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
