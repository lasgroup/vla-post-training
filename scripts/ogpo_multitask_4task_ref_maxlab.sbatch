#!/bin/bash
#SBATCH --partition=maxlab
#SBATCH --qos=maxlab_qos
#SBATCH --job-name=mt4_ref
#SBATCH --gres=gpu:1
#SBATCH --constraint=VRAM_96GB
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=48:00:00
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
# save_interval=100000, so the first checkpoint lands at step 100000 -- a
# preemption at hour 10 loses everything. The smoke was fine on preempt because
# it was 11 minutes. maxlab's queue wait is the cheaper risk here.
#
# Submit:  sbatch --export=ALL,ARM=ref,NUM_STEPS=100001 \
#              scripts/ogpo_multitask_4task_ref_maxlab.sbatch
# Arms:    see scripts/submit_mt4_ref_arms.sh
# All env knobs of ogpo_multitask_4task_ref.sh pass through via --export.
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_DIR=/home/pchellap/Projects/OGPO-VLA/vla-post-training
cd "$PROJECT_DIR"

# SLURM renumbers the allocated GPUs inside the cgroup; index 0 is ours.
export GPU="${GPU:-0}"
export NUM_STEPS="${NUM_STEPS:-100001}"

echo "[mt4-ref] node=$(hostname) job=${SLURM_JOB_ID:-?} arm=${ARM:-ref} steps=$NUM_STEPS"
echo "[mt4-ref] gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo n/a)"

exec bash scripts/ogpo_multitask_4task_ref.sh
