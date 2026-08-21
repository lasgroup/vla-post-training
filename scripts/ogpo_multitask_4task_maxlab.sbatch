#!/bin/bash
#SBATCH --partition=maxlab
#SBATCH --qos=maxlab_qos
#SBATCH --job-name=mt4_ogpo
#SBATCH --gres=gpu:1
#SBATCH --constraint=VRAM_96GB
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=48:00:00
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

exec bash scripts/ogpo_multitask_4task.sh
