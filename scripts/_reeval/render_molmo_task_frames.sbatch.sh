#!/bin/bash
#SBATCH --job-name=molmo_frames
#SBATCH --account=a0220
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --output=/capstor/scratch/cscs/%u/molmo_frames-%j.out
set -x
# EGL needs the graphics capability since the 2026-10-02 driver update
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
OUT=$SCRATCH/molmo_task_frames_tasks144
mkdir -p $OUT
srun --environment=vla-post-training \
  uv run scripts/_reeval/render_molmo_task_frames.py \
    --out $OUT
