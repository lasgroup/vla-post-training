#!/bin/bash
#SBATCH --job-name=molmo_frames
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --output=/capstor/scratch/cscs/ralfroemer/molmo_task_frames_v2/slurm-%j.out
set -x
mkdir -p /capstor/scratch/cscs/ralfroemer/molmo_task_frames_v2
srun --account=a0220 --environment=vla-post-training \
  uv run scripts/_reeval/render_molmo_task_frames.py \
    --out /capstor/scratch/cscs/ralfroemer/molmo_task_frames_v2
