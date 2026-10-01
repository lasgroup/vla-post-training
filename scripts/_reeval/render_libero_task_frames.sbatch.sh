#!/bin/bash
#SBATCH --job-name=libero_frames
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --output=/capstor/scratch/cscs/ralfroemer/libero_task_frames/slurm-%j.out
set -euo pipefail
cd /users/ralfroemer/vla-post-training
srun --account=ab037 --environment=vla-post-training \
  uv run scripts/_reeval/render_libero_task_frames.py
