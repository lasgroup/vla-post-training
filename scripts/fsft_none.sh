#!/bin/bash
#SBATCH --partition=maxlab
#SBATCH --qos=maxlab_qos
#SBATCH --nodelist=babel-m9-16
#SBATCH --job-name=fsft_none
#SBATCH --gres=gpu:4
#SBATCH --constraint=VRAM_96GB
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --output=/home/mananaga/logs/%j/.out
#SBATCH --error=/home/mananaga/logs/%j/.out
# ---------------------------------------------------------------------------
# Filtered SFT, PaliGemma tuning = none.
#
# Action expert only: PaliGemma LLM + SigLIP frozen (cast to bf16), ~430M
# trainable params. Cheapest arm -- the frozen backbone leaves a lot of VRAM
# free, so BATCH_SIZE can go up if you want it to.
#
# The action expert is trained here as in every arm; the recipe (5k steps,
# collect every 500, lr 2.5e-5, batch 256, libero_90_38) lives in
# scripts/fsft_libero.sh and is identical across the three arms.
#
# Submit:  sbatch scripts/fsft_none.sh
# Inspect: DRY=1 bash scripts/fsft_none.sh
# Knobs:   sbatch --export=ALL,SEED=1,BATCH_SIZE=128 scripts/fsft_none.sh
# ---------------------------------------------------------------------------
set -euo pipefail

export PROJECT_DIR="${PROJECT_DIR:-/home/mananaga/VLA/ogpo/vla-post-training}"
cd "$PROJECT_DIR"

export PG_TUNE=none
exec bash scripts/fsft_libero.sh "$@"
