#!/bin/bash
#SBATCH --partition=maxlab
#SBATCH --qos=maxlab_qos
#SBATCH --nodelist=babel-m9-16
#SBATCH --job-name=fsft_full
#SBATCH --gres=gpu:4
#SBATCH --constraint=VRAM_96GB
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --output=/home/mananaga/logs/%j/.out
#SBATCH --error=/home/mananaga/logs/%j/.out
# ---------------------------------------------------------------------------
# Filtered SFT, PaliGemma tuning = full.
#
# Full PaliGemma finetune: no freeze filter, so the LLM, the SigLIP tower and
# the action expert all move (~3.35G trainable). Same setup as the existing
# fsft_libero_babel.sh, so it doubles as this ablation's baseline.
#
# The action expert is trained here as in every arm; the recipe (5k steps,
# collect every 500, lr 2.5e-5, batch 256, libero_90_38) lives in
# scripts/fsft_libero.sh and is identical across the three arms.
#
# Submit:  sbatch scripts/fsft_full.sh
# Inspect: DRY=1 bash scripts/fsft_full.sh
# Knobs:   sbatch --export=ALL,SEED=1,BATCH_SIZE=128 scripts/fsft_full.sh
# ---------------------------------------------------------------------------
set -euo pipefail

export PROJECT_DIR="${PROJECT_DIR:-/home/mananaga/VLA/ogpo/vla-post-training}"
cd "$PROJECT_DIR"

export PG_TUNE=full
exec bash scripts/fsft_libero.sh "$@"
