#!/bin/bash
#SBATCH --partition=maxlab
#SBATCH --qos=maxlab_qos
#SBATCH --nodelist=babel-m9-16
#SBATCH --job-name=fsft_lora
#SBATCH --gres=gpu:4
#SBATCH --constraint=VRAM_96GB
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --output=/home/mananaga/logs/%j/.out
#SBATCH --error=/home/mananaga/logs/%j/.out
# ---------------------------------------------------------------------------
# Filtered SFT, PaliGemma tuning = lora.
#
# PaliGemma tuned through rank-16 LoRA adapters (~28M adapter params) on the
# LLM's attn+ffn; base LLM weights and SigLIP stay frozen. ~458M trainable.
# Adapters start from the model's own normal(0.01) init -- the released pi05
# checkpoint carries none.
#
# The action expert is trained here as in every arm; the recipe (5k steps,
# collect every 500, lr 2.5e-5, batch 256, libero_90_38) lives in
# scripts/fsft_libero.sh and is identical across the three arms.
#
# Submit:  sbatch scripts/fsft_lora.sh
# Inspect: DRY=1 bash scripts/fsft_lora.sh
# Knobs:   sbatch --export=ALL,SEED=1,BATCH_SIZE=128 scripts/fsft_lora.sh
# ---------------------------------------------------------------------------
set -euo pipefail

export PROJECT_DIR="${PROJECT_DIR:-/home/mananaga/VLA/ogpo/vla-post-training}"
cd "$PROJECT_DIR"

export PG_TUNE=lora
exec bash scripts/fsft_libero.sh "$@"
