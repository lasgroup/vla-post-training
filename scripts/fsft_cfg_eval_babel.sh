#!/bin/bash
#SBATCH --partition=maxlab
#SBATCH --qos=maxlab_qos
#SBATCH --nodelist=babel-m9-16
#SBATCH --job-name=fsft_cfg_eval
#SBATCH --gres=gpu:2
#SBATCH --constraint=VRAM_96GB
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=/home/mananaga/logs/%j/.out
#SBATCH --error=/home/mananaga/logs/%j/.out

set -euo pipefail

PROJECT_DIR=/home/mananaga/VLA/manan_babel/vla-post-training
STORE_ROOT=/data/group_data/maxlab/common_datasets/mananaga/vla-post-training
EXP_NAME=pi05_libero_online_filtered_sft_multitask4_cfg_seed0
CKPT_BASE_DIR=$STORE_ROOT/checkpoints/fsft_multitask_cfg

cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/openpi/packages/openpi-client/src:$PROJECT_DIR/openpi/src:$PROJECT_DIR/openpi/packages/openpi-client:$PROJECT_DIR/molmospaces"

export OPENPI_DATA_HOME="$STORE_ROOT/cache/openpi"
export HF_HOME="$STORE_ROOT/cache/huggingface"
export LIBERO_CONFIG_PATH="$HOME/.libero"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

export NCCL_CUMEM_ENABLE=0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES%%,*}"

echo "[fsft-cfg-eval] node=$(hostname) job=${SLURM_JOB_ID:-none} exp=${EXP_NAME}"

# Re-run the step-5000 guidance sweep of fsft_cfg_libero_babel.sh as a *paired*
# eval. The in-training sweep was unpaired: evaluate_policy did not seed the env,
# so each scale saw different initial states, and at 32 rollouts/task the
# per-scale spread (0.398-0.445 overall) sat inside one standard error.
#
# Two things changed in src/training/collect.py:
#   - evaluate_policy seeds the eval envs (config.seed + offset + step), so every
#     scale replays the same initial states.
#   - evaluate_policy_sweep rewinds the agent PRNG before each scale, so the
#     action noise is identical too.
# The scales now differ only in guidance, and episodes are matched one-to-one
# across scales -- pair them (McNemar) rather than comparing marginal rates.
#
# num_eval_rollouts 100 (vs 32 in training): 400 episodes/scale, per-task SE
# ~0.049 and overall SE ~0.025 even before pairing. Guided scales cost ~2x per
# denoising step; the 512-episode training sweep took ~27 min, so 1600 episodes
# here should land near ~1.5h. Raise this if you want to resolve <5 points.
#
# rl.buffer_capacity / rl.discount must match fsft_cfg_libero_babel.sh even
# though eval never reads the buffer: resuming restores the replay shards
# (filtered_sft_learner.py:273-278, not gated on requeue despite what eval.py's
# docstring says), and the default capacity of 1024 overflows on the first
# 7988-observation shard. Costs a few minutes and ~7G of RAM at startup.
#
# --resume is REQUIRED and load-bearing: the learner passes
# overwrite=not config.resume to initialize_checkpoint_dir, so running without
# it WIPES the checkpoint. Do not add --overwrite.
#
# W&B stays off so this does not overwrite the summary of the original run
# (3a4pmfrg) -- the same eval/cfg<scale>/... keys would be replaced in place.
# Metrics land in $CKPT_BASE_DIR/.../$EXP_NAME/eval_metrics_step5000.json;
# plot them with `python scripts/plot_cfg_eval.py --json <that file>`.

exec uv run scripts/eval.py \
  pi05_libero_online_filtered_sft \
  --exp_name "$EXP_NAME" \
  --checkpoint_base_dir "$CKPT_BASE_DIR" \
  --resume \
  --no-wandb_enabled \
  --seed 0 \
  --fsdp_devices 2 \
  --collect.tasks libero_90_79 libero_90_31 libero_90_82 libero_90_38 \
  --collect.eval_tasks libero_90_79 libero_90_31 libero_90_82 libero_90_38 \
  --collect.num_eval_rollouts 100 \
  --collect.eval_env_num 8 \
  --rl.online_ratio 1.0 \
  --rl.discount 0.995 \
  --rl.buffer_capacity 500000 \
  --rl.cfg_dropout_prob 0.1 \
  --rl.cfg_scale 1.0 1.5 2.0 3.0
