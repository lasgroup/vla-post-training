#!/bin/bash
#SBATCH --partition=maxlab
#SBATCH --qos=maxlab_qos
#SBATCH --nodelist=babel-m9-16
#SBATCH --job-name=fsft_libero
#SBATCH --gres=gpu:4
#SBATCH --constraint=VRAM_96GB
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --output=/home/mananaga/logs/%j/.out
#SBATCH --error=/home/mananaga/logs/%j/.out

set -euo pipefail

PROJECT_DIR=/home/mananaga/VLA/manan_babel/vla-post-training
STORE_ROOT=/data/group_data/maxlab/common_datasets/mananaga/vla-post-training
EXP_NAME=pi05_libero_online_filtered_sft_multitask4_seed0
CKPT_BASE_DIR=$STORE_ROOT/checkpoints/fsft_multitask

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
# Leave real VRAM headroom on the shared GPUs so MuJoCo/EGL offscreen framebuffers
# if the crash still bites.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES%%,*}"

mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME" "$CKPT_BASE_DIR"

echo "[fsft] node=$(hostname) job=${SLURM_JOB_ID:-none} exp=${EXP_NAME}"

# Filtered SFT has no critic, so none of the --rl.critic.* / --rl.beta knobs from
# awr_libero_babel.sh apply here. Schedule follows
# scripts/configs/multitask/fsft/libero/fsft_libero_tasks4-16_seed0_v0.yaml
# (5k steps, collect every 500) — its first sweep entry is AWR's 4-task set.

exec uv run scripts/exp.py \
  pi05_libero_online_filtered_sft \
  --project_name openpi \
  --group_name fsft_multitask_babel \
  --exp_name "$EXP_NAME" \
  --checkpoint_base_dir "$CKPT_BASE_DIR" \
  --seed 0 \
  --fsdp_devices 4 \
  --overwrite \
  --log_interval 25 \
  --save_interval 100000 \
  --num_train_steps 5000 \
  --lr_schedule.value 2.5e-5 \
  --max_runtime 169200 \
  --collect.tasks libero_90_38 \
  --collect.eval_tasks libero_90_38 \
  --collect.collect_interval 500 \
  --collect.num_rollouts 20 \
  --collect.env_num 8 \
  --collect.eval_env_num 8 \
  --collect.eval_interval 4999 \
  --rl.discount 0.995 \
  --rl.online_ratio 1.0 \
  --rl.buffer_capacity 500000 \
  --batch_size "${BATCH_SIZE:-256}"
