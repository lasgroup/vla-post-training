#!/bin/bash
#SBATCH --partition=maxlab
#SBATCH --qos=maxlab_qos
#SBATCH --nodelist=babel-m9-16
#SBATCH --job-name=ogpo_libero
#SBATCH --gres=gpu:4
#SBATCH --constraint=VRAM_96GB
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=/home/mananaga/logs/%j/.out
#SBATCH --error=/home/mananaga/logs/%j/.out

set -euo pipefail

PROJECT_DIR=/home/mananaga/vla-post-training
STORE_ROOT=/data/group_data/maxlab/common_datasets/mananaga/vla-post-training
EXP_NAME=pi05_libero_online_ogpo_sft_libero_90_59_seed1
CKPT_BASE_DIR=$STORE_ROOT/checkpoints/ogpo_sweep_babel

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
# can allocate during the collection phase. At 0.95 (~4.8GB free/GPU) the
# framebuffer alloc wedges the driver (D-state, unkillable -> node drain);
# 0.75 leaves ~24GB/GPU, far more than the render CUDA contexts need. Tune lower
# if the crash still bites.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES%%,*}"

mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME" "$CKPT_BASE_DIR"

echo "[ogpo] node=$(hostname) job=${SLURM_JOB_ID:-none} exp=${EXP_NAME}"

# OGPO: PPO on flow policies with on-policy SDE log-probs and a BC anchor on the
# same on-policy batch. Params mirror scripts/configs/ogpo.yaml (v1: G=1 with a
# V-baseline advantage, adv = Q - V). --save_interval / --max_runtime are babel
# scaffolding not present in the sweep yaml.
exec uv run scripts/exp.py \
  pi05_libero_online_ogpo_sft \
  --project_name ogpo_sweep \
  --group_name ogpo_sweep_babel \
  --exp_name "$EXP_NAME" \
  --checkpoint_base_dir "$CKPT_BASE_DIR" \
  --seed 1 \
  --fsdp_devices 4 \
  --overwrite \
  --log_interval 25 \
  --save_interval 10000 \
  --num_train_steps 10000 \
  --lr_schedule.value 2.5e-5 \
  --batch_size 256 \
  --ema_decay 0.995 \
  --max_runtime 169200 \
  --collect.tasks libero_90_59 \
  --collect.eval_tasks libero_90_59 \
  --collect.collect_interval 300 \
  --collect.eval_interval 300 \
  --collect.num_rollouts 1 \
  --collect.num_initial_rollouts 5 \
  --collect.num_eval_rollouts 8 \
  --collect.env_num 1 \
  --collect.eval_env_num 1 \
  --rl.buffer_capacity 250000 \
  --rl.online_ratio 1.0 \
  --rl.store_success_episodes_only \
  --rl.policy.training_start_step 900 \
  --rl.policy.update_interval 1 \
  --rl.critic.pre_training_steps 900 \
  --rl.critic.num_updates_per_batch 10 \
  --rl.critic.td_weight_schedule.switch_step 1000000 \
  --rl.critic.no-use_ema \
  --rl.group_num_samples 1 \
  --rl.clip_epsilon 0.01 \
  --rl.bc_coeff 1.0 \
  --rl.num_sde_steps 10 \
  --rl.noise_level 0.3 \
  --rl.adv_strategy subtract_v
