#!/bin/bash
#SBATCH --partition=maxlab
#SBATCH --qos=maxlab_qos
#SBATCH --nodelist=babel-m9-16
#SBATCH --job-name=awr_so_libero
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
EXP_NAME=pi05_libero_online_aw_sft_successonly_libero_90_44_seed0
CKPT_BASE_DIR=$STORE_ROOT/checkpoints/awr_multitask

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
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES%%,*}"

mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME" "$CKPT_BASE_DIR"

echo "[awr-success-only] node=$(hostname) job=${SLURM_JOB_ID:-none} exp=${EXP_NAME}"

# Single-variable change from awr_libero_babel.sh: --rl.store_success_episodes_only.
# Everything else (schedule, critic, beta, clip) is byte-identical so this is a
# clean ablation against both that run and fsft_libero_babel.sh.
#
# Why this is expected to help. On the mixed buffer every failed episode runs the
# full 400 steps and never terminates, so:
#   - fix_mc_returns rewrites its MC returns to the constant -1/(1-0.995) = -200,
#   - its transitions never zero the discount, leaving the TD fixed point at
#     r_chunk/(1-gamma^H) = -9.78/0.0489 = -200.
# Exactly one window per *successful* episode carries terminate=True, so only
# 0.077% of the buffer (0.8 samples per 1024 critic batch) is evidence that
# success exists. Measured on run magcgjnq: V tracked the MC mixture mean at
# -175 while td_weight was 0, then slid to -197 within 500 steps of the switch
# to td_weight=1 and parked there, below the -173 achievable floor, leaving
# advantage_std at 0.23 and AWR weights inside [0.97, 1.04].
#
# Storing successes only removes both failure modes: rewards are no longer
# constant (the terminating step scores 0) so fix_mc_returns stops firing, and
# every chain ends in a real terminal window. Values become discounted
# time-to-success in [-152, -8.8], comfortably inside the bound.
#
# Expected outcome is bounded below by FSFT: if the critic cannot separate fast
# from slow successes, A -> 0, the weights -> 1, and this reduces exactly to
# fsft_libero_babel.sh. beta=10 is the right scale *here* -- one chunk of delay
# is 9.78 of advantage, i.e. one unit of exponent -- which is the calibration
# the original comment assumed but the mixed buffer never delivered.
#
# Caveat: the critic now only ever sees successful actions, so Q(s,a) off the
# success manifold is unconstrained. This reweights toward efficient successes;
# it cannot learn from failure.
#
# Verify with:
#   python tests/awr/wandb_quantiles.py --run-name "$EXP_NAME"
# Read actor/advantage_std and the ESS/N line to pick beta from data rather than
# from the reward scale.

exec uv run scripts/exp.py \
  pi05_libero_online_aw_sft \
  --project_name openpi \
  --group_name awr_success_only_babel \
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
  --collect.tasks libero_90_79 libero_90_31 libero_90_82 libero_90_38 \
  --collect.eval_tasks libero_90_79 libero_90_31 libero_90_82 libero_90_38 \
  --collect.store_prefix_rep \
  --collect.collect_interval 500 \
  --collect.num_rollouts 20 \
  --collect.env_num 8 \
  --collect.eval_env_num 8 \
  --collect.eval_interval 4999 \
  --rl.discount 0.995 \
  --rl.online_ratio 1.0 \
  --rl.buffer_capacity 500000 \
  --rl.store_success_episodes_only \
  --rl.policy.update_interval 1 \
  --rl.policy.training_start_step 0 \
  --rl.critic.no-use_distributional_critic \
  --rl.critic.num_value_bins 1 \
  --rl.critic.batch_size 1024 \
  --rl.critic.use_bronet \
  --rl.critic.bronet_hidden_dim 1024 \
  --rl.beta 10.0 \
  --rl.advantage_scale 1.0 \
  --rl.weight_clip 3.0 \
  --batch_size 256
