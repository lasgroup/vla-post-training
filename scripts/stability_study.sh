#!/bin/bash
# ---------------------------------------------------------------------------
# Single-task stability study launcher (one arm per invocation, single GPU).
#
# Base recipe = abl_viii_r2, the single-task winner (frozen backbone, GRPO
# G=8 group-mean baseline, eps=0.1, success-buffer BC), PLUS periodic
# 32-episode EMA evals every 10k steps (the study's low-noise measurement).
#
# Interventions are toggled per-arm via env vars (all default OFF = baseline):
#   ARM        - run name suffix (required), e.g. base_s1, E, N, NC, G, ENCG
#   SEED       - default 0
#   GPU        - CUDA device index (required on multi-GPU nodes)
#   EMA        - actor ema_decay (default 0.99; E arms use 0.999)
#   NORM       - 1 => --rl.normalize_group_advantage (EMA-quantile scale)
#   CLIP_SYM   - float => --rl.adv_clip_sym <v> (symmetric clip, post-norm)
#   ACCUM      - int  => --rl.policy_grad_accum <M> (micro-batch grad accum)
#
# Usage:  ARM=N NORM=1 GPU=3 bash scripts/stability_study.sh
# ---------------------------------------------------------------------------
set -euo pipefail

: "${ARM:?set ARM (run name suffix)}"
: "${GPU:?set GPU (cuda device index)}"
SEED="${SEED:-0}"
EMA="${EMA:-0.99}"

# Non-interactive shells (nohup over ssh) miss ~/.local/bin.
export PATH="$HOME/.local/bin:$PATH"

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
STORE_ROOT="${STORE_ROOT:-$PROJECT_DIR/run_store}"
EXP_NAME="stab_${ARM}"
CKPT_BASE_DIR="$STORE_ROOT/checkpoints/stability_study"

export CUDA_VISIBLE_DEVICES="$GPU"
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/openpi/packages/openpi-client/src:$PROJECT_DIR/openpi/src:$PROJECT_DIR/openpi/packages/openpi-client:$PROJECT_DIR/molmospaces"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$STORE_ROOT/cache/openpi}"
export HF_HOME="${HF_HOME:-$STORE_ROOT/cache/huggingface}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$STORE_ROOT/libero}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$STORE_ROOT/cache/uv}"
export TORCH_HOME="${TORCH_HOME:-$STORE_ROOT/cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$STORE_ROOT/cache/triton}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$STORE_ROOT/cache/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$STORE_ROOT/cache/xdg}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$STORE_ROOT/config/xdg}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-$STORE_ROOT/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$STORE_ROOT/cache/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$STORE_ROOT/config/wandb}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="$MUJOCO_GL"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ "$MUJOCO_GL" = "egl" ] && [ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ]; then
  if [ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
  elif ldconfig -p 2>/dev/null | grep -q libEGL_nvidia; then
    mkdir -p "$STORE_ROOT/egl"
    printf '%s\n' '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
      > "$STORE_ROOT/egl/10_nvidia.json"
    export __EGL_VENDOR_LIBRARY_FILENAMES="$STORE_ROOT/egl/10_nvidia.json"
  fi
fi
export NCCL_CUMEM_ENABLE=0
export NCCL_IB_DISABLE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}"
export MUJOCO_EGL_DEVICE_ID="$GPU"

EXTRA_FLAGS=()
[ "${NORM:-0}" = "1" ] && EXTRA_FLAGS+=(--rl.normalize_group_advantage)
[ -n "${CLIP_SYM:-}" ] && EXTRA_FLAGS+=(--rl.adv_clip_sym "$CLIP_SYM")
[ -n "${ACCUM:-}" ] && EXTRA_FLAGS+=(--rl.policy_grad_accum "$ACCUM")

mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME" "$LIBERO_CONFIG_PATH" "$CKPT_BASE_DIR" \
         "$UV_CACHE_DIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" \
         "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" \
         "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

echo "[stability-study] node=$(hostname) gpu=$GPU arm=$ARM seed=$SEED ema=$EMA extra=${EXTRA_FLAGS[*]:-none}"

uv run scripts/exp.py \
  pi05_libero_online_ogpo_sft \
  --project_name ogpo_stability \
  --group_name stability_study \
  --exp_name "$EXP_NAME" \
  --checkpoint_base_dir "$CKPT_BASE_DIR" \
  --seed "$SEED" \
  --fsdp_devices 1 \
  --overwrite \
  --log_interval 25 \
  --save_interval 100000 \
  --keep_period 100000 \
  --num_train_steps 100000 \
  --ema_decay "$EMA" \
  --lr_schedule.value 2.5e-5 \
  --max_runtime 169200 \
  --collect.tasks libero_90_44 \
  --collect.eval_tasks libero_90_44 \
  --collect.store_prefix_rep \
  --collect.collect_interval 10000 \
  --collect.num_rollouts 20 \
  --collect.env_num 8 \
  --collect.eval_env_num 8 \
  --collect.eval_interval 10000 \
  --rl.beta 0.05 \
  --rl.discount 0.995 \
  --rl.online_ratio 1.0 \
  --rl.buffer_capacity 250000 \
  --rl.policy.update_interval 10 \
  --rl.policy.training_start_step 900 \
  --rl.critic.td_weight_schedule.init_value 1 \
  --rl.critic.td_weight_schedule.end_value 1 \
  --rl.critic.td_weight_schedule.switch_step 999999 \
  --rl.critic.no-use_distributional_critic \
  --rl.critic.num_value_bins 1 \
  --rl.critic.batch_size 1024 \
  --rl.critic.pre_training_steps 0 \
  --rl.critic.use_bronet \
  --rl.critic.bronet_hidden_dim 1024 \
  --rl.critic.inference_start_step 1 \
  --rl.group_num_samples 8 \
  --rl.clip_epsilon 0.1 \
  --rl.bc_coeff 1.0 \
  --rl.num_sde_steps 10 \
  --rl.noise_level 0.02 \
  --rl.adv_strategy vanilla \
  --rl.dedup_group_prefix \
  --rl.use_success_buffer \
  --rl.critic.value_target_type one_hot \
  --batch_size 32 \
  "${EXTRA_FLAGS[@]}"
