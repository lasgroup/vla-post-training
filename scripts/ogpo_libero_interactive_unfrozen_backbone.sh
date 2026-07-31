#!/bin/bash
# ---------------------------------------------------------------------------
# Interactive (single-node, no SLURM) launcher for OGPO online post-training
# of pi05 on LIBERO with the PaliGemma LLM BACKBONE UNFROZEN and only the
# SigLIP vision tower FROZEN. Run directly in a terminal on a GPU node:
#
#     bash scripts/ogpo_libero_interactive_unfrozen_backbone.sh
#
# EDIT the CONFIG block below (paths + which GPUs) before the first run.
# See docs/awr_libero_babel_walkthrough.md for prerequisites and details.
#
# WHAT'S DIFFERENT vs scripts/ogpo_libero_interactive.sh:
#   * Entry point is scripts/exp_ogpo_unfrozen_backbone.py, which registers the
#     config `pi05_libero_online_ogpo_sft_unfrozen_backbone` at runtime WITHOUT
#     editing src/training/config.py. That config is a copy of the frozen
#     `pi05_libero_online_ogpo_sft` with only the freeze_filter changed:
#       - FROZEN: SigLIP image tower (`.*PaliGemma/img.*`).
#       - TRAINABLE: PaliGemma Gemma LLM backbone (stack 0), the action expert
#         (stack 1), and the small action heads.
#     The frozen default additionally freezes the LLM backbone, so this run
#     trains the backbone in fp32 with optimizer state and uses substantially
#     MORE memory. If it OOMs on a single GPU, raise
#     XLA_PYTHON_CLIENT_MEM_FRACTION and/or lower --batch_size.
#   * Every other hyperparameter below is IDENTICAL to
#     scripts/ogpo_libero_interactive.sh; only the entry point, config name,
#     EXP_NAME, and CKPT_BASE_DIR differ so runs do not collide.
#
# Shared behavior with the other interactive launcher:
#   * No SLURM header; runs in the foreground of your terminal.
#   * Single-GPU by default (CUDA_VISIBLE_DEVICES=0); --fsdp_devices is
#     auto-derived from the GPU count, so no model sharding on one GPU.
#   * All caches/config are kept INSIDE the repo (under STORE_ROOT). Nothing
#     is written to $HOME, so it is safe to run on a shared machine.
# ---------------------------------------------------------------------------

set -euo pipefail

# ===========================================================================
# CONFIG -- edit these for your node
# ===========================================================================
# Repo checkout. Auto-detected from this file's location (scripts/..); override
# by exporting PROJECT_DIR before running.
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

# Large writable dir for checkpoints + all model/data/tool caches (needs many GB).
# Kept inside the repo so the whole run is self-contained on a shared machine.
STORE_ROOT="${STORE_ROOT:-$PROJECT_DIR/run_store}"

EXP_NAME="${EXP_NAME:-pi05_libero_online_ogpo_sft_unfrozen_backbone_libero_90_44_seed0}"
CKPT_BASE_DIR="${CKPT_BASE_DIR:-$STORE_ROOT/checkpoints/ogpo_sweep_unfrozen_backbone}"

# Which GPUs to use. You have ONE GPU, so the default is 0.
# (fsdp_devices is auto-derived from the number of GPUs listed here.)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ===========================================================================
# (Usually no edits needed below this line)
# ===========================================================================
cd "$PROJECT_DIR"

# Make the git submodules importable (molmospaces is unused for LIBERO; a
# missing path here is harmless).
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/openpi/packages/openpi-client/src:$PROJECT_DIR/openpi/src:$PROJECT_DIR/openpi/packages/openpi-client:$PROJECT_DIR/molmospaces"

# --- keep every cache/config INSIDE the repo (safe on a shared machine) -----
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$STORE_ROOT/cache/openpi}"
export HF_HOME="${HF_HOME:-$STORE_ROOT/cache/huggingface}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$STORE_ROOT/libero}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$STORE_ROOT/cache/uv}"
export TORCH_HOME="${TORCH_HOME:-$STORE_ROOT/cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$STORE_ROOT/cache/triton}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$STORE_ROOT/cache/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$STORE_ROOT/cache/xdg}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$STORE_ROOT/config/xdg}"
export WANDB_DIR="${WANDB_DIR:-$STORE_ROOT/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$STORE_ROOT/cache/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$STORE_ROOT/config/wandb}"

# MuJoCo rendering backend. Default: headless GPU rendering via EGL. If you
# don't have working EGL, launch with MUJOCO_GL=osmesa (CPU software rendering
# -- no EGL/display needed, just slower); PYOPENGL_PLATFORM follows it.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# EGL vendor ICD selection (only matters when MUJOCO_GL=egl). We want the NVIDIA
# vendor so rendering runs on the GPU. Some systems (e.g. this one) register only
# the Mesa ICD, so GLVND auto-discovery would silently pick SOFTWARE rendering,
# and forcing a nonexistent /usr/share/.../10_nvidia.json breaks EGL entirely.
# Resolution order: honor an explicit override; else use a system NVIDIA JSON if
# present; else, if the NVIDIA EGL lib exists, generate a vendor JSON under
# STORE_ROOT and point at it (no root needed).
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

# NCCL (single node / single GPU: no InfiniBand, no multi-GPU collectives).
export NCCL_CUMEM_ENABLE=0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

# JAX per-GPU VRAM preallocation fraction. On a single display GPU this must
# leave headroom for the desktop + MuJoCo/EGL offscreen framebuffers. If JAX
# training OOMs, RAISE this (e.g. 0.85/0.9); if EGL/MuJoCo fails to allocate,
# LOWER it. This is the primary single-GPU memory knob. NOTE: the unfrozen
# backbone needs more VRAM than the frozen config, so expect to tune this.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}"

# EGL renders on the first visible GPU.
export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES%%,*}"

# FSDP shards the model across the visible GPUs; auto-count unless overridden.
# With one GPU this is 1 (no sharding).
NUM_GPUS="$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")"
FSDP_DEVICES="${FSDP_DEVICES:-$NUM_GPUS}"

# wandb: set WANDB_API_KEY, or export WANDB_MODE=offline before running.
export WANDB_MODE="${WANDB_MODE:-online}"

mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME" "$LIBERO_CONFIG_PATH" "$CKPT_BASE_DIR" \
         "$UV_CACHE_DIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" \
         "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" \
         "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

echo "[ogpo-unfrozen-backbone] node=$(hostname) gpus=$CUDA_VISIBLE_DEVICES fsdp=$FSDP_DEVICES exp=$EXP_NAME"

uv run scripts/exp_ogpo_unfrozen_backbone.py \
  pi05_libero_online_ogpo_sft_unfrozen_backbone \
  --project_name ogpo_sweep \
  --group_name ogpo_sweep_babel \
  --exp_name "$EXP_NAME" \
  --checkpoint_base_dir "$CKPT_BASE_DIR" \
  --seed 0 \
  --fsdp_devices 1 \
  --overwrite \
  --log_interval 25 \
  --save_interval 100000 \
  --num_train_steps 100000 \
  --lr_schedule.value 2.5e-5 \
  --max_runtime 169200 \
  --collect.tasks libero_90_44 \
  --collect.eval_tasks libero_90_44 \
  --collect.store_prefix_rep \
  --collect.collect_interval 10000 \
  --collect.num_rollouts 20 \
  --collect.env_num 8 \
  --collect.eval_env_num 8 \
  --collect.eval_interval 99999 \
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
  --rl.group_num_samples 1 \
  --rl.clip_epsilon 0.01 \
  --rl.bc_coeff 1.0 \
  --rl.num_sde_steps 10 \
  --rl.noise_level 0.02 \
  --rl.adv_strategy subtract_v \
  --rl.critic.value_target_type one_hot \
  --batch_size 32
