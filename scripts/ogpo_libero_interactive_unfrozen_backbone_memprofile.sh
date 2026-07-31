#!/bin/bash
# ---------------------------------------------------------------------------
# MEMORY PROFILING launcher for the unfrozen-backbone OGPO run.
#
#     bash scripts/ogpo_libero_interactive_unfrozen_backbone_memprofile.sh
#
# Goal: find EXACTLY which tensor/op inside the policy update allocates the
# ~135 GB buffer that OOMs, and measure how it scales, before choosing a fix.
#
# It is scripts/ogpo_libero_interactive_unfrozen_backbone.sh with:
#   1. PROFILING env (marked below): don't preallocate, so the OOM lands at the
#      real allocation and the runtime prints its "Peak buffers" report; and
#      dump XLA HLO + buffer-assignment so the giant buffer maps back to an op
#      and source line.
#   2. A FAST path to the memory-heavy step (first POLICY update): lower
#      policy.training_start_step / update_interval, cut num_train_steps. This
#      changes only WHEN that step runs, not its memory.
#   3. NUM_SDE_STEPS and BATCH_SIZE exposed as env vars (defaults = the REAL
#      values 10 / 32, so the first run reproduces the real 135 GB peak). Sweep
#      them to confirm scaling and size the knobs, e.g.:
#         NUM_SDE_STEPS=2 bash scripts/..._memprofile.sh   # expect peak ~1/5
#         BATCH_SIZE=8     bash scripts/..._memprofile.sh   # expect peak ~1/4
#
# The MuJoCo/EGL rendering setup is IDENTICAL to the base launcher (untouched).
#
# HOW TO READ THE OUTPUT (see the note printed at the end of this script too):
#   * stderr, right after "RESOURCE_EXHAUSTED / ran out of memory": a list of
#     the largest live buffers with sizes + shapes. The ~135 GiB f32[...] entry
#     is the culprit.
#   * $HLO_DUMP_DIR: grep the *buffer-assignment* / *after_optimizations* files
#     for the huge f32[...] shape and read its metadata={op_name=...,
#     source_file=..., source_line=...} to land on the exact line (expect the
#     gemma attention einsum/softmax, gemma.py:217/228, reached via
#     src/rl/ogpo/sampling.py score_chain_under_model / the BC compute_loss).
# ---------------------------------------------------------------------------

set -euo pipefail

# ===========================================================================
# CONFIG -- edit these for your node
# ===========================================================================
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
STORE_ROOT="${STORE_ROOT:-$PROJECT_DIR/run_store}"

EXP_NAME="${EXP_NAME:-pi05_ogpo_unfrozen_backbone_MEMPROFILE}"
CKPT_BASE_DIR="${CKPT_BASE_DIR:-$STORE_ROOT/checkpoints/ogpo_memprofile}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# PROFILING knobs to sweep. Defaults reproduce the real 135 GB peak.
NUM_SDE_STEPS="${NUM_SDE_STEPS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"

# Where XLA writes the HLO + buffer-assignment dump for this run.
HLO_DUMP_DIR="${HLO_DUMP_DIR:-$STORE_ROOT/hlo_dump/$EXP_NAME}"

# ===========================================================================
# (Usually no edits needed below this line)
# ===========================================================================
cd "$PROJECT_DIR"

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

# ===========================================================================
# PROFILING env (the point of this script)
# ===========================================================================
# Don't preallocate: allocate on demand so the OOM lands at the real allocation
# and the runtime prints its "Peak buffers" report naming the largest buffers.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Let JAX allocate near capacity so the peak reflects the true requirement.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
# Dump HLO + buffer-assignment so the giant buffer maps to an op + source line.
mkdir -p "$HLO_DUMP_DIR"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_dump_to=$HLO_DUMP_DIR --xla_dump_hlo_as_text"
# Full C++ logs (incl. the peak-buffer report) and un-filtered Python traceback.
export TF_CPP_MIN_LOG_LEVEL=0
export JAX_TRACEBACK_FILTERING=off

export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES%%,*}"

NUM_GPUS="$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")"
FSDP_DEVICES="${FSDP_DEVICES:-$NUM_GPUS}"

# Profiling run: no wandb.
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME" "$LIBERO_CONFIG_PATH" "$CKPT_BASE_DIR" \
         "$UV_CACHE_DIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" \
         "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" \
         "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

echo "[ogpo-memprofile] node=$(hostname) gpus=$CUDA_VISIBLE_DEVICES fsdp=$FSDP_DEVICES num_sde_steps=$NUM_SDE_STEPS batch_size=$BATCH_SIZE"
echo "[ogpo-memprofile] HLO dump -> $HLO_DUMP_DIR"
echo "[ogpo-memprofile] watch VRAM in another terminal with:  nvidia-smi -l 1"

uv run scripts/exp_ogpo_unfrozen_backbone.py \
  pi05_libero_online_ogpo_sft_unfrozen_backbone \
  --project_name ogpo_sweep \
  --group_name ogpo_memprofile \
  --exp_name "$EXP_NAME" \
  --checkpoint_base_dir "$CKPT_BASE_DIR" \
  --seed 0 \
  --fsdp_devices 1 \
  --overwrite \
  --log_interval 1 \
  --save_interval 100000 \
  --num_train_steps 3 \
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
  --rl.policy.update_interval 1 \
  --rl.policy.training_start_step 1 \
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
  --rl.num_sde_steps "$NUM_SDE_STEPS" \
  --rl.noise_level 0.02 \
  --rl.adv_strategy subtract_v \
  --rl.critic.value_target_type one_hot \
  --batch_size "$BATCH_SIZE"

# ---------------------------------------------------------------------------
# After the run (it is EXPECTED to OOM at the policy update on the first pass):
#
#   # 1. The largest buffers XLA tried to hold (the ~135 GiB one is the culprit):
#   #    look in the console stderr for "Peak buffers" / "ran out of memory".
#
#   # 2. Map that buffer to an op + source line from the HLO dump:
#   grep -rE "f32\[[0-9,]+\]" "$HLO_DUMP_DIR"/*buffer-assignment* | sort -t'[' -k2 -rn | head
#   #    then open the module .txt and read metadata={op_name=..., source_line=...}
#
#   # 3. Confirm scaling to size the fix (peak should track num_sde_steps * batch):
#   NUM_SDE_STEPS=2 bash scripts/ogpo_libero_interactive_unfrozen_backbone_memprofile.sh
#   BATCH_SIZE=8     bash scripts/ogpo_libero_interactive_unfrozen_backbone_memprofile.sh
# ---------------------------------------------------------------------------
