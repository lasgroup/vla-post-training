#!/bin/bash
# Wrapper for scripts/egl_safe_probe.py that reproduces the EXACT env of
# scripts/awr_libero_babel.sh (PYTHONPATH, LIBERO/EGL vars), minus JAX's global
# mem-fraction preallocation (the probe controls VRAM pressure itself via
# --pin-vram-frac). Safe to run on a throwaway node: the probe self-reaps and a
# top-level watchdog SIGKILLs everything on its deadline, so it cannot drain the
# node.
#
# Examples:
#   # 1. Sanity: does EGL offscreen render work with full headroom?
#   scripts/egl_probe.sh --mode probe --env-num 8 --pin-vram-frac 0.0
#   # 2. Reproduce the 0x8cdd crash the way the real run starves VRAM:
#   scripts/egl_probe.sh --mode probe --env-num 8 --pin-vram-frac 0.95
#   # 3. Confirm 0.75 leaves enough headroom (should pass):
#   scripts/egl_probe.sh --mode probe --env-num 8 --pin-vram-frac 0.75
#   # 4. Validate close() never hangs even under the crash:
#   scripts/egl_probe.sh --mode teardown --env-num 8 --pin-vram-frac 0.95
#
# On maxlab later (4 GPUs): add --num-devices 4 and set CUDA_VISIBLE_DEVICES.
set -euo pipefail

PROJECT_DIR=/home/mananaga/vla-post-training
STORE_ROOT=/data/user_data/mananaga/vla-post-training
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

# IMPORTANT: do NOT export XLA_PYTHON_CLIENT_MEM_FRACTION here. The probe pins
# VRAM itself (--pin-vram-frac) in an isolated process so we can sweep the value.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES%%,*}"

echo "[egl_probe] node=$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID=$MUJOCO_EGL_DEVICE_ID"
exec uv run scripts/egl_safe_probe.py "$@"
