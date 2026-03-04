#!/bin/bash
# ────────────────────────────────────────────────────────────────────
# Shared environment setup for vla-post-training HTCondor jobs.
# Sourced by per-algorithm .sh scripts (not executed directly).
# ────────────────────────────────────────────────────────────────────
set -e

# ── Shell / virtualenv ──────────────────────────────────────────────
export PATH="/home/malbaba/python/bin:$PATH"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/usr/lib/nvidia"
export WORKON_HOME=/home/malbaba/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/home/malbaba/python/bin/python3
source /home/malbaba/python/bin/virtualenvwrapper.sh
source /home/malbaba/.virtualenvs/vlapt/bin/activate

# ── Project root ────────────────────────────────────────────────────
REPO_ROOT="/lustre/home/malbaba/vla-post-training"
cd "${REPO_ROOT}"

# ── GPU selection (HTCondor → CUDA_VISIBLE_DEVICES → fallback) ─────
default_gpu_ids=0
if [ -n "${GPU_IDS:-}" ]; then
  gpu_ids="${GPU_IDS}"
elif [ -n "${_CONDOR_AssignedGPUs:-}" ]; then
  gpu_ids="${_CONDOR_AssignedGPUs// /}"
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  gpu_ids="${CUDA_VISIBLE_DEVICES}"
else
  gpu_ids="${default_gpu_ids}"
fi
export CUDA_VISIBLE_DEVICES="${gpu_ids}"

# Resolve GPU-UUID form to numeric indices (needed by robosuite/mujoco)
if [[ "${CUDA_VISIBLE_DEVICES}" == *GPU-* ]]; then
  _resolved_gpu_ids="$(for tok in ${CUDA_VISIBLE_DEVICES//,/ }; do
    nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
      | awk -F", " -v u="$tok" '$2==u || index($2,u)==1 {print $1; exit}'
  done | paste -sd, -)"
  if [[ -n "${_resolved_gpu_ids}" ]]; then
    export CUDA_VISIBLE_DEVICES="${_resolved_gpu_ids}"
  fi
fi
_first_visible_gpu="${CUDA_VISIBLE_DEVICES%%,*}"
export MUJOCO_EGL_DEVICE_ID="${_first_visible_gpu:-0}"

# ── Display / rendering ────────────────────────────────────────────
export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# ── OpenPI assets & caches (local, avoid downloads) ────────────────
export OPENPI_DATA_HOME=/fast/malbaba/vlapt_cache/openpi_cache
export OPENPI_ASSETS_DIR=${OPENPI_DATA_HOME}/openpi-assets/checkpoints/pi05_libero/assets
export OPENPI_POLICY_CHECKPOINT_DIR=${OPENPI_DATA_HOME}/openpi-assets/checkpoints/pi05_libero

# ── HuggingFace / LeRobot caches ───────────────────────────────────
export HOME=/home/malbaba
export HF_HOME=/fast/malbaba/vlapt_cache/hfcache
export HUGGINGFACE_HUB_CACHE=/fast/malbaba/vlapt_cache/hfcache/hub
export HF_DATASETS_CACHE=/fast/malbaba/vlapt_cache/hfcache/datasets
export HF_LEROBOT_HOME=/fast/malbaba/vlapt_cache/hfcache/lerobot
export HF_DATASETS_USE_SOFT_FILELOCK=1
export HF_DATASETS_DISABLE_LOCKING=1
export XDG_CACHE_HOME=/fast/malbaba/vlapt_cache/xdg_cache

# ── Offline mode (all data is pre-cached, avoid Hub calls that hang) ─
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ── LIBERO paths ───────────────────────────────────────────────────
export LIBERO_CONFIG_PATH=/fast/malbaba/vlapt_cache/libero_config

# ── JAX / XLA settings ────────────────────────────────────────────
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# ── PYTHONPATH ─────────────────────────────────────────────────────
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/openpi/packages/openpi-client/src:${REPO_ROOT}/openpi/src:${REPO_ROOT}/openpi/packages/openpi-client:${PYTHONPATH:-}"

# ── Checkpoints ────────────────────────────────────────────────────
export CHECKPOINT_BASE_DIR="/fast/malbaba/vlapt_cache/checkpoints"

echo "──────────────────────────────────────────────────"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "_CONDOR_AssignedGPUs=${_CONDOR_AssignedGPUs:-<unset>}"
echo "MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID}"
echo "SEED=${SEED:-<unset>}"
echo "──────────────────────────────────────────────────"
