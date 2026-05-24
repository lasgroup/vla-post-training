#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "$SCRIPT_DIR/.env" ]]; then
  set -a
  source "$SCRIPT_DIR/.env"
  set +a
fi

export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
export PYTHONPATH=.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

cd "$SCRIPT_DIR"
uv run scripts/best_of_n_agent/exp.py pi05_libero_online_best_of_n_debug \
  --exp-name test \
  --overwrite
