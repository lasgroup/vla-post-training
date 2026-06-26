#!/bin/bash
# One-shot environment setup for Euler (ETH Zürich).
# Run from the repo root after cloning with submodules.
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n vla-post-training python=3.12 cmake uv -c conda-forge -y
conda activate vla-post-training

# cmake>=3.27 dropped support for old cmake_minimum_required versions used by
# egl-probe (transitive dep: hf-libero -> robomimic -> egl-probe).
CMAKE_POLICY_VERSION_MINIMUM=3.5 uv sync

# openpi pins jax==0.5.3; molmospaces requires jax>=0.6.2 — --no-deps avoids
# re-resolution and installs them against the already-resolved venv.
uv pip install --no-deps -e openpi/ -e molmospaces/
