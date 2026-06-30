#!/bin/bash
# One-shot environment setup for Euler (ETH Zürich).
# Run from the repo root after cloning with submodules.
#
# NOTE: conda activate inside this script does NOT affect your calling shell.
# After this script finishes, run manually:
#   conda activate vla-post-training
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | grep -q "^vla-post-training "; then
    echo "Conda env 'vla-post-training' already exists, skipping creation."
else
    conda create -n vla-post-training python=3.12 cmake uv -c conda-forge -y
fi
conda activate vla-post-training

# cmake>=3.27 dropped support for old cmake_minimum_required versions used by
# egl-probe (transitive dep: hf-libero -> robomimic -> egl-probe).
CMAKE_POLICY_VERSION_MINIMUM=3.5 uv sync

# openpi pins jax==0.5.3; molmospaces requires jax>=0.6.2 — --no-deps avoids
# re-resolution and installs them against the already-resolved venv.
# Re-run this line after any `uv sync` that wipes these editable installs.
uv pip install --no-deps -e openpi/ -e molmospaces/

echo ""
echo "Setup complete. Activate the environment with:"
echo "  conda activate vla-post-training"
