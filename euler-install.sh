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

# Install into the conda env, not a separate .venv, so `python scripts/...`
# just works.
export UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX"

# cmake>=3.27 dropped support for old cmake_minimum_required versions used by
# egl-probe (transitive dep: hf-libero -> robomimic -> egl-probe).
CMAKE_POLICY_VERSION_MINIMUM=3.5 uv sync

echo ""
echo "Setup complete. Activate the environment with:"
echo "  conda activate vla-post-training"
echo ""
echo "With the env active, run scripts with plain 'python' (NOT 'uv run', which"
echo "still resolves against a fresh .venv unless UV_PROJECT_ENVIRONMENT is set"
echo "in your shell too):"
echo "  python scripts/exp.py ...        # or python scripts/launcher.py ..."
echo "Submit Euler jobs from the activated env: the launcher emits 'python ...' and"
echo "sbatch (--export=ALL) inherits your active conda env on the compute node."
