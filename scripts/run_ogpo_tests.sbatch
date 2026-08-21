#!/bin/bash
#SBATCH --partition=general
#SBATCH --job-name=ogpo_tests
# The general partition refuses jobs with no GPU ("requires a minimum of 1 GPU"),
# so one is requested purely to satisfy the scheduler. The suite itself stays on
# CPU via JAX_PLATFORMS=cpu below -- it is CPU-only by design (dummy PaliGemma
# variants) and must not silently start exercising a device.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/home/pchellap/logs/%x_%j.out
#SBATCH --error=/home/pchellap/logs/%x_%j.out
# ---------------------------------------------------------------------------
# CPU-only runner for `pytest tests/ogpo` on a compute node.
#
# WHY THIS EXISTS: on the login node `pytest tests/ogpo` cannot complete.
# XLA sizes its LLVM codegen thread pool from nproc (=24 there) while the
# per-user RLIMIT_NPROC is 1000 (hard cap 2000), so compilation dies with
#   LLVM ERROR: pthread_create failed: Resource temporarily unavailable
# Constraining visible CPUs avoids it, but --xla_cpu_parallel_codegen_split_count=1
# is too tight for the larger graphs and instead produces
#   XlaRuntimeError: INTERNAL: Too many extra compilation parts ...
# A dedicated allocation with a clean cgroup and its own CPU set avoids both.
#
# The suite is CPU-only by design (dummy PaliGemma variants), so no GPU is
# requested. It does NOT train anything and does NOT touch checkpoints.
#
# Submit:  sbatch scripts/run_ogpo_tests.sbatch
# Subset:  sbatch --export=ALL,PYTEST_TARGET=tests/ogpo/test_sampling.py \
#              scripts/run_ogpo_tests.sbatch
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_DIR=/home/pchellap/Projects/OGPO-VLA/vla-post-training
cd "$PROJECT_DIR"

export PATH="$HOME/.local/bin:$PATH"
STORE_ROOT="${STORE_ROOT:-$PROJECT_DIR/run_store}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$STORE_ROOT/cache/uv}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$STORE_ROOT/cache/xdg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$STORE_ROOT/cache/matplotlib}"
export HF_HOME="${HF_HOME:-$STORE_ROOT/cache/huggingface}"

# Keep JAX on CPU: the suite never needs a device, and asking for one on a
# GPU-less allocation just adds a slow backend-probe failure.
export JAX_PLATFORMS=cpu
# Match the intra-op pools to the allocation rather than the physical node, so
# XLA does not size its pools from the machine's full core count.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

PYTEST_TARGET="${PYTEST_TARGET:-tests/ogpo}"

echo "[tests] node=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} ulimit-u=$(ulimit -u) target=$PYTEST_TARGET"
echo "[tests] git: $(git rev-parse --short HEAD) (+ uncommitted working tree)"

uv run pytest "$PYTEST_TARGET" -q -p no:cacheprovider --durations=10
