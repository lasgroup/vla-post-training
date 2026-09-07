#!/bin/bash
#SBATCH --partition=maxlab
#SBATCH --qos=maxlab_qos
#SBATCH --nodelist=babel-m9-16
#SBATCH --job-name=molmo_install
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/home/mananaga/logs/%j/.out
#SBATCH --error=/home/mananaga/logs/%j/.out
# Downloads the MolmoSpaces base assets (robot, THOR objects, DROID grasps,
# MS-Bench v1 JSONs, ~3.7 GB) into the paths scripts/ogpo_molmo.sh defaults
# to. ProcTHOR houses and Objaverse objects are fetched lazily at run time.
# Run on a compute node: /data/user_data is not mounted on the login node.
#
# Usage:  bash scripts/install_molmo.sh      (or sbatch scripts/install_molmo.sh)
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/mananaga/VLA/ogpo/vla-post-training}"
STORE_ROOT="${STORE_ROOT:-/data/user_data/mananaga/vla-post-training}"
PY="${PY:-/home/mananaga/VLA/manan_babel/vla-post-training/.venv/bin/python}"
[ -x "$PY" ] || { echo "[install] no interpreter at $PY" >&2; exit 1; }

export MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-$STORE_ROOT/molmospaces/assets}"
export MLSPACES_CACHE_DIR="${MLSPACES_CACHE_DIR:-$STORE_ROOT/cache/molmo-spaces-resources}"
export MLSPACES_AUTO_INSTALL=True
mkdir -p "$MLSPACES_ASSETS_DIR" "$MLSPACES_CACHE_DIR"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/openpi/src:$PROJECT_DIR/openpi/packages/openpi-client/src:$PROJECT_DIR/molmospaces"

echo "[install] node=$(hostname) assets=$MLSPACES_ASSETS_DIR cache=$MLSPACES_CACHE_DIR"
"$PY" -c "from molmo_spaces.molmo_spaces_constants import get_resource_manager; get_resource_manager()"

BENCH="$MLSPACES_ASSETS_DIR/benchmarks/molmospaces-bench-v1/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231"
if [ -d "$BENCH" ]; then
  N=$("$PY" -c "
from pathlib import Path
from molmo_spaces.evaluation.benchmark_schema import load_all_episodes
print(len(load_all_episodes(Path('$BENCH'))))")
  echo "[install] OK benchmark at $BENCH ($N episodes)"
else
  echo "[install] benchmark not at the expected path; contents of benchmarks/:" >&2
  find "$MLSPACES_ASSETS_DIR/benchmarks/" -maxdepth 4 | head -30 >&2
  echo "[install] set MLSPACES_BENCHMARK_DIR to the dir holding house_*/episode_*.json before running ogpo_molmo.sh" >&2
  exit 1
fi
