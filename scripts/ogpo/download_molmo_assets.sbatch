#!/bin/bash
#SBATCH --job-name=molmo_dl
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
# Add --partition / --account for your cluster.
# Assets for the molmo_pi05 recipes (scripts/ogpo_pi05/molmo_*.sh, scripts/awr_pi05/molmo_*.sh), idempotent:
#   1. pi05_droid_jointpos (11.6 GiB) from gs://openpi-assets and the PaliGemma tokenizer from gs://big_vision (needs gsutil)
#   2. MolmoSpaces base assets (~10 GB cache: robots, benchmarks, objects, grasps; cold ~20 min) + NLTK wordnet
#   3. the scenes / objects / grasps of TASKS (benchmark episode ids; houses are otherwise fetched lazily at reset)
# Where login nodes cap virtual memory (a 16 GB ulimit -v breaks molmo's LMDB maps), run it on a compute node.
# Run it from the OGPO checkout after scripts/ogpo/setup_molmo_venv.sh. Required environment, no defaults: OPENPI_CACHE (receives
# openpi-assets/ and big_vision/), MLSPACES_ASSETS_DIR, MLSPACES_CACHE_DIR, NLTK_DATA (the recipes need the same four).
# TASKS defaults to the 16 tasks of the mt16 recipes (the single-task recipe's task, 138, is one of them).
#   OPENPI_CACHE=... MLSPACES_ASSETS_DIR=... MLSPACES_CACHE_DIR=... NLTK_DATA=... sbatch scripts/ogpo/download_molmo_assets.sbatch
#   TASKS=164,86,... sbatch scripts/ogpo/download_molmo_assets.sbatch
mkdir -p logs
set -euo pipefail
TASKS=${TASKS:-267,366,138,143,176,373,93,394,48,185,24,62,4,260,221,348}
GSUTIL=${GSUTIL:-gsutil}
: "${OPENPI_CACHE:?set OPENPI_CACHE (receives openpi-assets/ and big_vision/)}"
: "${MLSPACES_ASSETS_DIR:?set MLSPACES_ASSETS_DIR}"
: "${MLSPACES_CACHE_DIR:?set MLSPACES_CACHE_DIR}"
: "${NLTK_DATA:?set NLTK_DATA}"
export MLSPACES_ASSETS_DIR MLSPACES_CACHE_DIR NLTK_DATA
export JAX_PLATFORMS=cpu PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$MLSPACES_ASSETS_DIR" "$MLSPACES_CACHE_DIR" "$NLTK_DATA" "$OPENPI_CACHE/openpi-assets/checkpoints" "$OPENPI_CACHE/big_vision"

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
[ -f "$ROOT/.venv-molmo/bin/python" ] || { echo "ERROR: submit from the OGPO checkout with .venv-molmo (scripts/ogpo/setup_molmo_venv.sh)"; exit 1; }
cd "$ROOT"
echo "[molmo_dl] host $(hostname) | ulimit -v $(ulimit -v) | $(df -h "$MLSPACES_CACHE_DIR" | tail -1)"
need_gsutil() { command -v "$GSUTIL" >/dev/null || { echo "ERROR: gsutil not found (set GSUTIL=/path/to/gsutil)"; exit 1; }; }

CKPT="$OPENPI_CACHE/openpi-assets/checkpoints/pi05_droid_jointpos"
if [ -f "$CKPT/assets/droid/norm_stats.json" ] && [ -d "$CKPT/params" ]; then
    echo "[molmo_dl] pi05_droid_jointpos already present"
else
    need_gsutil
    "$GSUTIL" -m cp -r gs://openpi-assets/checkpoints/pi05_droid_jointpos "$OPENPI_CACHE/openpi-assets/checkpoints/"
fi
du -sh "$CKPT"

TOK="$OPENPI_CACHE/big_vision/paligemma_tokenizer.model"
if [ -f "$TOK" ]; then
    echo "[molmo_dl] paligemma tokenizer already present"
else
    need_gsutil
    "$GSUTIL" cp gs://big_vision/paligemma_tokenizer.model "$OPENPI_CACHE/big_vision/"
fi
du -sh "$TOK"

.venv-molmo/bin/python scripts/ogpo/molmo_prefetch.py --tasks "$TASKS"
du -sh "$MLSPACES_CACHE_DIR" "$MLSPACES_ASSETS_DIR" "$NLTK_DATA"
echo DOWNLOAD_DONE
