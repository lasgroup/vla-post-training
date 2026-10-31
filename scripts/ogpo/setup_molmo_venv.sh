#!/bin/bash
# Build .venv-molmo, the Python env for molmo_pi05 runs:
# OGPO's exact locked deps (uv.lock, base + cuda12, so jax 0.9.0.1 as in .venv) with mujoco overridden to 3.4.0, plus
# MolmoSpaces' runtime deps (molmo_venv/requirements.txt: mujoco-mjx 3.4.0, jaxlie, CPU torch 2.8, ...) and
# molmospaces + openpi-client + OGPO installed editable WITHOUT their own pins (molmospaces pins jax~=0.6.2; its
# mjx IK runs on jax 0.9, checked by the import smoke). pyproject.toml / uv.lock are untouched; the LIBERO / robomimic
# extras are not installed here. Expected `uv pip check` complaints: molmo-spaces' skipped pins (jax, beaker-py,
# nbstripout, open-clip, pynput, torchvision) and dm-control wanting mujoco>=3.5 (not on the molmo path).
#
#   bash scripts/ogpo/setup_molmo_venv.sh --molmospaces=<path> --openpi_client=<path> [--venv=<dir>] [--python=3.12]
# --molmospaces (required): a lasgroup/molmospaces checkout at commit 2a828d5, the commit the molmo recipes were run with.
# --openpi_client (required): openpi/packages/openpi-client from an openpi checkout (imported by molmo_spaces' policy configs).
# --venv: where to build it (default <checkout>/.venv-molmo, where scripts/{ogpo,awr}_pi05/molmo_*.sh look for it).
set -euo pipefail
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
molmospaces=""
openpi_client=""
venv="$ROOT/.venv-molmo"
python=3.12
for arg in "$@"; do
  case $arg in
    --*=*) key="${arg%%=*}"; key="${key#--}"; eval "$key=\"${arg#*=}\"" ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done
[ -n "$molmospaces" ] || { echo "ERROR: --molmospaces=<path> is required: a lasgroup/molmospaces checkout at commit 2a828d5"; exit 1; }
[ -n "$openpi_client" ] || { echo "ERROR: --openpi_client=<path> is required: openpi/packages/openpi-client from an openpi checkout"; exit 1; }
for d in "$molmospaces/molmo_spaces" "$openpi_client/pyproject.toml"; do
    [ -e "$d" ] || { echo "ERROR: missing $d (--molmospaces= must be the molmospaces checkout, --openpi_client= the openpi-client package dir)"; exit 1; }
done

molmospaces="$(realpath "$molmospaces")"; openpi_client="$(realpath "$openpi_client")"; venv="$(realpath -m "$venv")"  # relative to the caller's cwd
cd "$ROOT"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
uv export --frozen --no-hashes --extra cuda12 --no-emit-project --no-header | grep -v '^mujoco==' > "$tmp/ogpo_base.txt"
uv venv -q "$venv" --python "$python"
uv pip install -q --python "$venv/bin/python" \
    -r "$tmp/ogpo_base.txt" -r scripts/ogpo/molmo_venv/requirements.txt -e "$openpi_client" \
    --override scripts/ogpo/molmo_venv/overrides.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
uv pip install -q --python "$venv/bin/python" --no-deps -e "$molmospaces" -e .
"$venv/bin/python" -c "import jax, mujoco; print(f'[setup_molmo_venv] jax {jax.__version__}, mujoco {mujoco.__version__}')"
uv pip check --python "$venv/bin/python" || true  # expected complaints listed above
