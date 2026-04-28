#!/usr/bin/env bash
# Build the shared Clariden container, import it to SquashFS, and write a base
# EDF that can be reused by branch-matched worktree wrappers.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

IMAGE_TAG="${IMAGE_TAG:-vla-post-training-shared:pi05-active-learning-docker-testing}"
BASE_ENV_NAME="${BASE_ENV_NAME:-vla-post-training-shared-pi05-al}"
ACTIVE_LEARNING_SOURCE="${ACTIVE_LEARNING_SOURCE:-${HOME}/worktrees/active_learning_vlas/docker-clariden}"
IMAGE_DIR="${IMAGE_DIR:-/capstor/scratch/cscs/${USER}/images/vla_post_training}"
BRANCH_NAME="${BRANCH_NAME:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo docker-testing)}"
BRANCH_SLUG="${BRANCH_NAME//\//-}"
SQSH_OUT="${SQSH_OUT:-${IMAGE_DIR}/vla_post_training_shared_${BRANCH_SLUG}.sqsh}"
EDF_OUT="${EDF_OUT:-${HOME}/.edf/${BASE_ENV_NAME}.toml}"
WORKTREE_MOUNT="${WORKTREE_MOUNT:-${REPO_ROOT}}"
RUN_SMOKE="${RUN_SMOKE:-1}"
PODMAN_STORAGE_DRIVER="${PODMAN_STORAGE_DRIVER:-overlay}"
PODMAN_RUNROOT="${PODMAN_RUNROOT:-/dev/shm/${USER}/runroot}"
PODMAN_GRAPHROOT="${PODMAN_GRAPHROOT:-/dev/shm/${USER}/root}"

if [[ ! -f "${ACTIVE_LEARNING_SOURCE}/pyproject.toml" ]]; then
  echo "active_learning_vlas source not found at ${ACTIVE_LEARNING_SOURCE}" >&2
  exit 1
fi

if [[ ! -d "${REPO_ROOT}/openpi/src/openpi/models_pytorch/transformers_replace" ]]; then
  echo "openpi transformers_replace files are missing; initialize submodules first" >&2
  exit 1
fi

if type module >/dev/null 2>&1; then
  module load eth_proxy >/dev/null 2>&1 || true
fi

mkdir -p "${HOME}/.config/containers" "${HOME}/.edf" "${IMAGE_DIR}" "${PODMAN_RUNROOT}" "${PODMAN_GRAPHROOT}"
cat > "${HOME}/.config/containers/storage.conf" <<STORAGE
[storage]
driver = "${PODMAN_STORAGE_DRIVER}"
runroot = "${PODMAN_RUNROOT}"
graphroot = "${PODMAN_GRAPHROOT}"
STORAGE

if command -v lfs >/dev/null 2>&1; then
  lfs setstripe -E 4G -c 1 -E -1 -c 4 "${IMAGE_DIR}" >/dev/null 2>&1 || true
fi

BUILD_CONTEXT="$(mktemp -d /dev/shm/${USER}/vla-pt-shared-build.XXXXXX)"
cleanup() {
  rm -rf "${BUILD_CONTEXT}"
}
trap cleanup EXIT

mkdir -p "${BUILD_CONTEXT}/active_learning_vlas" \
         "${BUILD_CONTEXT}/vla_post_training/openpi/src/openpi/models_pytorch"

cp "${REPO_ROOT}/docker/vla-pt.Dockerfile" "${BUILD_CONTEXT}/Dockerfile"
cp "${REPO_ROOT}/pyproject.toml" "${REPO_ROOT}/uv.lock" "${BUILD_CONTEXT}/vla_post_training/"
cp "${ACTIVE_LEARNING_SOURCE}/pyproject.toml" "${ACTIVE_LEARNING_SOURCE}/README.md" "${BUILD_CONTEXT}/active_learning_vlas/"
rsync -a --delete "${ACTIVE_LEARNING_SOURCE}/src/" "${BUILD_CONTEXT}/active_learning_vlas/src/"
rsync -a --delete \
  "${REPO_ROOT}/openpi/src/openpi/models_pytorch/transformers_replace/" \
  "${BUILD_CONTEXT}/vla_post_training/openpi/src/openpi/models_pytorch/transformers_replace/"

PODMAN_BUILD_ARGS=()
for proxy_var in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY; do
  if [[ -n "${!proxy_var:-}" ]]; then
    PODMAN_BUILD_ARGS+=(--build-arg "${proxy_var}=${!proxy_var}")
  fi
done

podman build "${PODMAN_BUILD_ARGS[@]}" -f "${BUILD_CONTEXT}/Dockerfile" -t "${IMAGE_TAG}" "${BUILD_CONTEXT}"

rm -f "${SQSH_OUT}"
if ! enroot import -x mount -o "${SQSH_OUT}" "podman://${IMAGE_TAG}"; then
  if [[ -s "${SQSH_OUT}" ]]; then
    printf 'WARNING: enroot import exited non-zero, but %s exists; continuing.\n' "${SQSH_OUT}" >&2
  else
    printf 'ERROR: enroot import failed and did not create %s.\n' "${SQSH_OUT}" >&2
    exit 1
  fi
fi

cat > "${EDF_OUT}" <<EOF
image = "${SQSH_OUT}"

mounts = [
  "${WORKTREE_MOUNT}:/app",
  "/capstor",
  "/iopsstor",
  "/users"
]

workdir = "/app"

[env]
PYTHONPATH = "/app/src:/app:/app/openpi/src:/app/openpi/packages/openpi-client/src:/app/openpi/packages/openpi-client:/app/molmospaces:/opt/active_learning_vlas/src"
HF_HOME = "/capstor/scratch/cscs/\${USER}/huggingface"
WANDB_DIR = "/app/wandb"
WANDB_PROJECT = "vla-post-training"
MUJOCO_GL = "egl"
PYOPENGL_PLATFORM = "egl"
EGL_PLATFORM = "surfaceless"
LIBERO_CONFIG_PATH = "/etc/libero"
TOKENIZERS_PARALLELISM = "false"
GIT_LFS_SKIP_SMUDGE = "1"
EOF

printf 'Built image: %s\n' "${IMAGE_TAG}"
printf 'Imported SquashFS: %s\n' "${SQSH_OUT}"
printf 'Wrote base EDF: %s\n' "${EDF_OUT}"

if [[ "${RUN_SMOKE}" == "1" ]]; then
  srun --environment="${EDF_OUT}" python scripts/clariden/smoke_pi05_container.py
fi
