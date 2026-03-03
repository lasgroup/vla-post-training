# Dockerfile for custon CSCS container

# Build the container:
# cd vla-post-training
# podman build . -f docker/vla-pt.Dockerfile -t vla-pt
# enroot import -x mount -o vla-pt.sqsh podman://vla-pt:latest
# mv vla-pt.sqsh /capstor/store/cscs/swissai/a143/project-vla-pt

# Run the container:
# podman run --rm -it --network=host -v .:/app --gpus=all vla-pt zsh

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

WORKDIR /app

# Needed because LeRobot uses git-lfs.
RUN apt-get update && apt-get install -y git git-lfs linux-headers-generic build-essential clang nano tmux zsh

# LIBERO dependencies
RUN apt-get update && \
    apt-get install -y \
    make \
    g++ \
    clang \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libegl1 \
    libglew-dev \
    libglfw3-dev \
    libgles2-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    cmake

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Write the virtual environment outside of the project directory so it doesn't
# leak out of the container when we mount the application code.
ENV UV_PROJECT_ENVIRONMENT=/.venv

# Install the project's dependencies using the lockfile and settings
RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen

# Copy transformers_replace files while preserving directory structure
COPY openpi/src/openpi/models_pytorch/transformers_replace/ /tmp/transformers_replace/
RUN /.venv/bin/python -c "import transformers; print(transformers.__file__)" | xargs dirname | xargs -I{} cp -r /tmp/transformers_replace/* {} && rm -rf /tmp/transformers_replace

# Add openpi, openpi-client, MolmoSpaces to PATH
ENV PYTHONPATH=/app:/app/openpi/packages/openpi-client/src:/app/openpi/src:/app/openpi/packages/openpi-client:/app/molmospaces

# Setup macros for robosuite
RUN uv run /.venv/lib/python3.11/site-packages/robosuite/scripts/setup_macros.py

# Update EGL vendor
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && echo '{"file_format_version" : "1.0.0", "ICD" : { "library_path" : "libEGL_nvidia.so.0" }}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Create a default config file to avoid an input prompt from LIBERO's init script.
# https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/__init__.py
ENV LIBERO_CONFIG_PATH=/etc/libero
RUN mkdir -p /etc/libero && cat <<'EOF2' > /etc/libero/config.yaml
benchmark_root: /.venv/lib/python3.11/site-packages/libero/libero
bddl_files: /.venv/lib/python3.11/site-packages/libero/libero/./bddl_files
init_states: /.venv/lib/python3.11/site-packages/libero/libero/./init_files
datasets: /.venv/lib/python3.11/site-packages/libero/libero/../datasets
assets: /.venv/lib/python3.11/site-packages/libero/libero/./assets
EOF2
