# Clariden/GH200 shared image for:
# 1. vla-post-training/docker-testing pi0.5 smokes
# 2. active_learning_vlas/docker-clariden smokes
#
# This Dockerfile is built from a staged context prepared by
# scripts/clariden/build_shared_clariden_container.sh. The context contains:
# - active_learning_vlas/{pyproject.toml,README.md,src}
# - vla_post_training/{pyproject.toml,uv.lock,openpi/src/.../transformers_replace}

FROM nvcr.io/nvidia/pytorch:25.04-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    EGL_PLATFORM=surfaceless \
    LIBERO_CONFIG_PATH=/etc/libero \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
      bash \
      build-essential \
      ca-certificates \
      clang \
      cmake \
      curl \
      ffmpeg \
      git \
      git-lfs \
      g++ \
      libasound2t64 \
      libegl1 \
      libegl1-mesa-dev \
      libgl1 \
      libgl1-mesa-dev \
      libglib2.0-0 \
      libglvnd0 \
      libglew-dev \
      libglfw3-dev \
      libosmesa6 \
      libsm6 \
      libusb-1.0-0 \
      libx11-6 \
      libx11-dev \
      libxcursor1 \
      libxext6 \
      libxext-dev \
      libxi6 \
      libxinerama1 \
      libxkbcommon0 \
      libxrandr2 \
      libxrender1 \
      libxtst6 \
      linux-headers-generic \
      make \
      nano \
      ninja-build \
      openssh-client \
      patchelf \
      pkg-config \
      tini \
      tmux \
      zsh \
    && rm -rf /var/lib/apt/lists/*

RUN PIP_CONSTRAINT= python -m pip install --upgrade pip setuptools wheel "packaging==24.2"

WORKDIR /opt/active_learning_vlas
COPY active_learning_vlas/pyproject.toml active_learning_vlas/README.md ./
COPY active_learning_vlas/src ./src

# Install the active-learning repo without its custom transformers dependency,
# then layer the OpenPI-compatible transformers patch used by vla-post-training.
RUN PIP_CONSTRAINT= python -m pip install -e ".[smolvla,libero,uncertainty]" \
    && PIP_CONSTRAINT= python -m pip install \
      "transformers==4.53.2" \
      "tokenizers>=0.21,<0.22" \
      seaborn \
      python-dotenv

# Add the extra runtime pieces needed by the vla-post-training/OpenPI stack
# without replacing the NGC torch build or the active_learning_vlas source.
RUN PIP_CONSTRAINT= python -m pip install \
      "augmax==0.4.1" \
      "beartype==0.19.0" \
      "compress-json==1.1.1" \
      "dm-tree==0.1.8" \
      "einops==0.8.1" \
      "equinox==0.12.2" \
      "flatbuffers>=25.2.10" \
      "flax==0.10.6" \
      "fsspec[gcs]==2025.3.0" \
      "gym-aloha==0.1.2" \
      "imageio==2.37.0" \
      "jax[cuda12]==0.5.3" \
      "jaxlie==1.5.0" \
      "jaxtyping==0.2.36" \
      "lmdb" \
      "ml_collections==1.0.0" \
      "mujoco==3.4.0" \
      "mujoco-mjx==3.4.0" \
      "nltk==3.9.2" \
      "numpy==1.26.4" \
      "numpydantic==1.6.9" \
      "orbax-checkpoint==0.11.13" \
      "pillow==11.2.1" \
      "polars==1.30.0" \
      "rich==14.0.0" \
      "scikit-image" \
      "sentencepiece==0.2.0" \
      "sentry_sdk==2.0.0" \
      "stringcase==1.2.0" \
      "teledex" \
      "tensorflow-probability>=0.25.0" \
      "treescope==0.1.9" \
      "tqdm-loggable==0.2" \
      "tyro==0.9.22" \
      "wandb==0.21.4" \
      "zstandard"

COPY vla_post_training/openpi/src/openpi/models_pytorch/transformers_replace/ /tmp/transformers_replace/
RUN python -c "import transformers; print(transformers.__file__)" \
    | xargs dirname \
    | xargs -I{} cp -r /tmp/transformers_replace/* {} \
    && rm -rf /tmp/transformers_replace

# Avoid interactive LIBERO setup and force EGL in Slurm/container-engine jobs.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d /etc/libero \
    && printf '%s\n' '{"file_format_version" : "1.0.0", "ICD" : { "library_path" : "libEGL_nvidia.so.0" }}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    && cat > /etc/libero/config.yaml <<'EOF'
benchmark_root: /usr/local/lib/python3.12/dist-packages/libero
datasets: /usr/local/lib/python3.12/dist-packages/libero/../datasets
bddl_files: /usr/local/lib/python3.12/dist-packages/libero/bddl_files
init_states: /usr/local/lib/python3.12/dist-packages/libero/init_files
assets: /usr/local/lib/python3.12/dist-packages/libero/assets
EOF

ENV PYTHONPATH=/app/src:/app:/app/openpi/src:/app/openpi/packages/openpi-client/src:/app/openpi/packages/openpi-client:/app/molmospaces:/opt/active_learning_vlas/src

WORKDIR /app
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
