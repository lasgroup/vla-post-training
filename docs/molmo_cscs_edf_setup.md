# MolmoSpaces CSCS / EDF Setup

This repo commonly runs MolmoSpaces through an EDF `.toml` file. To create a new setup, make a copy of your existing EDF config and save it under `~/.edf/`, for example:

```bash
cp ~/.edf/vla-post-training.toml ~/.edf/vla-post-training-molmo-spaces.toml
```

If you already have a working MolmoSpaces config, you can also copy that one and adjust only the parts you need.

## No install step needed

There is no need to run the Molmo install script in this repo. The shared MolmoSpaces assets and cache are already set up for everyone.

In particular, you do not need to run `scripts/install_molmo_assets.py` when using the shared EDF setup below.

## Minimal MolmoSpaces EDF config

Use a config like the one below and keep the `MLSPACES_*` paths as shown so you reuse the shared MolmoSpaces install:

```toml
# path to the container image
image = "/capstor/store/cscs/swissai/a143/project-vla-pt/vla-post-training-molmo.sqsh"

mounts = [
  "/users/${USER}/vla-post-training:/app",
  "/capstor",
  "/iopsstor",
  "/users",
]

workdir = "/app"

[env]
OPENPI_DATA_HOME = "/capstor/scratch/cscs/${USER}/openpi_assets"
HF_HOME = "/capstor/scratch/cscs/${USER}/huggingface"
HF_TOKEN = "<your_hf_token>"
LD_LIBRARY_PATH = "/usr/lib64:${LD_LIBRARY_PATH:-}"
MUJOCO_GL = "egl"
MLSPACES_ASSETS_DIR = "/capstor/store/cscs/swissai/a143/molmospaces/assets"
MLSPACES_CACHE_DIR = "/capstor/store/cscs/swissai/a143/molmospaces/cache"
XLA_PYTHON_CLIENT_MEM_FRACTION = "0.9"
```

## Shared MolmoSpaces paths

These two paths are already shared and should be reused:

```toml
MLSPACES_ASSETS_DIR = "/capstor/store/cscs/swissai/a143/molmospaces/assets"
MLSPACES_CACHE_DIR = "/capstor/store/cscs/swissai/a143/molmospaces/cache"
```

What they are used for:

- `MLSPACES_ASSETS_DIR`: the shared assets directory that MolmoSpaces reads from.
- `MLSPACES_CACHE_DIR`: the shared cache where MolmoSpaces stores downloaded resources and versions.

Keeping these values avoids re-downloading MolmoSpaces assets into a private directory and makes everyone use the same shared data. As long as your EDF file points to these shared paths, MolmoSpaces should work without an extra asset-install step.

## What to customize

Update only the user-specific values as needed:

- `HF_TOKEN`
- `HF_HOME`
- `OPENPI_DATA_HOME`
- The EDF file name itself

You normally do not need to change the shared `MLSPACES_ASSETS_DIR` or `MLSPACES_CACHE_DIR` values.
