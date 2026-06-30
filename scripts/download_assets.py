#!/usr/bin/env python3
"""Pre-download all assets required for training to the local cache.

Run this once on the Euler login node (which has internet access) before submitting
jobs. Compute nodes have no outbound internet, so all assets must be cached first.

By default assets go to ~/.cache/openpi. Override with --cache_dir to use scratch
(recommended on Euler where home quota is limited).

Usage:
    uv run scripts/download_assets.py
    uv run scripts/download_assets.py --cache_dir /cluster/scratch/asukhija/openpi_cache
"""

import argparse
import importlib.util
import os
import pathlib
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
# openpi package lives in openpi/src/ — add it explicitly so this script works
# even if the editable install was wiped by a subsequent `uv sync`.
sys.path.insert(0, os.path.join(_REPO_ROOT, "openpi", "src"))

ASSETS = [
    "gs://big_vision/paligemma_tokenizer.model",
    # LIBERO checkpoint (used by all pi05_libero_* configs)
    "gs://openpi-assets/checkpoints/pi05_libero/params",
    "gs://openpi-assets/checkpoints/pi05_libero/assets",
    # Molmo / DROID checkpoint (used by all pi05_molmo_* configs)
    "gs://openpi-assets/checkpoints/pi05_droid_jointpos/params",
    "gs://openpi-assets/checkpoints/pi05_droid_jointpos/assets",
]


def _download_libero_assets() -> None:
    """Download LIBERO scene/object assets into the installed hf-libero package dir.

    hf-libero ships without assets and downloads them from lerobot/libero-assets on
    first use. Compute nodes have no internet, so we do it here on the login node.
    The assets land in the venv package directory, which is on the shared filesystem
    and is therefore visible to all compute nodes.
    """
    spec = importlib.util.find_spec("libero.libero")
    if spec is None or spec.origin is None:
        print("Warning: libero.libero not found, skipping LIBERO asset download.")
        return

    assets_dir = pathlib.Path(spec.origin).parent / "assets"
    if (assets_dir / "scenes").exists() and any((assets_dir / "scenes").iterdir()):
        print(f"LIBERO assets already present at {assets_dir}, skipping.")
        return

    print(f"Fetching lerobot/libero-assets -> {assets_dir} ...")
    # Suppress per-file progress bars from huggingface_hub (they clutter the terminal).
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="lerobot/libero-assets",
        repo_type="dataset",
        local_dir=str(assets_dir),
        ignore_patterns=["*.gitattributes", ".gitattributes"],
    )
    print(f"  -> Done: {assets_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Override cache directory (sets OPENPI_DATA_HOME). "
        "Defaults to ~/.cache/openpi.",
    )
    args = parser.parse_args()

    if args.cache_dir:
        os.environ["OPENPI_DATA_HOME"] = args.cache_dir
        print(f"Cache dir: {args.cache_dir}")
        print(
            f"Add this to your Euler sbatch scripts or shell profile:\n"
            f"  export OPENPI_DATA_HOME={args.cache_dir}\n"
        )

    from openpi.shared.download import maybe_download

    for url in ASSETS:
        print(f"Fetching {url} ...")
        local = maybe_download(url)
        print(f"  -> {local}")

    _download_libero_assets()

    print("Done. All assets cached.")


if __name__ == "__main__":
    main()
