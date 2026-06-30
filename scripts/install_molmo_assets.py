"""Pre-download MolmoSpaces benchmark and scene assets to the local asset store.

Run once on the Euler login node (internet access required) before submitting
molmo jobs. Assets land in molmospaces/assets/ by default (shared filesystem,
visible to compute nodes). Override with MLSPACES_ASSETS_DIR if needed.

Usage:
    uv run python scripts/install_molmo_assets.py
"""

import os
import sys

# molmo_spaces package lives directly in the molmospaces/ submodule root.
# Add it to sys.path so this script works even if the editable install was
# wiped by a subsequent `uv sync`.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "molmospaces"))

from molmo_spaces.utils.lazy_loading_utils import (
    install_scene_with_objects_and_grasps_from_path,
)
from molmo_spaces.molmo_spaces_constants import get_scenes
from molmo_spaces.molmo_spaces_constants import get_resource_manager

install_scene_with_objects_and_grasps_from_path(get_scenes("ithor", "train")["train"][1])
get_resource_manager()
