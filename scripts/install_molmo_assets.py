"""Pre-download MolmoSpaces benchmark and scene assets to the local asset store.

Run once on the Euler login node (internet access required) before submitting
molmo jobs. Assets land in molmospaces/assets/ by default (shared filesystem,
visible to compute nodes). Override with MLSPACES_ASSETS_DIR if needed.

Usage:
    python scripts/install_molmo_assets.py
"""

from molmo_spaces.utils.lazy_loading_utils import (
    install_scene_with_objects_and_grasps_from_path,
)
from molmo_spaces.molmo_spaces_constants import get_scenes
from molmo_spaces.molmo_spaces_constants import get_resource_manager

install_scene_with_objects_and_grasps_from_path(get_scenes("ithor", "train")["train"][1])
get_resource_manager()
