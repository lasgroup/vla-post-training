from molmo_spaces.utils.lazy_loading_utils import (
    install_scene_with_objects_and_grasps_from_path,
)
from molmo_spaces.molmo_spaces_constants import get_scenes
from molmospaces.molmo_spaces_constants import get_resource_manager

install_scene_with_objects_and_grasps_from_path(get_scenes("ithor", "train")["train"][1])
get_resource_manager()
