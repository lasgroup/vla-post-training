from typing import Any

import numpy as np


def as_uint8_hwc(image: Any) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0) * 255.0 if image.max() <= 1.0 else image
    return np.clip(image, 0, 255).astype(np.uint8)


def resolve_qpos(obs: dict[str, Any]) -> dict[str, Any]:
    if "qpos" in obs:
        return obs["qpos"]
    if "robot_state" in obs and "qpos" in obs["robot_state"]:
        return obs["robot_state"]["qpos"]
    raise KeyError("Could not find qpos in observation. Expected 'qpos' or 'robot_state/qpos'.")


def resolve_exo_camera_key(obs: dict[str, Any], default_key: str) -> str:
    return (
        "droid_shoulder_light_randomization"
        if "droid_shoulder_light_randomization" in obs
        else default_key
    )


def resolve_wrist_camera_key(obs: dict[str, Any], default_key: str) -> str:
    return "wrist_camera_zed_mini" if "wrist_camera_zed_mini" in obs else default_key


def get_camera(obs: dict[str, Any], primary: str, fallback: tuple[str, ...] = ()) -> np.ndarray:
    if primary in obs:
        return as_uint8_hwc(obs[primary])
    for key in fallback:
        if key in obs:
            return as_uint8_hwc(obs[key])
    raise KeyError(f"Missing camera key '{primary}'. Available keys: {list(obs.keys())}")


def obs_to_openpi_input(
    obs: dict[str, Any],
    *,
    exo_camera_key: str = "exo_camera_1",
    wrist_camera_key: str = "wrist_camera",
    gripper_obs_norm: float = 0.824033,
    prompt: str | None = None,
) -> dict[str, Any]:
    qpos = resolve_qpos(obs)
    if "arm" not in qpos or "gripper" not in qpos:
        raise KeyError(f"Expected qpos to contain 'arm' and 'gripper'. Got: {list(qpos.keys())}")

    exo_key = resolve_exo_camera_key(obs, exo_camera_key)
    wrist_key = resolve_wrist_camera_key(obs, wrist_camera_key)
    exo = get_camera(obs, exo_key, ())
    wrist = get_camera(obs, wrist_key, ())

    gripper = np.asarray(qpos["gripper"], dtype=np.float32).reshape(-1)
    if gripper.size == 0:
        raise KeyError("qpos['gripper'] is empty.")
    grip = np.clip(float(gripper[0]) / max(float(gripper_obs_norm), 1e-6), 0.0, 1.0)

    out = {
        "observation/exterior_image_1_left": exo,
        "observation/wrist_image_left": wrist,
        "observation/joint_position": np.asarray(qpos["arm"][:7], dtype=np.float32),
        "observation/gripper_position": np.asarray([grip], dtype=np.float32),
    }
    if prompt is not None:
        out["prompt"] = prompt
    return out
