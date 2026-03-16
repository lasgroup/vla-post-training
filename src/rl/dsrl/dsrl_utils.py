"""Shared utilities for DSRL and Residual RL learners."""

from __future__ import annotations

import gc
import logging
import os
from typing import Any, Dict

import numpy as np
import PIL.Image
import jax
import jax.numpy as jnp

import flax.nnx as nnx
import flax.traverse_util as traverse_util

import openpi.shared.array_typing as at
import openpi.training.checkpoints as _checkpoints
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

from src.training.config import OnlineTrainConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _resize_image_np(img: np.ndarray, target_size: int) -> np.ndarray:
    """Resize a single HxWxC uint8 image to target_size x target_size."""
    if img.ndim == 3:
        h, w = img.shape[0], img.shape[1]
        if h == target_size and w == target_size:
            return img
        return np.array(
            PIL.Image.fromarray(img).resize((target_size, target_size)),
            dtype=img.dtype,
        )
    # Handle arbitrary leading dims (batch, temporal, etc.) by recursing.
    if img.ndim >= 4:
        return np.stack([_resize_image_np(img[i], target_size) for i in range(img.shape[0])])
    return img


# ---------------------------------------------------------------------------
# Observation extraction for replay buffers
# ---------------------------------------------------------------------------

def _extract_replay_observation(
    observation: Any,
    *,
    obs_prefix: str,
    sac_image_size: int = 0,
) -> Dict[str, np.ndarray]:
    """Convert env observation payloads into replay observations.

    Returns a dict with:
    - ``state`` (required)
    - optional ``image``, ``wrist_image`` (resized to *sac_image_size* when > 0)
    - optional ``base_action`` (residual RL specific, included when present)
    """
    if not isinstance(observation, dict):
        raise TypeError(
            f"Expected dict observation for replay extraction, got {type(observation)}."
        )

    obs_dict = (
        observation["observation"]
        if isinstance(observation.get("observation"), dict)
        else observation
    )

    def _get_from_obs(*keys: str) -> Any | None:
        for key in keys:
            if key in obs_dict:
                return obs_dict[key]
            if key in observation:
                return observation[key]
        return None

    extracted: Dict[str, np.ndarray] = {}

    image = _get_from_obs(
        f"{obs_prefix}/image",
        "image",
        "observation/image",
        "observation/exterior_image_1_left",
        "exterior_image_1_left",
        "pixels",
    )
    if image is not None:
        img = np.asarray(image)
        if sac_image_size > 0:
            img = _resize_image_np(img, sac_image_size)
        extracted["image"] = np.asarray(img, dtype=np.uint8)

    wrist_image = _get_from_obs(
        f"{obs_prefix}/wrist_image",
        "wrist_image",
        "observation/wrist_image",
        "observation/wrist_image_left",
        "wrist_image_left",
    )
    if wrist_image is not None:
        wimg = np.asarray(wrist_image)
        if sac_image_size > 0:
            wimg = _resize_image_np(wimg, sac_image_size)
        extracted["wrist_image"] = np.asarray(wimg, dtype=np.uint8)

    state = _get_from_obs(
        f"{obs_prefix}/state",
        "state",
        "observation/state",
    )
    if state is None:
        joint_position = _get_from_obs(
            "observation/joint_position",
            "joint_position",
        )
        gripper_position = _get_from_obs(
            "observation/gripper_position",
            "gripper_position",
        )
        if joint_position is not None and gripper_position is not None:
            state = np.concatenate(
                [
                    np.asarray(joint_position, dtype=np.float32),
                    np.asarray(gripper_position, dtype=np.float32),
                ],
                axis=-1,
            )
    if state is None:
        raise KeyError(
            "Replay extraction requires a state vector. Expected one of "
            f"['{obs_prefix}/state', 'state', 'observation/state']."
        )
    extracted["state"] = np.asarray(state, dtype=np.float32)

    # Preserve base_action when present (used by residual RL).
    base_action = _get_from_obs("base_action")
    if base_action is not None:
        extracted["base_action"] = np.asarray(base_action, dtype=np.float32)

    return extracted


def _finalize_replay_observation(
    current_obs: Dict[str, np.ndarray],
    next_obs: Dict[str, np.ndarray] | None,
) -> Dict[str, np.ndarray]:
    """Fill next-observation fields from current observation when missing."""
    if next_obs is None:
        next_obs = {}

    merged: Dict[str, np.ndarray] = {
        "state": np.asarray(next_obs.get("state", current_obs["state"]), dtype=np.float32)
    }
    for key, dtype in [("image", np.uint8), ("wrist_image", np.uint8), ("base_action", np.float32)]:
        if key in next_obs or key in current_obs:
            merged[key] = np.asarray(
                next_obs.get(key, current_obs.get(key)), dtype=dtype
            )
    return merged


def _build_replay_observation_template(
    observation: Any,
    *,
    obs_prefix: str,
    sac_image_size: int = 0,
) -> Dict[str, np.ndarray]:
    """Build fixed-shape replay template preserving image/state/base_action modalities."""
    extracted = _extract_replay_observation(
        observation, obs_prefix=obs_prefix, sac_image_size=sac_image_size,
    )
    template: Dict[str, np.ndarray] = {}
    for key in ("image", "wrist_image"):
        if key in extracted:
            template[key] = np.zeros_like(np.asarray(extracted[key]), dtype=np.uint8)
    template["state"] = np.zeros_like(np.asarray(extracted["state"]), dtype=np.float32)
    if "base_action" in extracted:
        template["base_action"] = np.zeros_like(np.asarray(extracted["base_action"]), dtype=np.float32)
    return template


def _copy_with_batch_dim(x: Any, *, dtype: Any | None = None) -> np.ndarray:
    arr = np.asarray(x, dtype=dtype)
    return np.array(arr[None, ...], copy=True)


# ---------------------------------------------------------------------------
# Train state initialization (shared between env files)
# ---------------------------------------------------------------------------

def _load_weights_and_validate(
    loader: _weight_loaders.WeightLoader, params_shape: at.Params
) -> at.Params:
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(
        expected=params_shape,
        got=loaded_params,
        check_shapes=True,
        check_dtypes=True,
    )
    return traverse_util.unflatten_dict(
        {
            k: v
            for k, v in traverse_util.flatten_dict(loaded_params).items()
            if not isinstance(v, jax.ShapeDtypeStruct)
        }
    )


@at.typecheck
def init_train_state(
    config: OnlineTrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    resume: bool = False,
) -> tuple[training_utils.TrainState, Any]:
    import openpi.shared.nnx_utils as nnx_utils

    tx = _optimizer.create_optimizer(
        config.optimizer,
        config.lr_schedule,
        weight_decay_mask=None,
    )

    def init(
        rng: at.KeyArrayLike,
        partial_params: at.Params | None = None,
    ) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            nnx.replace_by_pure_dict(state, partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(
            params,
            config.freeze_filter,
            lambda p: p.replace(p.value.astype(jnp.bfloat16)),
        )

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(nnx.filter_state(params, config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=not resume)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(
        config.weight_loader,
        nnx.to_pure_dict(train_state_shape.params),
    )
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(),
    )

    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


def _resolve_policy_checkpoint_dir(
    config: OnlineTrainConfig,
    checkpoint_manager: _checkpoints.CheckpointManager | None = None,
) -> str:
    checkpoint_dir = os.environ.get("OPENPI_POLICY_CHECKPOINT_DIR")
    if checkpoint_dir is not None:
        return checkpoint_dir
    if isinstance(config.weight_loader, _weight_loaders.CheckpointWeightLoader):
        path = config.weight_loader.params_path
        return path[: -len("/params")] if path.endswith("/params") else path
    if checkpoint_manager is not None:
        directory = checkpoint_manager._directory
        if (directory / "params").exists():
            return str(directory)
    raise FileNotFoundError(
        "Policy checkpoint not found. Set OPENPI_POLICY_CHECKPOINT_DIR or "
        "use CheckpointWeightLoader with a params path."
    )


# ---------------------------------------------------------------------------
# Environment factory helpers
# ---------------------------------------------------------------------------

def _normalize_task_descriptions(
    task_description: list[str] | str,
    env_num: int,
) -> list[str]:
    if isinstance(task_description, str):
        return [task_description] * env_num
    if not task_description:
        return [""] * env_num
    if len(task_description) == env_num:
        return [str(x) for x in task_description]
    if len(task_description) == 1:
        return [str(task_description[0])] * env_num
    return [str(task_description[i % len(task_description)]) for i in range(env_num)]
