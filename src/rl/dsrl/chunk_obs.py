from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from src.rl.prefix_embedding import PREFIX_EMBEDDING_NAME


def unwrap_dsrl_vector_observation(observations: Any) -> Any:
    """Extract the model observation payload from DSRLVectorEnv-style outputs."""
    if (
        isinstance(observations, dict)
        and "observation" in observations
        and isinstance(observations["observation"], dict)
    ):
        unpacked = dict(observations["observation"])
        # DSRLVectorEnv attaches prefix reps at the top level while model
        # observations live under "observation". Preserve prefix features.
        if PREFIX_EMBEDDING_NAME in observations:
            unpacked.setdefault(PREFIX_EMBEDDING_NAME, observations[PREFIX_EMBEDDING_NAME])
        if "prefix_rep" in observations:
            unpacked.setdefault("prefix_rep", observations["prefix_rep"])
        return unpacked
    return observations


def normalize_observation_for_model(observations: Any) -> Any:
    """Convert wrapper observations to model-ready SAC inputs.

    Keeps features used by the DSRL actor/critic:
    - `state` (flattened per-env vector)
    - optional `image` / `wrist_image` (latest frame for chunked envs)
    - optional `prefix_embedding`
    """
    observations = unwrap_dsrl_vector_observation(observations)
    if not isinstance(observations, dict):
        return observations
    normalized: dict[str, Any] = {}

    def _find_first(*keys: str) -> Any | None:
        for key in keys:
            if key in observations:
                return observations[key]
        return None

    state_source = _find_first("pi0/state", "observation/state", "state")
    if state_source is None:
        joint_position = _find_first(
            "observation/joint_position",
            "joint_position",
        )
        gripper_position = _find_first(
            "observation/gripper_position",
            "gripper_position",
        )
        if joint_position is not None and gripper_position is not None:
            joint_arr = jnp.asarray(joint_position, dtype=jnp.float32)
            gripper_arr = jnp.asarray(gripper_position, dtype=jnp.float32)
            state_source = jnp.concatenate([joint_arr, gripper_arr], axis=-1)
    if state_source is not None:
        state = jnp.asarray(state_source, dtype=jnp.float32)
        # Query-frequency wrappers add a temporal/chunk axis after batch.
        # For SAC/DSRL we model Q(s, a_chunk), so keep only the latest state.
        if state.ndim >= 3:
            state = state[:, -1, ...]
        if state.ndim == 1:
            state = state[jnp.newaxis, ...]
        # Flatten any remaining non-batch axes (e.g. trailing singleton dims).
        if state.ndim > 2:
            state = jnp.reshape(state, (state.shape[0], -1))
        normalized["state"] = state

    image_source = _find_first(
        "image",
        "observation/image",
        "pi0/image",
        "pixels",
        "observation/pixels",
        "observation/exterior_image_1_left",
        "exterior_image_1_left",
    )
    if image_source is not None:
        image = jnp.asarray(image_source)
        # Chunked envs: [B, T, H, W, C] -> keep latest frame.
        if image.ndim >= 5:
            image = image[:, -1, ...]
        if image.ndim == 3:
            image = image[jnp.newaxis, ...]
        normalized["image"] = image

    wrist_image_source = _find_first(
        "wrist_image",
        "observation/wrist_image",
        "pi0/wrist_image",
        "observation/wrist_image_left",
        "wrist_image_left",
    )
    if wrist_image_source is not None:
        wrist_image = jnp.asarray(wrist_image_source)
        if wrist_image.ndim >= 5:
            wrist_image = wrist_image[:, -1, ...]
        if wrist_image.ndim == 3:
            wrist_image = wrist_image[jnp.newaxis, ...]
        normalized["wrist_image"] = wrist_image

    # Keep optional prefix embedding for experiments that enable it.
    if PREFIX_EMBEDDING_NAME in observations:
        prefix = jnp.asarray(observations[PREFIX_EMBEDDING_NAME], dtype=jnp.float32)
        if prefix.ndim >= 4:
            prefix = prefix[:, -1, ...]
        if prefix.ndim == 4:
            prefix = jnp.reshape(prefix, (prefix.shape[0], -1, prefix.shape[-1]))
        if prefix.ndim == 3:
            prefix = jnp.mean(prefix, axis=1)
        normalized[PREFIX_EMBEDDING_NAME] = prefix
    elif "prefix_rep" in observations:
        # Backward-compatible alias used by DSRLVectorEnv.
        prefix = jnp.asarray(observations["prefix_rep"], dtype=jnp.float32)
        if prefix.ndim >= 4:
            prefix = prefix[:, -1, ...]
        if prefix.ndim == 4:
            prefix = jnp.reshape(prefix, (prefix.shape[0], -1, prefix.shape[-1]))
        if prefix.ndim == 3:
            prefix = jnp.mean(prefix, axis=1)
        normalized[PREFIX_EMBEDDING_NAME] = prefix

    return normalized if normalized else observations


def expected_chunk_action_shape(dummy_action: np.ndarray) -> tuple[int, ...]:
    raw_shape = tuple(np.asarray(dummy_action).shape[1:])
    if len(raw_shape) < 2:
        return raw_shape
    return (raw_shape[0], *tuple(dim for dim in raw_shape[1:] if dim != 1))


def normalize_action_batch_shape(
    actions: np.ndarray, expected_shape: tuple[int, ...]
) -> np.ndarray:
    expected_ndim = len(expected_shape) + 1  # include batch axis
    if actions.ndim == expected_ndim and tuple(actions.shape[1:]) == expected_shape:
        return actions
    if actions.ndim >= 2 and np.prod(actions.shape[1:]) == np.prod(expected_shape):
        return actions.reshape((actions.shape[0], *expected_shape))
    # Fallback for chunked environments: if the policy emits one action per env
    # (B, A) but the wrapped env expects (B, H, A), repeat across the horizon.
    if (
        actions.ndim == 2
        and len(expected_shape) >= 2
        and actions.shape[-1] == expected_shape[-1]
    ):
        repeated = np.repeat(actions[:, None, :], expected_shape[0], axis=1)
        if tuple(repeated.shape[1:]) == expected_shape:
            return repeated
        if np.prod(repeated.shape[1:]) == np.prod(expected_shape):
            return repeated.reshape((actions.shape[0], *expected_shape))
    return actions


def _fit_length(
    values: Any, length: int, *, dtype: Any, pad_value: Any
) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype).reshape(-1)
    if arr.size == length:
        return arr
    if arr.size == 0:
        return np.full((length,), pad_value, dtype=dtype)
    if arr.size == 1:
        return np.full((length,), arr.item(), dtype=dtype)
    if arr.size > length:
        return arr[:length]
    return np.pad(
        arr,
        (0, length - arr.size),
        mode="constant",
        constant_values=pad_value,
    )


def reduce_chunk_transition(
    *,
    reward: Any,
    terminated: Any,
    truncated: Any,
    discount: float,
    action: Any,
) -> tuple[float, bool, bool, bool, int]:
    reward_arr = np.asarray(reward if reward is not None else 0.0, dtype=np.float32).reshape(
        -1
    )
    term_arr = np.asarray(terminated, dtype=np.bool_).reshape(-1)
    trunc_arr = np.asarray(truncated, dtype=np.bool_).reshape(-1)
    rollout_len = int(max(reward_arr.size, term_arr.size, trunc_arr.size, 1))

    reward_arr = _fit_length(reward_arr, rollout_len, dtype=np.float32, pad_value=0.0)
    term_arr = _fit_length(term_arr, rollout_len, dtype=np.bool_, pad_value=False)
    trunc_arr = _fit_length(trunc_arr, rollout_len, dtype=np.bool_, pad_value=False)

    done_arr = np.logical_or(term_arr, trunc_arr)
    if bool(np.any(done_arr)):
        n_steps = int(np.argmax(done_arr) + 1)
    else:
        n_steps = rollout_len
        action_arr = np.asarray(action)
        if rollout_len == 1 and action_arr.ndim >= 2:
            n_steps = max(1, int(action_arr.shape[0]))

    reward_weights = np.power(np.float32(discount), np.arange(n_steps, dtype=np.float32))
    total_reward = float(np.sum(reward_arr[:n_steps] * reward_weights))
    last_terminated = bool(term_arr[n_steps - 1])
    last_truncated = bool(trunc_arr[n_steps - 1])
    done = bool(last_terminated or last_truncated)

    return total_reward, last_terminated, last_truncated, done, n_steps
