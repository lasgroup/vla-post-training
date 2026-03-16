"""DSRL vector environment with OpenPI policy action decoding.

Responsibilities:
1. DSRLActionDecoder: Manages the OpenPI model/policy lifecycle and converts
   latent noise -> real action chunks.
2. DSRLVectorEnv: A SubprocVectorEnv that uses the decoder in reset/step,
   attaching prefix representations to observations.
3. dsrl_wrap_env: Factory that builds the full wrapped environment.
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import mesh_utils

import flax.nnx as nnx

import openpi.training.sharding as sharding
from openpi.policies import policy_config
from openpi_client import image_tools

from src.envs.venv import SubprocVectorEnv
from src.envs.wrappers import (
    Pi0ObservationWrapper,
    QueryFrequencyWrapper,
    TimeToSuccessAsRewardWrapper,
)
import src.training.config as _config
from src.training.config import OnlineTrainConfig
from src.rl.dsrl.dsrl_utils import (
    init_train_state,
    _resolve_policy_checkpoint_dir,
    _normalize_task_descriptions,
)


class DSRLActionDecoder:
    """Decodes latent DSRL noise into action chunks via an OpenPI policy."""

    def __init__(self, config: OnlineTrainConfig) -> None:
        self._rng = jax.random.key(config.seed)
        init_rng, self._rng = jax.random.split(self._rng, 2)

        self._mesh = sharding.make_mesh(config.fsdp_devices)
        self._sharding_spec = jax.sharding.NamedSharding(
            jax.sharding.Mesh(
                mesh_utils.create_device_mesh((len(jax.devices()),)),
                axis_names=("batch",),
            ),
            jax.sharding.PartitionSpec(),
        )

        # Load model weights.
        train_state, _ = init_train_state(config, init_rng, self._mesh)
        jax.block_until_ready(train_state)

        params = train_state.ema_params if train_state.ema_params is not None else train_state.params
        self.model = nnx.merge(train_state.model_def, params)

        # Load policy (for tokenization / action decoding), then drop its
        # redundant copy of the model to free memory.
        checkpoint_dir = _resolve_policy_checkpoint_dir(config)
        self._policy = policy_config.create_trained_policy(config, checkpoint_dir)
        self._drop_policy_model()

        self.action_dim = int(self._policy.action_dim)
        self.action_horizon = int(self._policy.action_horizon)

    def _drop_policy_model(self) -> None:
        if getattr(self._policy, "_is_pytorch_model", False):
            return
        self._policy._model = None
        for attr in ("_sample_actions", "_get_prefix_rep"):
            if hasattr(self._policy, attr):
                setattr(self._policy, attr, None)
        gc.collect()

    def infer(self, obs: Dict[str, Any],noise: np.ndarray,) -> Dict[str, Any]:
        """Run policy inference, returning actions and prefix_rep."""
        outputs = self._policy.infer_with_model(
            model=self.model,
            obs=obs,
            noise=noise,
            return_prefix_rep=False,
            sharding_spec=self._sharding_spec,
        )
        if not isinstance(outputs, dict):
            if isinstance(outputs, (tuple, list)) and len(outputs) >= 1:
                normalized: Dict[str, Any] = {"actions": outputs[0]}
                if len(outputs) > 1:
                    normalized["prefix_rep"] = outputs[1]
                return normalized
            raise TypeError(
                f"Unexpected policy output type: {type(outputs)}."
            )

        actions = outputs.get("actions")
        if isinstance(actions, (tuple, list)) and len(actions) >= 1:
            normalized = dict(outputs)
            normalized["actions"] = actions[0]
            if len(actions) > 1 and "prefix_rep" not in normalized:
                normalized["prefix_rep"] = actions[1]
            return normalized
        return outputs


# ---------------------------------------------------------------------------
# DSRLVectorEnv
# ---------------------------------------------------------------------------

gym_old_venv_step_type = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
gym_new_venv_step_type = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class DSRLVectorEnv(SubprocVectorEnv):
    """Vector env that maps latent DSRL noise to OpenPI action chunks."""

    def __init__(
        self,
        env_fns,
        *,
        config: OnlineTrainConfig,
        task_description: list[str],
        **kwargs: Any,
    ) -> None:
        super().__init__(env_fns, **kwargs)

        assert len(task_description) == self.env_num, (
            f"Expected {self.env_num} task descriptions, got {len(task_description)}"
        )
        self._task_description = task_description
        self._query_frequency = int(config.collect.replan_steps)
        self._resize_image = int(config.collect.resize_image)

        self._decoder = DSRLActionDecoder(config)
        self._last_obs: Optional[Dict[str, Any]] = None
        self._prefix_rep_shape: tuple[int, ...] | None = None
        self._warned_missing_prefix_rep = False

    @property
    def policy_action_dim(self) -> int:
        return self._decoder.action_dim

    @property
    def policy_action_horizon(self) -> int:
        return self._decoder.action_horizon

    # ----- observation processing -----

    def _select_latest_frame(self, observations: Dict) -> Dict:
        """Query wrappers return [B, T, ...]; keep only the latest frame."""
        def _pick_last(x):
            arr = np.asarray(x)
            if arr.ndim >= 2 and arr.shape[1] == self._query_frequency:
                return arr[:, -1]
            return arr
        return jax.tree_util.tree_map(_pick_last, observations)

    def _process_obs_for_pi0(self, observations: Dict, env_ids: List[int] | None = None) -> Dict[str, Any]:
        current_obs = self._select_latest_frame(observations)

        _IMAGE_AND_STATE_KEYS = frozenset({
            "image", "wrist_image", "state",
            "exterior_image_1_left", "wrist_image_left",
            "joint_position", "gripper_position",
        })

        processed: dict[str, Any] = {}
        for key, val in current_obs.items():
            if key.startswith("observation/"):
                obs_key = key
            elif key.startswith("pi0/"):
                obs_key = f"observation/{key.split('pi0/', maxsplit=1)[-1]}"
            elif key in _IMAGE_AND_STATE_KEYS:
                obs_key = f"observation/{key}"
            elif key == "prompt":
                continue
            else:
                continue

            if "image" in obs_key and self._resize_image > 0:
                val = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(val, self._resize_image, self._resize_image)
                )
            processed[obs_key] = val

        if env_ids is None:
            task_descs = self._task_description
        else:
            task_descs = [self._task_description[i] for i in env_ids]
        processed["prompt"] = np.array(task_descs)
        return processed

    # ----- noise -> full-horizon expansion -----

    def _expand_noise(self, noise: np.ndarray) -> np.ndarray: # TODO: Simplify function
        """Expand (B, 1, A) or (B, H, A) noise to (B, policy_horizon, A)."""
        arr = np.asarray(noise, dtype=np.float32)
        B = self.env_num
        H = self._decoder.action_horizon
        A = self._decoder.action_dim

        # Normalize to 3D.
        if arr.ndim == 2:
            assert arr.shape == (B, A), f"Expected noise shape ({B}, {A}), got {arr.shape}"
            arr = arr[:, None, :]
        assert arr.ndim == 3 and arr.shape[0] == B and arr.shape[2] == A, (
            f"Expected noise shape ({B}, ?, {A}), got {arr.shape}"
        )

        # Expand horizon.
        if arr.shape[1] == H:
            return arr
        if arr.shape[1] == 1:
            return np.repeat(arr, H, axis=1)
        if arr.shape[1] < H:
            pad = np.repeat(arr[:, -1:, :], H - arr.shape[1], axis=1)
            return np.concatenate([arr, pad], axis=1)
        return arr[:, :H, :]

    # ----- prefix rep helpers -----

    @staticmethod
    def _normalize_prefix_rep_shape(
        prefix_rep: np.ndarray,
        *,
        batch_size: int,
    ) -> np.ndarray:
        # Keep prefix leaves writable because collect.py updates reset env slots in-place.
        prefix = np.array(prefix_rep, copy=True)
        if prefix.ndim == 0:
            prefix = prefix.reshape(1, 1)
        if batch_size <= 0:
            return prefix
        if prefix.ndim == 1:
            prefix = prefix[None, ...]
        if prefix.shape[0] == batch_size:
            return prefix
        if prefix.shape[0] == 1 and batch_size > 1:
            return np.repeat(prefix, batch_size, axis=0)
        # Unbatched policy output in single-env paths: [S, D] -> [1, S, D].
        if batch_size == 1:
            return prefix[None, ...]
        raise ValueError(
            f"Prefix batch {prefix.shape[0]} != obs batch {batch_size}."
        )

    @staticmethod
    def _attach_prefix_rep(
        observation: Dict[str, Any],
        prefix_rep: np.ndarray,
        batch_size: int,
    ) -> Dict[str, Any]:
        prefix = DSRLVectorEnv._normalize_prefix_rep_shape(prefix_rep, batch_size=batch_size)
        return {**observation, "prefix_rep": prefix}

    def _get_prefix_rep(
        self,
        outputs: Dict[str, Any],
        *,
        batch_size: int,
    ) -> np.ndarray:
        prefix_rep = outputs.get("prefix_rep")
        if prefix_rep is not None:
            prefix = self._normalize_prefix_rep_shape(prefix_rep, batch_size=batch_size)
            self._prefix_rep_shape = tuple(prefix.shape[1:])
            return prefix

        if not self._warned_missing_prefix_rep:
            logging.warning("Policy outputs are missing 'prefix_rep'; using zeros as fallback.")
            self._warned_missing_prefix_rep = True
        shape_tail = self._prefix_rep_shape or (1,)
        return np.zeros((batch_size, *shape_tail), dtype=np.float32)

    def _update_obs_cache(
        self,
        env_ids: List[int],
        new_obs: Dict[str, Any],
    ) -> None:
        if self._last_obs is None or len(env_ids) == self.env_num:
            self._last_obs = new_obs
            return

        idx = np.asarray(env_ids, dtype=np.int32)
        def _scatter_update(prev, new):
            updated = np.asarray(prev).copy()
            updated[idx] = np.asarray(new)
            return updated
        self._last_obs = jax.tree_util.tree_map(
            _scatter_update,
            self._last_obs,
            new_obs,
        )

    # ----- reset / step -----

    def reset(
        self,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
        **kwargs: Any,
    ):
        reset_ids = [int(i) for i in self._wrap_id(id)]
        reset_returns = super().reset(id, **kwargs)

        if isinstance(reset_returns, tuple):
            obs, info = reset_returns
        else:
            obs, info = reset_returns, None

        batch_size = len(reset_ids)
        processed_obs = self._process_obs_for_pi0(obs, env_ids=reset_ids)
        dummy_noise = np.zeros((batch_size, self._decoder.action_horizon, self._decoder.action_dim),dtype=np.float32)
        outputs = self._decoder.infer(processed_obs, dummy_noise)
        prefix_rep = self._get_prefix_rep(outputs, batch_size=batch_size)
        obs_with_prefix = self._attach_prefix_rep(obs, prefix_rep, batch_size)
        self._update_obs_cache(reset_ids, obs_with_prefix)

        return (obs_with_prefix, info) if info is not None else obs_with_prefix

    def step(
        self,
        noise: np.ndarray,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ):
        if id is not None:
            raise NotImplementedError("Partial stepping is not supported.")
        assert self._last_obs is not None, "Call reset() before step()."

        processed_obs = self._process_obs_for_pi0(self._last_obs, env_ids=list(range(self.env_num)))
        expanded_noise = self._expand_noise(noise) # TODO: This should happen outside of the wrapper

        outputs = self._decoder.infer(processed_obs, expanded_noise)
        actions = np.asarray(outputs["actions"])
        if actions.ndim == 2:
            actions = actions[np.newaxis, ...]

        return_stacks = super().step(actions, id)
        obs_stack = return_stacks[0]
        prefix_rep = self._get_prefix_rep(outputs, batch_size=self.env_num)
        obs_with_prefix = self._attach_prefix_rep(obs_stack, prefix_rep, self.env_num)
        self._last_obs = obs_with_prefix
        return (obs_with_prefix, *return_stacks[1:])


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def dsrl_wrap_env(
    env_fn,
    config: _config.OnlineTrainConfig,
    task_description: list[str] | str,
    env_num: int | None = None,
) -> tuple[DSRLVectorEnv, list[str]]:
    """Build a DSRLVectorEnv with all necessary wrappers."""
    env_num = int(env_num if env_num is not None else config.collect.env_num)
    replan_steps = int(config.collect.replan_steps)
    domain = str(config.collect.domain)

    task_description = _normalize_task_descriptions(task_description, env_num)

    env_factories = []
    for i in range(env_num):
        task_desc_i = task_description[i]

        def _make_env(rank=i, task_description_single=task_desc_i):
            base_env = env_fn(rank)
            if config.collect.use_time_to_success_as_reward:
                base_env = TimeToSuccessAsRewardWrapper(base_env)
            base_env = Pi0ObservationWrapper(
                env=base_env,
                env_class=domain,
                task_description=task_description_single,
                molmo_config=getattr(config, "molmo", None),
            )
            base_env = QueryFrequencyWrapper(
                env=base_env,
                query_frequency=replan_steps,
            )
            return base_env

        env_factories.append(_make_env)

    env = DSRLVectorEnv(
        env_factories,
        config=config,
        task_description=task_description,
    )
    env.seed(int(config.seed))
    return env, task_description