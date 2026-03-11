from __future__ import annotations

from src.envs import make_env
from src.envs.wrappers import (
    Pi0ObservationWrapper,
    QueryFrequencyWrapper,
    TimeToSuccessAsRewardWrapper,
)
import src.training.config as _config


import gc
import logging
import os
import weakref
from typing import Any, Dict, List, Optional, Tuple, Union

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import mesh_utils

import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
from openpi.policies import policy_config
from openpi_client import image_tools
from src.envs.venv import SubprocVectorEnv
from src.training.config import OnlineTrainConfig


gym_old_venv_step_type = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
gym_new_venv_step_type = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]

def _load_weights_and_validate(
    loader: _weight_loaders.WeightLoader, params_shape: at.Params
) -> at.Params:
    """Load checkpoint weights and return only concrete loaded leaves."""
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
) -> tuple[training_utils.TrainState, Any]:
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
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=False)

    partial_params = _load_weights_and_validate(
        config.weight_loader,
        nnx.to_pure_dict(train_state_shape.params),
    )
    replicated_sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(),
    )

    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


class DSRLVectorEnv(SubprocVectorEnv):
    """Vector env that maps latent DSRL noise to OpenPI action chunks."""

    def __init__(
        self,
        env_fns,
        *,
        config: OnlineTrainConfig,
        task_description: list[str] | str,
        **kwargs: Any,
    ) -> None:
        super().__init__(env_fns, **kwargs)
        self._config = config
        self._task_description = self._normalize_task_descriptions(
            task_description,
            self.env_num,
        )
        self._query_frequency = int(config.collect.replan_steps)
        self._resize_image = int(config.collect.resize_image)

        self._rng = jax.random.key(self._config.seed)
        init_rng, self._rng = jax.random.split(self._rng, 2)

        self._mesh = sharding.make_mesh(self._config.fsdp_devices)
        # Replicated sharding avoids per-env reset shape issues in collection loops.
        self._policy_sharding_spec = jax.sharding.NamedSharding(
            jax.sharding.Mesh(
                mesh_utils.create_device_mesh((len(jax.devices()),)),
                axis_names=("batch",),
            ),
            jax.sharding.PartitionSpec(),
        )

        self._train_state, self._train_state_sharding = init_train_state(
            self._config,
            init_rng,
            self._mesh,
        )
        jax.block_until_ready(self._train_state)

        params = (
            self._train_state.ema_params
            if self._train_state.ema_params is not None
            else self._train_state.params
        )
        self.model = nnx.merge(self._train_state.model_def, params)

        policy_checkpoint_dir = os.environ.get("OPENPI_POLICY_CHECKPOINT_DIR")
        if policy_checkpoint_dir is None and isinstance(
            self._config.weight_loader,
            _weight_loaders.CheckpointWeightLoader,
        ):
            params_path = self._config.weight_loader.params_path
            policy_checkpoint_dir = (
                params_path[: -len("/params")]
                if params_path.endswith("/params")
                else params_path
            )
        if policy_checkpoint_dir is None:
            raise FileNotFoundError(
                "Policy checkpoint not found. Set OPENPI_POLICY_CHECKPOINT_DIR or "
                "use CheckpointWeightLoader with a params path."
            )

        self._policy = policy_config.create_trained_policy(
            self._config,
            policy_checkpoint_dir,
        )
        self._drop_policy_model()

        self._policy_action_dim = int(self._policy.action_dim)
        self._policy_action_horizon = int(self._policy.action_horizon)
        self._last_obs = None

    @staticmethod
    def _normalize_task_descriptions(
        task_description: list[str] | str,
        env_num: int,
    ) -> list[str]:
        if isinstance(task_description, str):
            return [task_description for _ in range(env_num)]
        if not task_description:
            return ["" for _ in range(env_num)]
        if len(task_description) == env_num:
            return [str(x) for x in task_description]
        if len(task_description) == 1:
            return [str(task_description[0]) for _ in range(env_num)]
        return [str(task_description[i % len(task_description)]) for i in range(env_num)]

    @property
    def policy_action_dim(self) -> int:
        return self._policy_action_dim

    @property
    def policy_action_horizon(self) -> int:
        return self._policy_action_horizon

    def _drop_policy_model(self):
        # For PyTorch policies infer_with_model ignores provided model and uses
        # internal state, so dropping would be unsafe.
        if getattr(self._policy, "_is_pytorch_model", False):
            return

        model = getattr(self._policy, "_model", None)
        model_ref = None
        if model is not None:
            try:
                model_ref = weakref.ref(model)
            except TypeError:
                model_ref = None

        self._policy._model = None
        if hasattr(self._policy, "_sample_actions"):
            self._policy._sample_actions = None
        if hasattr(self._policy, "_get_prefix_rep"):
            self._policy._get_prefix_rep = None

        del model
        gc.collect()

        if model_ref is not None and model_ref() is not None:
            logging.warning(
                "Policy model object is still alive after cleanup; other references remain."
            )

    def _resolve_reset_ids(
        self,
        id: Optional[Union[int, List[int], np.ndarray]],
    ) -> List[int]:
        return [int(i) for i in self._wrap_id(id)]

    def _task_prompts_for_ids(self, env_ids: list[int]) -> list[str]:
        return [self._task_description[i] for i in env_ids]

    def _process_obs_for_pi0(
        self,
        observations: Dict,
        *,
        env_ids: list[int],
    ) -> Dict[str, Any]:
        # Query wrappers return [B, T, ...] for all leaves; keep latest frame.
        def _select_latest(x):
            arr = np.asarray(x)
            if arr.ndim >= 2 and arr.shape[1] == self._query_frequency:
                return arr[:, -1]
            return arr

        current_obs = jax.tree_util.tree_map(_select_latest, observations)

        processed_obs: dict[str, Any] = {}

        for key, val in current_obs.items():
            obs_key = None
            if key.startswith("observation/"):
                obs_key = key
            elif key.startswith("pi0/"):
                obs_key = f"observation/{key.split('pi0/', maxsplit=1)[-1]}"
            elif key in (
                "image",
                "wrist_image",
                "state",
                "exterior_image_1_left",
                "wrist_image_left",
                "joint_position",
                "gripper_position",
            ):
                obs_key = f"observation/{key}"
            elif key == "prompt":
                continue

            if obs_key is None:
                continue

            if "image" in obs_key and self._resize_image > 0:
                val = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(
                        val,
                        self._resize_image,
                        self._resize_image,
                    )
                )
            processed_obs[obs_key] = val

        # OpenPI tokenization expects a scalar prompt in this inference path.
        processed_obs["prompt"] = (
            str(self._task_description[0]) if self._task_description else ""
        )
        return processed_obs

    @staticmethod
    def _infer_batch_size(observations: Dict[str, Any]) -> int:
        leaves = jax.tree_util.tree_leaves(observations)
        if not leaves:
            return 1
        arr = np.asarray(leaves[0])
        if arr.ndim == 0:
            return 1
        return int(arr.shape[0])

    def _expand_noise_to_horizon(
        self,
        noise: np.ndarray,
        *,
        batch_size: int,
    ) -> np.ndarray:
        noise_arr = np.asarray(noise, dtype=np.float32)

        if noise_arr.ndim == 1:
            noise_arr = noise_arr.reshape(1, 1, -1)
        elif noise_arr.ndim == 2:
            if (
                noise_arr.shape[0] == batch_size
                and noise_arr.shape[1] == self._policy_action_dim
            ):
                noise_arr = noise_arr[:, None, :]
            elif (
                batch_size == 1
                and noise_arr.shape[0] == self._policy_action_horizon
                and noise_arr.shape[1] == self._policy_action_dim
            ):
                noise_arr = noise_arr[None, ...]
            elif (
                noise_arr.shape[0] == batch_size
                and noise_arr.shape[1] % self._policy_action_dim == 0
            ):
                horizon = noise_arr.shape[1] // self._policy_action_dim
                noise_arr = noise_arr.reshape(
                    batch_size,
                    horizon,
                    self._policy_action_dim,
                )
            else:
                raise ValueError(
                    "Unsupported compact noise shape for DSRLVectorEnv: "
                    f"{tuple(noise_arr.shape)}."
                )
        elif noise_arr.ndim != 3:
            raise ValueError(
                "Expected noise with shape (B, A), (B, H, A), or (A,), "
                f"got {tuple(noise_arr.shape)}."
            )

        if noise_arr.shape[0] != batch_size:
            if noise_arr.shape[0] == 1 and batch_size > 1:
                noise_arr = np.repeat(noise_arr, batch_size, axis=0)
            else:
                raise ValueError(
                    "Noise batch size does not match observations: "
                    f"noise_batch={noise_arr.shape[0]}, obs_batch={batch_size}."
                )

        if noise_arr.shape[-1] != self._policy_action_dim:
            raise ValueError(
                "Noise action dim does not match policy action dim: "
                f"noise={noise_arr.shape[-1]}, policy={self._policy_action_dim}."
            )

        target_horizon = self._policy_action_horizon
        if noise_arr.shape[1] == target_horizon:
            return noise_arr
        if noise_arr.shape[1] == 1:
            return np.repeat(noise_arr, target_horizon, axis=1)
        if noise_arr.shape[1] < target_horizon:
            pad = np.repeat(
                noise_arr[:, -1:, :],
                target_horizon - noise_arr.shape[1],
                axis=1,
            )
            return np.concatenate([noise_arr, pad], axis=1)
        return noise_arr[:, :target_horizon, :]

    @staticmethod
    def _normalize_prefix_rep_shape(
        prefix_rep: np.ndarray,
        *,
        batch_size: int,
    ) -> np.ndarray:
        # Ensure collect/reset merge can assign into per-env slots.
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
        # Fallback for unbatched policy outputs in single-env paths.
        if batch_size == 1:
            return prefix[None, ...]
        raise ValueError(
            "Prefix representation batch size mismatch: "
            f"prefix_batch={prefix.shape[0]}, obs_batch={batch_size}."
        )

    @staticmethod
    def _attach_prefix_rep(
        observation: Any,
        prefix_rep: np.ndarray,
        *,
        batch_size: int,
    ) -> Any:
        prefix = DSRLVectorEnv._normalize_prefix_rep_shape(
            prefix_rep,
            batch_size=batch_size,
        )
        if isinstance(observation, dict):
            obs_with_prefix = dict(observation)
            obs_with_prefix["prefix_rep"] = prefix
            return obs_with_prefix
        return np.concatenate([observation, prefix], axis=1)

    def _update_last_obs_cache(
        self,
        reset_ids: List[int],
        reset_obs_with_prefix: Any,
    ) -> None:
        # Full reset (or first reset): replace cache directly.
        if self._last_obs is None or len(reset_ids) == self.env_num:
            self._last_obs = reset_obs_with_prefix
            return

        id_index = np.asarray(reset_ids, dtype=np.int32)

        def _scatter_update(prev_leaf, new_leaf):
            prev = np.asarray(prev_leaf).copy()
            prev[id_index] = np.asarray(new_leaf)
            return prev

        try:
            self._last_obs = jax.tree_util.tree_map(
                _scatter_update,
                self._last_obs,
                reset_obs_with_prefix,
            )
        except Exception as exc:
            logging.warning(
                "Failed to merge partial reset into _last_obs cache; replacing cache. "
                "This may temporarily change batch shape. Error: %s",
                exc,
            )
            self._last_obs = reset_obs_with_prefix

    def reset(
        self,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Union[dict, List[dict]]]]:
        """Reset env(s) and append prefix representations to observations."""
        reset_ids = self._resolve_reset_ids(id)
        reset_returns = super().reset(id, **kwargs)
        if isinstance(reset_returns, tuple):
            obs, info = reset_returns
            returns_info = True
        else:
            obs = reset_returns
            info = None
            returns_info = False

        batch_size = self._infer_batch_size(obs)
        processed_obs = self._process_obs_for_pi0(obs, env_ids=reset_ids)
        dummy_noise = np.zeros(
            (batch_size, self._policy_action_horizon, self._policy_action_dim),
            dtype=np.float32,
        )
        outputs = self._policy.infer_with_model(
            model=self.model,
            obs=processed_obs,
            noise=dummy_noise,
            return_prefix_rep=True,
            sharding_spec=self._policy_sharding_spec,
        )
        obs_with_prefix = self._attach_prefix_rep(
            obs,
            outputs["prefix_rep"],
            batch_size=batch_size,
        )
        self._update_last_obs_cache(reset_ids, obs_with_prefix)

        if returns_info:
            return obs_with_prefix, info
        return obs_with_prefix

    def step(
        self,
        noise: np.ndarray,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ) -> Union[gym_old_venv_step_type, gym_new_venv_step_type]:
        """Step env by converting latent DSRL noise to action chunks."""
        if id is not None:
            raise NotImplementedError(
                "Partially stepping DSRLVectorEnv is not supported."
            )
        if self._last_obs is None:
            raise RuntimeError("Call reset() before step().")

        env_ids = list(range(self.env_num))
        processed_obs = self._process_obs_for_pi0(self._last_obs, env_ids=env_ids)
        batch_size = self._infer_batch_size(self._last_obs)
        expanded_noise = self._expand_noise_to_horizon(noise, batch_size=batch_size)

        outputs = self._policy.infer_with_model(
            model=self.model,
            obs=processed_obs,
            noise=expanded_noise,
            return_prefix_rep=True,
            sharding_spec=self._policy_sharding_spec,
        )
        actions = np.asarray(outputs["actions"])
        if batch_size == 1 and actions.ndim == 2:
            actions = actions[np.newaxis, ...]

        return_stacks = super().step(actions, id)
        obs_stack = return_stacks[0]
        obs_with_prefix = self._attach_prefix_rep(
            obs_stack,
            outputs["prefix_rep"],
            batch_size=batch_size,
        )
        self._last_obs = obs_with_prefix

        return (obs_with_prefix, *return_stacks[1:])  # type: ignore



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


def _get_pre_step_action_filter(domain: str):
    if domain == "libero":
        return lambda x: np.where(np.abs(x) < 0.0011, 0.0, x)
    return lambda x: x


def dsrl_wrap_env(
    env_fn,
    config: _config.OnlineTrainConfig,
    task_description: list[str] | str,
) -> tuple["DSRLVectorEnv", list[str]]:
    """Build a DSRLVectorEnv with all necessary wrappers.

    Returns the wrapped vector environment and the normalized task descriptions.
    """
    env_num = int(config.collect.env_num)
    replan_steps = int(config.collect.replan_steps)
    domain = str(config.collect.domain)
    pre_step_action_filter = _get_pre_step_action_filter(domain)

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
                pre_step_filter=pre_step_action_filter,
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