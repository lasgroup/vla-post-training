"""Residual RL vector environment with OpenPI policy base-action sampling.

Responsibilities:
1. ResidualRLBaseActionSampler: Manages the OpenPI model/policy lifecycle and
   samples base actions for the residual RL agent.
2. ResidualRLVectorEnv: A SubprocVectorEnv that uses the sampler in reset/step,
   attaching base actions to observations.
3. residual_rl_wrap_env: Factory that builds the full wrapped environment.
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import mesh_utils

import flax.nnx as nnx
import flax.traverse_util as traverse_util

import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
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


# ---------------------------------------------------------------------------
# Train state initialization (shared with learner)
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
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

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


# ---------------------------------------------------------------------------
# Base action sampler: owns OpenPI model + policy, samples base actions
# ---------------------------------------------------------------------------

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


class ResidualRLBaseActionSampler:
    """Samples base actions from an OpenPI policy for residual RL."""

    def __init__(
        self,
        config: OnlineTrainConfig,
        *,
        checkpoint_manager: _checkpoints.CheckpointManager | None = None,
        resuming: bool = False,
        data_loader: Any = None,
    ) -> None:
        self._rng = jax.random.key(config.seed)
        init_rng, self._rng = jax.random.split(self._rng, 2)

        self._mesh = sharding.make_mesh(config.fsdp_devices)
        self._sharding_spec = jax.sharding.NamedSharding(
            jax.sharding.Mesh(
                mesh_utils.create_device_mesh((len(jax.devices()),)),
                axis_names=("batch",),
            ),
            jax.sharding.PartitionSpec("batch"),
        )

        # Initialize checkpoint manager if not provided.
        if checkpoint_manager is None:
            checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
                config.checkpoint_dir,
                keep_period=config.keep_period,
                overwrite=config.overwrite,
                resume=config.resume,
            )
        self._checkpoint_manager = checkpoint_manager

        # Load model weights.
        train_state, self._train_state_sharding = init_train_state(
            config, init_rng, self._mesh, resume=resuming,
        )
        jax.block_until_ready(train_state)
        logging.info(
            f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}"
        )

        if resuming:
            train_state = _checkpoints.restore_state(
                checkpoint_manager, train_state, data_loader,
            )

        self._train_state = train_state
        params = (
            train_state.ema_params
            if train_state.ema_params is not None
            else train_state.params
        )
        self.model = nnx.merge(train_state.model_def, params)

        # Load policy (for tokenization / action sampling), then drop its
        # redundant copy of the model to free memory.
        checkpoint_dir = _resolve_policy_checkpoint_dir(config, checkpoint_manager)
        self._policy = policy_config.create_trained_policy(config, checkpoint_dir)
        self._drop_policy_model()

        self.action_dim = int(self._policy.action_dim)
        self.action_horizon = int(self._policy.action_horizon)

    @property
    def train_state(self) -> training_utils.TrainState:
        return self._train_state

    @property
    def train_state_sharding(self) -> Any:
        return self._train_state_sharding

    @property
    def checkpoint_manager(self) -> _checkpoints.CheckpointManager:
        return self._checkpoint_manager

    def _drop_policy_model(self) -> None:
        if getattr(self._policy, "_is_pytorch_model", False):
            return
        self._policy._model = None
        for attr in ("_sample_actions", "_get_prefix_rep"):
            if hasattr(self._policy, attr):
                setattr(self._policy, attr, None)
        gc.collect()

    def sample(self, obs: Dict[str, Any], batch_size: int) -> np.ndarray:
        """Sample base actions from the policy given processed observations."""
        rng, self._rng = jax.random.split(self._rng)
        noise = jax.random.normal(
            rng, (batch_size, self.action_horizon, self.action_dim),
        )
        outputs = self._policy.infer_with_model(
            model=self.model,
            obs=obs,
            noise=noise,
            sharding_spec=self._sharding_spec,
        )
        return outputs["actions"]


# ---------------------------------------------------------------------------
# ResidualRLVectorEnv
# ---------------------------------------------------------------------------

gym_old_venv_step_type = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
gym_new_venv_step_type = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class ResidualRLVectorEnv(SubprocVectorEnv):
    """Vector env that samples base actions and applies residual corrections."""

    def __init__(
        self,
        env_fns: List,
        *,
        config: OnlineTrainConfig,
        task_description: list[str],
        residual_action_clip_range: tuple[float, float] = (-1.0, 1.0),
        checkpoint_manager: _checkpoints.CheckpointManager | None = None,
        resuming: bool = False,
        data_loader: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(env_fns, **kwargs)

        assert len(task_description) == self.env_num, (
            f"Expected {self.env_num} task descriptions, got {len(task_description)}"
        )
        self._task_description = task_description
        self._query_frequency = int(config.collect.replan_steps)
        self._resize_image = int(config.collect.resize_image)
        self._residual_action_clip_range = residual_action_clip_range

        self._sampler = ResidualRLBaseActionSampler(
            config,
            checkpoint_manager=checkpoint_manager,
            resuming=resuming,
            data_loader=data_loader,
        )
        self._last_obs: Optional[Dict[str, Any]] = None
        self._base_actions: Optional[np.ndarray] = None
        self._query_count = 0

    @property
    def sampler(self) -> ResidualRLBaseActionSampler:
        return self._sampler

    # ----- observation processing -----

    def _select_latest_frame(self, observations: Dict) -> Dict:
        """Query wrappers return [B, T, ...]; keep only the latest frame."""
        def _pick_last(x):
            arr = np.asarray(x)
            if arr.ndim >= 2 and arr.shape[1] == self._query_frequency:
                return arr[:, -1]
            return arr
        return jax.tree_util.tree_map(_pick_last, observations)

    def _process_obs_for_pi0(self, observations: Dict) -> Dict[str, Any]:
        current_obs = self._select_latest_frame(observations)

        _IMAGE_AND_STATE_KEYS = frozenset({
            "image", "wrist_image", "state",
            "exterior_image_1_left", "wrist_image_left",
            "joint_position", "gripper_position",
        })

        processed: dict[str, Any] = {}
        for key, val in current_obs.items():
            if key == "base_action":
                continue
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

        processed["prompt"] = self._task_description[0] if self._task_description else ""
        return processed

    def _infer_batch_size(self, observations: Dict[str, Any]) -> int:
        """Infer batch size from processed observation tensors."""
        state = observations.get("observation/state")
        if state is not None:
            state_arr = np.asarray(state)
            return int(state_arr.shape[0]) if state_arr.ndim > 1 else 1

        for image_key in ("observation/image", "observation/wrist_image"):
            image = observations.get(image_key)
            if image is None:
                continue
            image_arr = np.asarray(image)
            return int(image_arr.shape[0]) if image_arr.ndim >= 4 else 1

        obs_leaves = jax.tree_util.tree_leaves(observations)
        if not obs_leaves:
            raise ValueError("No observation leaves found for policy inference.")
        first_leaf = np.asarray(obs_leaves[0])
        return int(first_leaf.shape[0]) if first_leaf.ndim > 1 else 1

    # ----- base action management -----

    def _clear_base_actions(self) -> None:
        self._base_actions = None
        self._query_count = 0

    def _sample_base_actions(self, obs: Dict) -> np.ndarray:
        processed_obs = self._process_obs_for_pi0(obs)
        batch_size = self._infer_batch_size(processed_obs)
        return self._sampler.sample(processed_obs, batch_size)

    @staticmethod
    def _attach_base_action(
        observation: Dict[str, Any],
        base_action: np.ndarray,
    ) -> Dict[str, Any]:
        return {**observation, "base_action": base_action}

    # ----- reset / step -----

    def reset(
        self,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
        **kwargs: Any,
    ):
        reset_returns = super().reset(id, **kwargs)
        self._clear_base_actions()

        if isinstance(reset_returns, tuple):
            obs, info = reset_returns
        else:
            obs, info = reset_returns, None

        logging.info("[reset: sampling base actions]")
        self._base_actions = self._sample_base_actions(obs)
        base_action = self._base_actions[:, [0]]
        obs_with_base = self._attach_base_action(obs, base_action)
        self._last_obs = obs_with_base

        return (obs_with_base, info) if info is not None else obs_with_base

    def step(
        self,
        residual_action: np.ndarray,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ):
        if id is not None:
            raise NotImplementedError(
                "Partial stepping is not supported due to state tracking complexity."
            )
        assert self._last_obs is not None, "Call reset() before step()."

        lo, hi = self._residual_action_clip_range
        residual_action = np.clip(residual_action, lo, hi)

        base_action = self._last_obs["base_action"]
        actions = base_action + residual_action

        return_stacks = super().step(actions, id)
        obs_stack = return_stacks[0]

        self._query_count += 1
        if self._query_count >= self._sampler.action_horizon:
            self._clear_base_actions()
            logging.info("[step: re-sampling base actions]")
            self._base_actions = self._sample_base_actions(obs_stack)

        next_base_action = self._base_actions[:, [self._query_count]]
        obs_with_base = self._attach_base_action(obs_stack, next_base_action)
        self._last_obs = obs_with_base

        return (obs_with_base, *return_stacks[1:])


# ---------------------------------------------------------------------------
# Environment factory
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


def residual_rl_wrap_env(
    env_fn,
    config: _config.OnlineTrainConfig,
    task_description: list[str] | str,
    residual_action_clip_range: tuple[float, float] = (-1.0, 1.0),
    env_num: int | None = None,
) -> tuple[ResidualRLVectorEnv, list[str]]:
    """Build a ResidualRLVectorEnv with all necessary wrappers."""
    if env_num is None:
        env_num = int(config.collect.env_num)
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

    env = ResidualRLVectorEnv(
        env_factories,
        config=config,
        task_description=task_description,
        residual_action_clip_range=residual_action_clip_range,
    )
    env.seed(int(config.seed))
    return env, task_description
