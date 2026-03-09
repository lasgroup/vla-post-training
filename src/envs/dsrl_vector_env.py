import functools
import gc
import logging
import os
import weakref
import cloudpickle
import ctypes
import gymnasium as gym
import warnings
import time
from typing import Any, Dict, List, Callable, Optional, Tuple, Union

import etils.epath as epath
import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import mesh_utils

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms
from openpi.policies import policy_config
from openpi_client import image_tools
from src.rl.agent import Agent
from src.rl.filtered_sft_agent.update import train_step
from src.rl.replay_buffer import ShardedReplayBuffer
from src.rl.types import StepData
from src.training.data_loader import create_data_loader
from src.envs.wrappers import Pi0ObservationWrapper, QueryFrequencyWrapper
from src.envs.venv import SubprocVectorEnv, DummyVectorEnv
from src.rl.agent import Agent, EnvFn
#from src.rl.dsrl_agent.dsrl_vector_env_config import EnvConfig
#from src.rl.dsrl_agent.dsrl_vector_env_config import _env_config


import dataclasses
import difflib
import tyro
from openpi.training.config import (
    _CONFIGS,
    TrainConfig,
    DataConfig,
    pi0_config,
    LeRobotLiberoDataConfig,
)
from typing import Sequence

import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders


@dataclasses.dataclass(frozen=True)
class EnvCollectionConfig:
    collect_interval: int = 200
    env_num: int = 4
    env_resolution: int = 256
    resize_image: int = 224
    add_states: bool = True
    num_rollouts: int = 50
    tasks: list[str] = dataclasses.field(default_factory=lambda: ["libero_90_59"])
    replan_steps: int = 5
    num_steps_wait: int = 10
    add_per_step_data: bool = True
    obs_prefix_key: str = "pi0"


@dataclasses.dataclass(frozen=True)
class EnvDataConfig(DataConfig):
    # additional LeRobot repo paths to include (keeps repos separate but concatenates them for training)
    additional_repo_paths: Sequence[str] = ()


@dataclasses.dataclass(frozen=True)
class EnvConfig(TrainConfig):
    # additional configs for online training
    collect: EnvCollectionConfig = EnvCollectionConfig()
    discount: float = 0.99


_env_config = EnvConfig(
            name="pi05_libero_online",
            model=pi0_config.Pi0Config(
                pi05=True, action_horizon=10, discrete_state_input=False
            ),
            data=LeRobotLiberoDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=EnvDataConfig(prompt_from_task=True),
                extra_delta_transform=False,
            ),
            batch_size=256,
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=100,  # override default warmup steps
                peak_lr=5e-5,
                decay_steps=1_000_000,
                decay_lr=5e-5,
            ),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_libero/params"
            ),
            pytorch_weight_path="/path/to/your/pytorch_weight_path",
            num_train_steps=10_000,
            num_workers=4,  # override default num_workers
            exp_name="test",
            resume=True,
            checkpoint_base_dir="/capstor/scratch/cscs/chenhli/checkpoints",
        )




gym_old_venv_step_type = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
gym_new_venv_step_type = Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]
warnings.simplefilter("once", DeprecationWarning)
_NP_TO_CT = {
    np.bool_: ctypes.c_bool,
    np.uint8: ctypes.c_uint8,
    np.uint16: ctypes.c_uint16,
    np.uint32: ctypes.c_uint32,
    np.uint64: ctypes.c_uint64,
    np.int8: ctypes.c_int8,
    np.int16: ctypes.c_int16,
    np.int32: ctypes.c_int32,
    np.int64: ctypes.c_int64,
    np.float32: ctypes.c_float,
    np.float64: ctypes.c_double,
}


def _load_weights_and_validate(
    loader: _weight_loaders.WeightLoader, params_shape: at.Params
) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(
        expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True
    )

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {
            k: v
            for k, v in traverse_util.flatten_dict(loaded_params).items()
            if not isinstance(v, jax.ShapeDtypeStruct)
        }
    )

@at.typecheck
def init_train_state(
    config: EnvConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(
        config.optimizer, config.lr_schedule, weight_decay_mask=None
    )

    def init(
        rng: at.KeyArrayLike, partial_params: at.Params | None = None
    ) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            nnx.replace_by_pure_dict(state, partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
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
        config.weight_loader, nnx.to_pure_dict(train_state_shape.params)
    )
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


class DSRLVectorEnv(SubprocVectorEnv):
    """Vectorized environment wrapper based on subprocess for DSRL."""
    def __init__(self, env_fns: List[Callable[[], gym.Env]], **kwargs: Any) -> None:
        super().__init__(env_fns, **kwargs)
        self._config = _env_config
        self._rng = jax.random.key(self._config.seed)
        init_rng, self._rng = jax.random.split(self._rng, 2)

        # set up sharding
        self._mesh = sharding.make_mesh(self._config.fsdp_devices)
        self._policy_sharding_spec = jax.sharding.NamedSharding(
            jax.sharding.Mesh(
                mesh_utils.create_device_mesh((len(jax.devices()),)),
                axis_names=("batch",),
            ),
            jax.sharding.PartitionSpec(
                "batch",
            ),
        )
        # Initialize checkpoint manager.
        self._checkpoint_manager, self._resuming = (
            _checkpoints.initialize_checkpoint_dir(
                self._config.checkpoint_dir,
                keep_period=self._config.keep_period,
                overwrite=self._config.overwrite,
                resume=self._config.resume,
            )
        )
        # Initialize train state.
        self._train_state, self._train_state_sharding = init_train_state(
            self._config, init_rng, self._mesh, resume=self._resuming
        )
        jax.block_until_ready(self._train_state)
        logging.info(
            f"Initialized train state:\n{training_utils.array_tree_to_info(self._train_state.params)}"
        )
        if self._resuming:
            self._train_state = _checkpoints.restore_state(
                self._checkpoint_manager, self._train_state, self._data_loader
            )
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
            if params_path.endswith("/params"):
                policy_checkpoint_dir = params_path[: -len("/params")]
            else:
                policy_checkpoint_dir = params_path
        if policy_checkpoint_dir is None:
            policy_checkpoint_dir = self._checkpoint_manager._directory
            if not (policy_checkpoint_dir / "params").exists():
                raise FileNotFoundError(
                    "Policy checkpoint not found. Set OPENPI_POLICY_CHECKPOINT_DIR to a checkpoint "
                    "containing 'params' (e.g. .../openpi-assets/checkpoints/pi05_libero)."
                )
        self._policy = policy_config.create_trained_policy(
            self._config,
            policy_checkpoint_dir,
        )
        # This learner always calls `infer_with_model(...)` with the current train-state model.
        # Drop policy-owned model references to avoid keeping an extra model copy in memory.
        self._drop_policy_model()
        self._last_obs = None

    def _drop_policy_model(self):
        # For PyTorch policies `infer_with_model` ignores the provided model and uses internal state,
        # so we cannot safely drop the internal model there.
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
        # These JAX callables are created from bound model methods and can capture model state.
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

    def _process_obs_for_pi0(
        self,
        observations: Dict,
        task_description: str | None = None,
    ) -> Dict[str, Any]:
        # With per-step collection enabled, each env step contains a short chunk of
        # observations. Use the most recent one for policy inference.
        if self._config.collect.add_per_step_data:
            current_obs = jax.tree_util.tree_map(
                lambda x: x[:, -1], observations["observation"]
            )
        else:
            current_obs = observations["observation"]

        processed_obs = {}
        prompt_in_obs = False
        for key, val in current_obs.items():
            # Extract all observations relevant for the policy
            if self._config.collect.obs_prefix_key in key:
                obs_key = key.split(f"{self._config.collect.obs_prefix_key}/")[-1]
                if obs_key == "prompt":
                    prompt_in_obs = True
                    processed_obs[obs_key] = val
                else:
                    if "image" in obs_key and self._config.collect.resize_image > 0:
                        # Rescale images
                        val = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(
                                val,
                                self._config.collect.resize_image,
                                self._config.collect.resize_image,
                            )
                        )
                    obs_key = f"observation/{obs_key}"
                    processed_obs[obs_key] = val
        # If prompt is not stored in obs, we add the default prompt here.
        if not prompt_in_obs:
            assert task_description is not None, "No task description is provided"
            processed_obs["prompt"] = task_description
        return processed_obs

    def _resolve_reset_ids(
        self, id: Optional[Union[int, List[int], np.ndarray]]
    ) -> List[int]:
        return [int(i) for i in self._wrap_id(id)]

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
                _scatter_update, self._last_obs, reset_obs_with_prefix
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
        """Reset the state of some envs and return initial observations.
        Perform a dummy step to get the prefix representation.
        """        
        reset_ids = self._resolve_reset_ids(id)
        reset_returns = super().reset(id, **kwargs)
        if isinstance(reset_returns, tuple):
            obs, info = reset_returns
            returns_info = True
        else:
            obs = reset_returns
            info = None
            returns_info = False

        processed_obs = self._process_obs_for_pi0(
            obs, task_description="dummy"
        )
        dummy_noise = jnp.zeros(
            (
                len(obs["observation"]["state"]),
                self._policy.action_horizon,
                self._policy.action_dim,
            )
        )
        outputs = self._policy.infer_with_model(
            model=self.model,
            obs=processed_obs,  # Use processed obs
            noise=dummy_noise,
            return_prefix_rep=True,
            sharding_spec=self._policy_sharding_spec,
        )
        prefix_rep = outputs["prefix_rep"]

        if isinstance(obs, dict):
            obs_with_prefix = dict(obs)
            obs_with_prefix["prefix_rep"] = np.array(prefix_rep)
        else:
            obs_with_prefix = np.concatenate([obs, prefix_rep], axis=1)

        # Keep a full-batch cache for policy inference in step(), even when reset(id=...)
        # returns only a subset.
        self._update_last_obs_cache(reset_ids, obs_with_prefix)

        if returns_info:
            return obs_with_prefix, info
        return obs_with_prefix

    def step(
        self,
        noise: np.ndarray,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ) -> Union[gym_old_venv_step_type, gym_new_venv_step_type]:
        """Run one timestep of the environment with the given the DSRL policy noise.
        The noise is passed to the base policy's infer_with_model method.

        Args:
            noise: The noise to take in the environment.
            id: The id of the environment to take the action in.

        Returns:
            The observations with prefix representation, rewards, dones, and infos.
        """
        if id is not None:
             raise NotImplementedError("Partially stepping DSRLVectorEnv is not supported yet because of state tracking complexity.")

        # Use the stored _last_obs
        processed_obs = self._process_obs_for_pi0(
             self._last_obs, task_description="dummy"
        )
        
        # Vector envs expect a batch dimension for actions. Policy inference
        # unbatches when batch_size == 1, so add it back for single-env runs.
        outputs = self._policy.infer_with_model(
            model=self.model,
            obs=processed_obs,
            noise=noise,
            return_prefix_rep=True,
            sharding_spec=self._policy_sharding_spec,
        )
        actions = outputs["actions"]
        prefix_rep = outputs["prefix_rep"]
        
        return_stacks = super().step(actions, id)
        obs_stack = return_stacks[0]
        
        if isinstance(obs_stack, dict):
             obs_stack["prefix_rep"] = np.array(prefix_rep)
             self._last_obs = obs_stack
             return (obs_stack, *return_stacks[1:]) # type: ignore
             
        obs_stack = np.concatenate([obs_stack, prefix_rep], axis=1)
        self._last_obs = obs_stack
        return (obs_stack, *return_stacks[1:])  # type: ignore
