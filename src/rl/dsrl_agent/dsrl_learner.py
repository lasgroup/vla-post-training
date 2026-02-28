import functools
import gc
import logging
import os
import weakref
from typing import Any, Dict

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
from src.rl.filtered_sft_agent.filtered_sft_learner import FilteredSFTLearner
from rl.dsrl_agent.update import train_step
from src.rl.replay_buffer import ShardedReplayBuffer
from src.rl.types import StepData
from src.training.config import OnlineTrainConfig
from src.training.data_loader import create_data_loader
from src.envs.wrappers import Pi0ObservationWrapper, QueryFrequencyWrapper
from src.envs.venv import SubprocVectorEnv, DummyVectorEnv
from src.rl.dsrl_agent.dsrl_vector_env import DSRLVectorEnv
from src.rl.agent import Agent, EnvFn


def get_env_and_agent_for_dsrl(env_fn, config, task_description, env_class):
    env = dsrl_wrap_env(
        env_fn=env_fn,
        config=config,
        task_description=task_description,
        env_class=env_class,
    )
    agent = DSRLLearner(config)
    return env, agent


def dsrl_wrap_env(env_fn: EnvFn, config, task_description: str, env_class: str):
    env_num = config.collect.env_num
    add_states = config.collect.add_states
    obs_prefix_key = config.collect.obs_prefix_key
    replan_steps = config.collect.replan_steps
    seed = config.seed
    discount = config.discount
    add_per_step_data = config.collect.add_per_step_data
    env_factories = []
    for i in range(env_num):

        def _make_env(rank=i):
            # Create the base environment
            base_env = env_fn(rank)
            # Add Pi related obs to the environment
            base_env = Pi0ObservationWrapper(
                env=base_env,
                env_class=env_class,
                task_description=task_description,
                add_states=add_states,
                pi0_obs_prefix=obs_prefix_key,
            )
            # Add query frequency wrapper to rollout action chunks
            base_env = QueryFrequencyWrapper(
                env=base_env,
                query_frequency=replan_steps,
                discount=discount,
                store_full_transitions=add_per_step_data,
                pre_step_filter=lambda x: np.where(np.abs(x) < 0.0011, 0.0, x),
            )
            return base_env

        env_factories.append(_make_env)

    env = (
        DSRLVectorEnv(env_factories)
        if env_num > 1
        else DummyVectorEnv(env_factories)
    )
    # This sets the seed for all environment all at once to be [seed, seed + i, ..., seed + num_envs]
    env.seed(
        seed
    )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    # re-use training seed
    return env


class DSRLLearner(FilteredSFTLearner):
    def __init__(self, config: OnlineTrainConfig):
        self._config = config

        if self._config.batch_size % jax.device_count() != 0:
            raise ValueError(
                f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
            )
        jax.config.update(
            "jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser())
        )
        self._rng = jax.random.key(self._config.seed)
        init_rng, self._rng = jax.random.split(self._rng, 2)

        # set up sharding
        self._mesh = sharding.make_mesh(self._config.fsdp_devices)
        self._data_sharding = jax.sharding.NamedSharding(
            self._mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
        )
        self._replicated_sharding = jax.sharding.NamedSharding(
            self._mesh, jax.sharding.PartitionSpec()
        )
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

        # initialize data loader
        self._data_loader = create_data_loader(
            config, sharding=self._data_sharding, shuffle=True
        )
        self._data_iter = iter(self._data_loader)
        self._online_data_buffer = self._get_online_replay_buffer(self._data_sharding)
        self._collection_success_episodes = 0
        # batch = next(data_iter)
        # logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")
        # log_images(batch)

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

        # prepare train_step
        self._train_step = jax.jit(
            functools.partial(train_step, config),
            in_shardings=(
                self._replicated_sharding,
                self._train_state_sharding,
                self._data_sharding,
            ),
            out_shardings=(self._train_state_sharding, self._replicated_sharding),
            donate_argnums=(1,),
        )

        # Create temporary episode storage
        self._episode_storage = [[] for _ in range(self._config.collect.env_num)]

        # Create policy for data collection
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

        self._policy = self._create_trained_policy(
            self._config,
        )
        
    def _create_trained_policy(self, config: OnlineTrainConfig):
        # TODO: implement a simple MLP policy for DSRL
        raise NotImplementedError
        
    def _sample_action(
        self,
        observations: Dict,
        rng: jax.random.PRNGKey,
        train_state: training_utils.TrainState,
        batch_actions: bool = True,
    ) -> np.ndarray:
        params = (
            train_state.ema_params
            if train_state.ema_params is not None
            else train_state.params
        )
        model = nnx.merge(train_state.model_def, params)
        batch_size = self._infer_policy_batch_size(observations)
        # noise = jax.random.normal(
        #     rng, (batch_size, self._policy.action_horizon, self._policy.action_dim)
        # )
        # Vector envs expect a batch dimension for actions. Policy inference
        # unbatches when batch_size == 1, so add it back for single-env runs.
        noises = self._policy.infer_with_model(
            model=model,
            obs=observations,
            # noise=noise,
            sharding_spec=self._policy_sharding_spec,
        )["noises"]
        if batch_noises and noises.ndim == 2:
            noises = noises[np.newaxis, ...]
        return noises

    def _online_batch_to_dsrl_batch(
        self, online_batch: Dict[str, Any]
    ) -> tuple[_model.Observation, _model.Actions]:
        return (
            _model.Observation.from_dict(online_batch["observation"]),
            online_batch["actions"],
        )

    def save_episode(self, is_success: bool = False, env_index: int = 0, **kwargs):
        # Keep all episodes for DSRL
        super().save_episode(True, env_index, **kwargs)

    def update(self):
        self.training_steps += 1
        batch = next(self._data_iter)
        use_online = (
            self._online_data_buffer.size >= self._online_data_buffer.batch_size
        )
        if use_online:
            online_batch_raw = self._online_data_buffer.sample()
            online_batch = self._online_batch_to_dsrl_batch(online_batch_raw)
            # online_ratio controls whether we fully switch to online data or mix by
            # simple concatenation along the batch dimension.
            online_ratio = float(getattr(self._config.collect, "online_ratio", 0.5))
            if online_ratio >= 1.0:
                batch = online_batch
            elif online_ratio > 0:
                batch = jax.tree.map(
                    lambda x, y: jnp.concatenate([x, y], axis=0),
                    batch,
                    online_batch,
                )
        train_rng, self._rng = jax.random.split(self._rng)
        train_state = self._train_state
        with sharding.set_mesh(self._mesh):
            train_state, info = self._train_step(train_rng, train_state, batch)
        self._train_state = train_state
        return info
