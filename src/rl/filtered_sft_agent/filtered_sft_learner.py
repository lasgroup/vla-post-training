from src.rl.agent import Agent
from src.rl.filtered_sft_agent.update import train_step
from src.rl.types import StepData
from src.training.config import OnlineTrainConfig
from src.training.data_loader import create_data_loader
from typing import Dict
import gc
import numpy as np
import os
import shutil
import weakref

import functools
import logging

import jax
import etils.epath as epath
import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax.numpy as jnp
from jax.experimental import mesh_utils
from typing import Any
import lerobot.datasets.lerobot_dataset as lerobot_dataset

import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
from openpi.policies import policy_config
from openpi_client import image_tools


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
    config: OnlineTrainConfig,
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


class FilteredSFTLearner(Agent):
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

        # initialize checkopointing, wandb
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
        self._collected_data_paths = []
        # batch = next(data_iter)
        # logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")
        # log_images(batch)

        # initialize training_state
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

        self._policy = policy_config.create_trained_policy(
            self._config,
            policy_checkpoint_dir,
        )
        # This learner always calls `infer_with_model(...)` with the current train-state model.
        # Drop policy-owned model references to avoid keeping an extra model copy in memory.
        self._drop_policy_model()

        # We use  a dummy data loader to store episodic data
        self._lerobot_dataset = None

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

    def _setup_lerobot_dataset(self, step: int | None = None):
        if step is None:
            step = self.training_steps
        allowed_keys = {"image", "wrist_image", "state", "actions"}
        new_data_path = self._checkpoint_manager._directory / "data" / str(step)
        if new_data_path.exists():
            shutil.rmtree(new_data_path)
        self._lerobot_dataset = lerobot_dataset.LeRobotDataset.create(
            repo_id=self._config.data.repo_id,
            root=new_data_path,
            robot_type="panda",
            fps=10,
            features={
                k: v
                for k, v in lerobot_dataset.LeRobotDatasetMetadata(
                    self._config.data.repo_id
                ).features.items()
                if k in allowed_keys
            },
            image_writer_threads=10,
            image_writer_processes=5,
        )

    def _process_obs_for_pi0(
        self,
        observations: Dict,
        task_description: str | None = None,
    ):
        # If we are stacking all the observations in the
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
                obs_key = key.split(self._config.collect.obs_prefix_key)[-1]
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

    def _sample_action(
        self,
        observations: Dict,
        rng: jax.random.PRNGKey,
        train_state: training_utils.TrainState,
        batch_actions: bool = True,
    ):
        # Define model
        model = nnx.merge(train_state.model_def, train_state.params)
        # Convert observation for the policy
        state = observations.get("observation/state")
        if state is None:
            raise KeyError("Expected 'observation/state' in processed observations.")
        batch_size = int(state.shape[0]) if state.ndim > 1 else 1
        noise = jax.random.normal(
            rng, (batch_size, self._policy.action_horizon, self._policy.action_dim)
        )
        # Vector envs expect a batch dimension for actions. Policy inference
        # un batches when batch_size == 1, so add it back for single-env runs.
        actions = self._policy.infer_with_model(
            model=model,
            obs=observations,
            noise=noise,
            sharding_spec=self._policy_sharding_spec,
        )["actions"]
        if batch_actions and actions.ndim == 2:
            actions = actions[np.newaxis, ...]
        return actions

    def eval_actions(self, observations: np.ndarray | Dict, **kwargs) -> np.ndarray:
        # For OpenPI sample and eval actions behave the same way.
        return self.sample_actions(observations, **kwargs)

    def sample_actions(self, observations: np.ndarray | Dict, **kwargs) -> np.ndarray:
        task_description = kwargs.get("task_description")
        batch_actions = kwargs.get("batch_actions")
        if batch_actions is None:
            batch_actions = False
        rng, self._rng = jax.random.split(self._rng)
        processed_obs = self._process_obs_for_pi0(
            observations, task_description=task_description
        )
        actions = self._sample_action(
            observations=processed_obs,
            rng=rng,
            train_state=self._train_state,
            batch_actions=batch_actions,
        )
        return np.asarray(actions, dtype=np.float32)

    def save_checkpoint(self, step: int | None = None):
        if step is None:
            step = self.training_steps
        _checkpoints.save_state(
            self._checkpoint_manager, self._train_state, self._data_loader, step
        )
        self._checkpoint_manager.wait_until_finished()

    def add_data(self, step_data: StepData):
        def get_env_value(vec, env_id):
            return jax.tree.map(lambda x: x[env_id], vec)

        for i in range(self._config.collect.env_num):
            self._episode_storage[i].append(get_env_value(step_data, i))

    def save_episode(self, is_success: bool = False, env_index: int = 0, **kwargs):
        # Extract episode data from storage
        episode_data = self._episode_storage[env_index]
        # Empty the storage now for the next episode
        self._episode_storage[env_index] = []
        if not is_success:
            # We are running filtered SFT to so we only add successful episode.
            return
        assert self._lerobot_dataset is not None, "LeRobot Dataset is not initiliazed"
        task_description = kwargs.get("task_description")

        def process_frame(ob):
            frame = {}
            # Extract actions and observations from total_obs
            obs, action = ob["observation"], ob["action"]
            for key, val in obs.items():
                if self._config.collect.obs_prefix_key in key:
                    obs_key = key.split(self._config.collect.obs_prefix_key)[-1]
                    frame[obs_key] = val
            frame["actions"] = action
            return frame

        if self._config.collect.add_per_step_data:
            # Add all the per time-step transitions one by one.
            total_frames = len(episode_data)
            for n_frame, ep in enumerate(episode_data):
                ep_obs, terminate, truncate = (
                    ep["observation"],
                    ep["terminate"],
                    ep["truncate"],
                )
                total_chunks = self._config.collect.replan_steps
                # For the last frame where termination occurred check at which step this was observed.
                if n_frame == total_frames - 1:
                    done = np.logical_or(terminate, truncate)
                    done_indices = np.where(done)[0]
                    if len(done_indices) > 0:
                        total_chunks = done_indices[0]
                for step in range(total_chunks):
                    obs = jax.tree.map(lambda x: x[step], ep_obs)
                    self._lerobot_dataset.add_frame(
                        process_frame(obs), task=str(task_description)
                    )
        else:
            for ep in episode_data:
                self._lerobot_dataset.add_frame(process_frame(ep["observation"]))
        self._lerobot_dataset.save_episode()

    def start_data_collection(self, step: int | None = None):
        # Reset episode storage
        self._episode_storage = [[] for _ in range(self._config.collect.env_num)]
        # Initialize lerobot dataset
        self._setup_lerobot_dataset(step)
        # TODO: Move train state to CPU and policy state to GPU?

    def end_data_collection(self, step: int | None = None):
        # Reset episode storage
        self._episode_storage = [[] for _ in range(self._config.collect.env_num)]
        if step is None:
            step = self.training_steps
        # Delete/Reset Lerobot dataset
        if self._lerobot_dataset.num_episodes > 0:
            self._collected_data_paths.append(
                self._checkpoint_manager._directory / "data" / str(step)
            )
        del self._lerobot_dataset
        self._lerobot_dataset = None
        # Reinitialize the data loader with the new data.
        self._data_loader = create_data_loader(
            self._config,
            sharding=self._data_sharding,
            shuffle=True,
            collected_data_paths=self._collected_data_paths,
        )
        self._data_iter = iter(self._data_loader)
        # TODO: Move policy state to CPU and train state to GPU?

    def update(self):
        self.training_steps += 1
        batch = next(self._data_iter)
        train_rng, self._rng = jax.random.split(self._rng)
        train_state = self._train_state
        with sharding.set_mesh(self._mesh):
            train_state, info = self._train_step(train_rng, train_state, batch)
        self._train_state = train_state
        return info
