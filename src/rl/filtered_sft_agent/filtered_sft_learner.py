import dataclasses
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
from src.rl.filtered_sft_agent.update import train_step
from src.rl.replay_buffer import ShardedReplayBuffer
from src.rl.types import StepData
from src.training.config import OnlineTrainConfig, FilteredSFTLearnerConfig
from src.training.data_loader import create_data_loader
from src.envs.wrappers import (
    TimeToSuccessAsRewardWrapper,
    Pi0ObservationWrapper,
    PrefixEmbeddingVectorEnvWrapper,
    QueryFrequencyWrapper,
)
from src.envs.venv import SubprocVectorEnv, DummyVectorEnv
from src.rl.agent import Agent, EnvFn
from src.rl.prefix_embedding import PREFIX_EMBEDDING_NAME
from src.training.runtime_state import (
    load_resume_state,
    restore_train_state,
)


def filtered_sft_wrap_env(
    env_fn: EnvFn,
    config,
    task_description: list[str],
    env_num: int | None = None,
):
    env_num = env_num if env_num is not None else config.collect.env_num
    replan_steps = config.collect.replan_steps
    env_class = config.collect.domain
    seed = config.seed
    env_factories = []
    for i in range(env_num):

        def _make_env(rank=i):
            # Create the base environment
            base_env = env_fn(rank)
            if config.collect.use_time_to_success_as_reward:
                base_env = TimeToSuccessAsRewardWrapper(base_env)
            # Add Pi related obs to the environment
            base_env = Pi0ObservationWrapper(
                env=base_env,
                env_class=env_class,
                task_description=task_description,
                molmo_config=getattr(config, "molmo", None),
            )
            # Add query-frequency wrapper to rollout action chunks.
            query_wrapper = (
                PrefixEmbeddingVectorEnvWrapper
                if config.collect.store_prefix_rep
                else QueryFrequencyWrapper
            )
            base_env = query_wrapper(
                env=base_env,
                query_frequency=replan_steps,
            )
            return base_env

        env_factories.append(_make_env)

    env = (
        SubprocVectorEnv(env_factories)
        if env_num > 1
        else DummyVectorEnv(env_factories)
    )
    # This sets the seed for all environment all at once to be [seed, seed + i, ..., seed + num_envs]
    env.seed(
        seed
    )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    # re-use training seed
    return env


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


def _get_post_step_action_filter(domain: str):
    if domain == "libero":
        # see https://arxiv.org/pdf/2501.09747, Appendix C - clipping low-magnitude actions
        # the LIBERO dataset seems to have been filtered accordingly
        return lambda x: np.where(np.abs(x) < 0.0011, 0.0, x)
    return lambda x: x


class FilteredSFTLearner(Agent):
    def __init__(self, config: OnlineTrainConfig):
        self._config = config
        self.post_step_action_filter = _get_post_step_action_filter(
            self._config.collect.domain
        )

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
        self._resume_state = None
        if self._config.resume and bool(getattr(self._config, "requeue", False)):
            self._resume_state = load_resume_state(self._config)
            if self._resume_state is not None:
                if not self._resuming:
                    raise ValueError(
                        "Found resumable runtime state, but checkpoint manager did not enter resume mode."
                    )
                logging.info(
                    "Found resumable state for %s at step %d (replay snapshot=%s, replay transitions=%d)",
                    self._config.checkpoint_dir,
                    self._resume_state.step,
                    self._resume_state.replay_snapshot_path,
                    self._resume_state.replay_size,
                )

        # initialize data loader
        assert (
            0.0 <= self._config.rl.online_ratio <= 1.0
        ), "Online ratio must be between 0 and 1."
        self._offline_batch_size = max(
            len(jax.devices()),
            int(self._config.batch_size * (1 - self._config.rl.online_ratio)),
        )
        self._data_loader = create_data_loader(
            config,
            batch_size=self._offline_batch_size,
            sharding=self._data_sharding,
            shuffle=True,
        )
        self._data_iter = iter(self._data_loader)
        self._online_data_buffer = self._get_online_replay_buffer()
        if self._resume_state is not None:
            restored_replay = self._online_data_buffer.restore_snapshot(
                self._resume_state.replay_snapshot_path
            )
            logging.info(
                "Restored replay buffer from %s (step=%d, transitions=%d)",
                restored_replay["path"],
                restored_replay["step"],
                restored_replay["size"],
            )
        self._collection_success_episodes = 0

        # Initialize train state.
        self._train_state, self._train_state_sharding = init_train_state(
            self._config, init_rng, self._mesh, resume=self._resuming
        )
        jax.block_until_ready(self._train_state)
        logging.info(
            f"Initialized train state:\n{training_utils.array_tree_to_info(self._train_state.params)}"
        )
        if self._resuming:
            self._train_state = restore_train_state(
                _checkpoints.restore_state,
                self._checkpoint_manager,
                self._train_state,
                self._data_loader,
                resume_state=self._resume_state,
            )
            if self._resume_state is not None:
                logging.info(
                    "Restored training checkpoint from %s at committed step %d",
                    self._config.checkpoint_dir,
                    self._resume_state.step,
                )
                self.training_steps = int(self._resume_state.step)
            else:
                self.training_steps = int(jax.device_get(self._train_state.step))
        else:
            self.training_steps = 0

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

        def _get_prefix_rep_with_model_fn(
            m: _model.BaseModel, observation: _model.Observation
        ):
            prefix_rep = m.get_prefix_rep(observation)
            # TODO: remove if below
            return prefix_rep[0] if isinstance(prefix_rep, tuple) else prefix_rep

        self._get_prefix_rep_with_model = nnx.jit(_get_prefix_rep_with_model_fn)

        # Create policy for data collection
        policy_checkpoint_dir = self._config.weight_loader.params_path[
            : -len("/params")
        ]
        self._policy = policy_config.create_trained_policy(
            self._config,
            policy_checkpoint_dir,
        )
        # This learner always calls `infer_with_model(...)` with the current train-state model.
        # Drop policy-owned model references to avoid keeping an extra model copy in memory.
        self._drop_policy_model()

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

    def _get_online_replay_buffer(
        self,
    ) -> ShardedReplayBuffer:

        # prepare transforms for preprocessing episode data into model input format
        data_config = self._data_loader.data_config()
        tt_types = (_transforms.TokenizePrompt, _transforms.TokenizeFASTInputs)
        token_transforms = [
            t for t in data_config.model_transforms.inputs if isinstance(t, tt_types)
        ]
        non_token_transforms = [
            t
            for t in data_config.model_transforms.inputs
            if not isinstance(t, tt_types)
        ]
        assert (
            len(token_transforms) == 1
        ), f"Expected exactly one token transform in the model transforms, but found {len(token_transforms)}."
        self._token_transform = token_transforms[0]
        self._pre_token_transform = _transforms.compose(
            [
                *data_config.repack_transforms.inputs,
                *data_config.data_transforms.inputs,
                _transforms.Normalize(
                    data_config.norm_stats, use_quantiles=data_config.use_quantile_norm
                ),
                *non_token_transforms,
            ]
        )
        self._token_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        # prepare dummy data for initializing the replay buffer
        # TODO: this might need to be updated to store prefixes
        obs_spec, act_spec = self._config.model.inputs_spec(batch_size=1)
        obs_spec_dict = obs_spec.to_dict()
        dummy_obs_dict = jax.tree.map(
            lambda spec: np.zeros(spec.shape, dtype=spec.dtype), obs_spec_dict
        )
        dummy_obs_dict = {k: v for k, v in dummy_obs_dict.items() if v is not None}
        if "image" in dummy_obs_dict:
            dummy_obs_dict["image"] = jax.tree.map(
                lambda v: v.astype(np.uint8), dummy_obs_dict["image"]
            )
        dummy_data = {
            "observation": dummy_obs_dict,
            "actions": np.zeros(act_spec.shape, dtype=act_spec.dtype),
            "next_observation": dummy_obs_dict,
            "reward": np.zeros((1,), dtype=np.float32),
            "mc_return": np.zeros((1,), dtype=np.float32),
            "discount": np.zeros((1,), dtype=np.float32),
        }
        logging.info(
            "Initializing online replay buffer (capacity=%d)",
            self._config.rl.buffer_capacity,
        )

        return ShardedReplayBuffer(
            dummy_data=dummy_data,
            max_capacity=self._config.rl.buffer_capacity,
            data_sharding=self._data_sharding,
            seed=self._config.seed,
            preprocess_fn=None,
            postprocess_fn=None,
            freeze_dict=False,
        )

    def _process_obs_for_pi0(
        self,
        observations: Dict,
        task_description: list[str],
    ) -> Dict[str, Any]:
        # With per-step collection enabled, each env step contains a short chunk of
        # observations. Use the most recent one for policy inference.
        obs = jax.tree_util.tree_map(lambda x: x[:, -1], observations)
        size = int(self._config.collect.resize_image)
        resize_fn = lambda x: image_tools.convert_to_uint8(
            image_tools.resize_with_pad(x, size, size)
        )
        obs = {k: resize_fn(v) if "image" in k else v for k, v in obs.items()}
        obs["prompt"] = task_description
        return obs

    def _sample_action(
        self,
        observations: Dict,
        rng: jax.random.PRNGKey,
        train_state: training_utils.TrainState,
        return_prefix_rep: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        params = (
            train_state.ema_params
            if train_state.ema_params is not None
            else train_state.params
        )
        model = nnx.merge(train_state.model_def, params)
        model.eval()
        first_obs = next(iter(observations.values()), None)
        if first_obs is None:
            raise ValueError("Observation dictionary is empty.")
        first_obs = np.asarray(first_obs)
        batch_size = first_obs.shape[0] if first_obs.ndim > 1 else 1
        noise = jax.random.normal(
            rng, (batch_size, self._policy.action_horizon, self._policy.action_dim)
        )
        # Vector envs expect a batch dimension for actions. Policy inference
        # unbatches when batch_size == 1, so add it back for single-env runs.
        num_devices = len(jax.devices())
        sharding_spec = (
            self._policy_sharding_spec if batch_size % num_devices == 0 else None
        )
        actions = self._policy.infer_with_model(
            model=model,
            obs=observations,
            noise=noise,
            return_prefix_rep=return_prefix_rep,
            sharding_spec=sharding_spec,
        )["actions"]

        if return_prefix_rep:
            actions, prefix = actions

        if batch_size == 1 and actions.ndim == 2:
            actions = actions[np.newaxis, ...]

        return (actions, prefix) if return_prefix_rep else actions

    def _generate_actions(
        self,
        observations: np.ndarray | Dict,
        task_description: list[str],
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        rng, self._rng = jax.random.split(self._rng)
        processed_obs = self._process_obs_for_pi0(
            observations, task_description=task_description
        )
        actions = self._sample_action(
            observations=processed_obs,
            rng=rng,
            train_state=self._train_state,
            return_prefix_rep=self._config.collect.store_prefix_rep,
        )
        # TODO: if store_prefix_rep is True, this will crash because (i) openpi output transforms
        # cannot process tuples and (ii) venvs do not accept tuples as input

        # TODO: check if casting is necessary
        if isinstance(actions, (tuple, list)):
            return tuple(np.asarray(x, dtype=np.float32) for x in actions)

        return np.asarray(actions, dtype=np.float32)

    def eval_actions(self, observations: np.ndarray | Dict, **kwargs) -> np.ndarray:
        return self._generate_actions(observations, **kwargs)

    def sample_actions(
        self, observations: np.ndarray | Dict, **kwargs
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        return self._generate_actions(observations, **kwargs)

    def _online_batch_to_sft_batch(
        self, online_batch: Dict[str, Any]
    ) -> tuple[_model.Observation, _model.Actions]:
        return (
            _model.Observation.from_dict(online_batch["observation"]),
            online_batch["actions"],
        )

    def save_checkpoint(self, step: int | None = None):
        if step is None:
            step = self.training_steps
        _checkpoints.save_state(
            self._checkpoint_manager, self._train_state, self._data_loader, step
        )
        self._checkpoint_manager.wait_until_finished()

    def add_data(self, step_data: StepData):
        for i in range(self._config.collect.env_num):
            self._episode_storage[i].append(jax.tree.map(lambda x: x[i], step_data))

    def _attach_prefix_embeddings_to_episode_data(
        self,
        episode_data: list[Dict[str, Any]],
        task_description: str,
    ) -> None:
        """Unpack `(actions, prefix)` payloads and align prefixes to transitions.

        - Replaces each `ep["action"]` payload with pure actions.
        - Writes the unpacked prefix embedding to `ep["observation"]`.
        - Writes the shifted prefix embedding to `ep["next_observation"]`
          using the next step's payload.
        - Computes the final next-observation prefix with the current model.
        """

        # TODO: this function is untested
        # TODO: check if last observation needs to be taken
        next_observation = jax.tree.map(
            lambda x: x[[-1]], episode_data[-1]["next_observation"]
        )
        processed_obs = self._process_obs_for_pi0(next_observation, task_description)
        params = (
            self._train_state.ema_params
            if self._train_state.ema_params is not None
            else self._train_state.params
        )
        model = nnx.merge(self._train_state.model_def, params)
        model.eval()
        inputs = self._policy._input_transform(processed_obs)
        # TODO: check if batch dim needs to be added
        observation = _model.Observation.from_dict(inputs)
        next_prefix = self._get_prefix_rep_with_model(m=model, observation=observation)
        # TODO: check if this check and cast is necessary
        if next_prefix.ndim >= 3 and next_prefix.shape[0] == 1:
            next_prefix = next_prefix[0]
        next_prefix = np.asarray(next_prefix, dtype=np.float32)

        for idx in reversed(range(len(episode_data))):
            ep = episode_data[idx]
            ep["action"], prefix = ep["action"]
            horizon = ep["observation"]["state"].shape[0]
            ep["observation"][PREFIX_EMBEDDING_NAME] = np.repeat(
                prefix[None, ...], horizon, axis=0
            )
            ep["next_observation"][PREFIX_EMBEDDING_NAME] = np.repeat(
                next_prefix[None, ...], horizon, axis=0
            )
            next_prefix = prefix

    def save_episode(self, is_success: bool, env_index: int, task_description: str):

        assert env_index in range(
            len(self._episode_storage)
        ), f"env_index must be between 0 and {len(self._episode_storage) - 1}, but got {env_index}."
        # extract episode data from storage and empty it
        episode_data = self._episode_storage[env_index]
        self._episode_storage[env_index] = []
        # filtered SFT keeps only successful episodes.
        if is_success:
            self._save_episode_in_buffer(episode_data, task_description)

    def _save_episode_in_buffer(self, episode_data, task_description):

        assert isinstance(self._config.rl, FilteredSFTLearnerConfig), (
            "Only Filtered SFT config should be passed " "to the filtered SFT agent"
        )

        if self._config.collect.store_prefix_rep:
            self._attach_prefix_embeddings_to_episode_data(
                episode_data, task_description=task_description
            )

        # concatenate all chunks
        episode_data = jax.tree_util.tree_map(
            lambda *xs: np.concatenate(xs, axis=0), *episode_data
        )
        done = np.logical_or(episode_data["terminate"], episode_data["truncate"])
        n_steps = np.where(done)[0][0] + 1
        act_h = int(self._config.model.action_horizon)
        last_gamma = float(self._config.rl.discount**act_h)
        all_gammas = np.array([self._config.rl.discount**i for i in range(n_steps)])
        w_gammas = all_gammas[:act_h]
        n_windows = n_steps - act_h + 1
        if n_windows <= 0:
            return

        # process elements to account for action chunks
        _obs = {
            k[len("observation/") :]: v[:n_windows]
            for k, v in episode_data["observation"].items()
        }
        _next_obs = {
            k[len("observation/") :]: v[act_h - 1 : act_h - 1 + n_windows]
            for k, v in episode_data["next_observation"].items()
        }
        _actions = np.stack(
            [
                episode_data["action"][start : start + act_h]
                for start in range(n_windows)
            ]
        )
        _actions = self.post_step_action_filter(_actions)
        _reward = np.asarray(
            [
                (episode_data["reward"][start : start + act_h] * w_gammas).sum()
                for start in range(n_windows)
            ]
        )
        _discount = np.asarray(
            [
                0.0 if np.any(done[start : start + act_h]) else last_gamma
                for start in range(n_windows)
            ]
        )
        _mc_return = (
            (all_gammas * episode_data["reward"][:n_steps])[::-1].cumsum()[::-1]
            / all_gammas
        )[:n_windows]

        def transform(obs, act, prompt):
            obs.update({"actions": act, "prompt": prompt})
            obs = self._pre_token_transform(obs)
            obs["image_mask"] = {
                k: np.full((n_windows,), bool(v)) for k, v in obs["image_mask"].items()
            }
            if isinstance(self._token_transform, _transforms.TokenizePrompt):
                if prompt not in self._token_cache:
                    tok = self._token_transform({"prompt": prompt})
                    self._token_cache[prompt] = (
                        tok["tokenized_prompt"],
                        tok["tokenized_prompt_mask"],
                    )
                tokens, token_masks = self._token_cache[prompt]
                obs["tokenized_prompt"] = np.broadcast_to(
                    tokens, (n_windows,) + tokens.shape
                ).copy()
                obs["tokenized_prompt_mask"] = np.broadcast_to(
                    token_masks, (n_windows,) + token_masks.shape
                ).copy()
            else:
                raise TypeError(
                    f"Unsupported token transform: {type(self._token_transform)}"
                )
            actions = obs.pop("actions")
            prompt = obs.pop("prompt")
            return obs, actions

        # process observations and actions according to pi0 preprocessing
        _next_obs, _ = transform(_next_obs, _actions, str(task_description))
        _obs, _actions = transform(_obs, _actions, str(task_description))

        self._online_data_buffer.insert(
            {
                "observation": _obs,
                "actions": _actions.astype(np.float32),
                "next_observation": _next_obs,
                "reward": _reward.astype(np.float32),
                "mc_return": _mc_return.astype(np.float32),
                "discount": _discount.astype(np.float32),
            }
        )
        self._collection_success_episodes += 1

    def start_data_collection(self, step: int | None = None):
        # Reset episode storage
        self._episode_storage = [[] for _ in range(self._config.collect.env_num)]
        self._collection_success_episodes = 0

    def end_data_collection(self, step: int | None = None) -> int:
        collected_episodes = int(self._collection_success_episodes)
        # Reset episode storage and counter for the next collection round.
        self._episode_storage = [[] for _ in range(self._config.collect.env_num)]
        self._collection_success_episodes = 0
        return collected_episodes

    def update(self):
        self.training_steps += 1
        update_policy = (
            self.training_steps >= self._config.rl.policy_training_start_step
            and self.training_steps % self._config.rl.policy_update_interval == 0
        )
        if not update_policy:
            return {"online_buffer_size": self._online_data_buffer.size}

        if self._online_data_buffer.size == 0:
            return {}
        online_ratio = self._config.rl.online_ratio
        if online_ratio < 1.0:
            batch = next(self._data_iter)
        if online_ratio > 0.0:
            online_batch_size = int(self._config.batch_size * min(1.0, online_ratio))
            online_batch_raw = self._online_data_buffer.sample(
                batch_size=online_batch_size
            )
            online_batch = self._online_batch_to_sft_batch(online_batch_raw)
            batch = (
                online_batch
                if online_ratio >= 1.0
                else jax.tree.map(
                    lambda x, y: jnp.concatenate([x, y], axis=0),
                    batch,
                    online_batch,
                )
            )

        train_rng, self._rng = jax.random.split(self._rng)
        rl_config = self._config.rl
        assert isinstance(rl_config, FilteredSFTLearnerConfig)
        with sharding.set_mesh(self._mesh):
            policy_state, info = self._train_step(train_rng, self._train_state, batch)
        self._train_state = policy_state
        info = info | {
            "online_buffer_size": jnp.asarray(
                float(self._online_data_buffer.size), dtype=jnp.float32
            )
        }
        return info
