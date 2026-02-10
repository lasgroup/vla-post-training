import copy
import inspect
import os
from typing import Any, Dict, Callable

from src.rl_training.agent import Agent, StepData
from src.rl_training.update_actor import update_actor
from src.rl_training.types import OnlineTrainingConfig, OnlineLearningConfig
from src.rl_training.replay_buffer import ShardedReplayBuffer

import openpi.training.config as _config
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.shared.nnx_utils as nnx_utils
import openpi.shared.array_typing as at
import openpi.training.weight_loaders as _weight_loaders
import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi_client import image_tools

import functools
import logging
import jax
import flax.nnx as nnx
import flax.traverse_util as traverse_util
import tqdm
from flax.training import common_utils
import jax.numpy as jnp
import numpy as np
from openpi.policies import policy_config
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from src.rl_training.venv import BaseVectorEnv, SubprocVectorEnv, DummyVectorEnv
from src.rl_training.env_utils import (
    QueryFrequencyWrapper,
    Pi0ObservationWrapper,
    WarmUpOnResetWrapper,
    ensure_gymnasium_env,
)


def _pad_actions_to_horizon(actions: np.ndarray, action_horizon: int) -> np.ndarray:
    actions = np.asarray(actions)
    if actions.ndim == 2:
        actions = actions[None, ...]
    if actions.shape[1] < action_horizon:
        pad = action_horizon - actions.shape[1]
        last = actions[:, -1:, :]
        actions = np.concatenate([actions, np.repeat(last, pad, axis=1)], axis=1)
    elif actions.shape[1] > action_horizon:
        actions = actions[:, :action_horizon, :]
    return actions


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
    config: _config.TrainConfig,
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
            state.replace_by_pure_dict(partial_params)
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
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(
        config.weight_loader, train_state_shape.params.to_pure_dict()
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


class PiFilteredSFTLearner(Agent):
    def __init__(self, config: OnlineTrainingConfig):
        self._obs_prefix_key = config.obs_prefix_key
        self._rng = jax.random.key(config.base_policy_config.seed)
        self._config = config

        # setup sharding
        self._mesh = sharding.make_mesh(self.base_policy_config.fsdp_devices)
        self._data_sharding = jax.sharding.NamedSharding(
            self._mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
        )
        self._replicated_sharding = jax.sharding.NamedSharding(
            self._mesh, jax.sharding.PartitionSpec()
        )

        # setup training data loader and states
        self._checkpoint_manager, self._resuming = (
            _checkpoints.initialize_checkpoint_dir(
                self.base_policy_config.checkpoint_dir,
                keep_period=self.base_policy_config.keep_period,
                overwrite=self.base_policy_config.overwrite,
                resume=self.base_policy_config.resume,
            )
        )

        self._offline_data_loader = self._get_offline_data_loader()
        self._online_data_buffer = self._get_online_replay_buffer(self._data_sharding)
        self._train_state, self._train_state_sharding = self._get_train_state(
            self._resuming
        )
        self._update_actor = jax.jit(
            functools.partial(update_actor, self.base_policy_config.trainable_filter),
            in_shardings=(
                self._replicated_sharding,
                self._train_state_sharding,
                self._data_sharding,
            ),
            out_shardings=(self._train_state_sharding, self._replicated_sharding),
            donate_argnums=(1,),
        )

        # Setup data collection pipeline
        default_prompt = self.online_learning_config.default_prompt
        if default_prompt is None and self.online_learning_config.variant is not None:
            default_prompt = getattr(
                self.online_learning_config.variant, "task_description", None
            )
        if default_prompt is None:
            default_prompt = self.base_policy_config.default_prompt
        policy_checkpoint_dir = os.environ.get("OPENPI_POLICY_CHECKPOINT_DIR")
        if policy_checkpoint_dir is None and isinstance(
            self.base_policy_config.weight_loader,
            _weight_loaders.CheckpointWeightLoader,
        ):
            params_path = self.base_policy_config.weight_loader.params_path
            if params_path.endswith("/params"):
                policy_checkpoint_dir = params_path[: -len("/params")]
            else:
                policy_checkpoint_dir = params_path
        if policy_checkpoint_dir is None:
            policy_checkpoint_dir = self._checkpoint_manager._directory
            if not os.path.exists(os.path.join(policy_checkpoint_dir, "params")):
                raise FileNotFoundError(
                    "Policy checkpoint not found. Set OPENPI_POLICY_CHECKPOINT_DIR to a checkpoint "
                    "containing 'params' (e.g. .../openpi-assets/checkpoints/pi0_libero)."
                )

        self.actor = policy_config.create_trained_policy(
            self.base_policy_config,
            policy_checkpoint_dir,
            default_prompt=default_prompt,
        )
        self._episode_storage = [
            [] for _ in range(self.online_learning_config.num_envs)
        ]

    def _get_online_replay_buffer(
        self, data_sharding: jax.sharding.NamedSharding
    ) -> ShardedReplayBuffer:
        train_config = self.base_policy_config
        olc = self.online_learning_config
        default_prompt = olc.default_prompt
        if default_prompt is None and olc.variant is not None:
            default_prompt = getattr(olc.variant, "task_description", None)
        if default_prompt is None:
            default_prompt = train_config.default_prompt
        if default_prompt is None:
            raise ValueError("default_prompt is required for online data insertion.")

        data_config = self._offline_data_loader.data_config()

        token_transform = None
        non_token_model_transforms = []
        for t in data_config.model_transforms.inputs:
            if isinstance(
                t, (_transforms.TokenizePrompt, _transforms.TokenizeFASTInputs)
            ):
                token_transform = t
            else:
                non_token_model_transforms.append(t)

        pre_token_transform = _transforms.compose(
            [
                *data_config.repack_transforms.inputs,
                *data_config.data_transforms.inputs,
                _transforms.Normalize(
                    data_config.norm_stats, use_quantiles=data_config.use_quantile_norm
                ),
                *non_token_model_transforms,
            ]
        )

        obs_spec, act_spec = train_config.model.inputs_spec(batch_size=1)
        obs_spec_dict = obs_spec.to_dict()

        def _zeros_like_spec(spec, *, override_dtype=None):
            dtype = override_dtype if override_dtype is not None else spec.dtype
            return np.zeros(spec.shape, dtype=dtype)

        dummy_obs_dict = {
            "image": {
                k: _zeros_like_spec(v, override_dtype=np.uint8)
                for k, v in obs_spec_dict["image"].items()
            },
            "image_mask": {
                k: _zeros_like_spec(v) for k, v in obs_spec_dict["image_mask"].items()
            },
            "state": _zeros_like_spec(obs_spec_dict["state"]),
        }
        for k in (
            "tokenized_prompt",
            "tokenized_prompt_mask",
            "token_ar_mask",
            "token_loss_mask",
        ):
            if k in obs_spec_dict and obs_spec_dict[k] is not None:
                dummy_obs_dict[k] = _zeros_like_spec(obs_spec_dict[k])

        dummy_actions = np.zeros(act_spec.shape, dtype=act_spec.dtype)

        batch_size = int(olc.online_batch_size or train_config.batch_size)
        max_capacity = max(batch_size, 256, batch_size * 8, int(olc.online_max_samples))

        token_cache = {}
        action_horizon = int(train_config.model.action_horizon)

        def _preprocess_insert(episode_data: Dict[str, Any]):
            raw = dict(episode_data)
            if "actions" not in raw and "action" in raw:
                raw["actions"] = raw.pop("action")

            prompt = raw.get("prompt", default_prompt)
            if not isinstance(prompt, str):
                prompt_arr = np.asarray(prompt)
                prompt = prompt_arr.reshape(-1)[0].item() if prompt_arr.size else ""
            prompt = str(prompt)
            raw["prompt"] = prompt

            if "observation" not in raw:
                obs = {}
                if "image" in raw:
                    obs["image"] = raw.pop("image")
                if "wrist_image" in raw:
                    obs["wrist_image"] = raw.pop("wrist_image")
                if "state" in raw:
                    obs["state"] = raw.pop("state")
                if obs:
                    raw["observation"] = obs

            raw = {k: (np.asarray(v) if k != "prompt" else v) for k, v in raw.items()}

            data = pre_token_transform(raw)

            if "actions" in data:
                data["actions"] = _pad_actions_to_horizon(
                    data["actions"], action_horizon
                )

            # Ensure batched image masks.
            batch_shape = tuple(np.asarray(data["state"]).shape[:-1])
            if "image_mask" in data:
                for k, v in data["image_mask"].items():
                    v = np.asarray(v)
                    if v.ndim == 0:
                        data["image_mask"][k] = np.full(
                            batch_shape, bool(v), dtype=np.bool_
                        )

            if token_transform is None:
                raise ValueError(
                    "Model transforms must include a prompt tokenization transform."
                )

            if isinstance(token_transform, _transforms.TokenizePrompt):
                cached = token_cache.get(prompt)
                if cached is None:
                    tok = token_transform({"prompt": prompt})
                    cached = (tok["tokenized_prompt"], tok["tokenized_prompt_mask"])
                    token_cache[prompt] = cached
                tokens, token_masks = cached
                data.pop("prompt", None)
                data["tokenized_prompt"] = np.broadcast_to(
                    tokens, batch_shape + tokens.shape
                ).copy()
                data["tokenized_prompt_mask"] = np.broadcast_to(
                    token_masks, batch_shape + token_masks.shape
                ).copy()
            elif isinstance(token_transform, _transforms.TokenizeFASTInputs):
                data.pop("prompt", None)
                state = np.asarray(data["state"])
                actions = np.asarray(data.get("actions"))
                if actions is None:
                    raise ValueError("FAST tokenization requires actions.")
                t = int(state.shape[0])
                toks, masks, ar_masks, loss_masks = [], [], [], []
                for i in range(t):
                    out = token_transform(
                        {"prompt": prompt, "state": state[i], "actions": actions[i]}
                    )
                    toks.append(out["tokenized_prompt"])
                    masks.append(out["tokenized_prompt_mask"])
                    ar_masks.append(out["token_ar_mask"])
                    loss_masks.append(out["token_loss_mask"])
                data["tokenized_prompt"] = np.stack(toks, axis=0)
                data["tokenized_prompt_mask"] = np.stack(masks, axis=0)
                data["token_ar_mask"] = np.stack(ar_masks, axis=0)
                data["token_loss_mask"] = np.stack(loss_masks, axis=0)
            else:
                raise TypeError(f"Unsupported token transform: {type(token_transform)}")

            actions = np.asarray(data.pop("actions"), dtype=np.float32)
            data["state"] = np.asarray(data["state"], dtype=np.float32)
            return data, actions

        def _postprocess_sample(batch):
            obs_dict, actions = batch
            return _model.Observation.from_dict(obs_dict), actions

        return ShardedReplayBuffer(
            dummy_data=(dummy_obs_dict, dummy_actions),
            max_capacity=max_capacity,
            batch_size=batch_size,
            data_sharding=data_sharding,
            seed=train_config.seed,
            preprocess_fn=_preprocess_insert,
            postprocess_fn=_postprocess_sample,
            freeze_dict=False,
        )

    @property
    def resuming(self):
        return self._resuming

    def _process_obs_for_pi0(self, observations: Dict):
        element = {}
        for key, val in observations.items():
            # Extract all observations relevant for the policy
            if self._obs_prefix_key in key:
                obs_key = key.split(self._obs_prefix_key)[-1]
                if obs_key == "prompt":
                    continue
                if "image" in obs_key:
                    # Rescale images
                    val = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(val, 224, 224)
                    )
                obs_key = f"observation/{obs_key}"
                element[obs_key] = val
        return element

    def _sample_action(
        self,
        observations: Dict,
        rng: jax.random.PRNGKey,
        train_state: training_utils.TrainState,
        noise_level: float = 0.0,
        num_steps: int | None = None,
        batch_actions: bool = True,
    ):
        # Define model
        model = nnx.merge(train_state.model_def, train_state.params)
        # Convert observation for the policy

        kwargs = dict(model=model, obs=observations, rng=rng, noise_level=noise_level)
        if num_steps is not None:
            kwargs["num_steps"] = int(num_steps)
        # Vector envs expect a batch dimension for actions. Policy inference
        # un batches when batch_size == 1, so add it back for single-env runs.
        actions = self.actor.infer_with_model(**kwargs)["actions"]
        if batch_actions and actions.ndim == 2:
            actions = actions[np.newaxis, ...]
        return actions

    @property
    def online_learning_config(self) -> OnlineLearningConfig:
        return self._config.online_learning_config

    @property
    def base_policy_config(self) -> _config.TrainConfig:
        return self._config.base_policy_config

    def eval_actions(
        self, observations: Dict, batch_actions: bool = True
    ) -> np.ndarray:
        rng, self._rng = jax.random.split(self._rng)
        element = self._process_obs_for_pi0(observations)
        actions = self._sample_action(
            observations=element,
            rng=rng,
            train_state=self._train_state,
            num_steps=self.online_learning_config.pi0_num_steps,
            batch_actions=batch_actions,
        )
        return np.asarray(actions, dtype=np.float32)

    def sample_actions(
        self, observations: Dict, batch_actions: bool = True
    ) -> np.ndarray:
        rng, self._rng = jax.random.split(self._rng)
        element = self._process_obs_for_pi0(observations)
        actions = self._sample_action(
            observations=element,
            rng=rng,
            train_state=self._train_state,
            noise_level=self.online_learning_config.fm_noise_level,
            num_steps=self.online_learning_config.pi0_num_steps,
            batch_actions=batch_actions,
        )
        return np.asarray(actions, dtype=np.float32)

    def _get_train_state(self, resuming: bool = False):
        init_rng, self._rng = jax.random.split(self._rng)
        train_state, train_state_sharding = init_train_state(
            self.base_policy_config, init_rng, self._mesh, resume=resuming
        )
        if resuming:
            # `init_train_state(..., resume=True)` returns a TrainState *shape* that orbax uses as a restore target.
            # Don't reference `self._train_state` here (it isn't initialized yet).
            train_state = _checkpoints.restore_state(
                self._checkpoint_manager, train_state, self._offline_data_loader
            )
        jax.block_until_ready(train_state)
        logging.info(
            f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}"
        )
        return train_state, train_state_sharding

    def _get_offline_data_loader(self):
        logging.info(
            "Initializing offline data loader (may download dataset if missing)..."
        )
        data_loader = _data_loader.create_data_loader(
            self.base_policy_config,
            sharding=self._data_sharding,
            shuffle=True,
        )
        data_iter = iter(data_loader)
        batch = next(data_iter)
        logging.info(
            f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}"
        )
        return data_loader

    def save_checkpoint(self, step, keep_every_n_steps: int | None = None):
        logging.warning(
            "The dir argument is ignored here as the default dir in the config is used."
        )
        _checkpoints.save_state(
            self._checkpoint_manager, self._train_state, self._offline_data_loader, step
        )
        self._checkpoint_manager.wait_until_finished()

    def start_data_collection(self):
        super().start_data_collection()
        # Restart episode counter for all environments
        self._episode_storage = [
            [] for _ in range(self.online_learning_config.num_envs)
        ]
        return True

    def start_training(self):
        super().start_training()
        return True

    def add_data(self, step_data: StepData):
        assert self.online_learning_config.num_envs == step_data.terminate.shape[0], (
            f"Agent expects step data from {self.online_learning_config.num_envs} "
            f"but only {step_data.terminate.shape[0]} envs are provided."
        )

        def process_obs(observation: Dict, env_index: int):
            obs_dict = {}
            for key, val in observation.items():
                if self._obs_prefix_key in key:
                    obs_key = key.split(self._obs_prefix_key)[-1]
                    obs_dict[obs_key] = val[env_index]
            return obs_dict

        # We assume obs is a dictionary where of numpy arrays, each with dim (num_envs, dim_key)
        obs = step_data.obs
        for env in range(self.online_learning_config.num_envs):
            env_obs, act = process_obs(obs, env), step_data.action[env]
            self._episode_storage[env].append({"observations": env_obs, "actions": act})
        self.env_steps += self.online_learning_config.num_envs

    def save_episode(self, is_success: bool = False, env_index: int = 0):
        # If successful we add the episode to the buffer
        super().save_episode(is_success, env_index)
        if is_success:
            episode_data = self._episode_storage[env_index]
            # Stack to have a dictionary of arrays
            episode_data = jax.tree_util.tree_map(
                lambda *xs: np.stack(xs, axis=0), *episode_data
            )
            self._online_data_buffer.insert(episode_data)
        # Empty the data buffer the episode storage for this env to collect the next buffer.
        self._episode_storage[env_index] = []

    def _update(
        self,
        rng: at.KeyArrayLike,
        train_state: training_utils.TrainState,
        batch: tuple[_model.Observation, _model.Actions],
    ):
        train_state, info = self._update_actor(rng, train_state, batch)
        return train_state, info

    def update(self):
        olc = self.online_learning_config
        should_update = False
        if olc.episode_update_frequency is not None:
            should_update = should_update or (
                self.episodes % olc.episode_update_frequency == 0
            )
        if olc.env_steps_update_frequency is not None:
            should_update = should_update or (
                self.env_steps % olc.env_steps_update_frequency == 0
            )

        if should_update:
            self.start_training()
            train_state = self._train_state
            pbar = tqdm.tqdm(
                range(
                    self.online_learning_config.start_step,
                    self.online_learning_config.num_train_steps_per_update,
                ),
                initial=self.online_learning_config.start_step,
                total=self.online_learning_config.num_train_steps_per_update,
                dynamic_ncols=True,
            )
            infos = []
            offline_data_iter = iter(self._offline_data_loader)
            for step in pbar:
                train_rng, self._rng = jax.random.split(self._rng)
                offline_batch = next(offline_data_iter)
                use_online = (
                    olc.online_ratio > 0
                    and self._online_data_buffer.size
                    >= self._online_data_buffer.batch_size
                )
                if use_online:
                    online_batch = self._online_data_buffer.sample()
                    if olc.online_ratio >= 1.0:
                        batch = online_batch
                    else:
                        batch = jax.tree.map(
                            lambda x, y: jnp.concatenate([x, y], axis=0),
                            offline_batch,
                            online_batch,
                        )
                else:
                    batch = offline_batch
                with sharding.set_mesh(self._mesh):
                    train_state, info = self._update(train_rng, train_state, batch)
                infos.append(info)
            self._train_state = train_state
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            return reduced_info
        else:
            return {}

    def wrap_env(
        self, env_or_factory: gym.Env | Callable[[], gym.Env]
    ) -> BaseVectorEnv:
        olc = self.online_learning_config
        variant = olc.variant
        replan_steps = olc.replan_steps
        discount = olc.discount
        num_steps_wait_upon_reset = variant.num_steps_wait_upon_reset
        warm_up_action = variant.warm_up_action
        time_limit = olc.time_limit

        render_gpu_device_mod = 4
        pass_render_gpu_device_id = False
        if callable(env_or_factory) and variant.env == "libero":
            try:
                sig = inspect.signature(env_or_factory)
                pass_render_gpu_device_id = (
                    "render_gpu_device_id" in sig.parameters
                    or any(
                        p.kind == inspect.Parameter.VAR_KEYWORD
                        for p in sig.parameters.values()
                    )
                )
            except (TypeError, ValueError):
                pass_render_gpu_device_id = False

        env_factories = []
        for i in range(olc.num_envs):

            def _make_env(
                rank=i,
                env_or_factory=env_or_factory,
                variant=variant,
                replan_steps=replan_steps,
                discount=discount,
                num_steps_wait_upon_reset=num_steps_wait_upon_reset,
                warm_up_action=warm_up_action,
                pass_render_gpu_device_id=pass_render_gpu_device_id,
                render_gpu_device_mod=render_gpu_device_mod,
            ):
                if callable(env_or_factory):
                    if pass_render_gpu_device_id:
                        base_env = env_or_factory(
                            render_gpu_device_id=rank % render_gpu_device_mod
                        )
                    else:
                        base_env = env_or_factory()
                else:
                    base_env = copy.deepcopy(env_or_factory)
                # This is crucial to deal with gym/gymnasium issues
                base_env = ensure_gymnasium_env(base_env)
                # Add time limit to avoid executing infinitely long episodes
                base_env = TimeLimit(base_env, max_episode_steps=time_limit)
                env_i = QueryFrequencyWrapper(
                    base_env,
                    query_frequency=replan_steps,
                    discount=discount,
                )
                env_i = Pi0ObservationWrapper(env_i, variant=variant)
                if num_steps_wait_upon_reset > 0:
                    env_i = WarmUpOnResetWrapper(
                        env_i,
                        num_steps_wait=num_steps_wait_upon_reset,
                        warm_up_action=warm_up_action,
                    )

                return env_i

            env_factories.append(_make_env)
        env = (
            DummyVectorEnv(env_factories)
            if olc.num_envs == 1
            else SubprocVectorEnv(env_factories)
        )
        env.seed([self.base_policy_config.seed + i for i in range(olc.num_envs)])
        return env
