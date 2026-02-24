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
from src.rl.agent import Agent
from src.rl.filtered_sft_agent.update import train_step
from src.rl.replay_buffer import ShardedReplayBuffer
from src.rl.types import StepData
from src.training.config import OnlineTrainConfig
from src.training.data_loader import create_data_loader
from src.envs.wrappers import Pi0ObservationWrapper, QueryFrequencyWrapper
from src.envs.venv import SubprocVectorEnv, DummyVectorEnv
from src.rl.agent import Agent, EnvFn


def get_env_and_agent_for_filtered_sft(env_fn, config, task_description):
    env = filtered_sft_wrap_env(
        env_fn=env_fn,
        config=config,
        task_description=task_description,
    )
    agent = FilteredSFTLearner(config)
    return env, agent


def filtered_sft_wrap_env(env_fn: EnvFn, config, task_description: str):
    env_num = config.collect.env_num
    replan_steps = config.collect.replan_steps
    env_class = config.domain
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
            )
            # Add query frequency wrapper to rollout action chunks
            base_env = QueryFrequencyWrapper(
                env=base_env,
                query_frequency=replan_steps,
                discount=discount,
                store_full_transitions=add_per_step_data,
                post_step_filter=lambda x: np.where(np.abs(x) < 0.0011, 0.0, x),
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
        assert 0.0 <= self._config.online_ratio <= 1.0, "Online ratio must be between 0 and 1."
        self._offline_batch_size = max(1, int(self._config.batch_size * (1 - self._config.online_ratio)))
        self._data_loader = create_data_loader(
            config, batch_size=self._offline_batch_size, sharding=self._data_sharding, shuffle=True
        )
        self._data_iter = iter(self._data_loader)
        self._online_data_buffer = self._get_online_replay_buffer()
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
        policy_checkpoint_dir = self._config.weight_loader.params_path[: -len("/params")]
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
        train_config = self._config
        data_config = self._data_loader.data_config()

        token_transform: (
            _transforms.TokenizePrompt | _transforms.TokenizeFASTInputs | None
        ) = None
        non_token_model_transforms = []
        for t in data_config.model_transforms.inputs:
            if isinstance(
                t, (_transforms.TokenizePrompt, _transforms.TokenizeFASTInputs)
            ):
                token_transform = t
            else:
                non_token_model_transforms.append(t)
        if token_transform is None:
            raise ValueError(
                "Model transforms must include a prompt tokenization transform."
            )

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
        transition_state_dim = int(obs_spec_dict["state"].shape[-1])
        dummy_next_obs_dict = {
            "state": np.zeros((1, transition_state_dim), dtype=np.float32)
        }
        dummy_rewards = np.zeros((1,), dtype=np.float32)
        dummy_discounts = np.zeros((1,), dtype=np.float32)
        # Keep enough online data for stable sampling when mixing with offline batches.
        max_capacity = self._config.online_buffer_size
        logging.info(
            "Initializing online replay buffer (capacity=%d)",
            max_capacity,
        )
        token_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        action_horizon = int(train_config.model.action_horizon)
        transition_gamma = float(train_config.discount)

        def _pad_feature_dim(values: Any, target_dim: int, *, name: str) -> np.ndarray:
            arr = np.asarray(values, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[None, :]
            if arr.ndim < 2:
                raise ValueError(
                    f"{name} must be at least 2-D with shape [batch, features], got {arr.shape}."
                )
            feature_dim = int(arr.shape[-1])
            if feature_dim < target_dim:
                pad_shape = [(0, 0)] * arr.ndim
                pad_shape[-1] = (0, target_dim - feature_dim)
                arr = np.pad(arr, pad_shape, mode="constant", constant_values=0.0)
            elif feature_dim > target_dim:
                arr = arr[..., :target_dim]
            return arr

        def _ensure_batch_scalar(
            values: Any | None, *, batch_size: int, default: float
        ) -> np.ndarray:
            if values is None:
                return np.full((batch_size,), default, dtype=np.float32)

            arr = np.asarray(values, dtype=np.float32)
            if arr.ndim == 0:
                return np.full((batch_size,), float(arr), dtype=np.float32)
            if arr.shape[0] != batch_size:
                raise ValueError(
                    f"Scalar batch has mismatched size: expected {batch_size}, got {arr.shape[0]}."
                )
            if arr.ndim > 1:
                arr = arr.reshape(batch_size, -1)[:, 0]
            return arr.astype(np.float32, copy=False)

        def _preprocess_insert(episode_data: Dict[str, Any]):
            raw = episode_data
            raw["observation"] = {k: raw[k] for k in ("image", "wrist_image", "state")}
            prompt = raw["prompt"]
            raw = {k: (np.asarray(v) if k != "prompt" else v) for k, v in raw.items()}
            transition_state = _pad_feature_dim(
                raw["state"], transition_state_dim, name="state"
            )
            data = pre_token_transform(raw)

            def _pad_actions_to_horizon(actions: np.ndarray, action_horizon: int) -> np.ndarray:
                """Pad or truncate actions to match the policy action horizon."""
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

            data["actions"] = _pad_actions_to_horizon(
                data["actions"], action_horizon
            )

            # Ensure batched image masks.
            batch_shape = tuple(np.asarray(data["state"]).shape[:-1])
            for k, v in data["image_mask"].items():
                v = np.asarray(v)
                if v.ndim == 0:
                    data["image_mask"][k] = np.full(
                        batch_shape, bool(v), dtype=np.bool_
                    )

            data.pop("prompt", None)
            if isinstance(token_transform, _transforms.TokenizePrompt):
                if prompt not in token_cache:
                    tok = token_transform({"prompt": prompt})
                    token_cache[prompt] = (tok["tokenized_prompt"], tok["tokenized_prompt_mask"])
                tokens, token_masks = token_cache[prompt]
                data["tokenized_prompt"] = np.broadcast_to(
                    tokens, batch_shape + tokens.shape
                ).copy()
                data["tokenized_prompt_mask"] = np.broadcast_to(
                    token_masks, batch_shape + token_masks.shape
                ).copy()
            elif isinstance(token_transform, _transforms.TokenizeFASTInputs):
                state = np.asarray(data["state"])
                actions = np.asarray(data["actions"])
                t = int(state.shape[0])
                keys = ["tokenized_prompt", "tokenized_prompt_mask", "token_ar_mask", "token_loss_mask"]
                buff = {k: [] for k in keys}
                for i in range(t):
                    out = token_transform(
                        {"prompt": prompt, "state": state[i], "actions": actions[i]}
                    )
                    buff = {k: buff[k] + [out[k]] for k in keys}
                data =  {**data, **{k: np.stack(v, axis=0) for k, v in buff.items()}}
            else:
                raise TypeError(f"Unsupported token transform: {type(token_transform)}")

            actions = np.asarray(data.pop("actions"), dtype=np.float32)
            data["state"] = np.asarray(data["state"], dtype=np.float32)
            insert_batch_size = int(actions.shape[0])
            if transition_state.shape[0] != insert_batch_size:
                raise ValueError(
                    f"Transition state batch mismatch: {transition_state.shape[0]} vs {insert_batch_size}."
                )
            transition_reward = _ensure_batch_scalar(
                raw.get("reward"), batch_size=insert_batch_size, default=0.0
            )
            transition_discount = _ensure_batch_scalar(
                raw.get("discount"),
                batch_size=insert_batch_size,
                default=transition_gamma,
            )

            return {
                "observation": data,
                "actions": actions,
                "next_observation": {
                    "state": transition_state.astype(np.float32, copy=False)
                },
                "reward": transition_reward,
                "discount": transition_discount,
            }

        return ShardedReplayBuffer(
            dummy_data={
                "observation": dummy_obs_dict,
                "actions": dummy_actions,
                "next_observation": dummy_next_obs_dict,
                "reward": dummy_rewards,
                "discount": dummy_discounts,
            },
            max_capacity=max_capacity,
            data_sharding=self._data_sharding,
            seed=train_config.seed,
            preprocess_fn=_preprocess_insert,
            postprocess_fn=None,
            freeze_dict=False,
        )

    def _process_obs_for_pi0(
        self,
        observations: Dict,
        task_description: str,
    ) -> Dict[str, Any]:
        # With per-step collection enabled, each env step contains a short chunk of
        # observations. Use the most recent one for policy inference.
        obs = observations["observation"]
        if self._config.collect.add_per_step_data:
            obs = jax.tree_util.tree_map(lambda x: x[:, -1], obs)
        size = int(self._config.collect.resize_image)
        resize_fn = lambda x: image_tools.convert_to_uint8(image_tools.resize_with_pad(x, size, size))
        obs = {k: resize_fn(v) if "image" in k else v for k, v in obs.items()}
        obs["prompt"] = task_description
        # TODO: return prompt and resized image from the environment
        return obs

    def _sample_action(
        self,
        observations: Dict,
        rng: jax.random.PRNGKey,
        train_state: training_utils.TrainState,
    ) -> np.ndarray:
        params = (
            train_state.ema_params
            if train_state.ema_params is not None
            else train_state.params
        )
        model = nnx.merge(train_state.model_def, params)
        assert "observation/state" in observations, "Observation must contain 'observation/state' key to infer batch size."
        batch_size = observations["observation/state"].shape[0] if observations["observation/state"].ndim > 1 else 1
        noise = jax.random.normal(
            rng, (batch_size, self._policy.action_horizon, self._policy.action_dim)
        )
        # Vector envs expect a batch dimension for actions. Policy inference
        # unbatches when batch_size == 1, so add it back for single-env runs.
        actions = self._policy.infer_with_model(
            model=model,
            obs=observations,
            noise=noise,
            sharding_spec=self._policy_sharding_spec,
        )["actions"]
        if batch_size == 1 and actions.ndim == 2:
            actions = actions[np.newaxis, ...]
        return actions

    def _generate_actions(
        self, observations: np.ndarray | Dict,
        task_description: str,
    ) -> np.ndarray:
        rng, self._rng = jax.random.split(self._rng)
        processed_obs = self._process_obs_for_pi0(
            observations, task_description=task_description
        )
        actions = self._sample_action(
            observations=processed_obs,
            rng=rng,
            train_state=self._train_state,
        )
        return np.asarray(actions, dtype=np.float32)

    def eval_actions(self, observations: np.ndarray | Dict, **kwargs) -> np.ndarray:
        return self._generate_actions(observations, **kwargs)

    def sample_actions(self, observations: np.ndarray | Dict, **kwargs) -> np.ndarray:
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

    def save_episode(self, is_success: bool = False, env_index: int = 0, task_description: str | None = None):
        if env_index < 0 or env_index >= len(self._episode_storage):
            raise IndexError(
                f"env_index={env_index} is out of range for {len(self._episode_storage)} environments."
            )

        # Extract episode data from storage
        episode_data = self._episode_storage[env_index]
        # Empty the storage now for the next episode
        self._episode_storage[env_index] = []
        if not is_success:
            # Filtered SFT keeps only successful episodes.
            return
        discount_gamma = float(self._config.discount)

        def _extract_policy_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
            extracted = {}
            for key, val in obs.items():
                if not key.startswith("observation/"):
                    continue
                obs_key = key[len("observation/"):]
                extracted[obs_key] = val
            return extracted

        def process_frame(
            ob: Dict[str, Any],
            *,
            actions: Any,
            next_ob: Dict[str, Any] | None,
            reward: float,
            done: bool,
            discount: float,
        ) -> Dict[str, Any]:
            # Extract actions and observations from total_obs.
            obs = ob["observation"]
            frame = _extract_policy_obs(obs)
            if "state" not in frame:
                raise KeyError(
                    "Cannot construct transitions: current observation is missing state."
                )

            frame["actions"] = np.asarray(actions, dtype=np.float32)
            next_state = frame["state"]
            if next_ob is not None:
                next_obs = _extract_policy_obs(next_ob["observation"])
                if "state" in next_obs:
                    next_state = next_obs["state"]
            frame["next_observation"] = {"state": next_state}
            frame["reward"] = np.float32(reward)
            frame["done"] = np.bool_(done)
            frame["discount"] = np.float32(discount)
            return frame

        def _stack_transitions(frames):
            return jax.tree_util.tree_map(lambda *xs: np.stack(xs, axis=0), *frames)

        transitions = []
        if self._config.collect.add_per_step_data:
            # Build one-step transitions first, then convert to H-step sliding windows.
            for ep in episode_data:
                ep_obs, ep_next_obs, ep_rewards, terminate, truncate = (
                    ep["observation"],
                    ep.get("next_observation"),
                    ep.get("reward"),
                    ep["terminate"],
                    ep["truncate"],
                )
                done_mask = np.asarray(
                    np.logical_or(terminate, truncate), dtype=np.bool_
                )
                valid_steps = int(done_mask.shape[0])
                done_indices = np.where(done_mask)[0]
                if done_indices.size > 0:
                    # Keep the terminal step, drop only wrapper-introduced padding after termination.
                    valid_steps = int(done_indices[0]) + 1

                for step in range(valid_steps):
                    step_obs = jax.tree.map(lambda x: x[step], ep_obs)
                    step_next_obs = (
                        jax.tree.map(lambda x: x[step], ep_next_obs)
                        if ep_next_obs is not None
                        else None
                    )
                    step_reward = (
                        float(np.asarray(ep_rewards, dtype=np.float32)[step])
                        if ep_rewards is not None
                        else 0.0
                    )
                    step_done = bool(done_mask[step])
                    transitions.append(
                        process_frame(
                            step_obs,
                            actions=np.asarray(
                                ep_obs["action"][step], dtype=np.float32
                            ),
                            next_ob=step_next_obs,
                            reward=step_reward,
                            done=step_done,
                            discount=0.0 if step_done else discount_gamma,
                        )
                    )
            if not transitions:
                return
            episode_batch = _stack_transitions(transitions)
            action_horizon = int(self._config.model.action_horizon)
            actions = np.asarray(episode_batch["actions"], dtype=np.float32)
            rewards = np.asarray(episode_batch["reward"], dtype=np.float32)
            dones = np.asarray(episode_batch["done"], dtype=np.bool_)
            next_states = np.asarray(
                episode_batch["next_observation"]["state"], dtype=np.float32
            )
            num_steps = int(actions.shape[0])
            num_windows = num_steps - action_horizon + 1
            if num_windows <= 0:
                return

            windowed_batch = {
                key: np.asarray(value)[:num_windows]
                for key, value in episode_batch.items()
                if key
                not in {
                    "actions",
                    "next_observation",
                    "reward",
                    "done",
                    "discount",
                }
            }
            windowed_batch["actions"] = np.stack(
                [
                    actions[start : start + action_horizon]
                    for start in range(num_windows)
                ],
                axis=0,
            )
            windowed_batch["reward"] = np.asarray(
                [
                    rewards[start : start + action_horizon].sum()
                    for start in range(num_windows)
                ],
                dtype=np.float32,
            )
            windowed_batch["discount"] = np.asarray(
                [
                    (
                        0.0
                        if np.any(dones[start : start + action_horizon])
                        else float(discount_gamma**action_horizon)
                    )
                    for start in range(num_windows)
                ],
                dtype=np.float32,
            )
            windowed_batch["next_observation"] = {
                "state": next_states[
                    action_horizon - 1 : action_horizon - 1 + num_windows
                ]
            }
            episode_batch = windowed_batch
        else:
            for ep in episode_data:
                reward = ep.get("reward")
                terminate = ep.get("terminate", False)
                truncate = ep.get("truncate", False)
                done = bool(
                    np.asarray(terminate).reshape(-1)[-1]
                    or np.asarray(truncate).reshape(-1)[-1]
                )
                reward_value = (
                    float(np.asarray(reward).reshape(-1)[0])
                    if reward is not None
                    else 0.0
                )
                chunk_horizon = int(self._config.collect.replan_steps)
                discount_value = 0.0 if done else float(discount_gamma**chunk_horizon)
                transitions.append(
                    process_frame(
                        ep["observation"],
                        actions=np.asarray(
                            ep["observation"]["action"], dtype=np.float32
                        ),
                        next_ob=ep.get("next_observation"),
                        reward=reward_value,
                        done=done,
                        discount=discount_value,
                    )
                )
            if not transitions:
                return
            episode_batch = _stack_transitions(transitions)
            episode_batch.pop("done", None)
        if task_description is not None:
            episode_batch["prompt"] = str(task_description)
        self._online_data_buffer.insert(episode_batch)
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
        if self._online_data_buffer.size == 0:
            return {}
        online_ratio = self._config.online_ratio
        if online_ratio < 1.0:
            batch = next(self._data_iter)
        if online_ratio > 0.0:
            online_batch_size = int(self._config.batch_size * min(1.0, online_ratio))
            online_batch_raw = self._online_data_buffer.sample(batch_size=online_batch_size)
            online_batch = self._online_batch_to_sft_batch(online_batch_raw)
            batch = online_batch if online_ratio >= 1.0 else jax.tree.map(
                lambda x, y: jnp.concatenate([x, y], axis=0),
                batch,
                online_batch,
            )

        train_rng, self._rng = jax.random.split(self._rng)
        with sharding.set_mesh(self._mesh):
            self._train_state, info = self._train_step(train_rng, self._train_state, batch)
        return info
