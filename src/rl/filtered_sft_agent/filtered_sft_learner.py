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
from src.envs.wrappers import (
    Pi0ObservationWrapper,
    PrefixEmbeddingVectorEnvWrapper,
    QueryFrequencyWrapper,
)
from src.envs.venv import SubprocVectorEnv, DummyVectorEnv
from src.rl.agent import Agent, EnvFn
from src.rl.prefix_embedding import PREFIX_EMBEDDING_NAME, unpack_action_and_prefix


def get_env_and_agent_for_filtered_sft(env_fn, config, task_description, env_class):
    env = filtered_sft_wrap_env(
        env_fn=env_fn,
        config=config,
        task_description=task_description,
        env_class=env_class,
    )
    agent = FilteredSFTLearner(config)
    return env, agent


def filtered_sft_wrap_env(env_fn: EnvFn, config, task_description: str, env_class: str):
    env_num = config.collect.env_num
    add_states = config.collect.add_states
    obs_prefix_key = config.collect.obs_prefix_key
    replan_steps = config.collect.replan_steps
    seed = config.seed
    discount = config.discount
    add_per_step_data = config.collect.add_per_step_data
    return_prefix_rep = bool(getattr(config, "return_prefix_rep", False))
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
            # Add query-frequency wrapper to rollout action chunks.
            query_wrapper = (
                PrefixEmbeddingVectorEnvWrapper
                if return_prefix_rep
                else QueryFrequencyWrapper
            )
            base_env = query_wrapper(
                env=base_env,
                query_frequency=replan_steps,
                discount=discount,
                store_full_transitions=add_per_step_data,
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
        # TODO(Marco Bagatella): Add a link to the Pi paper where they tell why should we have the action filter
        self.post_step_action_filter = lambda x: np.where(np.abs(x) < 0.0011, 0.0, x)
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

        def _get_prefix_rep_with_model_fn(
            m: _model.BaseModel,
            *,
            observation: _model.Observation,
        ):
            if not hasattr(m, "get_prefix_rep"):
                raise AttributeError(
                    f"Model type {type(m)} does not implement get_prefix_rep."
                )
            prefix_rep = m.get_prefix_rep(observation)
            return prefix_rep[0] if isinstance(prefix_rep, tuple) else prefix_rep

        self._get_prefix_rep_with_model = nnx.jit(_get_prefix_rep_with_model_fn)

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
        prefix_embedding_template = self._infer_prefix_embedding_template()
        self._online_data_buffer = self._get_online_replay_buffer(
            self._data_sharding,
            prefix_embedding_template=prefix_embedding_template,
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

    def _infer_prefix_embedding_template(self) -> np.ndarray | None:
        if not bool(getattr(self._config, "return_prefix_rep", False)):
            return None
        if getattr(self._policy, "_is_pytorch_model", False):
            logging.warning(
                "Prefix representations are not supported for PyTorch policies; skipping."
            )
            return None
        get_prefix_rep = getattr(self._policy, "_get_prefix_rep", None)
        if get_prefix_rep is None:
            logging.warning("Policy does not expose get_prefix_rep; skipping.")
            return None

        try:
            fake_obs = self._config.model.fake_obs(batch_size=1)
            prefix_rep = get_prefix_rep(fake_obs)
            if isinstance(prefix_rep, tuple):
                prefix_rep = prefix_rep[0]
            prefix_rep = np.asarray(prefix_rep, dtype=np.float32)
            return np.zeros_like(prefix_rep, dtype=np.float32)
        except Exception as exc:  # pragma: no cover - defensive startup path
            logging.warning(
                "Failed to infer prefix embedding shape for online buffer: %s", exc
            )
            return None

    def _get_online_replay_buffer(
        self,
        data_sharding: jax.sharding.NamedSharding,
        *,
        prefix_embedding_template: np.ndarray | None = None,
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

        obs_repack_transforms = []
        for transform in data_config.repack_transforms.inputs:
            if isinstance(transform, _transforms.RepackTransform):
                # Keep the original flat slash-key structure expected by LiberoInputs
                # while removing the actions mapping for observation-only preprocessing.
                structure = {
                    k: v
                    for k, v in transform.structure.items()
                    if k != "actions" and v != "actions"
                }
                transform = _transforms.RepackTransform(structure)
            obs_repack_transforms.append(transform)
        obs_pre_token_transform = _transforms.compose(
            [
                *obs_repack_transforms,
                *data_config.data_transforms.inputs,
                _transforms.Normalize(
                    data_config.norm_stats, use_quantiles=data_config.use_quantile_norm
                ),
                *non_token_model_transforms,
            ]
        )
        next_obs_pre_token_transform = _transforms.compose(
            [
                *obs_repack_transforms,
                *data_config.data_transforms.inputs,
                _transforms.Normalize(
                    data_config.norm_stats, use_quantiles=data_config.use_quantile_norm
                ),
                *non_token_model_transforms,
            ]
        )
        action_normalize_transform = _transforms.Normalize(
            data_config.norm_stats, use_quantiles=data_config.use_quantile_norm
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
        if prefix_embedding_template is not None:
            dummy_obs_dict[PREFIX_EMBEDDING_NAME] = np.asarray(
                prefix_embedding_template, dtype=np.float32
            )
        store_prefix_embedding = prefix_embedding_template is not None

        dummy_actions = np.zeros(act_spec.shape, dtype=act_spec.dtype)
        transition_state_dim = int(obs_spec_dict["state"].shape[-1])
        dummy_next_obs_dict = {
            "image": {
                k: _zeros_like_spec(v, override_dtype=np.uint8)
                for k, v in obs_spec_dict["image"].items()
            },
            "image_mask": {
                k: _zeros_like_spec(v) for k, v in obs_spec_dict["image_mask"].items()
            },
            "state": np.zeros((1, transition_state_dim), dtype=np.float32),
        }
        for k in (
            "tokenized_prompt",
            "tokenized_prompt_mask",
            "token_ar_mask",
            "token_loss_mask",
        ):
            if k in obs_spec_dict and obs_spec_dict[k] is not None:
                dummy_next_obs_dict[k] = _zeros_like_spec(obs_spec_dict[k])
        if prefix_embedding_template is not None:
            dummy_next_obs_dict[PREFIX_EMBEDDING_NAME] = np.asarray(
                prefix_embedding_template, dtype=np.float32
            )
        dummy_rewards = np.zeros((1,), dtype=np.float32)
        dummy_discounts = np.zeros((1,), dtype=np.float32)
        batch_size = int(train_config.batch_size)
        # Keep enough online data for stable sampling when mixing with offline batches.
        max_capacity = max(batch_size, 256, batch_size * 8)
        logging.info(
            "Initializing online replay buffer (batch_size=%d, capacity=%d)",
            batch_size,
            max_capacity,
        )
        token_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        action_horizon = int(train_config.model.action_horizon)
        default_prompt = getattr(train_config, "default_prompt", None)
        transition_gamma = float(getattr(train_config, "discount", 1.0))

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

        def _normalize_observation_layout(
            raw_episode: Dict[str, Any],
        ) -> Dict[str, Any]:
            """Normalize accepted observation layouts into a single structure."""
            raw = dict(raw_episode)
            aliases = (
                ("observation.image", "image"),
                ("observation.wrist_image", "wrist_image"),
                ("observation.state", "state"),
                (
                    f"observation.{PREFIX_EMBEDDING_NAME}",
                    PREFIX_EMBEDDING_NAME,
                ),
                ("observation/image", "image"),
                ("observation/wrist_image", "wrist_image"),
                ("observation/state", "state"),
                (
                    f"observation/{PREFIX_EMBEDDING_NAME}",
                    PREFIX_EMBEDDING_NAME,
                ),
            )
            for source, target in aliases:
                if target not in raw and source in raw:
                    raw[target] = raw[source]

            obs = raw.get("observation")
            if not isinstance(obs, dict):
                obs = {}

            for key in ("image", "wrist_image", "state", PREFIX_EMBEDDING_NAME):
                if key not in raw and key in obs:
                    raw[key] = obs[key]
                if key in raw:
                    obs[key] = raw[key]

            if obs:
                raw["observation"] = obs

            next_obs = raw.get("next_observation")
            if isinstance(next_obs, dict):
                normalized_next_obs = dict(next_obs)
                for source, target in aliases:
                    if (
                        target not in normalized_next_obs
                        and source in normalized_next_obs
                    ):
                        normalized_next_obs[target] = normalized_next_obs[source]
                raw["next_observation"] = normalized_next_obs
            return raw

        def _preprocess_insert(episode_data: Dict[str, Any]):
            raw = _normalize_observation_layout(episode_data)
            if "actions" not in raw and "action" in raw:
                raw["actions"] = raw.pop("action")
            prefix_embedding = (
                raw.get(PREFIX_EMBEDDING_NAME) if store_prefix_embedding else None
            )

            prompt = raw.get("prompt", default_prompt)
            if prompt is None:
                raise ValueError(
                    "Prompt is required for online insertion. Provide task_description during "
                    "collection or configure a default prompt."
                )
            if not isinstance(prompt, str):
                prompt_arr = np.asarray(prompt)
                prompt = prompt_arr.reshape(-1)[0].item() if prompt_arr.size else ""
            prompt = str(prompt)
            raw["prompt"] = prompt

            raw = {
                k: (np.asarray(v) if k != "prompt" and not isinstance(v, dict) else v)
                for k, v in raw.items()
            }
            state_source = raw.get("state")
            next_observation = raw.get("next_observation")
            next_state_source = raw.get("next_state", state_source)
            next_prefix_embedding = None
            if isinstance(next_observation, dict):
                next_state_source = next_observation.get("state", next_state_source)
                if store_prefix_embedding:
                    next_prefix_embedding = next_observation.get(PREFIX_EMBEDDING_NAME)

            obs_transform_input = {
                "image": raw.get("image"),
                "wrist_image": raw.get("wrist_image"),
                "state": state_source,
                "prompt": prompt,
            }
            data = obs_pre_token_transform(obs_transform_input)

            raw_actions = raw.get("actions")
            if raw_actions is None:
                raise ValueError("Online insertion requires 'actions'.")
            actions = _pad_actions_to_horizon(raw_actions, action_horizon).astype(
                np.float32, copy=False
            )
            actions = action_normalize_transform({"actions": actions}).get(
                "actions", actions
            )
            actions = _transforms.pad_to_dim(
                np.asarray(actions, dtype=np.float32),
                int(act_spec.shape[-1]),
                axis=-1,
            )
            data["actions"] = actions

            next_observation = (
                next_observation if isinstance(next_observation, dict) else {}
            )
            next_obs_transform_input = {
                "image": next_observation.get("image", raw.get("image")),
                "wrist_image": next_observation.get(
                    "wrist_image", raw.get("wrist_image")
                ),
                "state": next_state_source,
                "prompt": prompt,
            }
            next_data = next_obs_pre_token_transform(next_obs_transform_input)

            # Ensure batched image masks.
            batch_shape = tuple(np.asarray(data["state"]).shape[:-1])
            if "image_mask" in data:
                for k, v in data["image_mask"].items():
                    v = np.asarray(v)
                    if v.ndim == 0:
                        data["image_mask"][k] = np.full(
                            batch_shape, bool(v), dtype=np.bool_
                        )
            if "image_mask" in next_data:
                for k, v in next_data["image_mask"].items():
                    v = np.asarray(v)
                    if v.ndim == 0:
                        next_data["image_mask"][k] = np.full(
                            batch_shape, bool(v), dtype=np.bool_
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
                actions = data.get("actions")
                if actions is None:
                    raise ValueError("FAST tokenization requires actions.")
                actions = np.asarray(actions)
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
            data["state"] = _pad_feature_dim(
                data["state"], transition_state_dim, name="state"
            ).astype(np.float32, copy=False)
            next_data["state"] = _pad_feature_dim(
                next_data["state"], transition_state_dim, name="next_state"
            ).astype(np.float32, copy=False)
            insert_batch_size = int(actions.shape[0])
            if prefix_embedding is not None:
                prefix_embedding = np.asarray(prefix_embedding, dtype=np.float32)
                if prefix_embedding.ndim == 2:
                    prefix_embedding = prefix_embedding[None, ...]
                if prefix_embedding.shape[0] != insert_batch_size:
                    if prefix_embedding.shape[0] == 1:
                        prefix_embedding = np.broadcast_to(
                            prefix_embedding,
                            (insert_batch_size,) + prefix_embedding.shape[1:],
                        )
                    else:
                        raise ValueError(
                            "Prefix embedding batch mismatch: "
                            f"{prefix_embedding.shape[0]} vs {insert_batch_size}."
                        )
                data[PREFIX_EMBEDDING_NAME] = prefix_embedding.astype(
                    np.float32, copy=False
                )

            processed_next_prefix = None
            if next_prefix_embedding is not None:
                next_prefix_embedding = np.asarray(
                    next_prefix_embedding, dtype=np.float32
                )
                if next_prefix_embedding.ndim == 2:
                    next_prefix_embedding = next_prefix_embedding[None, ...]
                if next_prefix_embedding.shape[0] != insert_batch_size:
                    if next_prefix_embedding.shape[0] == 1:
                        next_prefix_embedding = np.broadcast_to(
                            next_prefix_embedding,
                            (insert_batch_size,) + next_prefix_embedding.shape[1:],
                        )
                    else:
                        raise ValueError(
                            "Next prefix embedding batch mismatch: "
                            f"{next_prefix_embedding.shape[0]} vs {insert_batch_size}."
                        )
                processed_next_prefix = next_prefix_embedding.astype(
                    np.float32, copy=False
                )
            if data["state"].shape[0] != insert_batch_size:
                raise ValueError(
                    f"Transition state batch mismatch: {data['state'].shape[0]} vs {insert_batch_size}."
                )
            if next_data["state"].shape[0] != insert_batch_size:
                raise ValueError(
                    "Transition next_state batch mismatch: "
                    f"{next_data['state'].shape[0]} vs {insert_batch_size}."
                )
            transition_reward = _ensure_batch_scalar(
                raw.get("reward"), batch_size=insert_batch_size, default=0.0
            )
            transition_discount = _ensure_batch_scalar(
                raw.get("discount"),
                batch_size=insert_batch_size,
                default=transition_gamma,
            )

            transition_next_observation = {
                "image": next_data["image"],
                "image_mask": next_data["image_mask"],
                "state": next_data["state"].astype(np.float32, copy=False),
            }
            # Copy tokenized prompt fields so that get_prefix_rep can include
            # language tokens when computing the next-obs prefix embedding.
            # The prompt is the same for current and next obs within an episode.
            for tok_key in (
                "tokenized_prompt",
                "tokenized_prompt_mask",
                "token_ar_mask",
                "token_loss_mask",
            ):
                if tok_key in data:
                    transition_next_observation[tok_key] = data[tok_key]
            if processed_next_prefix is not None:
                transition_next_observation[PREFIX_EMBEDDING_NAME] = (
                    processed_next_prefix
                )

            return {
                "observation": data,
                "actions": actions,
                "next_observation": transition_next_observation,
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
            batch_size=batch_size,
            data_sharding=data_sharding,
            seed=train_config.seed,
            preprocess_fn=_preprocess_insert,
            postprocess_fn=None,
            freeze_dict=False,
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
                lambda x: x[:, -1] if x.ndim >= 2 else x, observations
            )
        else:
            current_obs = observations

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
            if task_description is not None:
                processed_obs["prompt"] = task_description
            else:
                processed_obs["prompt"] = self.task_description
        return processed_obs

    def _infer_policy_batch_size(self, observations: Dict[str, Any]) -> int:
        """Infer policy batch size from processed observation tensors."""
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

    def _prepare_policy_inputs(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        inputs = jax.tree.map(lambda x: x, observations)
        inputs = self._policy._input_transform(inputs)

        if np.asarray(inputs["state"]).ndim > 1:
            batch_size = int(np.asarray(inputs["state"]).shape[0])

            def _add_batch_dim(x):
                return jnp.broadcast_to(x[jnp.newaxis, ...], (batch_size,) + x.shape)

            inputs = jax.tree.map(lambda x: jnp.asarray(x), inputs)
            for key in list(inputs.keys()):
                if key not in ("image", "state"):
                    inputs[key] = jax.tree.map(_add_batch_dim, inputs[key])
        else:
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        return inputs

    def _compute_prefix_rep_with_model(
        self,
        *,
        model: _model.BaseModel,
        observations: Dict[str, Any],
    ) -> np.ndarray:
        if getattr(self._policy, "_is_pytorch_model", False):
            raise NotImplementedError(
                "Prefix representation extraction is only supported for JAX policies."
            )
        inputs = self._prepare_policy_inputs(observations)
        observation = _model.Observation.from_dict(inputs)
        prefix_rep = self._get_prefix_rep_with_model(
            m=model,
            observation=observation,
        )
        return np.asarray(prefix_rep, dtype=np.float32)

    def _sample_action(
        self,
        observations: Dict,
        rng: jax.random.PRNGKey,
        train_state: training_utils.TrainState,
        batch_actions: bool = True,
        return_prefix_rep: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        params = (
            train_state.ema_params
            if train_state.ema_params is not None
            else train_state.params
        )
        model = nnx.merge(train_state.model_def, params)
        batch_size = self._infer_policy_batch_size(observations)
        noise = jax.random.normal(
            rng, (batch_size, self._policy.action_horizon, self._policy.action_dim)
        )
        sampled_actions = self._policy.infer_with_model(
            model=model,
            obs=observations,
            noise=noise,
            return_prefix_rep=return_prefix_rep,
            sharding_spec=self._policy_sharding_spec,
        )["actions"]
        if not batch_actions:
            return sampled_actions

        # Vector envs expect a batch dimension for actions. Policy inference
        # can unbatch when batch_size == 1, so add it back for single-env runs.
        if isinstance(sampled_actions, (tuple, list)):
            actions = np.asarray(sampled_actions[0])
            if actions.ndim == 2:
                actions = actions[np.newaxis, ...]
            return (actions, *sampled_actions[1:])

        actions = np.asarray(sampled_actions)
        if actions.ndim == 2:
            actions = actions[np.newaxis, ...]
        return actions

    def _generate_actions(
        self, observations: np.ndarray | Dict, **kwargs
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        task_description = kwargs.get("task_description")
        batch_actions = kwargs.get("batch_actions")
        return_prefix_rep = kwargs.get("return_prefix_rep")
        if batch_actions is None:
            batch_actions = False
        if return_prefix_rep is None:
            return_prefix_rep = bool(getattr(self._config, "return_prefix_rep", False))
        rng, self._rng = jax.random.split(self._rng)
        processed_obs = self._process_obs_for_pi0(
            observations, task_description=task_description
        )
        sampled = self._sample_action(
            observations=processed_obs,
            rng=rng,
            train_state=self._train_state,
            batch_actions=batch_actions,
            return_prefix_rep=bool(return_prefix_rep),
        )
        if isinstance(sampled, (tuple, list)):
            return tuple(np.asarray(x, dtype=np.float32) for x in sampled)
        return np.asarray(sampled, dtype=np.float32)

    def eval_actions(self, observations: np.ndarray | Dict, **kwargs) -> np.ndarray:
        # kwargs.setdefault("return_prefix_rep", False)
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

    def sample_online_transitions(self) -> Dict[str, Any]:
        """Sample transitions stored in the online replay buffer."""
        if self._online_data_buffer.size == 0:
            raise ValueError(
                "Cannot sample transitions from an empty online replay buffer."
            )
        batch = self._online_data_buffer.sample()
        return {
            "observation": batch["observation"],
            "actions": batch["actions"],
            "next_observation": batch["next_observation"],
            "reward": batch["reward"],
            "discount": batch["discount"],
        }

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

    def _broadcast_prefix_embedding(
        self, prefix_embedding: Any, observation: Dict[str, Any]
    ) -> np.ndarray:
        """Shape prefix embeddings to match observation layout.

        For per-step collection, policy prefix embeddings are per observation
        (shape [S, E] or [1, S, E]) while observations are sequences over the
        query horizon. This helper repeats the prefix embedding across that
        horizon so it can be stored alongside per-step observations.
        """
        prefix = np.asarray(prefix_embedding, dtype=np.float32)
        if not self._config.collect.add_per_step_data:
            if prefix.ndim == 3 and prefix.shape[0] == 1:
                return prefix[0]
            return prefix

        obs_leaves = jax.tree_util.tree_leaves(observation)
        horizon = next(
            (
                int(np.asarray(leaf).shape[0])
                for leaf in obs_leaves
                if np.asarray(leaf).ndim >= 1
            ),
            None,
        )
        if horizon is None:
            raise ValueError(
                "Cannot infer per-step horizon for prefix embedding broadcast."
            )

        if prefix.ndim == 2:
            return np.repeat(prefix[None, ...], horizon, axis=0)
        if prefix.ndim == 3:
            if prefix.shape[0] == 1:
                return np.repeat(prefix, horizon, axis=0)
            if prefix.shape[0] == horizon:
                return prefix
            raise ValueError(
                "Per-step prefix embedding has incompatible leading dimension: "
                f"got {prefix.shape[0]}, expected 1 or {horizon}."
            )
        raise ValueError(
            "Per-step prefix embedding must be rank-2 or rank-3, "
            f"got shape {prefix.shape}."
        )

    def _compute_prefix_for_observation(
        self,
        *,
        observation: Dict[str, Any] | None,
        task_description: str | None,
    ) -> np.ndarray | None:
        """Compute policy prefix embedding for one observation dict.

        Used for the final transition in an episode, where the next-step
        prefix embedding is not available from action payloads.
        """
        if observation is None:
            return None
        if getattr(self._policy, "_is_pytorch_model", False):
            return None
        prompt = task_description or getattr(self._config, "default_prompt", None)
        if prompt is None:
            return None

        # For per-step collection the observation is a short horizon [H, ...].
        # We only need the final state in that horizon for the terminal next-observation prefix.
        if self._config.collect.add_per_step_data:
            observation = jax.tree.map(
                lambda x: (arr[-1] if (arr := np.asarray(x)).ndim > 0 else arr),
                observation,
            )

        batched_obs = jax.tree.map(lambda x: np.asarray(x)[None, ...], observation)
        processed_obs = self._process_obs_for_pi0(
            batched_obs,
            task_description=str(prompt),
        )
        params = (
            self._train_state.ema_params
            if self._train_state.ema_params is not None
            else self._train_state.params
        )
        model = nnx.merge(self._train_state.model_def, params)
        prefix_rep = self._compute_prefix_rep_with_model(
            model=model,
            observations=processed_obs,
        )
        if prefix_rep.ndim >= 3 and prefix_rep.shape[0] == 1:
            prefix_rep = prefix_rep[0]
        return np.asarray(prefix_rep, dtype=np.float32)

    def _attach_prefix_embeddings_to_episode_data(
        self,
        episode_data: list[Dict[str, Any]],
        *,
        task_description: str | None,
    ) -> None:
        """Unpack `(actions, prefix)` payloads and align prefixes to transitions.

        - Replaces each `ep["action"]` payload with pure actions.
        - Writes the unpacked prefix embedding to `ep["observation"]`.
        - Writes the shifted prefix embedding to `ep["next_observation"]`
          using the next step's payload.
        - Computes the final next-observation prefix with the current model.
        """
        if not episode_data:
            return

        step_prefixes: list[np.ndarray | None] = []
        for ep in episode_data:
            actions, prefix_embedding = unpack_action_and_prefix(ep["action"])
            ep["action"] = np.asarray(actions, dtype=np.float32)
            step_prefixes.append(
                None
                if prefix_embedding is None
                else np.asarray(prefix_embedding, dtype=np.float32)
            )

        final_next_prefix = self._compute_prefix_for_observation(
            observation=episode_data[-1].get("next_observation"),
            task_description=task_description,
        )

        for idx, ep in enumerate(episode_data):
            obs = ep.get("observation")
            curr_prefix = step_prefixes[idx]
            if curr_prefix is not None and isinstance(obs, dict):
                obs[PREFIX_EMBEDDING_NAME] = self._broadcast_prefix_embedding(
                    curr_prefix, obs
                )

            next_obs = ep.get("next_observation")
            if not isinstance(next_obs, dict):
                continue
            next_prefix = (
                step_prefixes[idx + 1]
                if idx + 1 < len(step_prefixes)
                else final_next_prefix
            )
            if next_prefix is not None:
                next_obs[PREFIX_EMBEDDING_NAME] = self._broadcast_prefix_embedding(
                    next_prefix, next_obs
                )

    def save_episode(self, is_success: bool = False, env_index: int = 0, **kwargs):
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
        task_description = kwargs.get("task_description")
        obs_prefix = self._config.collect.obs_prefix_key
        discount_gamma = float(self._config.discount)
        if bool(getattr(self._config, "return_prefix_rep", False)):
            self._attach_prefix_embeddings_to_episode_data(
                episode_data,
                task_description=task_description,
            )

        def _extract_policy_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
            extracted = {}
            for key, val in obs.items():
                if key == PREFIX_EMBEDDING_NAME:
                    extracted[key] = np.asarray(val, dtype=np.float32)
                    continue
                if obs_prefix not in key:
                    continue
                obs_key = key.split(f"{obs_prefix}/")[-1]
                if obs_key == "prompt":
                    continue
                extracted[obs_key] = val
            return extracted

        def process_frame(
            obs: Dict[str, Any],
            *,
            actions: Any,
            next_obs: Dict[str, Any] | None,
            reward: float,
            done: bool,
            discount: float,
        ) -> Dict[str, Any]:
            # Extract actions and observations from total_obs.
            frame = _extract_policy_obs(obs)
            if "state" not in frame:
                raise KeyError(
                    "Cannot construct transitions: current observation is missing state."
                )

            frame["actions"] = self.post_step_action_filter(
                np.asarray(actions, dtype=np.float32)
            )
            next_frame = {
                "image": frame.get("image"),
                "wrist_image": frame.get("wrist_image"),
                "state": frame["state"],
            }
            next_prefix_embedding = frame.get(PREFIX_EMBEDDING_NAME)
            if next_obs is not None:
                next_obs = _extract_policy_obs(next_obs)
                if "image" in next_obs:
                    next_frame["image"] = next_obs["image"]
                if "wrist_image" in next_obs:
                    next_frame["wrist_image"] = next_obs["wrist_image"]
                if "state" in next_obs:
                    next_frame["state"] = next_obs["state"]
                if PREFIX_EMBEDDING_NAME in next_obs:
                    next_prefix_embedding = next_obs[PREFIX_EMBEDDING_NAME]
            frame["next_observation"] = {
                k: v for k, v in next_frame.items() if v is not None
            }
            if next_prefix_embedding is not None:
                frame["next_observation"][PREFIX_EMBEDDING_NAME] = np.asarray(
                    next_prefix_embedding, dtype=np.float32
                )
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
                ep_actions = np.asarray(ep["action"], dtype=np.float32)
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
                            actions=ep_actions[step],
                            next_obs=step_next_obs,
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
            windowed_batch["next_observation"] = jax.tree_util.tree_map(
                lambda x: np.asarray(x)[
                    action_horizon - 1 : action_horizon - 1 + num_windows
                ],
                episode_batch["next_observation"],
            )
            episode_batch = windowed_batch
        else:
            for ep in episode_data:
                ep_actions = np.asarray(ep["action"], dtype=np.float32)
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
                        actions=ep_actions,
                        next_obs=ep.get("next_observation"),
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
        batch = next(self._data_iter)
        use_online = (
            self._online_data_buffer.size >= self._online_data_buffer.batch_size
        )
        if use_online:
            online_batch_raw = self._online_data_buffer.sample()
            online_batch = self._online_batch_to_sft_batch(online_batch_raw)
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
