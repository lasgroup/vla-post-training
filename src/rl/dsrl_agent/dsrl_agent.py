from __future__ import annotations

import functools
from typing import Any, Dict

import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from jax.experimental import mesh_utils

from src.rl.agent import Agent
from src.rl.dsrl_agent.update_actor import (
    init_policy_state,
    train_actor_step,
    PolicyDef,
)
from src.rl.dsrl_agent.update_alpha import (
    alpha_autotune_enabled,
    alpha_value,
    init_alpha_state,
    resolve_target_entropy,
    train_alpha_step,
)
from src.rl.dsrl_agent.update_critic import (
    StateActionCriticDef,
    init_state_action_critic_train_state,
    train_q_step,
)
from src.rl.dsrl_agent.chunk_ops import (
    expected_chunk_action_shape,
    normalize_action_batch_shape,
    normalize_observation_for_model,
    reduce_chunk_transition,
)
from src.rl.prefix_embedding import PREFIX_EMBEDDING_NAME
from src.rl.replay_buffer import ShardedReplayBuffer
from src.rl.networks.rl_networks import ActionType, ObsType
from src.rl.types import StepData
from src.training.config import OnlineTrainConfig


def _extract_replay_observation(
    observation: Any,
    *,
    obs_prefix: str,
) -> Dict[str, np.ndarray]:
    """Convert env observation payloads into AWR-style replay observations.

    Returns a dict with:
    - `state` (required)
    - optional `image`, `wrist_image`
    - optional `prefix_embedding`
    """
    if not isinstance(observation, dict):
        raise TypeError(
            f"Expected dict observation for replay extraction, got {type(observation)}."
        )

    obs_dict = (
        observation["observation"]
        if isinstance(observation.get("observation"), dict)
        else observation
    )

    def _get_from_obs(*keys: str) -> Any | None:
        for key in keys:
            if key in obs_dict:
                return obs_dict[key]
        return None

    extracted: Dict[str, np.ndarray] = {}

    image = _get_from_obs(
        f"{obs_prefix}/image",
        "image",
        "observation/image",
        "pixels",  # fallback for non-LIBERO pixel envs
    )
    if image is not None:
        extracted["image"] = np.asarray(image)

    wrist_image = _get_from_obs(
        f"{obs_prefix}/wrist_image",
        "wrist_image",
        "observation/wrist_image",
    )
    if wrist_image is not None:
        extracted["wrist_image"] = np.asarray(wrist_image)

    state = _get_from_obs(
        f"{obs_prefix}/state",
        "state",
        "observation/state",
    )
    if state is None:
        raise KeyError(
            "Replay extraction requires a state vector. Expected one of "
            f"['{obs_prefix}/state', 'state', 'observation/state']."
        )
    extracted["state"] = np.asarray(state, dtype=np.float32)

    for src in (observation, obs_dict):
        if PREFIX_EMBEDDING_NAME in src:
            extracted[PREFIX_EMBEDDING_NAME] = np.asarray(
                src[PREFIX_EMBEDDING_NAME], dtype=np.float32
            )
            break
        if "prefix_rep" in src:
            extracted[PREFIX_EMBEDDING_NAME] = np.asarray(
                src["prefix_rep"], dtype=np.float32
            )
            break

    return extracted


def _finalize_replay_observation(
    current_obs: Dict[str, np.ndarray],
    next_obs: Dict[str, np.ndarray] | None,
) -> Dict[str, np.ndarray]:
    """Fill next-observation fields from current observation when missing."""
    if next_obs is None:
        next_obs = {}

    merged: Dict[str, np.ndarray] = {
        "state": np.asarray(next_obs.get("state", current_obs["state"]), dtype=np.float32)
    }
    if "image" in next_obs or "image" in current_obs:
        merged["image"] = np.asarray(
            next_obs.get("image", current_obs.get("image")), dtype=np.uint8
        )
    if "wrist_image" in next_obs or "wrist_image" in current_obs:
        merged["wrist_image"] = np.asarray(
            next_obs.get("wrist_image", current_obs.get("wrist_image")), dtype=np.uint8
        )
    prefix = next_obs.get(PREFIX_EMBEDDING_NAME, current_obs.get(PREFIX_EMBEDDING_NAME))
    if prefix is not None:
        merged[PREFIX_EMBEDDING_NAME] = np.asarray(prefix, dtype=np.float32)
    return merged


def _build_replay_observation_template(
    observation: Any,
    *,
    obs_prefix: str,
) -> Dict[str, np.ndarray]:
    """Build fixed-shape replay template preserving image/state modalities."""
    extracted = _extract_replay_observation(observation, obs_prefix=obs_prefix)
    template: Dict[str, np.ndarray] = {}
    if "image" in extracted:
        template["image"] = np.zeros_like(np.asarray(extracted["image"]), dtype=np.uint8)
    if "wrist_image" in extracted:
        template["wrist_image"] = np.zeros_like(
            np.asarray(extracted["wrist_image"]), dtype=np.uint8
        )
    template["state"] = np.zeros_like(np.asarray(extracted["state"]), dtype=np.float32)
    if PREFIX_EMBEDDING_NAME in extracted:
        template[PREFIX_EMBEDDING_NAME] = np.zeros_like(
            np.asarray(extracted[PREFIX_EMBEDDING_NAME]), dtype=np.float32
        )
    return template


def _copy_with_batch_dim(x: Any, *, dtype: Any | None = None) -> np.ndarray:
    arr = np.asarray(x, dtype=dtype)
    return np.array(arr[None, ...], copy=True)


class DSRLLearner(Agent):
    
    def __init__(self, 
        config: OnlineTrainConfig,
        dummy_obs: ObsType,
        dummy_act: ActionType,
        state_action_critic_def: StateActionCriticDef,
        policy_def: PolicyDef,
        task_description: str,):
        self._config = config
        self._rng = jax.random.key(config.seed)
        devices = mesh_utils.create_device_mesh((jax.device_count(),))
        self._mesh = jax.sharding.Mesh(devices, axis_names=("batch",))
        raw_dummy_obs = jax.tree.map(lambda x: np.asarray(x), dummy_obs)
        replay_dummy_obs = _build_replay_observation_template(
            raw_dummy_obs,
            obs_prefix=str(self._config.collect.obs_prefix_key),
        )
        dummy_obs = normalize_observation_for_model(raw_dummy_obs)
        self._dummy_obs = dummy_obs
        self._dummy_act = dummy_act
        self._expected_action_shape = expected_chunk_action_shape(np.asarray(dummy_act))
        self._action_dim = int(np.prod(np.asarray(dummy_act).shape[1:]))

        replay_dummy_obs = jax.tree_util.tree_map(lambda x: np.asarray(x), replay_dummy_obs)
        self._online_data_buffer = ShardedReplayBuffer(
            dummy_data={
                "observation": replay_dummy_obs,
                "actions": np.zeros((1, self._action_dim), dtype=np.float32),
                "real_actions": np.zeros(
                    (1, *self._expected_action_shape), dtype=np.float32
                ),
                "next_observation": replay_dummy_obs,
                "reward": np.zeros((1,), dtype=np.float32),
                "discount": np.zeros((1,), dtype=np.float32),
                "terminated": np.zeros((1,), dtype=np.bool_),
                "truncated": np.zeros((1,), dtype=np.bool_),
                "done": np.zeros((1,), dtype=np.bool_),
                "n_steps": np.ones((1,), dtype=np.int32),
            },
            max_capacity=100_000,
            batch_size=int(getattr(self._config, "batch_size", 128)),
            data_sharding=None,
            seed=int(getattr(self._config, "seed", 0)),
            preprocess_fn=None,
            postprocess_fn=None,
            freeze_dict=False,
        )
        # Keep attribute parity with existing DSRL collection code paths.
        self.replay = self._online_data_buffer
        self._episode_storage = [[] for _ in range(self._config.collect.env_num)]
        self._collection_success_episodes = 0

        q_init_rng, policy_init_rng, alpha_init_rng, self._rng = jax.random.split(
            self._rng, 4
        )
        self._state_action_critic_state, self._state_action_critic_state_sharding = init_state_action_critic_train_state(
                self._config,
                q_init_rng,
                self._mesh,
                critic_def=state_action_critic_def,
                dummy_obs=dummy_obs,
                dummy_act=dummy_act,
                use_sharding=False
            )

        self._policy_state, self._policy_state_sharding = init_policy_state(
            self._config,
            policy_init_rng,
            self._mesh,
            policy_def=policy_def,
            dummy_obs=dummy_obs,
            dummy_act=dummy_act,
            use_sharding=False
        )
        self._alpha_state, self._alpha_state_sharding = init_alpha_state(
            self._config,
            alpha_init_rng,
            self._mesh,
            use_sharding=False,
        )
        self._target_entropy = resolve_target_entropy(self._config, self._action_dim)
        self._autotune_alpha = alpha_autotune_enabled(self._config)

        def _sample_policy_actions(params, obs, rng):
            policy = nnx.merge(self._policy_state.model_def, params)
            policy.eval()
            dist = policy(obs)
            return dist.sample(seed=rng)
        
        def _eval_policy_actions(params, obs):
            policy = nnx.merge(self._policy_state.model_def, params)
            policy.eval()
            dist = policy(obs)
            if hasattr(dist, "mode"):
                return dist.mode()
            if hasattr(dist, "mean"):
                return dist.mean()
            return dist.sample(seed=jax.random.PRNGKey(0))

        self._sample_policy_actions_jit = jax.jit(_sample_policy_actions)
        self._eval_policy_actions_jit = jax.jit(_eval_policy_actions)
        warmup_obs = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), dummy_obs)
        warmup_rng = jax.random.fold_in(self._rng, 0)
        _ = jax.block_until_ready(
            self._sample_policy_actions_jit(self._policy_state.params, warmup_obs, warmup_rng)
        )
        _ = jax.block_until_ready(
            self._eval_policy_actions_jit(self._policy_state.params, warmup_obs)
        )
        self._train_critic_step = jax.jit(functools.partial(train_q_step, self._config))
        self._train_actor_step = jax.jit(functools.partial(train_actor_step, self._config))
        self._train_alpha_step = jax.jit(functools.partial(train_alpha_step, self._config))

    def _sample_action(
        self,
        observations: Dict[str, Any] | np.ndarray,
        rng: jax.random.PRNGKey,
        *,
        deterministic: bool = False,
        batch_actions: bool = True,
    ) -> np.ndarray:
        processed_obs = normalize_observation_for_model(observations)
        obs = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), processed_obs)

        if deterministic:
            sampled_actions = self._eval_policy_actions_jit(self._policy_state.params, obs)
        else:
            sampled_actions = self._sample_policy_actions_jit(self._policy_state.params, obs, rng)

        actions = np.asarray(sampled_actions, dtype=np.float32)
        if not batch_actions:
            return actions

        if actions.ndim == 1:
            actions = actions[None, ...]
        actions = normalize_action_batch_shape(actions, self._expected_action_shape)
        return actions

    def _generate_actions(
        self, observations: np.ndarray | Dict, **kwargs
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        deterministic = kwargs.get("deterministic")
        batch_actions = kwargs.get("batch_actions")
        if deterministic is None:
            deterministic = False
        if batch_actions is None:
            batch_actions = True
        rng, self._rng = jax.random.split(self._rng)
        return np.asarray(
            self._sample_action(
                observations=observations,
                rng=rng,
                deterministic=bool(deterministic),
                batch_actions=bool(batch_actions),
            ),
            dtype=np.float32,
        )

    def eval_actions(self, observations, **kwargs):
        return self._generate_actions(observations, deterministic=True, **kwargs)

    def sample_actions(self, observations, **kwargs):
        return self._generate_actions(observations, deterministic=False, **kwargs)

    def sample_online_transitions(self) -> Dict[str, Any]:
        """Sample transitions from the shared online replay buffer interface."""
        if self._online_data_buffer.size == 0:
            raise ValueError(
                "Cannot sample transitions from an empty online replay buffer."
            )
        batch = self._online_data_buffer.sample()

        return {
            "observation": batch["observation"],
            "actions": np.asarray(batch["actions"], dtype=np.float32),
            "real_actions": np.asarray(batch["real_actions"], dtype=np.float32),
            "next_observation": batch["next_observation"],
            "reward": np.asarray(batch["reward"], dtype=np.float32),
            "discount": np.asarray(batch["discount"], dtype=np.float32),
            "terminated": np.asarray(batch["terminated"], dtype=np.bool_),
            "truncated": np.asarray(batch["truncated"], dtype=np.bool_),
            "done": np.asarray(batch["done"], dtype=np.bool_),
            "n_steps": np.asarray(batch["n_steps"], dtype=np.int32),
        }

    def add_data(self, step_data: StepData):
        def get_env_value(vec, env_id):
            return jax.tree.map(lambda x: x[env_id], vec)

        for i in range(self._config.collect.env_num):
            self._episode_storage[i].append(get_env_value(step_data, i))

    def _get_critic_update_frequency(self) -> int:
        rl = getattr(self._config, "rl", None)
        return int(getattr(rl, "critic_update_frequency", 1))

    def _get_actor_update_frequency(self) -> int:
        rl = getattr(self._config, "rl", None)
        return int(getattr(rl, "actor_update_frequency", 1))

    def _current_alpha(self) -> jax.Array:
        return alpha_value(self._alpha_state)

    def _update_alpha(self, entropy: jax.Array) -> dict[str, jax.Array]:
        entropy = jnp.asarray(entropy, dtype=jnp.float32)
        if not self._autotune_alpha:
            alpha = self._current_alpha()
            return {
                "alpha": alpha,
                "alpha_loss": jnp.asarray(0.0, dtype=jnp.float32),
                "entropy": entropy,
                "log_prob_mean": -entropy,
                "target_entropy": jnp.asarray(self._target_entropy, dtype=jnp.float32),
            }

        train_rng, self._rng = jax.random.split(self._rng)
        self._alpha_state, alpha_info = self._train_alpha_step(
            train_rng,
            self._alpha_state,
            entropy,
            jnp.asarray(self._target_entropy, dtype=jnp.float32),
        )
        return alpha_info

    def update(self):
        self.training_steps += 1

        batch_size = int(getattr(self._config, "batch_size", 128))
        if self._online_data_buffer.size < batch_size:
            return {}

        info = {}
        latest_actor_observation = None

        if self.training_steps % self._get_critic_update_frequency() == 0:
            if batch_size != int(self._online_data_buffer.batch_size):
                raise ValueError(
                    "Configured train batch_size does not match replay batch_size: "
                    f"{batch_size} vs {self._online_data_buffer.batch_size}."
                )
            batch = self._online_data_buffer.sample()
            batch_observation = normalize_observation_for_model(batch["observation"])
            batch_next_observation = normalize_observation_for_model(
                batch["next_observation"]
            )
            observation = jax.tree.map(
                lambda x: jnp.asarray(x, dtype=jnp.float32), batch_observation
            )
            actions = jnp.asarray(batch["actions"], dtype=jnp.float32)
            next_observation = jax.tree.map(
                lambda x: jnp.asarray(x, dtype=jnp.float32), batch_next_observation
            )
            reward = jnp.asarray(batch["reward"], dtype=jnp.float32)
            discount = jnp.asarray(batch["discount"], dtype=jnp.float32)

            critic_batch = (observation, actions, next_observation, reward, discount)

            train_rng, self._rng = jax.random.split(self._rng)
            self._state_action_critic_state, critic_info = self._train_critic_step(
                train_rng,
                self._state_action_critic_state,
                self._policy_state,
                critic_batch,
                self._current_alpha(),
            )
            latest_actor_observation = observation
            info.update({f"critic/{k}": v for k, v in critic_info.items()})

        if self.training_steps % self._get_actor_update_frequency() == 0:
            if latest_actor_observation is None:
                batch = self._online_data_buffer.sample()
                batch_observation = normalize_observation_for_model(batch["observation"])
                observation = jax.tree.map(
                    lambda x: jnp.asarray(x, dtype=jnp.float32), batch_observation
                )
            else:
                observation = latest_actor_observation

            train_rng, self._rng = jax.random.split(self._rng)
            self._policy_state, actor_info = self._train_actor_step(
                train_rng,
                self._policy_state,
                self._state_action_critic_state,
                observation,
                self._current_alpha(),
            )
            info.update({f"actor/{k}": v for k, v in actor_info.items()})
            if "entropy" in actor_info:
                alpha_info = self._update_alpha(actor_info["entropy"])
                info.update({f"alpha/{k}": v for k, v in alpha_info.items()})
            elif "log_prob_mean" in actor_info:
                alpha_info = self._update_alpha(-actor_info["log_prob_mean"])
                info.update({f"alpha/{k}": v for k, v in alpha_info.items()})

        info.setdefault("alpha/value", self._current_alpha())
        return info
        
    def save_episode(self, is_success: bool = False, env_index: int = 0, **kwargs):
        episode = self._episode_storage[env_index]
        self._episode_storage[env_index] = []

        if not episode:
            return

        obs_prefix = str(self._config.collect.obs_prefix_key)
        base_discount = float(self._config.discount)

        for ep in episode:
            obs = _extract_replay_observation(ep["observation"], obs_prefix=obs_prefix)
            next_obs_raw = ep.get("next_observation", ep["observation"])
            try:
                next_obs_extracted = _extract_replay_observation(
                    next_obs_raw,
                    obs_prefix=obs_prefix,
                )
            except (TypeError, KeyError):
                next_obs_extracted = obs
            next_obs = _finalize_replay_observation(obs, next_obs_extracted)
            act = ep.get("action", ep.get("actions"))
            if act is None:
                raise KeyError("Episode transition is missing `action` / `actions`.")
            real_action = ep.get("real_action", ep.get("real_actions", act))

            rew = ep.get("reward", 0.0)
            term = ep.get("terminate", False)
            trunc = ep.get("truncate", False)
            r, terminated, truncated, done, n_steps = reduce_chunk_transition(
                reward=rew,
                terminated=term,
                truncated=trunc,
                discount=base_discount,
                action=act,
            )
            discount = (base_discount ** int(n_steps)) * (1.0 - float(done))
            policy_actions = np.asarray(act, dtype=np.float32).reshape(1, -1)
            real_actions = np.asarray(real_action, dtype=np.float32)
            if real_actions.shape != self._expected_action_shape:
                if int(np.prod(real_actions.shape, dtype=np.int64)) == self._action_dim:
                    real_actions = real_actions.reshape(self._expected_action_shape)
            real_actions = np.asarray(real_actions, dtype=np.float32)[None, ...]

            self._online_data_buffer.insert(
                {
                    "observation": jax.tree_util.tree_map(_copy_with_batch_dim, obs),
                    "actions": policy_actions,
                    "real_actions": real_actions,
                    "next_observation": jax.tree_util.tree_map(
                        _copy_with_batch_dim, next_obs
                    ),
                    "reward": np.asarray([r], dtype=np.float32),
                    "discount": np.asarray([discount], dtype=np.float32),
                    "terminated": np.asarray([terminated], dtype=np.bool_),
                    "truncated": np.asarray([truncated], dtype=np.bool_),
                    "done": np.asarray([done], dtype=np.bool_),
                    "n_steps": np.asarray([n_steps], dtype=np.int32),
                }
            )

        self._collection_success_episodes += int(is_success)
    
    def start_data_collection(self, step: int | None = None):
        self._episode_storage = [[] for _ in range(self._config.collect.env_num)]
        self._collection_success_episodes = 0

    def end_data_collection(self, step: int | None = None) -> int:
        collected_episodes = int(self._collection_success_episodes)
        self._episode_storage = [[] for _ in range(self._config.collect.env_num)]
        self._collection_success_episodes = 0
        return collected_episodes
