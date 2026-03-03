from __future__ import annotations

import functools
from dataclasses import dataclass
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
from src.rl.replay_buffer import ShardedReplayBuffer
from src.rl.networks.rl_networks import ActionType, ObsType
from src.rl.types import StepData
from src.training.config import OnlineTrainConfig


@dataclass
class TransitionBatch:
    observation: Any
    action: np.ndarray
    next_observation: Any
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    done: np.ndarray
    n_steps: np.ndarray


class ReplayBuffer:
    """Adapter around ShardedReplayBuffer with TransitionBatch API."""

    def __init__(
        self,
        *,
        dummy_observation: Any,
        flat_action_dim: int,
        batch_size: int,
        capacity: int = 100_000,
        seed: int = 0,
    ):
        self.capacity = int(capacity)
        self._flat_action_dim = int(flat_action_dim)

        dummy_obs = jax.tree_util.tree_map(
            lambda x: np.asarray(x, dtype=np.float32), dummy_observation
        )
        dummy_data = {
            "observation": dummy_obs,
            "action": np.zeros((1, self._flat_action_dim), dtype=np.float32),
            "next_observation": dummy_obs,
            "reward": np.zeros((1,), dtype=np.float32),
            "terminated": np.zeros((1,), dtype=np.bool_),
            "truncated": np.zeros((1,), dtype=np.bool_),
            "done": np.zeros((1,), dtype=np.bool_),
            "n_steps": np.ones((1,), dtype=np.int32),
        }

        self._buffer = ShardedReplayBuffer(
            dummy_data=dummy_data,
            max_capacity=self.capacity,
            batch_size=int(batch_size),
            data_sharding=None,
            seed=int(seed),
            preprocess_fn=self._preprocess_insert,
            postprocess_fn=None,
            freeze_dict=False,
        )

    @property
    def size(self) -> int:
        return int(self._buffer.size)

    @staticmethod
    def _fit_vector(
        values: Any,
        length: int,
        *,
        dtype: Any,
        fill_value: Any,
    ) -> np.ndarray:
        arr = np.asarray(values, dtype=dtype).reshape(-1)
        if arr.size == length:
            return arr
        if arr.size == 0:
            return np.full((length,), fill_value, dtype=dtype)
        if arr.size == 1:
            return np.full((length,), arr.item(), dtype=dtype)
        if arr.size > length:
            return arr[:length]
        return np.pad(
            arr,
            (0, length - arr.size),
            mode="constant",
            constant_values=fill_value,
        )

    def _preprocess_insert(self, transition_batch: Dict[str, Any]) -> Dict[str, Any]:
        obs = jax.tree_util.tree_map(
            lambda x: np.asarray(x, dtype=np.float32), transition_batch["observation"]
        )
        next_obs = jax.tree_util.tree_map(
            lambda x: np.asarray(x, dtype=np.float32), transition_batch["next_observation"]
        )

        actions = np.asarray(transition_batch["action"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, ...]
        actions = actions.reshape(actions.shape[0], -1)
        batch_size = int(actions.shape[0])

        reward = self._fit_vector(
            transition_batch.get("reward", 0.0),
            batch_size,
            dtype=np.float32,
            fill_value=0.0,
        )
        terminated = self._fit_vector(
            transition_batch.get("terminated", False),
            batch_size,
            dtype=np.bool_,
            fill_value=False,
        )
        truncated = self._fit_vector(
            transition_batch.get("truncated", False),
            batch_size,
            dtype=np.bool_,
            fill_value=False,
        )
        done = self._fit_vector(
            transition_batch.get("done", np.logical_or(terminated, truncated)),
            batch_size,
            dtype=np.bool_,
            fill_value=False,
        )
        n_steps = self._fit_vector(
            transition_batch.get("n_steps", 1),
            batch_size,
            dtype=np.int32,
            fill_value=1,
        )

        return {
            "observation": obs,
            "action": actions,
            "next_observation": next_obs,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "done": done,
            "n_steps": n_steps,
        }

    def insert(self, transition: Dict[str, Any]) -> None:
        batched = jax.tree_util.tree_map(lambda x: np.asarray(x)[None, ...], transition)
        self._buffer.insert(batched)

    def sample(self, batch_size: int) -> TransitionBatch:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        if int(batch_size) != int(self._buffer.batch_size):
            raise ValueError(
                f"ReplayBuffer sample batch_size={batch_size} does not match configured "
                f"batch_size={self._buffer.batch_size}."
            )
        stacked = self._buffer.sample()
        return TransitionBatch(
            observation=stacked["observation"],
            action=np.asarray(stacked["action"], dtype=np.float32),
            next_observation=stacked["next_observation"],
            reward=np.asarray(stacked["reward"], dtype=np.float32).reshape(-1),
            terminated=np.asarray(stacked["terminated"], dtype=np.bool_).reshape(-1),
            truncated=np.asarray(stacked["truncated"], dtype=np.bool_).reshape(-1),
            done=np.asarray(stacked["done"], dtype=np.bool_).reshape(-1),
            n_steps=np.asarray(stacked["n_steps"], dtype=np.int32).reshape(-1),
        )
    
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
        dummy_obs = normalize_observation_for_model(dummy_obs)
        self._dummy_obs = dummy_obs
        self._dummy_act = dummy_act
        self._expected_action_shape = expected_chunk_action_shape(np.asarray(dummy_act))
        self._action_dim = int(np.prod(np.asarray(dummy_act).shape[1:]))
        self.replay = ReplayBuffer(
            dummy_observation=dummy_obs,
            flat_action_dim=self._action_dim,
            batch_size=int(getattr(self._config, "batch_size", 128)),
            capacity=100_000,
            seed=int(getattr(self._config, "seed", 0)),
        )

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
        if self.replay.size < batch_size:
            return {}

        info = {}
        latest_actor_observation = None

        if self.training_steps % self._get_critic_update_frequency() == 0:
            batch = self.replay.sample(batch_size=batch_size)

            batch_observation = normalize_observation_for_model(batch.observation)
            batch_next_observation = normalize_observation_for_model(
                batch.next_observation
            )
            observation = jax.tree.map(
                lambda x: jnp.asarray(x, dtype=jnp.float32), batch_observation
            )
            actions = jnp.asarray(batch.action, dtype=jnp.float32)
            next_observation = jax.tree.map(
                lambda x: jnp.asarray(x, dtype=jnp.float32), batch_next_observation
            )
            reward = jnp.asarray(batch.reward, dtype=jnp.float32)
            done = jnp.asarray(batch.done, dtype=jnp.float32)
            n_steps = jnp.asarray(batch.n_steps, dtype=jnp.float32)
            base_discount = jnp.asarray(float(self._config.discount), dtype=jnp.float32)
            discount = jnp.power(base_discount, n_steps) * (1.0 - done)

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
                batch = self.replay.sample(batch_size=batch_size)
                batch_observation = normalize_observation_for_model(batch.observation)
                observation = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), batch_observation)
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

        def _copy_leaf(x, *, dtype=None):
            arr = np.asarray(x, dtype=dtype)
            return np.array(arr, copy=True)

        def _normalize_single_observation(obs_like: Any) -> Any:
            """Normalize one transition observation using batched code-path.

            DSRLVectorEnv emits per-step prefix reps with shape (S, E) for a single
            env transition. Running normalization on unbatched tensors keeps that
            full payload, which is expensive to store/sample. Add a synthetic batch
            axis so normalize_observation_for_model() performs its pooled/batched
            reductions, then remove that axis.
            """
            if isinstance(obs_like, dict):
                batched = jax.tree_util.tree_map(
                    lambda x: np.asarray(x)[None, ...], obs_like
                )
                normalized = normalize_observation_for_model(batched)
                if isinstance(normalized, dict):
                    return jax.tree_util.tree_map(
                        lambda x: np.asarray(x)[0], normalized
                    )
                return np.asarray(normalized)[0]
            return normalize_observation_for_model(obs_like)

        for ep in episode:
            obs = _normalize_single_observation(ep["observation"])
            next_obs = _normalize_single_observation(
                ep.get("next_observation", ep["observation"])
            )
            act = ep.get("action", ep.get("actions"))

            rew = ep.get("reward", 0.0)
            term = ep.get("terminate", False)
            trunc = ep.get("truncate", False)
            r, terminated, truncated, done, n_steps = reduce_chunk_transition(
                reward=rew,
                terminated=term,
                truncated=trunc,
                discount=float(self._config.discount),
                action=act,
            )

            self.replay.insert({
                "observation": jax.tree_util.tree_map(_copy_leaf, obs),
                "action": _copy_leaf(act, dtype=np.float32),
                "next_observation": jax.tree_util.tree_map(_copy_leaf, next_obs),
                "reward": np.float32(r),
                "terminated": np.bool_(terminated),
                "truncated": np.bool_(truncated),
                "done": np.bool_(done),
                "n_steps": np.int32(n_steps),
            })

        self._collection_success_episodes += int(is_success)
    
    def start_data_collection(self, step: int | None = None):
        self._episode_storage = [[] for _ in range(self._config.collect.env_num)]
        self._collection_success_episodes = 0

    def end_data_collection(self, step: int | None = None) -> int:
        collected_episodes = int(self._collection_success_episodes)
        self._episode_storage = [[] for _ in range(self._config.collect.env_num)]
        self._collection_success_episodes = 0
        return collected_episodes
