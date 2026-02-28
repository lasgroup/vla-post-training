from __future__ import annotations

import functools
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict

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


class MinimalReplayBuffer:
    """Stores transitions of the form (obs, act, next_obs, reward, done)."""

    def __init__(self, capacity: int = 100_000, seed: int = 0):
        self.capacity = int(capacity)
        self._buf: Deque[Dict[str, Any]] = deque(maxlen=self.capacity)
        self._rng = np.random.default_rng(seed)

    @property
    def size(self) -> int:
        return len(self._buf)

    def insert(self, transition: Dict[str, Any]) -> None:
        # store numpy arrays / pytrees of numpy arrays
        self._buf.append(transition)

    def sample(self, batch_size: int) -> TransitionBatch:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        b = min(int(batch_size), self.size)
        idx = self._rng.integers(0, self.size, size=b, endpoint=False)
        samples = [self._buf[i] for i in idx]

        # stack pytree leaves (works for dict obs, arrays, etc.)
        stacked = jax.tree_util.tree_map(lambda *xs: np.stack(xs, axis=0), *samples)
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
        self.replay = MinimalReplayBuffer(capacity=100000, seed=int(getattr(self._config, "seed", 0)))
        self._rng = jax.random.key(config.seed)
        devices = mesh_utils.create_device_mesh((jax.device_count(),))
        self._mesh = jax.sharding.Mesh(devices, axis_names=("batch",))
        self._dummy_obs = dummy_obs
        self._dummy_act = dummy_act
        self._action_dim = int(np.prod(np.asarray(dummy_act).shape[1:]))

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

    def eval_actions(self, observations, **kwargs):
        obs = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), observations)
        actions = self._eval_policy_actions_jit(self._policy_state.params, obs)
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, ...]
        return actions

    def sample_actions(self, observations, **kwargs):
        obs = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), observations)
        rng, self._rng = jax.random.split(self._rng)
        actions = self._sample_policy_actions_jit(self._policy_state.params, obs, rng)
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, ...]  # single-env safety
        
        return np.asarray(actions, dtype=np.float32)

    def _generate_actions(
        self, observations: np.ndarray | Dict, **kwargs
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if bool(kwargs.get("deterministic", False)):
            return self.eval_actions(observations, **kwargs)
        return self.sample_actions(observations, **kwargs)

    def add_data(self, step_data: StepData):
        def get_env_value(vec, env_id):
            return jax.tree.map(lambda x: x[env_id], vec)

        for i in range(self._config.collect.env_num): #self._config.collect.env_num
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

            observation = jax.tree.map(
                lambda x: jnp.asarray(x, dtype=jnp.float32), batch.observation
            )
            actions = jnp.asarray(batch.action, dtype=jnp.float32)
            next_observation = jax.tree.map(
                lambda x: jnp.asarray(x, dtype=jnp.float32), batch.next_observation
            )
            reward = jnp.asarray(batch.reward, dtype=jnp.float32)

            # Time-limit truncations should not zero the bootstrap term.
            # Only true environment terminations should set discount to zero.
            done = jnp.asarray(batch.terminated, dtype=jnp.float32)
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
                observation = jax.tree.map(
                    lambda x: jnp.asarray(x, dtype=jnp.float32), batch.observation
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

        def _copy_leaf(x, *, dtype=None):
            arr = np.asarray(x, dtype=dtype)
            return np.array(arr, copy=True)

        for ep in episode:
            obs = ep["observation"]
            next_obs = ep.get("next_observation", obs)
            act = ep.get("action", ep.get("actions"))

            rew = ep.get("reward", 0.0)
            term = ep.get("terminate", False)
            trunc = ep.get("truncate", False)

            # Support both scalar and per-substep payloads from collection.
            def _fit_length(arr, length, *, dtype, pad_value):
                values = np.asarray(arr, dtype=dtype).reshape(-1)
                if values.size == length:
                    return values
                if values.size == 0:
                    return np.full((length,), pad_value, dtype=dtype)
                if values.size == 1:
                    return np.full((length,), values.item(), dtype=dtype)
                if values.size > length:
                    return values[:length]
                return np.pad(
                    values,
                    (0, length - values.size),
                    mode="constant",
                    constant_values=pad_value,
                )

            reward_arr = np.asarray(
                rew if rew is not None else 0.0, dtype=np.float32
            ).reshape(-1)
            term_arr = np.asarray(term, dtype=np.bool_).reshape(-1)
            trunc_arr = np.asarray(trunc, dtype=np.bool_).reshape(-1)
            rollout_len = int(max(reward_arr.size, term_arr.size, trunc_arr.size, 1))
            reward_arr = _fit_length(
                reward_arr, rollout_len, dtype=np.float32, pad_value=0.0
            )
            term_arr = _fit_length(
                term_arr, rollout_len, dtype=np.bool_, pad_value=False
            )
            trunc_arr = _fit_length(
                trunc_arr, rollout_len, dtype=np.bool_, pad_value=False
            )

            done_arr = np.logical_or(term_arr, trunc_arr)
            if bool(np.any(done_arr)):
                n_steps = int(np.argmax(done_arr) + 1)
            else:
                n_steps = rollout_len
                # For chunked-action env wrappers that collapse rewards/dones to scalars,
                # infer the effective transition length from the action horizon.
                action_arr = np.asarray(act)
                if rollout_len == 1 and action_arr.ndim >= 2:
                    n_steps = max(1, int(action_arr.shape[0]))

            reward_weights = np.power(
                np.float32(self._config.discount),
                np.arange(n_steps, dtype=np.float32),
            )
            r = float(np.sum(reward_arr[:n_steps] * reward_weights))
            terminated = bool(term_arr[n_steps - 1])
            truncated = bool(trunc_arr[n_steps - 1])
            done = bool(terminated or truncated)

            self.replay.insert({
                # Copy leaves to avoid aliasing with collector buffers that are
                # mutated in-place after per-env resets.
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
        # Reset episode storage
        self._episode_storage = [[] for _ in range(self._config.collect.env_num)]
        self._collection_success_episodes = 0

    def end_data_collection(self, step: int | None = None) -> int:
        collected_episodes = int(self._collection_success_episodes)
        # Reset episode storage and counter for the next collection round.
        self._episode_storage = [[] for _ in range(self._config.collect.env_num)]
        self._collection_success_episodes = 0
        return collected_episodes
