from __future__ import annotations

import functools
import gc
import logging
import os
import weakref
from typing import Any, Dict
from dataclasses import dataclass
from collections import deque
from typing import Any, Dict, Deque, Optional
import numpy as np
import jax
import jax.numpy as jnp
import optax

import etils.epath as epath
import flax.nnx as nnx
import flax.traverse_util as traverse_util
from jax.experimental import mesh_utils

from src.rl.agent import Agent
from src.rl.networks.rl_networks import ObsType, ActionType
from src.rl.prefix_embedding import PREFIX_EMBEDDING_NAME
from src.training.config import OnlineTrainConfig
from src.rl.dsrl_agent.update_actor import (
    train_actor_step, 
    init_policy_state,
    PolicyDef
)
from src.rl.dsrl_agent.update_critic import (
    init_state_action_critic_train_state,
    train_q_step,
    StateActionCriticDef,
    CriticBatch,
)


@dataclass
class TransitionBatch:
    observation: Any
    action: np.ndarray
    next_observation: Any
    reward: np.ndarray
    done: np.ndarray


@dataclass
class AlphaState:
    log_alpha: jax.Array
    opt_state: optax.OptState
    tx: optax.GradientTransformation

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
            done=np.asarray(stacked["done"], dtype=np.bool_).reshape(-1),
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

        q_init_rng, polciy_init_rng, self._rng = jax.random.split(self._rng, 3)
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
            polciy_init_rng,
            self._mesh,
            policy_def=policy_def,
            dummy_obs=dummy_obs,
            dummy_act=dummy_act,
            use_sharding=False
        )
        def _sample_policy_actions(params, obs, rng):
            policy = nnx.merge(self._policy_state.model_def, params)
            policy.eval()
            dist = policy(obs)
            return dist.sample(seed=rng)

        self._sample_policy_actions_jit = jax.jit(_sample_policy_actions)
        warmup_obs = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), dummy_obs)
        warmup_rng = jax.random.fold_in(self._rng, 0)
        _ = jax.block_until_ready(
            self._sample_policy_actions_jit(self._policy_state.params, warmup_obs, warmup_rng)
        )
        self._train_critic_step = jax.jit(functools.partial(train_q_step, self._config))
        self._train_actor_step = jax.jit(functools.partial(train_actor_step, self._config))
        self._alpha_state = self._init_alpha_state()
        self._target_entropy = self._get_target_entropy()

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
        pass        

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

    def _get_utd_ratio(self) -> int:
        rl = getattr(self._config, "rl", None)
        return max(1, int(getattr(rl, "utd_ratio", 1)))

    def _alpha_autotune_enabled(self) -> bool:
        rl = getattr(self._config, "rl", None)
        return bool(getattr(rl, "autotune_alpha", True))

    def _get_init_alpha(self) -> float:
        rl = getattr(self._config, "rl", None)
        return max(1e-6, float(getattr(rl, "init_alpha", 0.1)))

    def _get_alpha_lr(self) -> float:
        rl = getattr(self._config, "rl", None)
        return max(1e-8, float(getattr(rl, "alpha_lr", 3e-4)))

    def _get_target_entropy(self) -> float:
        rl = getattr(self._config, "rl", None)
        target_entropy = getattr(rl, "target_entropy", "auto")
        if target_entropy in (None, "auto"):
            return -float(self._action_dim)
        return float(target_entropy)

    def _init_alpha_state(self) -> AlphaState:
        init_alpha = self._get_init_alpha()
        log_alpha = jnp.asarray(np.log(init_alpha), dtype=jnp.float32)
        tx = optax.adam(self._get_alpha_lr())
        opt_state = tx.init(log_alpha)
        return AlphaState(log_alpha=log_alpha, opt_state=opt_state, tx=tx)

    def _current_alpha(self) -> jax.Array:
        return jnp.exp(self._alpha_state.log_alpha)

    def _update_alpha(self, log_prob_mean: jax.Array) -> dict[str, jax.Array]:
        if not self._alpha_autotune_enabled():
            alpha = self._current_alpha()
            entropy = -jnp.asarray(log_prob_mean, dtype=jnp.float32)
            return {
                "alpha": alpha,
                "alpha_loss": jnp.asarray(0.0, dtype=jnp.float32),
                "entropy_mean": entropy,
                "target_entropy": jnp.asarray(self._target_entropy, dtype=jnp.float32),
            }

        log_prob_mean = jax.lax.stop_gradient(jnp.asarray(log_prob_mean, dtype=jnp.float32))
        entropy_mean = -log_prob_mean
        target_entropy_mag = jnp.asarray(-self._target_entropy, dtype=jnp.float32)

        def alpha_loss_fn(log_alpha):
            alpha = jnp.exp(log_alpha)
            # Increase alpha when observed entropy exceeds the target.
            return alpha * (target_entropy_mag - entropy_mean)

        alpha_loss, grads = jax.value_and_grad(alpha_loss_fn)(self._alpha_state.log_alpha)
        updates, new_opt_state = self._alpha_state.tx.update(
            grads, self._alpha_state.opt_state, self._alpha_state.log_alpha
        )
        new_log_alpha = optax.apply_updates(self._alpha_state.log_alpha, updates)
        new_log_alpha = jnp.clip(new_log_alpha, -20.0, 2.0)
        self._alpha_state = AlphaState(
            log_alpha=new_log_alpha,
            opt_state=new_opt_state,
            tx=self._alpha_state.tx,
        )

        return {
            "alpha": jnp.exp(new_log_alpha),
            "alpha_loss": alpha_loss,
            "entropy_mean": entropy_mean,
            "target_entropy": jnp.asarray(self._target_entropy, dtype=jnp.float32),
        }

    def update(self):
        self.training_steps += 1

        batch_size = int(getattr(self._config, "batch_size", 128))
        if self.replay.size < batch_size:
            return {}

        info = {}

        # ---- critic update(s): UTD ratio controls repeated critic steps per learner step ----
        if self.training_steps % self._get_critic_update_frequency() == 0:
            for _ in range(self._get_utd_ratio()):
                batch = self.replay.sample(batch_size=batch_size)

                observation = jax.tree.map(
                    lambda x: jnp.asarray(x, dtype=jnp.float32), batch.observation
                )
                actions = jnp.asarray(batch.action, dtype=jnp.float32)
                next_observation = jax.tree.map(
                    lambda x: jnp.asarray(x, dtype=jnp.float32), batch.next_observation
                )
                reward = jnp.asarray(batch.reward, dtype=jnp.float32)

                done = jnp.asarray(batch.done, dtype=jnp.float32)
                discount = jnp.asarray(
                    float(self._config.discount) * (1.0 - done), dtype=jnp.float32
                )

                critic_batch = (observation, actions, next_observation, reward, discount)

                train_rng, self._rng = jax.random.split(self._rng)
                self._state_action_critic_state, critic_info = self._train_critic_step(
                    train_rng,
                    self._state_action_critic_state,
                    self._policy_state,
                    critic_batch,
                    self._current_alpha(),
                )
                info.update({f"critic/{k}": v for k, v in critic_info.items()})

        # ---- actor update (AWR-style: periodic separate step) ----
        if self.training_steps % self._get_actor_update_frequency() == 0:
            batch = self.replay.sample(batch_size=batch_size)

            observation = jax.tree.map(
                lambda x: jnp.asarray(x, dtype=jnp.float32), batch.observation
            )

            train_rng, self._rng = jax.random.split(self._rng)
            self._policy_state, actor_info = self._train_actor_step(
                train_rng,
                self._policy_state,
                self._state_action_critic_state,
                observation,
                self._current_alpha(),
            )
            info.update({f"actor/{k}": v for k, v in actor_info.items()})
            if "log_prob_mean" in actor_info:
                alpha_info = self._update_alpha(actor_info["log_prob_mean"])
                info.update({f"alpha/{k}": v for k, v in alpha_info.items()})

        info.setdefault("alpha/value", self._current_alpha())
        info.setdefault("sac/utd_ratio", jnp.asarray(self._get_utd_ratio(), dtype=jnp.float32))

        return info

        
    def save_episode(self, is_success: bool = False, env_index: int = 0, **kwargs):
        episode = self._episode_storage[env_index]
        self._episode_storage[env_index] = []

        if not episode:
            return

        for ep in episode:
            obs = ep["observation"]
            next_obs = ep.get("next_observation", obs)
            act = ep.get("action", ep.get("actions"))

            rew = ep.get("reward", 0.0)
            term = ep.get("terminate", False)
            trunc = ep.get("truncate", False)

            r = float(np.asarray(rew).reshape(-1)[0]) if rew is not None else 0.0
            done = bool(np.asarray(term).reshape(-1)[0] or np.asarray(trunc).reshape(-1)[0])

            self.replay.insert({
                "observation": jax.tree_util.tree_map(np.asarray, obs),
                "action": np.asarray(act, dtype=np.float32),
                "next_observation": jax.tree_util.tree_map(np.asarray, next_obs),
                "reward": np.float32(r),
                "done": np.bool_(done),
            })

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
