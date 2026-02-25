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
        # self._train_actor_step = jax.jit(
        #     functools.partial(train_actor_step, config),
        #     # in_shardings=(
        #     #     self._replicated_sharding,
        #     #     self._train_state_sharding,
        #     #     self._data_sharding,
        #     # ),
        #     #out_shardings=(self._train_state_sharding, self._replicated_sharding),
        #     donate_argnums=(1,),
        # )
        # self._train_critic_step = jax.jit(
        #     functools.partial(train_q_step, config),
        #     # in_shardings=(
        #     #     self._replicated_sharding,
        #     #     self._train_state_sharding,
        #     #     self._data_sharding,
        #     # ),
        #     #out_shardings=(self._train_state_sharding, self._replicated_sharding),
        #     donate_argnums=(1,),
        # )
        # self._online_data_buffer = self._get_online_replay_buffer(
        #     self._data_sharding,
        #     prefix_embedding_template= None, # prefix_embedding_template
        # )

    def sample_actions(self, observations, **kwargs):
        #actions = np.stack([self._dummy_act for _ in range(1)], axis=0)
        #return np.asarray(self._dummy_act, dtype=np.float32)
        obs = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), observations)
        # policy = nnx.merge(self._policy_state.model_def, self._policy_state.params)
        # policy.eval()
        # dist = policy(obs)
        # rng, self._rng = jax.random.split(self._rng)
        # actions = dist.sample(seed=rng)
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

    def update(self):
        return {}
        # if self.replay.size < 128:
        #     return {}
        # # self.training_steps += 1
        # # #use_online = True
        # # #if use_online:
        # # #    online_batch_raw
        
        # #batch = self.replay.sample()
        # batch = self.replay.sample(batch_size=128) # self._config.batch_size
        
        # train_rng, self._rng = jax.random.split(self._rng)

        # train_state = self._train_state
        # with sharding.set_mesh(self._mesh):
        #     #train_state, info = self._train_actor_step(train_rng, train_state, batch)
        #     #train_state, info = self._train_critic_step(train_rng, train_state, batch)
        #     self._q_state, info = self._train_critic_step(train_rng, self._q_state, self.actor_model_for_training, batch)
        # self._train_state = train_state
        # return info
        #return {}
    
    def save_episode(self, is_success: bool = False, env_index: int = 0, **kwargs):
        # For now, store *all* episodes. If you want success-only, uncomment:
        # if not is_success: self._episode_storage[env_index] = []; return

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
