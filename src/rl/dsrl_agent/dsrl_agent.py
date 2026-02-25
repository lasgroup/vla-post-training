from __future__ import annotations

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
from src.rl.dsrl_agent.update_actor import train_actor_step
from src.rl.dsrl_agent.update_critic import train_q_step
from src.rl.replay_buffer import ShardedReplayBuffer
from src.rl.types import StepData
from src.training.config import OnlineTrainConfig
from src.training.data_loader import create_data_loader
from src.envs.venv import SubprocVectorEnv, DummyVectorEnv
from src.rl.agent import Agent, EnvFn
from src.rl.prefix_embedding import PREFIX_EMBEDDING_NAME, unpack_action_and_prefix
from src.rl.networks.rl_networks import StateActionCritic, Policy
from src.rl.networks.encoders.encoders import ImageEncoder
from src.rl.networks.decoders.policies.normal_policy import NormalPolicyDecoder

from dataclasses import dataclass
from collections import deque
from typing import Any, Dict, Deque, Optional
import numpy as np
import jax

@dataclass
class TransitionBatch:
    observation: Any
    action: np.ndarray
    next_observation: Any
    reward: np.ndarray
    done: np.ndarray


# def make_dummy_from_env(obs_example: dict, action_space, batch_size: int):
#     # obs_example is what env.reset() returns: dict of np arrays
#     dummy_obs = {k: jnp.zeros((batch_size,) + np.asarray(v).shape, dtype=jnp.float32)
#                  for k, v in obs_example.items()}
#     # Box action space -> shape like (act_dim,)
#     act_shape = action_space.shape
#     dummy_act = jnp.zeros((batch_size,) + act_shape, dtype=jnp.float32)
#     return dummy_obs, dummy_act

# def init_policy_from_env(env, encoder_def, decoder_def, seed: int = 0):
#     obs0, _ = env.reset()
#     # env.action_space is a list for BaseVectorEnv; take first
#     space = env.action_space[0] if isinstance(env.action_space, (list, tuple)) else env.action_space
#     dummy_obs, dummy_act = make_dummy_from_env(obs0, space, batch_size=1)

#     rng = jax.random.key(seed)
#     rngs = nnx.Rngs(params=rng, dropout=rng)  # add others if your modules use them
#     policy = Policy(
#         observation=dummy_obs,
#         action=dummy_act,
#         encoder_def=encoder_def,
#         decoder_def=decoder_def,
#         rngs=rngs,
#     )
#     return policy


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


class Critic(nnx.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(obs_dim + act_dim, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, hidden, rngs=rngs)
        self.out = nnx.Linear(hidden, 1, rngs=rngs)

    def __call__(self, obs, act):
        x = nnx.relu(self.fc1(jnp.concatenate([obs, act], -1)))
        return self.out(nnx.relu(self.fc2(x))).squeeze(-1)

class Actor(nnx.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(obs_dim, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, hidden, rngs=rngs)
        self.mean_head = nnx.Linear(hidden, act_dim, rngs=rngs)
        self.log_std_head = nnx.Linear(hidden, act_dim, rngs=rngs)
        self.rng = jax.random.key(0)

    def __call__(self, obs, training):
        x = nnx.relu(self.fc2(nnx.relu(self.fc1(obs))))
        mean = self.mean_head(x)
        log_std = jnp.clip(self.log_std_head(x), -5.0, 2.0)
        std = jnp.exp(log_std)
        noise = jax.random.normal(self.rng, mean.shape)
        raw = mean + std * noise
        action = jnp.tanh(raw)
        log_prob = jnp.sum(
            -0.5 * (noise ** 2 + jnp.log(2.0 * jnp.pi)) - log_std
            - jnp.log(1.0 - action ** 2 + 1e-6),
            axis=-1,
        )
        return action, log_prob
    
def flatten_dmc_obs(obs: dict) -> jnp.ndarray:
    # obs leaves are (E, dim)
    pos = jnp.asarray(obs["position"], dtype=jnp.float32)
    vel = jnp.asarray(obs["velocity"], dtype=jnp.float32)
    return jnp.concatenate([pos, vel], axis=-1)

class DSRLLearner(Agent):
    
    def __init__(self, config: OnlineTrainConfig, env):
        self._config = config
        self._action_space = env.action_space
        self.replay = MinimalReplayBuffer(capacity=100000, seed=int(getattr(self._config, "seed", 0)))
        self._rng, init = jax.random.split(jax.random.key(config.seed))
        self.actor = Actor(obs_dim=5, act_dim=1, hidden=256, rngs=nnx.Rngs(params=init))
        self._actor_apply = nnx.jit(lambda m, x: m(x, training=False))
        self.critic = Critic(obs_dim=5, act_dim=1, hidden=256, rngs=nnx.Rngs(params=self._rng))
        #encoder_def = lambda obs, rngs: ImageEncoder(obs, rngs=rngs, layers=2, units=256)
        #decoder_def = lambda obs, rngs: NormalPolicyDecoder(obs, rngs=rngs, layers=2, units=256)
        #self.actor = Policy(observation=None, action=None, encoder_def=None, decoder_def=None, rngs=None)
        # self.critic = StateActionCritic(observation=None, action=None, encoder_def=None, decoder_def=None, rngs=None)

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
        self._train_critic_step = jax.jit(
            functools.partial(train_q_step, config),
            # in_shardings=(
            #     self._replicated_sharding,
            #     self._train_state_sharding,
            #     self._data_sharding,
            # ),
            #out_shardings=(self._train_state_sharding, self._replicated_sharding),
            donate_argnums=(1,),
        )
        # self._online_data_buffer = self._get_online_replay_buffer(
        #     self._data_sharding,
        #     prefix_embedding_template= None, # prefix_embedding_template
        # )

    def sample_actions(self, observations, **kwargs):
        #actions = np.stack([self._action_space[0].sample() for _ in range(1)], axis=0)
        #return actions
        observations = flatten_dmc_obs(observations)
        obs_jax = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float32), observations)
        actions, logprob = self._actor_apply(self.actor, obs_jax)

        # Sampling needs a seed; TFP uses `seed=` in .sample
        #rng, sub = jax.random.split(jax.random.key(0))
        #actions = dist.sample(seed=sub)   # shape (E, act_dim) for Normal/TanhNormal decoders
        return actions #rng, np.asarray(actions, dtype=np.float32)

    def _generate_actions(
        self, observations: np.ndarray | Dict, **kwargs
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        pass        

    def add_data(self, step_data: StepData):
        def get_env_value(vec, env_id):
            return jax.tree.map(lambda x: x[env_id], vec)

        for i in range(1): #self._config.collect.env_num
            self._episode_storage[i].append(get_env_value(step_data, i))

    def update(self):
        #return {}
        if self.replay.size < 128:
            return {}
        # self.training_steps += 1
        # #use_online = True
        # #if use_online:
        # #    online_batch_raw
        
        #batch = self.replay.sample()
        batch = self.replay.sample(batch_size=128) # self._config.batch_size
        
        train_rng, self._rng = jax.random.split(self._rng)

        train_state = self._train_state
        with sharding.set_mesh(self._mesh):
            #train_state, info = self._train_actor_step(train_rng, train_state, batch)
            #train_state, info = self._train_critic_step(train_rng, train_state, batch)
            self._q_state, info = self._train_critic_step(train_rng, self._q_state, self.actor_model_for_training, batch)
        self._train_state = train_state
        return info
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
