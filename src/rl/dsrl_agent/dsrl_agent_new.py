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


def _pytree_size_mb(tree) -> float:
    """Return total size of all arrays in a pytree, in megabytes."""
    leaves = jax.tree.leaves(tree)
    total_bytes = sum(
        leaf.size * leaf.dtype.itemsize for leaf in leaves if hasattr(leaf, "size")
    )
    return total_bytes / (1024 * 1024)


@dataclass
class TransitionBatch:
    observation: Any
    action: np.ndarray
    next_observation: Any
    reward: np.ndarray
    done: np.ndarray


# ---------------------------------------------------------------------------
# Simple train-state container (mirrors the role of the AWR learner's
# init_state_action_critic_train_state / init_state_value_train_state)
# ---------------------------------------------------------------------------

@dataclass
class SimpleTrainState:
    """Lightweight container that holds a model, its optimizer state, and step."""
    model_def: nnx.GraphDef
    params: nnx.State
    opt_state: Any          # optax optimizer state
    opt_def: Any            # optax optimizer (GradientTransformation)
    step: int = 0

    # Convenience: merge params back into a live module
    def to_model(self) -> nnx.Module:
        return nnx.merge(self.model_def, self.params)


# Register SimpleTrainState as a JAX pytree so jit/grad can trace through it.
# "data_fields" are array leaves that JAX will trace; "meta_fields" are static.
jax.tree_util.register_dataclass(
    SimpleTrainState,
    data_fields=["params", "opt_state", "step"],
    meta_fields=["model_def", "opt_def"],
)


def _init_train_state(
    module: nnx.Module,
    learning_rate: float,
) -> SimpleTrainState:
    """Split an nnx.Module into (graph_def, params) and create an optax optimizer."""
    import optax

    model_def, params = nnx.split(module)
    opt_def = optax.adam(learning_rate)
    opt_state = opt_def.init(params)
    return SimpleTrainState(
        model_def=model_def,
        params=params,
        opt_state=opt_state,
        opt_def=opt_def,
        step=jnp.int32(0),
    )


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
        self._buf.append(transition)

    def sample(self, batch_size: int) -> TransitionBatch:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        b = min(int(batch_size), self.size)
        idx = self._rng.integers(0, self.size, size=b, endpoint=False)
        samples = [self._buf[i] for i in idx]

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
    pos = jnp.asarray(obs["position"], dtype=jnp.float32)
    vel = jnp.asarray(obs["velocity"], dtype=jnp.float32)
    return jnp.concatenate([pos, vel], axis=-1)


class DSRLLearner(Agent):

    def __init__(self, config: OnlineTrainConfig, env):
        self._config = config
        self._action_space = env.action_space
        self.training_steps = 0

        self.replay = MinimalReplayBuffer(
            capacity=100_000,
            seed=int(getattr(self._config, "seed", 0)),
        )

        # ----- RNG bookkeeping (same pattern as AWR learner) -----
        actor_init_rng, critic_init_rng, self._rng = jax.random.split(
            jax.random.key(config.seed), 3
        )

        # ----- Hyperparams (mirror AWR's _get_*_update_frequency) -----
        rl_cfg = getattr(self._config, "rl", None)
        self._critic_update_frequency = max(
            1, int(getattr(rl_cfg, "critic_update_frequency", 1))
        )
        self._policy_update_frequency = max(
            1, int(getattr(rl_cfg, "policy_update_frequency", 1))
        )
        actor_lr = float(getattr(rl_cfg, "actor_lr", 3e-4))
        critic_lr = float(getattr(rl_cfg, "critic_lr", 3e-4))

        # ----- Instantiate raw modules -----
        obs_dim = 5   # TODO: derive from env observation space
        act_dim = 1   # TODO: derive from env action space
        hidden = 256

        actor_module = Actor(
            obs_dim=obs_dim, act_dim=act_dim, hidden=hidden,
            rngs=nnx.Rngs(params=actor_init_rng),
        )
        critic_module = Critic(
            obs_dim=obs_dim, act_dim=act_dim, hidden=hidden,
            rngs=nnx.Rngs(params=critic_init_rng),
        )

        # ----- Create train states (graph_def + params + optimizer) -----
        #   This mirrors AWR's:
        #     self._state_action_critic_state, self._state_action_critic_state_sharding = ...
        #     self._value_state, self._value_state_sharding = ...
        self._actor_state = _init_train_state(actor_module, learning_rate=actor_lr)
        self._critic_state = _init_train_state(critic_module, learning_rate=critic_lr)

        # Materialise on device and block until ready (same as AWR)
        jax.block_until_ready(self._actor_state.params)
        jax.block_until_ready(self._critic_state.params)

        logging.info(
            f"[DSRL init] "
            f"actor_state: {_pytree_size_mb(self._actor_state.params):.2f} MB, "
            f"critic_state: {_pytree_size_mb(self._critic_state.params):.2f} MB"
        )

        # ----- Keep a live actor module for fast inference -----
        self._actor_apply = nnx.jit(lambda m, x: m(x, training=False))

        # ----- JIT-compile train steps (mirrors AWR's self._q_train_step etc.) -----
        self._train_critic_step = jax.jit(
            functools.partial(train_q_step, config),
            donate_argnums=(1,),
        )
        self._train_actor_step = jax.jit(
            functools.partial(train_actor_step, config),
            donate_argnums=(1,),
        )

        # Episode bookkeeping
        self._episode_storage = [[] for _ in range(self._config.collect.env_num)]
        self._collection_success_episodes = 0

    # ------------------------------------------------------------------
    # Helpers to go between train-state ↔ live module
    # ------------------------------------------------------------------

    def _get_actor_model(self) -> Actor:
        """Merge actor params back into a live module (call sparingly)."""
        return self._actor_state.to_model()

    def _get_critic_model(self) -> Critic:
        """Merge critic params back into a live module (call sparingly)."""
        return self._critic_state.to_model()

    # ------------------------------------------------------------------
    # Action sampling
    # ------------------------------------------------------------------

    def sample_actions(self, observations, **kwargs):
        observations = flatten_dmc_obs(observations)
        obs_jax = jax.tree_util.tree_map(
            lambda x: jnp.asarray(x, dtype=jnp.float32), observations
        )
        actor = self._get_actor_model()
        actions, _logprob = self._actor_apply(actor, obs_jax)
        return actions

    def _generate_actions(
        self, observations: np.ndarray | Dict, **kwargs
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        pass

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def add_data(self, step_data: StepData):
        def get_env_value(vec, env_id):
            return jax.tree.map(lambda x: x[env_id], vec)

        for i in range(1):  # self._config.collect.env_num
            self._episode_storage[i].append(get_env_value(step_data, i))

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
            done = bool(
                np.asarray(term).reshape(-1)[0] or np.asarray(trunc).reshape(-1)[0]
            )

            self.replay.insert({
                "observation": jax.tree_util.tree_map(np.asarray, obs),
                "action": np.asarray(act, dtype=np.float32),
                "next_observation": jax.tree_util.tree_map(np.asarray, next_obs),
                "reward": np.float32(r),
                "done": np.bool_(done),
            })

        self._collection_success_episodes += 1

    def start_data_collection(self, step: int | None = None):
        self._episode_storage = [[] for _ in range(self._config.collect.env_num)]
        self._collection_success_episodes = 0

    def end_data_collection(self, step: int | None = None) -> int:
        collected_episodes = int(self._collection_success_episodes)
        self._episode_storage = [[] for _ in range(self._config.collect.env_num)]
        self._collection_success_episodes = 0
        return collected_episodes

    # ------------------------------------------------------------------
    # Update  (mirrors AWR's update → _update_critics / _update_policy)
    # ------------------------------------------------------------------

    def _update_critic(self, batch) -> dict[str, Any]:
        """Single critic gradient step. Mirrors AWR's _update_critics."""
        critic_rng, self._rng = jax.random.split(self._rng)
        critic_state, critic_info = self._train_critic_step(
            critic_rng,
            self._critic_state,
            self._actor_state,   # actor params needed for target actions
            batch,
        )
        self._critic_state = critic_state
        return {f"critic/{k}": v for k, v in critic_info.items()}

    def _update_actor(self, batch) -> dict[str, Any]:
        """Single actor gradient step. Mirrors AWR's _update_policy."""
        actor_rng, self._rng = jax.random.split(self._rng)
        actor_state, actor_info = self._train_actor_step(
            actor_rng,
            self._actor_state,
            self._critic_state,   # critic params needed for Q-values
            batch,
        )
        self._actor_state = actor_state
        return {f"actor/{k}": v for k, v in actor_info.items()}

    def update(self) -> dict[str, Any]:
        if self.replay.size < 128:
            return {}

        self.training_steps += 1

        # Periodic memory debugging (same as AWR)
        if self.training_steps % 100 == 1:
            logging.info(
                f"[DSRL step={self.training_steps}] "
                f"actor_state: {_pytree_size_mb(self._actor_state.params):.2f} MB, "
                f"critic_state: {_pytree_size_mb(self._critic_state.params):.2f} MB, "
                f"replay_size: {self.replay.size}"
            )

        update_critic = self.training_steps % self._critic_update_frequency == 0
        update_actor = self.training_steps % self._policy_update_frequency == 0

        if not update_critic and not update_actor:
            return {}

        batch = self.replay.sample(batch_size=128)

        critic_info, actor_info = {}, {}
        if update_critic:
            critic_info = self._update_critic(batch)
        if update_actor:
            actor_info = self._update_actor(batch)

        return actor_info | critic_info