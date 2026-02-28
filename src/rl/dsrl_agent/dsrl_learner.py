import functools
import logging
import weakref
from typing import Any, Dict

import etils.epath as epath
import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from flax.training.train_state import TrainState

from src.rl.agent import Agent, EnvFn
from src.rl.replay_buffer import ShardedReplayBuffer
from src.rl.types import StepData
from src.envs.wrappers import Pi0ObservationWrapper, QueryFrequencyWrapper
from src.envs.venv import DummyVectorEnv, SubprocVectorEnv
from src.rl.dsrl_agent.dsrl_learner_cfg import DSRLTrainConfig, SACModelConfig
from src.rl.dsrl_agent.dsrl_vector_env import DSRLVectorEnv

# ----------------------- SAC Architectures ----------------------- #

class MLPActor(nnx.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, rngs: nnx.Rngs=None):
        self.fc1 = nnx.Linear(obs_dim, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.mean = nnx.Linear(hidden_dim, action_dim, rngs=rngs)
        self.log_std = nnx.Linear(hidden_dim, action_dim, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.fc1(x))
        x = nnx.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = jnp.clip(log_std, -20, 2)
        return mean, log_std

class MLPCritic(nnx.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, rngs: nnx.Rngs=None):
        self.fc1 = nnx.Linear(obs_dim + action_dim, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.q = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, obs, action):
        x = jnp.concatenate([obs, action], axis=-1)
        x = nnx.relu(self.fc1(x))
        x = nnx.relu(self.fc2(x))
        return self.q(x)

class SACModel(nnx.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, rngs: nnx.Rngs=None):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.actor = MLPActor(obs_dim, action_dim, hidden_dim, rngs=rngs)
        self.critic1 = MLPCritic(obs_dim, action_dim, hidden_dim, rngs=rngs)
        self.critic2 = MLPCritic(obs_dim, action_dim, hidden_dim, rngs=rngs)
        self.target_critic1 = MLPCritic(obs_dim, action_dim, hidden_dim, rngs=rngs)
        self.target_critic2 = MLPCritic(obs_dim, action_dim, hidden_dim, rngs=rngs)

        # Initialize target networks
        for target, source in [
            (self.target_critic1, self.critic1),
            (self.target_critic2, self.critic2)
        ]:
            nnx.update(nnx.state(target), nnx.state(source))

    def sample_actions(self, obs, rng):
        mean, log_std = self.actor(obs)
        std = jnp.exp(log_std)
        normal = jax.random.normal(rng, shape=mean.shape)
        action = mean + std * normal
        return action

# -------------------- Utility Functions -------------------- #

class SACState(TrainState):
    model_def: Any = None
    ema_params: Any = None
    ema_decay: float = 0.995

def init_train_state(
    config: DSRLTrainConfig,
    init_rng: jax.Array,
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
) -> tuple[SACState, Any]:
    
    # Cosine decay schedule with warmup
    lr_schedule = optax.cosine_decay_schedule(
        init_value=config.lr,
        decay_steps=config.decay_steps,
        alpha=0.0
    )
    if config.warmup_steps > 0:
        lr_schedule = optax.join_schedules(
            schedules=[optax.linear_schedule(0.0, config.lr, config.warmup_steps), lr_schedule],
            boundaries=[config.warmup_steps]
        )

    tx = optax.chain(
        optax.clip_by_global_norm(config.clip_gradient_norm),
        optax.adamw(learning_rate=lr_schedule)
    )

    def init(rng: jax.Array) -> SACState:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)
        params = nnx.state(model)
        
        return SACState.create(
            apply_fn=lambda *args, **kwargs: None, # generic
            params=params,
            tx=tx,
            model_def=nnx.graphdef(model),
            ema_params=params if config.ema_decay is not None else None,
            ema_decay=config.ema_decay if config.ema_decay is not None else 0.0
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    
    # For SAC, simply replicate across axis
    replicated_sharding = NamedSharding(mesh, PartitionSpec())
    state_sharding = jax.tree.map(lambda x: replicated_sharding, train_state_shape)

    if resume:
        return train_state_shape, state_sharding

    train_state = jax.jit(
        init,
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng)

    return train_state, state_sharding

def get_env_and_agent_for_dsrl(env_fn, config, task_description, env_class):
    env = dsrl_wrap_env(
        env_fn=env_fn,
        config=config,
        task_description=task_description,
        env_class=env_class,
    )
    agent = DSRLLearner(config)
    return env, agent

def dsrl_wrap_env(env_fn: EnvFn, config: DSRLTrainConfig, task_description: str, env_class: str):
    env_num = config.collect.env_num
    add_states = config.collect.add_states
    obs_prefix_key = config.collect.obs_prefix_key
    replan_steps = config.collect.replan_steps
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
                add_states=add_states,
                pi0_obs_prefix=obs_prefix_key,
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
        DSRLVectorEnv(env_factories)
        if env_num > 1
        else DummyVectorEnv(env_factories)
    )
    # This sets the seed for all environment all at once to be [seed, seed + i, ..., seed + num_envs]
    env.seed(seed)
    # re-use training seed
    return env

# -------------------- Learner -------------------- #

class DSRLLearner(Agent):
    def __init__(self, config: DSRLTrainConfig):
        self._config = config

        if self._config.batch_size % jax.device_count() != 0:
            raise ValueError(
                f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
            )
        jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
        
        self._rng = jax.random.key(self._config.seed)
        init_rng, self._rng = jax.random.split(self._rng, 2)

        # set up sharding
        self._mesh = Mesh(np.array(jax.devices()).reshape(-1), ('data',))
        self._data_sharding = NamedSharding(self._mesh, PartitionSpec('data'))
        self._replicated_sharding = NamedSharding(self._mesh, PartitionSpec())

        # Initialize checkpoint manager using orbax
        options = ocp.CheckpointManagerOptions(max_to_keep=config.keep_period, create=True)
        self._checkpoint_manager = ocp.CheckpointManager(
            epath.Path(config.checkpoint_dir).absolute(), 
            options=options,
            item_names=('train_state',)
        )
        self._resuming = config.resume and self._checkpoint_manager.latest_step() is not None

        self._online_data_buffer = self._get_online_replay_buffer(self._data_sharding)
        self._collection_success_episodes = 0
        self.training_steps = 0

        # Dummy data loader for compatibility with training scripts strictly logging shape interfaces
        class DummyDataLoader:
            def __init__(self, buffer):
                self.buffer = buffer
            def __iter__(self):
                while True:
                    yield self.buffer._dummy_data
        
        self._data_loader = DummyDataLoader(self._online_data_buffer)

        # Initialize train state.
        self._train_state, self._train_state_sharding = init_train_state(
            self._config, init_rng, self._mesh, resume=self._resuming
        )
        jax.block_until_ready(self._train_state)
        logging.info("Initialized SAC train state")
        
        if self._resuming:
            latest_step = self._checkpoint_manager.latest_step()
            restored = self._checkpoint_manager.restore(
                latest_step, args=ocp.args.StandardRestore(self._train_state)
            )
            self._train_state = restored['train_state']
            self.training_steps = latest_step

        # Create temporary episode storage
        self._episode_storage = [[] for _ in range(self._config.collect.env_num)]

    def _get_online_replay_buffer(
        self, data_sharding: jax.sharding.NamedSharding
    ) -> ShardedReplayBuffer:
        train_config = self._config
        batch_size = int(train_config.batch_size)
        max_capacity = max(batch_size, 256, batch_size * 8)
        
        obs_dim = train_config.model.obs_dim
        dummy_obs_dict = {
            "prefix_rep": np.zeros((1, obs_dim), dtype=np.float32),
        }
        
        flat_action_dim = train_config.model.action_horizon * train_config.model.action_dim
        dummy_actions = np.zeros((1, flat_action_dim), dtype=np.float32)
        dummy_next_obs_dict = {
            "prefix_rep": np.zeros((1, obs_dim), dtype=np.float32),
        }
        dummy_rewards = np.zeros((1,), dtype=np.float32)
        dummy_discounts = np.zeros((1,), dtype=np.float32)
        transition_gamma = float(getattr(train_config, "discount", 1.0))

        def _preprocess_insert(episode_data: Dict[str, Any]):
            prefix_rep = np.asarray(episode_data["observation"]["prefix_rep"], dtype=np.float32)
            actions = np.asarray(episode_data["actions"], dtype=np.float32)
            actions_flat = actions.reshape(actions.shape[0], -1)
            next_prefix_rep = np.asarray(episode_data["next_observation"]["prefix_rep"], dtype=np.float32)
            
            insert_batch_size = int(actions.shape[0])
            
            transition_reward = np.asarray(episode_data.get("reward", 0.0), dtype=np.float32)
            if transition_reward.ndim == 0:
                 transition_reward = np.full((insert_batch_size,), float(transition_reward), dtype=np.float32)
                 
            transition_discount = np.asarray(episode_data.get("discount", transition_gamma), dtype=np.float32)
            if transition_discount.ndim == 0:
                 transition_discount = np.full((insert_batch_size,), float(transition_discount), dtype=np.float32)

            return {
                "observation": {"prefix_rep": prefix_rep},
                "actions": actions_flat,
                "next_observation": {"prefix_rep": next_prefix_rep},
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

    def _sample_action(
        self,
        observations: Dict,
        rng: jax.random.PRNGKey,
        train_state: SACState,
        batch_actions: bool = True,
    ) -> np.ndarray:
        params = (
            train_state.ema_params
            if train_state.ema_params is not None
            else train_state.params
        )
        model = nnx.merge(train_state.model_def, params)
        
        obs_prefix = observations.get("observation/prefix_rep")
        if obs_prefix is None:
            obs_prefix = observations.get("prefix_rep")
            
        obs_arr = np.asarray(obs_prefix, dtype=np.float32)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr[np.newaxis, ...]
            
        flat_action = model.sample_actions(obs_arr, rng)
        
        action_horizon = self._config.model.action_horizon
        action_dim = self._config.model.action_dim
        batch_size = obs_arr.shape[0]
        noises = np.asarray(flat_action).reshape(batch_size, action_horizon, action_dim)

        if batch_actions and noises.ndim == 2:
            noises = noises[np.newaxis, ...]
        return noises

    def _generate_actions(
        self, observations: np.ndarray | Dict, **kwargs
    ) -> np.ndarray:
        batch_actions = kwargs.get("batch_actions", False)
        rng, self._rng = jax.random.split(self._rng)
        
        actions = self._sample_action(
            observations=observations,
            rng=rng,
            train_state=self._train_state,
            batch_actions=batch_actions,
        )
        return np.asarray(actions, dtype=np.float32)

    def eval_actions(self, observations: np.ndarray | Dict, **kwargs) -> np.ndarray:
        return self._generate_actions(observations, **kwargs)

    def sample_actions(self, observations: np.ndarray | Dict, **kwargs) -> np.ndarray:
        return self._generate_actions(observations, **kwargs)

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
        
    def add_data(self, step_data: StepData):
        def get_env_value(vec, env_id):
            return jax.tree.map(lambda x: x[env_id], vec)

        for i in range(self._config.collect.env_num):
            self._episode_storage[i].append(get_env_value(step_data, i))

    def save_episode(self, is_success: bool = False, env_index: int = 0, **kwargs):
        if env_index < 0 or env_index >= len(self._episode_storage):
            raise IndexError(
                f"env_index={env_index} is out of range for {len(self._episode_storage)} environments."
            )

        episode_data = self._episode_storage[env_index]
        self._episode_storage[env_index] = []
        
        discount_gamma = float(self._config.discount)

        def process_frame(
            ob: Dict[str, Any],
            *,
            actions: Any,
            next_ob: Dict[str, Any] | None,
            reward: float,
            done: bool,
            discount: float,
        ) -> Dict[str, Any]:
            obs_dict = ob if "prefix_rep" in ob else ob["observation"]
            prefix_rep = obs_dict["prefix_rep"]
            frame = {"prefix_rep": np.asarray(prefix_rep, dtype=np.float32)}

            frame["actions"] = np.asarray(actions, dtype=np.float32)
            next_state = frame["prefix_rep"]
            if next_ob is not None:
                next_obs_dict = next_ob if "prefix_rep" in next_ob else next_ob["observation"]
                if "prefix_rep" in next_obs_dict:
                    next_state = next_obs_dict["prefix_rep"]
                
            frame["next_observation"] = {"prefix_rep": np.asarray(next_state, dtype=np.float32)}
            frame["reward"] = np.float32(reward)
            frame["done"] = np.bool_(done)
            frame["discount"] = np.float32(discount)
            return frame

        def _stack_transitions(frames):
            return jax.tree_util.tree_map(lambda *xs: np.stack(xs, axis=0), *frames)

        transitions = []
        if self._config.collect.add_per_step_data:
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
                                ep_obs.get("action", ep_obs.get("actions"))[step], dtype=np.float32
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
            episode_batch_formatted = {
                "observation": {"prefix_rep": episode_batch["prefix_rep"]},
                "actions": episode_batch["actions"],
                "next_observation": {"prefix_rep": episode_batch["next_observation"]["prefix_rep"]},
                "reward": episode_batch["reward"],
                "discount": episode_batch["discount"]
            }
            self._online_data_buffer.insert(episode_batch_formatted)
        self._collection_success_episodes += 1

    def start_data_collection(self, step: int | None = None):
        self._episode_storage = [[] for _ in range(self._config.collect.env_num)]
        self._collection_success_episodes = 0

    def end_data_collection(self, step: int | None = None) -> int:
        collected_episodes = int(self._collection_success_episodes)
        self._episode_storage = [[] for _ in range(self._config.collect.env_num)]
        self._collection_success_episodes = 0
        return collected_episodes

    def update(self):
        self.training_steps += 1
        
        # Determine if we have enough data to train
        if self._online_data_buffer.size < self._online_data_buffer.batch_size:
            return {"actor_loss": 0.0, "critic_loss": 0.0}

        batch = self._online_data_buffer.sample()
        
        # train_rng, self._rng = jax.random.split(self._rng)
        # train_state = self._train_state
        # with Mesh(self._mesh.devices, ('data',)):
        #     train_state, info = self._train_step(train_rng, train_state, batch)
        # self._train_state = train_state
        
        # TODO: Implement actual SAC train_step integration
        info = {"actor_loss": 0.0, "critic_loss": 0.0}
        
        return info

    def save_checkpoint(self, step: int | None = None):
        if step is None:
            step = self.training_steps
        
        self._checkpoint_manager.save(
            step, 
            args=ocp.args.StandardSave(self._train_state)
        )
        self._checkpoint_manager.wait_until_finished()
