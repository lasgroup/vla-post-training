from typing import Any, Dict, List, Callable
import gymnasium as gym
import jax
import logging
import math
import numpy as np

from src.rl.prefix_embedding import (
    PREFIX_EMBEDDING_NAME,
    unpack_action_and_prefix,
)


class GymnasiumEnvAdapter(gym.Env):
    """Wraps non-Gymnasium envs to satisfy gymnasium.Env checks."""

    def __init__(self, env):
        self.env = env

    @property
    def observation_space(self):
        return getattr(self.env, "observation_space", None)

    @property
    def action_space(self):
        if hasattr(self.env, "action_space"):
            return self.env.action_space
        elif hasattr(self.env, "env") and hasattr(self.env.env, "action_spec"):
            return gym.spaces.Box(
                low=self.env.env.action_spec[0],
                high=self.env.env.action_spec[1],
            )
        else:
            return None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.env.seed(seed)
        obs = self.env.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, bool(done), False, info

    def render(self, *args, **kwargs):
        if hasattr(self.env, "render"):
            return self.env.render(*args, **kwargs)
        return None

    def close(self):
        if hasattr(self.env, "close"):
            return self.env.close()
        return None

    def __getattr__(self, name):
        """Fallback to the wrapped environment for any unknown attributes."""
        return getattr(self.env, name, None)


def ensure_gymnasium_env(env):
    if isinstance(env, gym.Env):
        return env
    return GymnasiumEnvAdapter(env)


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def obs_to_img(obs, env_class: str = "libero"):
    """Convert raw observation to resized image for DSRL actor/critic"""
    if env_class == "libero":
        curr_image = obs["agentview_image"][::-1, ::-1]
    else:
        raise NotImplementedError()
    return curr_image


def obs_to_pi_zero_input(
    obs, env_class: str, task_description: str, *, include_prompt: bool = True
):
    if env_class == "libero":
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        obs_pi_zero = {
            "image": img,
            "wrist_image": wrist_img,
            "state": np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ),
                dtype=np.float32,
            ),
        }
        if include_prompt:
            obs_pi_zero["prompt"] = np.asarray(str(task_description))
    else:
        raise NotImplementedError()
    return obs_pi_zero


def obs_to_qpos(obs, env_class):
    if env_class == "libero":
        qpos = np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        )
    else:
        raise NotImplementedError()
    return qpos


class QueryFrequencyWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        query_frequency: int,
        discount: float = 0.99,
        store_full_transitions: bool = False,
        pre_step_filter: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    ):
        super().__init__(env)
        self._query_frequency = query_frequency
        self._discount = discount
        self._store_full_transitions = store_full_transitions
        self._pre_step_filter = pre_step_filter

    @property
    def return_full_transitions(self) -> bool:
        return self._store_full_transitions

    @property
    def expand_space(self, space):
        # We define a function to expand a single space leaf (e.g., a Box)
        if isinstance(space, gym.spaces.Box):
            # Expand Box: Shape becomes (query_frequency, *original_shape)
            # We repeat the low/high bounds to match the new shape
            return gym.spaces.Box(
                low=np.repeat(space.low[None, ...], self._query_frequency, axis=0),
                high=np.repeat(space.high[None, ...], self._query_frequency, axis=0),
                dtype=space.dtype,
            )
        elif isinstance(space, gym.spaces.Discrete):
            # Expand Discrete: Becomes MultiDiscrete with 'query_frequency' dimensions
            return gym.spaces.MultiDiscrete([space.n] * self._query_frequency)
        else:
            raise NotImplementedError(
                f"Space type {type(space)} not supported for expansion."
            )

    @property
    def action_space(self):
        return jax.tree_util.tree_map(self.expand_space, self.env.action_space)

    @property
    def observation_space(self):
        if self._store_full_transitions:
            obs_space = jax.tree_util.tree_map(
                self.expand_space, self.env.observation_space
            )
        else:
            obs_space = self.env.observation_space
        return obs_space

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        # 1. Reset the underlying environment
        obs, info = self.env.reset(seed=seed, options=options)

        # 3. Handle the observation based on store_full_transitions
        if self._store_full_transitions:
            # If we store full transitions, the observation space expects
            # a sequence of shape (query_frequency, ...).
            # We tile the initial observation to fill the buffer.
            obs = jax.tree_util.tree_map(
                lambda x: np.repeat(x[None, ...], self._query_frequency, axis=0),
                obs,
            )

        return obs, info

    def step(self, action):
        """
        Takes the stacked action (chunk_size, action_dim), where chunk_size >= query_frequency
        unrolls it, and steps the environment query_frequency times.
        """
        data = []

        # We assume action is indexable along the first dimension (the query frequency)
        # In a JAX context, you might need to convert this to a numpy array first if it's a DeviceArray
        for i in range(self._query_frequency):
            # Extract the sub-action for this specific step
            # tree_map handles nested actions (dict/tuple) by slicing the i-th element of every leaf
            sub_action = jax.tree_util.tree_map(lambda x: x[i], action)
            sub_action = self._pre_step_filter(sub_action)
            obs, reward, terminated, truncated, info = self.env.step(sub_action)
            data.append(
                {
                    "observation": obs,
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "info": info,
                }
            )

            # If the episode ends mid-query, we stop early
            # TODO: This is hacky
            if terminated or truncated:
                # Repeat the data for padding.
                for _ in range(i + 1, self._query_frequency):
                    data.append(
                        {
                            "observation": obs,
                            "reward": 0.0,
                            "terminated": terminated,
                            "truncated": truncated,
                            "info": info,
                        }
                    )
                break

        return self.step_response(data)

    def step_response(self, data: List[Dict]):
        stacked = jax.tree.map(lambda *xs: np.stack(xs), *data)
        if self._store_full_transitions:
            # Returns the observation tree where every leaf has shape (query_freq, ...)
            return (
                stacked["observation"],
                stacked["reward"],
                stacked["terminated"],
                stacked["truncated"],
                stacked["info"],
            )
        else:
            # 1. Discounted sum of rewards: sum(r_t * gamma^t)
            rewards = stacked["reward"]
            discounts = self._discount ** np.arange(len(rewards))
            discounted_reward = np.sum(rewards * discounts)

            # 2. Extract only the final state values
            # We use [-1] to get the state at the end of the query sequence
            last_obs = jax.tree.map(lambda x: x[-1], stacked["observation"])
            last_term = bool(stacked["terminated"][-1])
            last_trunc = bool(stacked["truncated"][-1])
            last_info = jax.tree.map(lambda x: x[-1], stacked["info"])

            return last_obs, discounted_reward, last_term, last_trunc, last_info


class PrefixEmbeddingVectorEnvWrapper(gym.Env):
    """Adds prefix embeddings to vectorized observations while stripping them from actions."""

    def __init__(
        self,
        env,
        *,
        add_per_step_data: bool,
        prefix_embedding_name: str = PREFIX_EMBEDDING_NAME,
    ):
        super().__init__()
        self.env = env
        self._add_per_step_data = add_per_step_data
        self._prefix_embedding_name = prefix_embedding_name
        self._prefix_tail_shape: tuple[int, ...] | None = None
        self.metadata = getattr(env, "metadata", {})
        self.render_mode = getattr(env, "render_mode", None)

    def _unwrap_vector_space(self, space):
        # BaseVectorEnv exposes Gym reserved attrs as lists (one per worker).
        if isinstance(space, list):
            if not space:
                raise ValueError("Wrapped vector env has an empty space list.")
            return space[0]
        return space

    @property
    def action_space(self):
        return self._unwrap_vector_space(getattr(self.env, "action_space", None))

    def _prefix_leading_shape_from_obs_space(
        self, obs_space: gym.spaces.Dict
    ) -> tuple[int, ...]:
        if not self._add_per_step_data:
            return ()
        for sub_space in obs_space.spaces.values():
            if isinstance(sub_space, gym.spaces.Box) and len(sub_space.shape) >= 1:
                return (int(sub_space.shape[0]),)
        return (1,)

    @property
    def observation_space(self):
        base_space = self._unwrap_vector_space(
            getattr(self.env, "observation_space", None)
        )
        if not isinstance(base_space, gym.spaces.Dict):
            return base_space

        spaces = dict(base_space.spaces)
        tail = (
            self._prefix_tail_shape if self._prefix_tail_shape is not None else (1, 1)
        )
        spaces[self._prefix_embedding_name] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self._prefix_leading_shape_from_obs_space(base_space) + tail,
            dtype=np.float32,
        )
        return gym.spaces.Dict(spaces)

    def _first_array_leaf(self, observation: Dict[str, Any]) -> np.ndarray:
        for leaf in jax.tree_util.tree_leaves(observation):
            if hasattr(leaf, "shape"):
                return np.asarray(leaf)
        raise ValueError("Cannot infer batch shape from an empty observation tree.")

    def _batch_leading_shape(self, observation: Dict[str, Any]) -> tuple[int, ...]:
        ref = self._first_array_leaf(observation)
        if self._add_per_step_data:
            if ref.ndim < 2:
                raise ValueError(
                    "Expected at least 2 dimensions for per-step observations."
                )
            return int(ref.shape[0]), int(ref.shape[1])
        if ref.ndim < 1:
            raise ValueError("Expected at least 1 dimension for batched observations.")
        return (int(ref.shape[0]),)

    def _placeholder_prefix(self, observation: Dict[str, Any]) -> np.ndarray:
        leading = self._batch_leading_shape(observation)
        tail = (
            self._prefix_tail_shape if self._prefix_tail_shape is not None else (1, 1)
        )
        return np.zeros(leading + tail, dtype=np.float32)

    def _format_prefix_for_observation(
        self,
        prefix_embedding: Any,
        observation: Dict[str, Any],
    ) -> np.ndarray:
        prefix = np.asarray(prefix_embedding, dtype=np.float32)
        leading = self._batch_leading_shape(observation)

        if self._add_per_step_data:
            batch_size, horizon = leading
            if prefix.ndim == 2:
                prefix = np.broadcast_to(
                    prefix[None, ...], (batch_size,) + prefix.shape
                )
            if prefix.ndim == 3 and prefix.shape[0] == batch_size:
                prefix = np.repeat(prefix[:, None, ...], horizon, axis=1)
            elif prefix.ndim == 4 and prefix.shape[:2] == (batch_size, horizon):
                pass
            else:
                raise ValueError(
                    "Prefix embedding has incompatible shape for per-step observations: "
                    f"{prefix.shape}."
                )
        else:
            (batch_size,) = leading
            if prefix.ndim == 2:
                prefix = np.broadcast_to(
                    prefix[None, ...], (batch_size,) + prefix.shape
                )
            elif prefix.ndim == 3 and prefix.shape[0] == batch_size:
                pass
            else:
                raise ValueError(
                    "Prefix embedding has incompatible shape for batched observations: "
                    f"{prefix.shape}."
                )

        self._prefix_tail_shape = tuple(int(x) for x in prefix.shape[len(leading) :])
        return prefix.astype(np.float32, copy=False)

    def _inject_prefix(
        self, observation: Dict[str, Any], prefix_embedding: np.ndarray
    ) -> Dict[str, Any]:
        updated = dict(observation)
        updated[self._prefix_embedding_name] = prefix_embedding
        return updated

    def reset(self, *args, **kwargs):
        result = self.env.reset(*args, **kwargs)
        if isinstance(result, (tuple, list)) and len(result) == 2:
            observation, info = result
            observation = self._inject_prefix(
                observation, self._placeholder_prefix(observation)
            )
            return observation, info
        observation = self._inject_prefix(result, self._placeholder_prefix(result))
        return observation

    def step(self, action, *args, **kwargs):
        env_action, prefix_embedding = unpack_action_and_prefix(action)
        observation, reward, terminated, truncated, info = self.env.step(
            env_action, *args, **kwargs
        )
        if prefix_embedding is None:
            prefix = self._placeholder_prefix(observation)
        else:
            prefix = self._format_prefix_for_observation(prefix_embedding, observation)
        observation = self._inject_prefix(observation, prefix)
        return observation, reward, terminated, truncated, info

    def __len__(self) -> int:
        return len(self.env)

    def render(self, *args, **kwargs):
        if hasattr(self.env, "render"):
            return self.env.render(*args, **kwargs)
        return None

    def close(self):
        if hasattr(self.env, "close"):
            return self.env.close()
        return None

    def __getattr__(self, name: str):
        return getattr(self.env, name)


class Pi0ObservationWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        env_class: str,
        task_description: str,
        add_states: bool = True,
        include_prompt_in_obs: bool = False,
        pi0_obs_prefix: str = "pi0",
    ):
        super().__init__(env)
        self.task_description = task_description
        self._env_class = env_class
        self._add_states = add_states
        self._include_prompt_in_obs = include_prompt_in_obs
        self._pi0_obs_prefix = pi0_obs_prefix
        logging.info(f"\nTask: {self.task_description}")

        # produced by the helper functions (obs_to_img, etc.)
        if getattr(env, "observation_space", None) is not None and hasattr(
            env.observation_space, "sample"
        ):
            dummy_obs = env.observation_space.sample()
        else:
            reset_out = env.reset()
            dummy_obs = (
                reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out
            )
        final_obs = self.observation(dummy_obs)
        spaces = {}
        for key, val in final_obs.items():
            if "prompt" in key:
                spaces[key] = gym.spaces.Text(max_length=256_000)
                continue
            if "image" in key or "pixels" in key:
                low, high = 0, 255
            else:
                low, high = -np.inf, np.inf
            spaces[key] = gym.spaces.Box(
                low=low, high=high, shape=val.shape, dtype=val.dtype
            )

        self.observation_space = gym.spaces.Dict(spaces)

    def observation(self, observation):
        curr_image = obs_to_img(observation, env_class=self._env_class)
        qpos = obs_to_qpos(observation, env_class=self._env_class)
        obs_dict = {"pixels": curr_image[np.newaxis, ..., np.newaxis]}
        if self._add_states:
            obs_dict["state"] = qpos[np.newaxis, ..., np.newaxis]

        # Do not inject prompt into env observations; prompt should be provided via default_prompt.
        obs_pi_zero = obs_to_pi_zero_input(
            observation,
            env_class=self._env_class,
            task_description=self.task_description,
            include_prompt=self._include_prompt_in_obs,
        )
        obs_pi_zero = {
            f"{self._pi0_obs_prefix}/{key}": val for key, val in obs_pi_zero.items()
        }
        obs_dict = obs_dict | obs_pi_zero
        return obs_dict


class WarmUpOnResetWrapper(gym.Wrapper):
    def __init__(self, env, warm_up_action: np.ndarray, num_steps_wait: int = 10):
        super().__init__(env)
        self._num_steps_wait = num_steps_wait
        self._warm_up_action = warm_up_action

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        for _ in range(self._num_steps_wait):
            obs, _, _, _, info = self.env.step(self._warm_up_action)
        return obs, info


class SetInitialStateWrapper(gym.Wrapper):
    def __init__(self, env, initial_states: np.ndarray):
        super().__init__(env)
        self._init_states = initial_states

    def _set_init_state(self):
        assert hasattr(
            self.env, "set_init_state"
        ), "The environment must have a set_init_state method to use SetInitialStateWrapper"
        random_index = self.np_random.integers(low=0, high=self._init_states.shape[0])
        init_state = self._init_states[random_index]
        return self.env.set_init_state(init_state)

    def seed(self, seed):
        self.env.seed(seed)
        self.np_random = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = self._set_init_state()
        return obs, info
