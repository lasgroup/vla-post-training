from typing import Any, Dict, List
import gymnasium as gym
import jax
import logging
import math
import numpy as np


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


def obs_to_pi_zero_input(
    obs,
    env_class: str,
):
    if env_class == "libero":
        obs_pi_zero = {
            "observation/image": np.ascontiguousarray(
                obs["agentview_image"][::-1, ::-1]
            ),
            "observation/wrist_image": np.ascontiguousarray(
                obs["robot0_eye_in_hand_image"][::-1, ::-1]
            ),
            "observation/state": np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ),
                dtype=np.float32,
            ),
        }
    elif env_class == "molmo":
        gripper_obs_norm: float = 0.824033    
        qpos = obs["qpos"]
        gripper = np.asarray(qpos["gripper"], dtype=np.float32)
        grip = np.clip(float(gripper[0]) / gripper_obs_norm, 0.0, 1.0)
        obs_pi_zero = {
            "observation/exterior_image_1_left": obs["exo_camera_1"],
            "observation/wrist_image_left": obs["wrist_camera"],
            "observation/joint_position": np.asarray(qpos["arm"][:7], dtype=np.float32),
            "observation/gripper_position": np.asarray([grip], dtype=np.float32),
        }
    else:
        raise NotImplementedError
    return obs_pi_zero


class QueryFrequencyWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        query_frequency: int,
    ):
        super().__init__(env)
        self._query_frequency = query_frequency

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
        obs_space = jax.tree_util.tree_map(
            self.expand_space, self.env.observation_space
        )
        return obs_space

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        # 1. Reset the underlying environment
        obs, info = self.env.reset(seed=seed, options=options)

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
        # Returns the dictionary where every leaf has shape (query_freq, ...)
        return (
            stacked["observation"],
            stacked["reward"],
            stacked["terminated"],
            stacked["truncated"],
            stacked["info"],
        )


class PrefixEmbeddingVectorEnvWrapper(QueryFrequencyWrapper):
    """Query wrapper that ignores prefix payload when stepping the underlying env."""

    def step(self, action):
        env_action, _ = action
        return super().step(env_action)


class TimeToSuccessAsRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env=env)

    def step(self, action):
        obs, _, terminate, truncate, info = self.env.step(action)
        time_to_success_reward = 0.0 if terminate else -1.0
        return obs, time_to_success_reward, terminate, truncate, info


class Pi0ObservationWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        env_class: str,
    ):
        super().__init__(env)
        self._env_class = env_class

        dummy_obs, _ = env.reset()
        final_obs = self.observation(dummy_obs)
        spaces = {}
        for key, val in final_obs.items():
            if "prompt" in key:
                spaces[key] = gym.spaces.Text(max_length=256_000)
                continue
            if "image" in key:
                low, high = 0, 255
            else:
                low, high = -np.inf, np.inf
            spaces[key] = gym.spaces.Box(
                low=low, high=high, shape=val.shape, dtype=val.dtype
            )

        self.observation_space = gym.spaces.Dict(spaces)

    def observation(self, observation):
        return obs_to_pi_zero_input(
            observation,
            env_class=self._env_class,
        )


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
