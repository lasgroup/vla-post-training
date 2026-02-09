from typing import Any
import gymnasium as gym
import jax
import numpy as np
import math
import PIL
import logging


class GymnasiumEnvAdapter(gym.Env):
    """Wraps non-Gymnasium envs to satisfy gymnasium.Env checks."""

    def __init__(self, env):
        self.env = env
        self.metadata = getattr(env, "metadata", {})
        self.reward_range = getattr(env, "reward_range", (-float("inf"), float("inf")))
        self.spec = getattr(env, "spec", None)

    @property
    def observation_space(self):
        return getattr(self.env, "observation_space", None)

    @property
    def action_space(self):
        return getattr(self.env, "action_space", None)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None and hasattr(self.env, "seed"):
            self.env.seed(seed)
        out = self.env.reset()
        if isinstance(out, (tuple, list)) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}
        return obs, info

    def step(self, action):
        out = self.env.step(action)
        if isinstance(out, (tuple, list)) and len(out) == 5:
            return out
        if isinstance(out, (tuple, list)) and len(out) == 4:
            obs, reward, done, info = out
            return obs, reward, bool(done), False, info
        raise ValueError("Env.step returned unsupported format.")

    def render(self, *args, **kwargs):
        if hasattr(self.env, "render"):
            return self.env.render(*args, **kwargs)
        return None

    def close(self):
        if hasattr(self.env, "close"):
            return self.env.close()
        return None


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


def obs_to_img(obs, variant):
    """
    Convert raw observation to resized image for DSRL actor/critic
    """
    if variant.env == "libero":
        curr_image = obs["agentview_image"][::-1, ::-1]
    elif variant.env == "aloha_cube":
        curr_image = obs["pixels"]["top"]
    else:
        raise NotImplementedError()
    if variant.resize_image > 0:
        curr_image = np.array(
            PIL.Image.fromarray(curr_image).resize(
                (variant.resize_image, variant.resize_image)
            )
        )
    return curr_image


def obs_to_pi_zero_input(obs, variant, *, include_prompt: bool = True):
    if variant.env == "libero":
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
                )
            ),
        }
        if include_prompt:
            obs_pi_zero["prompt"] = str(variant.task_description)
    elif variant.env == "aloha_cube":
        img = np.ascontiguousarray(obs["pixels"]["top"])
        obs_pi_zero = {
            "state": obs["agent_pos"],
            "images": {"cam_high": np.transpose(img, (2, 0, 1))},
        }
    else:
        raise NotImplementedError()
    return obs_pi_zero


def _stack_pi0_obs(pi0_obs_list):
    return jax.tree_util.tree_map(lambda *xs: np.stack(xs, axis=0), *pi0_obs_list)


def obs_to_qpos(obs, variant):
    if variant.env == "libero":
        qpos = np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        )
    elif variant.env == "aloha_cube":
        qpos = obs["agent_pos"]
    else:
        raise NotImplementedError()
    return qpos


class QueryFrequencyWrapper(gym.Wrapper):
    def __init__(self, base_env: gym.Env, query_frequency: int, discount: float = 0.99):
        super().__init__(base_env)
        self._query_frequency = query_frequency
        self._discount = discount

    @property
    def action_space(self):
        # We define a function to expand a single space leaf (e.g., a Box)
        def expand_space(space):
            if isinstance(space, gym.spaces.Box):
                # Expand Box: Shape becomes (query_frequency, *original_shape)
                # We repeat the low/high bounds to match the new shape
                return gym.spaces.Box(
                    low=np.repeat(space.low[None, ...], self._query_frequency, axis=0),
                    high=np.repeat(
                        space.high[None, ...], self._query_frequency, axis=0
                    ),
                    dtype=space.dtype,
                )
            elif isinstance(space, gym.spaces.Discrete):
                # Expand Discrete: Becomes MultiDiscrete with 'query_frequency' dimensions
                return gym.spaces.MultiDiscrete([space.n] * self._query_frequency)
            else:
                raise NotImplementedError(
                    f"Space type {type(space)} not supported for expansion."
                )

        # Apply this expansion to the entire structure of the action space
        return jax.tree_util.tree_map(expand_space, self.env.action_space)

    def step(self, action):
        """
        Takes the stacked action (chunk_size, action_dim), where chunk_size >= query_frequency
        unrolls it, and steps the environment query_frequency times.
        """
        total_reward = 0.0
        terminated = False
        truncated = False
        last_obs = None
        info = {}

        # We assume action is indexable along the first dimension (the query frequency)
        # In a JAX context, you might need to convert this to a numpy array first if it's a DeviceArray
        for i in range(self._query_frequency):
            # Extract the sub-action for this specific step
            # tree_map handles nested actions (dict/tuple) by slicing the i-th element of every leaf
            sub_action = jax.tree_util.tree_map(lambda x: x[i], action)

            step_out = self.env.step(sub_action)
            # Support both old Gym API (obs, reward, done, info) and Gymnasium API
            # (obs, reward, terminated, truncated, info).
            assert len(step_out) == 5, (
                "QueryFrequencyWrapper only works with gymnasium environments. Please check "
                "the base environment passed"
            )
            obs, reward, terminated, truncated, info = step_out

            total_reward += reward * (self._discount**i)
            last_obs = obs

            # If the episode ends mid-query, we stop early
            if terminated or truncated:
                break

        return last_obs, total_reward, terminated, truncated, info


class Pi0ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, variant):
        super().__init__(env)
        self._variant = variant
        logging.info(f"\nTask: {self._variant.task_description}")

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
            # Heuristic: if it looks like an image, assume 0-255
            if "image" in key or "pixels" in key:
                low, high = 0, 255
            else:
                low, high = -np.inf, np.inf
            spaces[key] = gym.spaces.Box(
                low=low, high=high, shape=val.shape, dtype=val.dtype
            )

        # Assign the final Dict space to self.observation_space
        self._observation_space = gym.spaces.Dict(spaces)

    @property
    def observation_space(self):
        return self._observation_space

    def observation(self, observation):
        curr_image = obs_to_img(observation, self._variant)

        qpos = obs_to_qpos(observation, self._variant)

        if self._variant.add_states:
            obs_dict = {
                "pixels": curr_image[np.newaxis, ..., np.newaxis],
                "state": qpos[np.newaxis, ..., np.newaxis],
            }
        else:
            obs_dict = {
                "pixels": curr_image[np.newaxis, ..., np.newaxis],
            }

        # Do not inject prompt into env observations; prompt should be provided via default_prompt.
        obs_pi_zero = obs_to_pi_zero_input(
            observation, self._variant, include_prompt=False
        )
        obs_pi_zero = {f"pi0/{key}": val for key, val in obs_pi_zero.items()}
        obs_dict = obs_dict | obs_pi_zero
        return obs_dict


class WarmUpOnResetWrapper(gym.Wrapper):
    def __init__(
        self, env, num_steps_wait: int = 10, warm_up_action: np.ndarray | None = None
    ):
        super().__init__(env)
        self._num_steps_wait = num_steps_wait
        if warm_up_action is None:
            warm_up_action = self.env.action_space.sample()
        self._warm_up_action = warm_up_action

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        t = 0
        obs, info = None, {}
        if t < self._num_steps_wait:
            obs, reward, terminate, truncate, info = self.env.step(self._warm_up_action)
            t += 1
        return obs, info
