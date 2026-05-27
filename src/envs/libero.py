import gymnasium as gym
from gymnasium.utils import seeding
from gymnasium.wrappers import TimeLimit
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
import pathlib
import os
import torch

from src.envs.wrappers import (
    ensure_gymnasium_env,
    WarmUpOnResetWrapper,
)


class LiberoWrapper(gym.Wrapper):

    def __init__(self, **args):
        self._args = args
        env = OffScreenRenderEnv(**args)
        super().__init__(env)
        self.rng, _ = seeding.np_random(0)

    def seed(self, seed=None):
        if seed is not None:
            self.rng, _ = seeding.np_random(seed)

    def reset(self, seed=None, options={}):
        task_id = options["task_id"] if (options is not None and "task_id" in options) else "libero_90_0"
        task_id = int(task_id.split("_")[-1])
        task_suite = benchmark.get_benchmark_dict()["libero_90"]()
        task = task_suite.get_task(task_id)
        self._args["bddl_file_name"] = (
            pathlib.Path(get_libero_path("bddl_files"))
            / task.problem_folder
            / task.bddl_file
        )
        env = OffScreenRenderEnv(**self._args)
        super().__init__(env)
        self.env.reset()
        init_states = get_task_init_states(task_suite, task_id)
        random_index = self.rng.integers(low=0, high=init_states.shape[0])
        init_state = init_states[random_index]
        obs = self.env.set_init_state(init_state)
        self._task_description = task.language
        info = {"task_description": self._task_description}
        return obs, info

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info["task_description"] = self._task_description
        return obs, reward, done, False, info


def get_task_init_states(task_suite, task_id: int):
    init_states_path = os.path.join(
        get_libero_path("init_states"),
        task_suite.tasks[task_id].problem_folder,
        task_suite.tasks[task_id].init_states_file,
    )
    torch.serialization.add_safe_globals(
        [
            np.core.multiarray._reconstruct,  # noqa
            np.ndarray,
            np.dtype,
            np.dtypes.Float64DType,
        ]
    )
    init_states = torch.load(init_states_path)
    return init_states


def get_libero_warm_start_action():
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])


def make_env_libero(config, tasks, num_devices: int = 4):
    benchmark_dict = benchmark.get_benchmark_dict()
    warm_start_action = get_libero_warm_start_action()

    task = tasks[0]
    task_suite_name = "_".join(task.split("_")[:-1]) 
    task_id = int(task.split("_")[-1])
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": config.collect.env_resolution,
        "camera_widths": config.collect.env_resolution,
    }
    # Episode length until truncation. Prefer the explicit config value; fall back
    # to the per-suite default when it is not set.
    max_steps = config.collect.max_episode_steps
    if max_steps is None:
        max_steps = get_max_steps_libero(task_suite_name)

    def env_fn(rank: int):
        args = env_args.copy()
        args["render_gpu_device_id"] = rank % num_devices
        env = LiberoWrapper(**args)
        # Converts gym envs to gymnasium style envs
        env = ensure_gymnasium_env(env)
        # Warm ups upon reset
        env = WarmUpOnResetWrapper(
            env=env,
            num_steps_wait=config.collect.num_steps_wait,
            warm_up_action=warm_start_action,
        )
        # Add timelimit wrapper
        env = TimeLimit(
            env,
            max_episode_steps=max_steps,
        )
        return env

    return env_fn


def get_max_steps_libero(task_suite_name):
    _max_steps_map = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    if task_suite_name not in _max_steps_map:
        raise ValueError(
            f"Unknown task suite name {task_suite_name}. Max steps for known task suites: {_max_steps_map}"
        )
    return _max_steps_map[task_suite_name]