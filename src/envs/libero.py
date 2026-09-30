import gymnasium as gym
from gymnasium.utils import seeding
from gymnasium.wrappers import TimeLimit
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
import pathlib
import os
import torch

from src.envs.wrappers import WarmUpOnResetWrapper

# All tasks come from the libero_90 suite; task ids look like "libero_90_<i>".
_TASK_SUITE = "libero_90"
# Gripper-open no-op action executed while the scene settles after a reset.
_WARM_START_ACTION = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])


def _get_task(task_id: str):
    task_suite = benchmark.get_benchmark_dict()[_TASK_SUITE]()
    task_index = int(task_id.split("_")[-1])
    task = task_suite.get_task(task_index)
    bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    return task_suite, task_index, task, bddl_file


class LiberoWrapper(gym.Wrapper):

    def __init__(self, **args):
        self._args = args
        env = OffScreenRenderEnv(**args)
        super().__init__(env)
        self._current_bddl_file = self._args.get("bddl_file_name")
        self.rng, _ = seeding.np_random(0)

    def seed(self, seed=None):
        if seed is not None:
            self.rng, _ = seeding.np_random(seed)

    def reset(self, seed=None, options=None):
        task_id = options["task_id"] if (options is not None and "task_id" in options) else f"{_TASK_SUITE}_0"
        task_suite, task_index, task, bddl_file = _get_task(task_id)
        if bddl_file != self._current_bddl_file:
            self.env.close()
            self._args["bddl_file_name"] = bddl_file
            env = OffScreenRenderEnv(**self._args)
            super().__init__(env)
            self._current_bddl_file = bddl_file
        self.env.reset()
        init_states = get_task_init_states(task_suite, task_index)
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


def make_env_libero(config, tasks, num_devices: int = 4):
    _, _, _, bddl_file = _get_task(tasks[0])
    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": config.collect.env_resolution,
        "camera_widths": config.collect.env_resolution,
    }

    def env_fn(rank: int):
        args = env_args.copy()
        args["render_gpu_device_id"] = rank % num_devices
        env = LiberoWrapper(**args)
        # Warm ups upon reset
        env = WarmUpOnResetWrapper(
            env=env,
            num_steps_wait=config.collect.num_steps_wait,
            warm_up_action=_WARM_START_ACTION,
        )
        env = TimeLimit(env, max_episode_steps=config.collect.max_episode_steps)
        return env

    return env_fn
