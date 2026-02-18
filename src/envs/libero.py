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
    SetInitialStateWrapper,
)


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


def make_env_libero(config, num_devices: int = 4):
    env_args_multitask = []
    max_steps_multitask = []
    initial_states_multitask = []
    task_descriptions = []

    for task in config.collect.tasks:
        task_suite_name = "_".join(task.split("_")[:-1])
        task_id = int(task.split("_")[-1])
        max_steps = get_max_steps_libero(task)
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[task_suite_name]()
        task = task_suite.get_task(task_id)
        initial_states = get_task_init_states(task_suite, task_id)
        warm_start_action = get_libero_warm_start_action()
        task_description = task.language
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

        env_args_multitask.append(env_args)
        max_steps_multitask.append(max_steps)
        initial_states_multitask.append(initial_states)
        task_descriptions.append(task_description)

    def env_fn(rank: int):
        task_index = rank % len(config.collect.tasks)
        args = env_args_multitask[task_index].copy()
        args["render_gpu_device_id"] = rank % num_devices
        env = OffScreenRenderEnv(**args)
        # Converts gym envs to gymnasium style envs
        env = ensure_gymnasium_env(env)
        # Sets initial states for the environment
        env = SetInitialStateWrapper(env, initial_states=initial_states_multitask[task_index])
        # Warm ups upon reset
        env = WarmUpOnResetWrapper(
            env=env,
            num_steps_wait=config.collect.num_steps_wait,
            warm_up_action=warm_start_action,
        )
        # Add timelimit wrapper
        env = TimeLimit(
            env,
            max_episode_steps=max_steps_multitask[task_index],
        )
        return env

    return env_fn, task_descriptions


def get_max_steps_libero(task_name):
    _max_steps_map = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    task_name = "_".join(task_name.split("_")[:-1])
    if task_name not in _max_steps_map:
        raise ValueError(
            f"Unknown task name {task_name}. Max steps for known tasks: {_max_steps_map}"
        )
    return _max_steps_map[task_name]
