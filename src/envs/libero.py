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


def get_task_init_states(init_root: str = None, problem_folder: str = None, init_states_file: str = None):
    init_states_path = os.path.join(
        init_root,
        problem_folder,
        init_states_file,
    )
    torch.serialization.add_safe_globals(
        [
            np.core.multiarray._reconstruct,  # noqa
            np.ndarray,
            np.dtype,
            np.dtypes.Float64DType,
        ]
    )
    init_states = torch.load(init_states_path, weights_only=False)
    return init_states


def get_libero_warm_start_action():
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])


def make_env_libero(config, tasks, num_devices: int = 4):
    benchmark_dict = benchmark.get_benchmark_dict()
    warm_start_action = get_libero_warm_start_action()

    env_args_multitask = []
    max_steps_multitask = []
    initial_states_multitask = []
    task_descriptions = []

    for task in tasks:
        perturbation = ""
        if any([task.startswith("libero_" + k) for k in ["swap", "object", "position"]]):
            perturbation = "_" + task.split("_")[1]
            task = task.replace(perturbation, "")
        task_suite_name = "_".join(task.split("_")[:-1]) 
        task_id = int(task.split("_")[-1])
        task_suite = benchmark_dict[task_suite_name]()
        task = task_suite.get_task(task_id)
        problem_folder = task_suite_name + perturbation
        task_bddl_file = (
            pathlib.Path(get_libero_path("bddl_files"))
            / problem_folder
            / task.bddl_file
        )
        env_args_multitask.append({
            "bddl_file_name": task_bddl_file,
            "camera_heights": config.collect.env_resolution,
            "camera_widths": config.collect.env_resolution,
        })
        max_steps_multitask.append(get_max_steps_libero(task_suite_name))
        initial_states_multitask.append(
            get_task_init_states(
                get_libero_path("init_states"),
                problem_folder,
                task_suite.tasks[task_id].init_states_file
            )
        )
        task_descriptions.append(task.language)

    def env_fn(rank: int):
        task_index = rank % len(tasks)
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