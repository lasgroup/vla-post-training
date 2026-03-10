from gymnasium.wrappers import TimeLimit
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import logging
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


def _get_physical_cuda_device_ids(default: int = 0) -> list[int]:
    """Return the physical GPU indices from CUDA_VISIBLE_DEVICES.

    EGL rendering requires physical device IDs, not the 0-indexed
    remapped IDs that CUDA exposes after CUDA_VISIBLE_DEVICES filtering.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return [default]
    ids = []
    for dev in visible.split(","):
        dev = dev.strip()
        if dev and dev != "-1":
            try:
                ids.append(int(dev))
            except ValueError:
                # GPU-UUID form – fall back to default
                pass
    return ids if ids else [default]


def _infer_num_visible_cuda_devices(default: int = 1) -> int:
    return max(default, len(_get_physical_cuda_device_ids()))


def make_env_libero(config, num_devices: int | None = None):
    if num_devices is None:
        num_devices = _infer_num_visible_cuda_devices(default=1)
    logging.info("LIBERO render device pool size: %d", int(num_devices))
    assert len(config.collect.tasks) == 1, "Only single-task collection is supported."
    task_suite_name = "_".join(config.collect.tasks[0].split("_")[:-1])
    task_id = int(config.collect.tasks[0].split("_")[-1])
    max_steps = get_max_steps_libero(config.collect.tasks[0])
    benchmark_dict = benchmark.get_benchmark_dict()
    warm_start_action = get_libero_warm_start_action()

    env_args_multitask = []
    max_steps_multitask = []
    initial_states_multitask = []
    task_descriptions = []

    for task in config.collect.tasks:
        task_suite_name = "_".join(task.split("_")[:-1]) 
        task_id = int(task.split("_")[-1])
        task_suite = benchmark_dict[task_suite_name]()
        task = task_suite.get_task(task_id)
        task_bddl_file = (
            pathlib.Path(get_libero_path("bddl_files"))
            / task.problem_folder
            / task.bddl_file
        )
        env_args_multitask.append({
            "bddl_file_name": task_bddl_file,
            "camera_heights": config.collect.env_resolution,
            "camera_widths": config.collect.env_resolution,
        })
        max_steps_multitask.append(get_max_steps_libero(task_suite_name))
        initial_states_multitask.append(get_task_init_states(task_suite, task_id))
        task_descriptions.append(task.language)

    physical_gpu_ids = _get_physical_cuda_device_ids()

    def env_fn(rank: int):
        args = env_args.copy()
        args["render_gpu_device_id"] = physical_gpu_ids[rank % len(physical_gpu_ids)]
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