from gymnasium.wrappers import TimeLimit
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
import pathlib
import os
import torch

from src.envs.venv import SubprocVectorEnv
from src.envs.wrappers import ensure_gymnasium_env, WarmUpOnResetWrapper, \
    SetInitialStateWrapper, Pi0ObservationWrapper, QueryFrequencyWrapper


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


def make_env_libero(config, discount: float = 0.99):
    env_args_multitask = []
    for task in config.collect.tasks:
        if not task.startswith("libero_"):
            raise ValueError(f"Unexpected task {task} in config.collect.tasks. Expected tasks to start with 'libero_'.")
        task_suite_name = "_".join(task.split("_")[:-1])
        task_id = int(task.split("_")[-1])
        max_steps = get_max_steps_libero(task)
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[task_suite_name]()
        task = task_suite.get_task(task_id)
        initial_states = get_task_init_states(task_suite, task_id)
        warm_start_action = get_libero_warm_start_action()
        task_description = task.language
        task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": config.collect.env_resolution,
            "camera_widths": config.collect.env_resolution,
        }
        env_args_multitask.append(env_args)

    env_factories = []
    num_envs_multitask = len(config.collect.tasks) 
    for i in range(num_envs_multitask):
        def _make_env(rank=i):
            args = env_args_multitask[rank].copy()
            args["render_gpu_device_id"] = rank % 4
            # Create Libero environment
            base_env = OffScreenRenderEnv(**args)
            # Converts gym envs to gymnasium style envs
            base_env = ensure_gymnasium_env(base_env)
            # Sets initial states for the environment
            base_env = SetInitialStateWrapper(base_env, initial_states=initial_states)
            # Add Pi related obs to the environment
            base_env = Pi0ObservationWrapper(
                env=base_env,
                env_class="libero",
                task_description=task_description,
                add_states=config.collect.add_states
            )
            # Warm ups upon reset
            base_env = WarmUpOnResetWrapper(
                env=base_env,
                num_steps_wait=config.collect.num_steps_wait,
                warm_up_action=warm_start_action,
            )
            # Add timelimit wrapper
            base_env = TimeLimit(
                base_env,
                max_episode_steps=max_steps,
            )
            # Add query frequency wrapper to rollout action chunks
            base_env = QueryFrequencyWrapper(
                env=base_env,
                query_frequency=config.collect.replan_steps,
                discount=discount,
                store_full_transitions=config.collect.add_per_step_data,
                post_step_filter=lambda x: np.where(np.abs(x) < 0.0011, 0.0, x),
            )
            return base_env

        env_factories.append(_make_env)
        
    env = SubprocVectorEnv(env_factories)
    # This sets the seed for all environment all at once to be [seed, seed + i, ..., seed + num_envs]
    env.seed(config.seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    # re-use training seed
    return env, task_description


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
        raise ValueError(f"Unknown task name {task_name}. Max steps for known tasks: {_max_steps_map}")
    return _max_steps_map[task_name]