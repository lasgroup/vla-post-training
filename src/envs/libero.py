import jax
import jax.numpy as jnp
import numpy as np
import pathlib
import torch

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.transform_utils import quat2axisangle

from openpi_client import image_tools
from src.envs.wrappers import ensure_gymnasium_env, WarmUpOnResetWrapper, SetInitialState, Pi0ObservationWrapper, QueryFrequencyWrapper
from gymnasium.wrappers import TimeLimit
from src.envs.venv import SubprocVectorEnv


# wrapper to load init states inside worker processes (works with spawn)
class OffScreenRenderEnvWithInit(OffScreenRenderEnv):
    def __init__(self, *args, init_states_path: pathlib.Path = None, **kwargs):
        super().__init__(*args, **kwargs)
    #     self._init_states_path = init_states_path
    #     self._init_states = None
    #     torch.serialization.add_safe_globals(
    #         [
    #             np.core.multiarray._reconstruct,  # noqa
    #             np.ndarray,
    #             np.dtype,
    #             np.dtypes.Float64DType,
    #         ]
    #     )
    #
    # def _ensure_init_states_loaded(self):
    #     if self._init_states is None:
    #         import torch
    #         # load locally inside the worker process
    #         self._init_states = torch.load(str(self._init_states_path))

    def set_init_state(self, init_state):
        # # accept either an integer index (preferred) or a full init-state object
        # self._ensure_init_states_loaded()
        # if isinstance(init_state_or_index, int):
        #     init_state = self._init_states[init_state_or_index]
        # else:
        #     init_state = init_state_or_index
        return super().set_init_state(init_state)


def get_libero_warm_start_action():
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])


def make_env_libero(config, discount: float = 0.99):
    assert len(config.tasks) == 1, "Only single-task collection is supported."
    task_suite_name = "_".join(config.tasks[0].split("_")[:-1])
    task_id = int(config.tasks[0].split("_")[-1])
    max_steps = get_max_steps_libero(config.tasks[0])
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    warm_start_action = get_libero_warm_start_action()
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    init_states_path = pathlib.Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": config.env_resolution,
        "camera_widths": config.env_resolution,
    }
    env_factories = []
    for i in range(config.env_num):
        def _make_env(rank=i):
            args = env_args.copy()
            args["render_gpu_device_id"] = rank % 4
            # Create Libero environment
            base_env = OffScreenRenderEnvWithInit(**args, init_states_path=init_states_path)
            # Converts gym envs to gymnasium style envs
            base_env = ensure_gymnasium_env(base_env)
            # Sets initial states for the environment
            base_env = SetInitialState(base_env, initial_states=initial_states)
            # Warm ups upon reset
            base_env = WarmUpOnResetWrapper(
                env=base_env,
                num_steps_wait=config.num_steps_wait,
                warm_up_action=warm_start_action,
            )
            # Add timelimit wrapper
            base_env = TimeLimit(
                base_env,
                max_episode_steps=max_steps,
            )
            base_env = Pi0ObservationWrapper(
                env=base_env,
                env_class="libero",
                task_description=task_description,
                resize_image=config.resize_image,
                add_states=config.add_states
            )
            base_env = QueryFrequencyWrapper(
                env=base_env,
                query_frequency=config.replan_steps,
                discount=discount,
                store_full_transitions=config.store_full_transitions,
                post_step_filter=lambda x: np.where(np.abs(x) < 0.0011, 0.0, x),
            )
            return base_env

        env_factories.append(_make_env)
    env = SubprocVectorEnv(env_factories)
    # env.seed(42)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
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