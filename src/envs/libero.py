import jax
import jax.numpy as jnp
import numpy as np
import pathlib
import torch

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from robosuite.utils.transform_utils import quat2axisangle

import openpi.models.model as _model
from openpi_client import image_tools


# wrapper to load init states inside worker processes (works with spawn)
class OffScreenRenderEnvWithInit(OffScreenRenderEnv):
    def __init__(self, *args, init_states_path: pathlib.Path = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_states_path = init_states_path
        self._init_states = None
        torch.serialization.add_safe_globals(
            [
                np.core.multiarray._reconstruct,  # noqa
                np.ndarray,
                np.dtype,
                np.dtypes.Float64DType,
            ]
        )

    def _ensure_init_states_loaded(self):
        if self._init_states is None:
            import torch
            # load locally inside the worker process
            self._init_states = torch.load(str(self._init_states_path))

    def set_init_state(self, init_state_or_index):
        # accept either an integer index (preferred) or a full init-state object
        self._ensure_init_states_loaded()
        if isinstance(init_state_or_index, int):
            init_state = self._init_states[init_state_or_index]
        else:
            init_state = init_state_or_index
        return super().set_init_state(init_state)


def make_env_libero(config):
    assert len(config.tasks) == 1, "Only single-task collection is supported."
    task_suite_name = "_".join(config.tasks[0].split("_")[:-1])
    task_id = int(config.tasks[0].split("_")[-1])
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
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
            return OffScreenRenderEnvWithInit(**args, init_states_path=init_states_path)
        env_factories.append(_make_env)
    env = SubprocVectorEnv(env_factories)
    # TODO: check seeding behavior
    env.seed(42)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, initial_states, task_description


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


def init_state_libero(env, initial_states, episode_idx, config):
    env.reset()
    start = episode_idx * config.collect.env_num
    end = (episode_idx + 1) * config.collect.env_num
    _initial_states = np.stack([initial_states[i % len(initial_states)] for i in range(start, end)], 0)
    obs = env.set_init_state(_initial_states)
    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
    for _ in range(config.collect.num_steps_wait):
        obs, _, _, _ = env.step(np.array([[0.0]*6+[-1.] for _ in range(config.collect.env_num)]))
    return obs


def get_action_chunk_libero(obs, task_description, policy, config, sharding_spec):
    obs = {k: np.stack([o[k] for o in obs], 0) for k, v in obs[0].items()}
    # Get preprocessed image
    # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = np.ascontiguousarray(obs["agentview_image"][:, ::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][:, ::-1, ::-1])
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, 224, 224))
    element = {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                np.stack([quat2axisangle(o) for o in obs["robot0_eef_quat"]], 0),
                obs["robot0_gripper_qpos"],
            ), -1
        ),
        "prompt": str(task_description),
    }
    # TODO: Move into batched infer functionality
    # call policy inference in a reasonable way - I could not find batch inference methods
    # manually replicating policy.infer
    # very hacky, only for benchmarking purposes
    inputs = policy._input_transform(element)
    inputs['image_mask'] = jax.tree.map(lambda x: jnp.stack([jnp.asarray(x)]*config.collect.env_num, 0), inputs['image_mask'])
    inputs['tokenized_prompt'] = jax.tree.map(lambda x: jnp.stack([jnp.asarray(x)]*config.collect.env_num, 0), inputs['tokenized_prompt'])
    inputs['tokenized_prompt_mask'] = jax.tree.map(lambda x: jnp.stack([jnp.asarray(x)]*config.collect.env_num, 0), inputs['tokenized_prompt_mask'])
    
    if sharding_spec:
        inputs = jax.device_put(inputs, sharding_spec)

    policy._rng, sample_rng_or_pytorch_device = jax.random.split(policy._rng)
    sample_kwargs = dict(policy._sample_kwargs)
    observation = _model.Observation.from_dict(inputs)
    outputs = {
        "state": inputs["state"],
        "actions": policy._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
    }

    actions_list = []
    # TODO: Improve slow python loop
    for b in range(config.collect.env_num):
        out_b = {k: (v[b, ...] if hasattr(v, "shape") and v.shape[0] == config.collect.env_num else v) for k, v in outputs.items()}
        out_b = policy._output_transform(out_b)
        actions_list.append(out_b["actions"])
    action_chunk = np.stack(actions_list, axis=0)

    return action_chunk


def get_frame_libero(obs, action, task_description):
    obs = {k: np.stack([o[k] for o in obs], 0) for k, v in obs[0].items()}
    og_img = np.ascontiguousarray(obs["agentview_image"][:, ::-1, ::-1])
    og_wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][:, ::-1, ::-1])
    processed_action = np.where(np.abs(action) < 0.0011, 0.0, action)
    return {
        "image": og_img,
        "wrist_image": og_wrist_img,
        "state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                np.stack([quat2axisangle(o) for o in obs["robot0_eef_quat"]], 0),
                obs["robot0_gripper_qpos"],
            ), -1
        ).astype(np.float32),
        "actions": np.asarray(processed_action, dtype=np.float32),
    }
