"""LIBERO env matching vla-post-training's pi05 setup.

Naming: env_name = "libero_pi05-<suite>-<task_id>", e.g. "libero_pi05-libero_90-59".

Differences from `libero_utils.LiberoImageWrapper` (reward/termination unchanged):
  - 256x256 render (VLA `collect.env_resolution`)
  - agentview + eye-in-hand images, both flipped 180 deg, uint8
  - 10 warm-up steps with a gripper-open no-op after reset, not counted
    toward max_episode_length (VLA `WarmUpOnResetWrapper`)
  - language instruction exposed as `self.prompt`

Obs dict: {'image', 'wrist_image', 'state', 'full_state'}. Resizing to 224,
quantile normalisation and tokenisation happen in the pi05 obs pipeline
(`make_env(..., vector=True)` moves the resize into the env, see `VectorSlotWrapper`).
"""

import pathlib

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict

from envs.libero_utils import (
    SUITE_CONFIG,
    LiberoImageWrapper,
    _flip_image,
    _quat2axisangle,
)

PREFIX = "libero_pi05-"
ROBOT = "libero"  # OGPOPi05Agent pi05_robot this family needs (main_pi05 checks)
IMAGE_SIZE = 256
NUM_STEPS_WAIT = 10
WARM_UP_ACTION = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)


def _parse_env_name(env_name):
    """`libero_pi05-libero_90-59` -> ("libero_90", 59)."""
    if not env_name.startswith(PREFIX):
        raise ValueError(f"Not a libero_pi05 env: {env_name}")
    suite, task_id_str = env_name[len(PREFIX):].rsplit("-", 1)
    if suite not in SUITE_CONFIG:
        raise ValueError(
            f"Unknown LIBERO suite '{suite}'. Known: {list(SUITE_CONFIG.keys())}"
        )
    return suite, int(task_id_str)


def is_libero_pi05_env(env_name):
    try:
        _parse_env_name(env_name)
        return True
    except (ValueError, AttributeError):
        return False


def get_prompt(env_name):
    """Task language instruction without building the sim (= the env's `prompt`)."""
    from libero.libero import benchmark

    suite_name, task_id = _parse_env_name(env_name)
    return benchmark.get_benchmark_dict()[suite_name]().get_task(task_id).language


def make_env(env_name, seed=0, post_success_steps=0, vector=False):
    """vector=True: wrapped in `VectorSlotWrapper`, for one slot of an AsyncVectorEnv (pi05 `env.vector_envs`)."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite_name, task_id = _parse_env_name(env_name)
    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    task = task_suite.get_task(task_id)
    bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    libero_env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=IMAGE_SIZE,
        camera_widths=IMAGE_SIZE,
    )
    wrapped = LiberoPi05Wrapper(
        env=libero_env,
        task_suite=task_suite,
        task_id=task_id,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        max_episode_length=SUITE_CONFIG[suite_name]["max_steps"],
        post_success_steps=post_success_steps,
    )
    wrapped.seed(seed)
    return VectorSlotWrapper(wrapped) if vector else wrapped


class LiberoPi05Wrapper(LiberoImageWrapper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        img_h, img_w = self.image_size
        img_space = Box(low=0, high=255, shape=(img_h, img_w, 3), dtype=np.uint8)
        state_space = Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.observation_space = Dict({
            "image": img_space,
            "wrist_image": img_space,
            "state": state_space,
            "full_state": state_space,
        })
        self.prompt = self.task_description

    def _make_obs(self, raw_obs):
        proprio = np.concatenate([
            np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32),
            _quat2axisangle(raw_obs["robot0_eef_quat"]),
            np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32),
        ], axis=0)
        return {
            "image": _flip_image(raw_obs["agentview_image"]),
            "wrist_image": _flip_image(raw_obs["robot0_eye_in_hand_image"]),
            "state": proprio,
            "full_state": proprio.copy(),
        }

    def reset(self, seed=None, options=None, **kwargs):
        _, info = super().reset(seed=seed, options=options, **kwargs)
        # Step the raw env so warm-up does not touch t / return / success.
        for _ in range(NUM_STEPS_WAIT):
            raw_obs, _, _, _ = self.env.step(WARM_UP_ACTION)
        return self._make_obs(raw_obs), info


class VectorSlotWrapper(gym.Wrapper):
    """One AsyncVectorEnv slot of the pi05 vector path.

    - Freeze after done: AsyncVectorEnv (AutoresetMode.DISABLED) steps every slot, so a finished slot
      would keep stepping the sim until its wave ends. That costs a sim step + two renders per idle
      step and corrupts the episode record (t / length / return keep growing, and after a timeout a
      later success step flips info['success'] to 1). Steps after done return the cached terminal
      transition until the next reset.
    - Images resized to 224 in the worker with the same `resize_images` the runner uses
      (byte-identical buffer images); the runner's resize then short-circuits instead of running
      a serial per-image PIL resize in the trainer.
    """

    def __init__(self, env):
        super().__init__(env)
        from ogpo.networks.pi05.transforms import IMAGE_RESOLUTION, resize_images
        self._resize = resize_images
        img_space = Box(low=0, high=255, shape=(IMAGE_RESOLUTION, IMAGE_RESOLUTION, 3), dtype=np.uint8)
        self.observation_space = Dict({**env.observation_space.spaces, "image": img_space, "wrist_image": img_space})
        self._final = None

    def _obs(self, obs):
        return {**obs, "image": self._resize(obs["image"]), "wrist_image": self._resize(obs["wrist_image"])}

    def reset(self, *, seed=None, options=None):
        self._final = None
        obs, info = self.env.reset(seed=seed, options=options)
        return self._obs(obs), info

    def step(self, action):
        if self._final is not None:
            return self._final
        obs, reward, terminated, truncated, info = self.env.step(action)
        out = (self._obs(obs), reward, terminated, truncated, info)
        if terminated or truncated:
            self._final = out
        return out
