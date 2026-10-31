"""LIBERO env + dataset adapter for OGPO.

Mirrors `envs/robomimic_utils.py` so that the rest of OGPO (main.py,
_pre_encode_dataset, runners) can use LIBERO with no further changes.

Naming: env_name = "libero-<suite>-<task_id>", e.g. "libero-libero_90-59".

Image orientation: LIBERO renders agentview / eye-in-hand images flipped.
We apply `[::-1, ::-1]` to both env and dataset images so they match what
the pi05_libero PaliGemma encoder was trained on (see VLA's
`src/envs/wrappers.py:obs_to_pi_zero_input`).

Dependencies (install into OGPO venv before launching):
    pip install libero  # plus its robosuite/mujoco deps
The LIBERO benchmark assets (BDDL + demo HDF5s) ship with the package;
demo HDF5s are located via `libero.libero.get_libero_path("datasets")`.
"""

import math
import os
from collections import deque

import gymnasium as gym
import h5py
import numpy as np
from gymnasium.spaces import Box, Dict

from ogpo.utils.datasets import Dataset


# Per-suite configuration. image_size is the render resolution; PaliGemma
# resizes to 224 internally so any reasonable size works, but the env render
# size must match the dataset image size.
SUITE_CONFIG = {
    "libero_90":      {"max_steps": 400, "image_size": (128, 128)},
    "libero_10":      {"max_steps": 520, "image_size": (128, 128)},
    "libero_goal":    {"max_steps": 300, "image_size": (128, 128)},
    "libero_object":  {"max_steps": 280, "image_size": (128, 128)},
    "libero_spatial": {"max_steps": 220, "image_size": (128, 128)},
}


def _parse_env_name(env_name):
    """`libero-libero_90-59` -> ("libero_90", 59)."""
    if not env_name.startswith("libero-"):
        raise ValueError(f"Not a LIBERO env: {env_name}")
    rest = env_name[len("libero-"):]
    # rest = "libero_90-59"
    suite, task_id_str = rest.rsplit("-", 1)
    if suite not in SUITE_CONFIG:
        raise ValueError(
            f"Unknown LIBERO suite '{suite}'. Known: {list(SUITE_CONFIG.keys())}"
        )
    return suite, int(task_id_str)


def is_libero_env(env_name):
    try:
        _parse_env_name(env_name)
        return True
    except (ValueError, AttributeError):
        return False


def _quat2axisangle(quat):
    """xyzw quaternion -> axis-angle (3-vec). Matches VLA's
    src/envs/wrappers.py:_quat2axisangle (from robosuite)."""
    q = np.asarray(quat, dtype=np.float64).copy()
    if q[3] > 1.0:
        q[3] = 1.0
    elif q[3] < -1.0:
        q[3] = -1.0
    den = math.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((q[:3] * 2.0 * math.acos(q[3])) / den).astype(np.float32)


def _flip_image(img):
    """LIBERO renders are upside-down (and mirrored). Match VLA convention."""
    return np.ascontiguousarray(img[..., ::-1, ::-1, :])


def make_env(env_name, seed=0, post_success_steps=0):
    """Build a LIBERO env wrapped to match the OGPO image-env contract."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import pathlib

    suite_name, task_id = _parse_env_name(env_name)
    cfg = SUITE_CONFIG[suite_name]
    img_h, img_w = cfg["image_size"]

    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    task = task_suite.get_task(task_id)
    bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )

    env_args = {
        "bddl_file_name": str(bddl_file),
        "camera_heights": img_h,
        "camera_widths": img_w,
    }
    libero_env = OffScreenRenderEnv(**env_args)

    wrapped = LiberoImageWrapper(
        env=libero_env,
        task_suite=task_suite,
        task_id=task_id,
        image_size=(img_h, img_w),
        max_episode_length=cfg["max_steps"],
        post_success_steps=post_success_steps,
    )
    wrapped.seed(seed)
    return wrapped


def get_dataset(env, env_name):
    """Load LIBERO demo HDF5(s) for a single task into the OGPO image-env
    dataset schema.

    Expected HDF5 layout (LIBERO standard):
        data/demo_N/actions       (T, 7)
        data/demo_N/dones         (T,)
        data/demo_N/rewards       (T,)        # optional; derive from dones if absent
        data/demo_N/obs/agentview_rgb         (T, H, W, 3) uint8
        data/demo_N/obs/eye_in_hand_rgb       (T, H, W, 3) uint8       # unused
        data/demo_N/obs/ee_pos                (T, 3)
        data/demo_N/obs/ee_ori                (T, 3 or 4)
        data/demo_N/obs/gripper_states        (T, 2)
    """
    from libero.libero import get_libero_path

    suite_name, task_id = _parse_env_name(env_name)
    task_suite = env.task_suite if hasattr(env, "task_suite") else None
    # Recover task_suite when the env was wrapped (EpisodeMonitor etc.)
    if task_suite is None:
        from libero.libero import benchmark
        task_suite = benchmark.get_benchmark_dict()[suite_name]()
    task = task_suite.get_task(task_id)

    dataset_root = os.environ.get("LIBERO_DATASET_ROOT") or get_libero_path("datasets")
    bddl_stem = os.path.splitext(task.bddl_file)[0]
    # libero_10 and libero_90 demos are distributed together under a
    # combined `libero_100/` directory on HuggingFace
    # (yifengzhu-hf/LIBERO-datasets). Other suites use their own dir.
    candidates = []
    if suite_name in ("libero_10", "libero_90"):
        candidates.append(os.path.join(dataset_root, "libero_100", f"{bddl_stem}_demo.hdf5"))
    candidates.append(os.path.join(dataset_root, suite_name, f"{bddl_stem}_demo.hdf5"))
    demo_path = next((p for p in candidates if os.path.exists(p)), None)
    if demo_path is None:
        raise FileNotFoundError(
            f"LIBERO demo HDF5 not found for suite={suite_name}, task={bddl_stem}.\n"
            f"Tried: {candidates}\n"
            f"Download with:\n"
            f"  huggingface-cli download yifengzhu-hf/LIBERO-datasets \\\n"
            f"      --repo-type dataset --local-dir {dataset_root}\n"
            f"Or set LIBERO_DATASET_ROOT to point at an existing copy."
        )

    print(f"[libero] loading demos: {demo_path}")
    f = h5py.File(demo_path, "r")
    demos = list(f["data"].keys())
    # demos are named demo_0, demo_1, ...; sort numerically
    demos.sort(key=lambda s: int(s.split("_")[-1]))

    num_timesteps = sum(int(f[f"data/{ep}/actions"].shape[0]) for ep in demos)
    print(f"[libero] {len(demos)} demos, {num_timesteps} transitions")

    observations, next_observations = [], []
    full_states, next_full_states = [], []
    actions, rewards, terminals, masks = [], [], [], []
    images_arr = None
    next_images_arr = None
    img_cursor = 0

    for ep in demos:
        g = f[f"data/{ep}"]
        a = np.asarray(g["actions"], dtype=np.float32)
        n = a.shape[0]
        if "dones" in g:
            dones = np.asarray(g["dones"], dtype=np.float32)
        else:
            dones = np.zeros(n, dtype=np.float32)
            dones[-1] = 1.0
        if "rewards" in g:
            r_raw = np.asarray(g["rewards"], dtype=np.float32)
        else:
            # Construct sparse 0/1 reward from dones (1.0 on terminal step).
            r_raw = dones.astype(np.float32)
        # Transform rewards [0, 1] -> [-1, 0] (matches Robomimic OGPO convention).
        r = r_raw - 1.0

        obs_g = g["obs"]
        img = np.asarray(obs_g["agentview_rgb"])  # (T, H, W, 3) uint8
        # The HDF5 stores T frames; we need (obs_t, next_obs_t) pairs. LIBERO
        # demos typically store next_obs implicitly via shifting. We build
        # next_images by shifting and padding the last frame.
        next_img = np.empty_like(img)
        next_img[:-1] = img[1:]
        next_img[-1] = img[-1]
        img = _flip_image(img)
        next_img = _flip_image(next_img)

        if images_arr is None:
            images_arr = np.empty((num_timesteps,) + img.shape[1:], dtype=img.dtype)
            next_images_arr = np.empty((num_timesteps,) + next_img.shape[1:], dtype=next_img.dtype)
        images_arr[img_cursor:img_cursor + n] = img
        next_images_arr[img_cursor:img_cursor + n] = next_img
        img_cursor += n

        ee_pos = np.asarray(obs_g["ee_pos"], dtype=np.float32)         # (T, 3)
        ee_ori_raw = np.asarray(obs_g["ee_ori"], dtype=np.float32)     # (T, 3) or (T, 4)
        if ee_ori_raw.shape[-1] == 4:
            ee_ori = np.stack([_quat2axisangle(q) for q in ee_ori_raw], axis=0)
        else:
            ee_ori = ee_ori_raw
        gripper = np.asarray(obs_g["gripper_states"], dtype=np.float32)  # (T, 2)
        proprio = np.concatenate([ee_pos, ee_ori, gripper], axis=-1)     # (T, 8)
        next_proprio = np.empty_like(proprio)
        next_proprio[:-1] = proprio[1:]
        next_proprio[-1] = proprio[-1]

        observations.append(proprio)
        next_observations.append(next_proprio)
        # No flat object-state in LIBERO HDF5s by default; full_state == proprio.
        # Two-tier obs encoder still works — it just sees a smaller proprio
        # tower input.
        full_states.append(proprio)
        next_full_states.append(next_proprio)

        actions.append(a)
        rewards.append(r.astype(np.float32))
        terminals.append(dones.astype(np.float32))
        masks.append((1.0 - dones).astype(np.float32))

    f.close()
    assert img_cursor == num_timesteps

    dataset_fields = dict(
        observations=np.concatenate(observations, axis=0),
        actions=np.concatenate(actions, axis=0),
        rewards=np.concatenate(rewards, axis=0),
        terminals=np.concatenate(terminals, axis=0),
        masks=np.concatenate(masks, axis=0),
        next_observations=np.concatenate(next_observations, axis=0),
        images=images_arr,
        next_images=next_images_arr,
        full_states=np.concatenate(full_states, axis=0),
        next_full_states=np.concatenate(next_full_states, axis=0),
    )
    return Dataset.create(**dataset_fields)


class LiberoImageWrapper(gym.Env):
    """Gymnasium adapter for LIBERO's OffScreenRenderEnv.

    Returns dict obs {'state', 'image', 'full_state'} matching
    RobomimicImageWrapper so OGPO's runners can route to PaliGemma encode_fn.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        env,
        task_suite,
        task_id,
        image_size=(128, 128),
        max_episode_length=400,
        post_success_steps=0,
    ):
        self.env = env
        self.task_suite = task_suite
        self.task_id = task_id
        self.image_size = image_size
        self.max_episode_length = max_episode_length
        self.post_success_steps = post_success_steps

        # Spaces. LIBERO actions: 6 deltas + 1 gripper, all in [-1, 1].
        self.action_space = Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        img_h, img_w = image_size
        # proprio: ee_pos(3) + axisangle(3) + gripper(2) = 8
        proprio_dim = 8
        self.observation_space = Dict({
            "state": Box(low=-np.inf, high=np.inf, shape=(proprio_dim,), dtype=np.float32),
            "image": Box(low=0, high=255, shape=(img_h, img_w, 3), dtype=np.float32),
            "full_state": Box(low=-np.inf, high=np.inf, shape=(proprio_dim,), dtype=np.float32),
        })

        self._init_states = self._load_init_states()
        self._rng = np.random.default_rng(0)
        self.t = 0
        self.t_succ = None
        self.episode_return = 0.0
        self.episode_length = 0
        self.env_step = 0
        self.n_episodes = 0
        self.task_description = task_suite.get_task(task_id).language

    def _load_init_states(self):
        import torch
        from libero.libero import get_libero_path
        task = self.task_suite.tasks[self.task_id]
        path = os.path.join(
            get_libero_path("init_states"),
            task.problem_folder,
            task.init_states_file,
        )
        return torch.load(path, weights_only=False)

    def seed(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def _make_obs(self, raw_obs):
        img = _flip_image(raw_obs["agentview_image"]).astype(np.float32)
        ee_pos = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32)
        ee_quat = np.asarray(raw_obs["robot0_eef_quat"], dtype=np.float32)
        ee_ori = _quat2axisangle(ee_quat)
        gripper = np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32)
        proprio = np.concatenate([ee_pos, ee_ori, gripper], axis=0)
        return {"state": proprio, "image": img, "full_state": proprio.copy()}

    def reset(self, seed=None, options=None, **kwargs):
        if seed is not None:
            self.seed(seed)
        self.env.reset()
        idx = int(self._rng.integers(low=0, high=self._init_states.shape[0]))
        raw_obs = self.env.set_init_state(self._init_states[idx])
        self.t = 0
        self.t_succ = None
        self.episode_return = 0.0
        self.episode_length = 0
        self.n_episodes += 1
        obs = self._make_obs(raw_obs)
        info = {"task_description": self.task_description}
        return obs, info

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(np.asarray(action, dtype=np.float32))
        # LIBERO native reward is 0/1. Match Robomimic OGPO convention: -> [-1, 0].
        reward = float(reward) - 1.0

        is_success_step = reward > -0.5
        if self.t_succ is None:
            if is_success_step:
                self.t_succ = self.t
                info["success"] = 1
                reward += 5.0
                if self.post_success_steps <= 0:
                    done = True
            else:
                info["success"] = 0
        else:
            info["success"] = 1
            steps_since_succ = self.t - self.t_succ
            if is_success_step and steps_since_succ <= self.post_success_steps:
                reward += 5.0
            if steps_since_succ >= self.post_success_steps or not is_success_step:
                done = True

        self.t += 1
        self.env_step += 1
        self.episode_return += reward
        self.episode_length += 1

        obs = self._make_obs(raw_obs)
        terminated = bool(done)
        truncated = self.t >= self.max_episode_length
        if terminated or truncated:
            info["return"] = self.episode_return
            info["length"] = self.episode_length
        info["task_description"] = self.task_description
        if terminated:
            return obs, reward, True, False, info
        if truncated:
            return obs, reward, False, True, info
        return obs, reward, False, False, info

    def render(self, mode="rgb_array"):
        # OffScreenRenderEnv exposes images via obs; render just returns the
        # current agentview image to satisfy the gym API.
        raw_obs = self.env.env.get_observations() if hasattr(self.env, "env") else None
        if raw_obs is None:
            return None
        return _flip_image(raw_obs["agentview_image"])

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()
