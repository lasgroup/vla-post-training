"""MolmoSpaces env for the pi05 agent with `pi05_robot: droid` (checkpoint pi05_droid_jointpos).

Naming: env_name = "molmo_pi05-<episode_id>", an episode index of the FrankaPickDroidMiniBench JSON benchmark
(e.g. "molmo_pi05-164" = vla-post-training's task "molmo_164").

Ported from vla-post-training (github.com/lasgroup/vla-post-training @ origin/cleanup, sim = lasgroup/molmospaces
@ 2a828d5), folded into one env:
  - src/envs/molmo.py: MolmoSpacesBenchmarkGymEnv (episode / sampler / reset / step, init randomisation, sensor
    pruning, output suppression), MolmoActionAdapter (8-D action -> {'arm', 'gripper'}), make_env_molmo (450 steps)
  - src/envs/wrappers.py: obs_to_pi_zero_input(env_class="molmo") (DroidInputs layout),
    TimeToSuccessAsRewardWrapper (reward 0 on the success step, -1 otherwise; no success bonus)
Episodes terminate on success; no post-success steps, no warm-up.

Obs dict (libero_pi05 contract): {'image': exo_camera_1, 'wrist_image': wrist_camera (uint8 352x624, upright),
'state' = 'full_state': [7 arm joints, gripper qpos / 0.824033 clipped to [0, 1]]}. Actions: 8-D absolute joint
targets + gripper (> 0.5 closes); the agent turns the model's joint deltas into these. info: 'success', 'task_description'
every step, 'return' / 'length' on the terminal step.

Process model: the sim imports jax / mujoco.mjx at module load, turns on jax_enable_x64 globally and builds CPU mjx
kinematics per robot (docs/changes/2026-10-14-molmo-pi05/BLAST-RADIUS.md §2), so it must never load in the trainer.
Only the vector path is supported (`vector=True`, one AsyncVectorEnv worker per slot), and the env is lazy:
gymnasium builds slot 0 once in the parent to read the spaces, so __init__ imports nothing; the first reset() or
get_prompt() (in a worker) pins jax to CPU, imports molmo_spaces and loads the scene (~113 s import + 80-200 s on a
cold worker; re-resets ~0.7 s, steps ~0.11 s with both renders).

Requires MLSPACES_ASSETS_DIR, MLSPACES_CACHE_DIR and NLTK_DATA (scripts/ogpo/download_molmo_assets.sbatch fills
them): unset, the first import silently downloads ~12 GB of assets into the repo / $HOME.
"""

import contextlib
import importlib
import logging
import os
import warnings
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict

PREFIX = "molmo_pi05-"
ROBOT = "droid"  # OGPOPi05Agent pi05_robot this family needs (main_pi05 checks)
BENCHMARK_REL = ("benchmarks/molmospaces-bench-v1/procthor-10k/FrankaPickDroidMiniBench/"
                 "FrankaPickDroidMiniBench_json_benchmark_20251231")
EVAL_CONFIG = "molmo_spaces.evaluation.configs.evaluation_configs:PiPolicyEvalConfig"
REQUIRED_ENV_VARS = ("MLSPACES_ASSETS_DIR", "MLSPACES_CACHE_DIR", "NLTK_DATA")
MAX_EPISODE_STEPS = 450  # vla-post-training make_env_molmo TimeLimit
IMAGE_HW = (352, 624)  # episode.img_resolution (624, 352) for all 1000 benchmark episodes
GRIPPER_QPOS_CLOSED = 0.824033  # gripper qpos when closed: state dim 7 = qpos / this, in [0, 1] (DroidInputs)
GRIPPER_CLOSE_THRESHOLD = 0.5  # action dim 7 > this -> command 255 (closed), else 0
# Initial-state randomisation, vla-post-training MolmoSpacesGymConfig defaults.
INIT_OBJECT_XY_NOISE = 0.01  # m, +/- uniform per axis
INIT_QPOS_NOISE = 0.01  # rad, +/- uniform per joint
INIT_MAX_ATTEMPTS = 20
DROP_SENSOR_UUIDS = ("object_image_points",)  # segmentation points, unused by pi05
# molmo Task per-step history lists (Task.get_and_cache_all_step_information / step), emptied by _drop_task_caches
TASK_CACHES = ("observation_cache", "action_cache", "reward_cache", "terminal_cache", "truncated_cache", "success_cache")


def _drop_task_caches(task):
    """Empty a molmo Task's per-step history. Task.get_and_cache_all_step_information appends every step's
    observations (all camera frames, ~1.3 MB / step) for get_history(), which only molmo's data-generation pipeline
    reads; the one other reader, step()'s first-step check of observation_cache[0], runs inside the first step()
    before this is called. Kept, a worker holds ~0.58 GB per 450-step episode, and a replaced task outlives the next
    reset in a reference cycle (task._registered_policy.task) that only a full gc frees, which a worker rarely runs:
    144 workers ran a TRI node out of host memory in ~2 h (molmo mt144, 2026-10-24)."""
    for name in TASK_CACHES:
        cache = getattr(task, name, None)
        if isinstance(cache, list):
            cache.clear()


def _parse_env_name(env_name):
    """`molmo_pi05-164` -> 164."""
    if not isinstance(env_name, str) or not env_name.startswith(PREFIX):
        raise ValueError(f"Not a molmo_pi05 env: {env_name!r} (expected '{PREFIX}<episode_id>')")
    episode = env_name[len(PREFIX):]
    if not episode.isdigit():
        raise ValueError(f"molmo_pi05 env {env_name!r}: episode id {episode!r} is not a non-negative integer")
    return int(episode)


def is_molmo_pi05_env(env_name):
    try:
        _parse_env_name(env_name)
        return True
    except ValueError:
        return False


def _check_env_vars():
    missing = [k for k in REQUIRED_ENV_VARS if not os.environ.get(k)]
    if missing:
        raise ValueError(f"molmo_pi05 envs need {missing} set (asset / cache / NLTK dirs, e.g. under "
                         f"/data/user_data/$USER/molmospaces); unset, importing molmo_spaces downloads ~12 GB "
                         f"into the repo / $HOME. See scripts/ogpo/download_molmo_assets.sbatch.")


@contextlib.contextmanager
def _quiet():
    """vla-post-training's _suppress_molmo_spaces_output: keep errors, drop molmo_spaces' INFO / print noise."""
    logging.getLogger("molmo_spaces").setLevel(logging.ERROR)
    with open(os.devnull, "w") as devnull, warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Using objaverse data version .*", category=UserWarning)
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


class _RegisteredPolicyAdapter:
    """Minimal policy interface for MolmoSpaces' policy-dependent sensors (vla-post-training, verbatim)."""

    def __init__(self):
        self.target_poses = {"grasp": np.eye(4, dtype=np.float32)}
        self.task = None
        self.retry_count = 0

    def reset(self):
        return

    def get_phase(self):
        return "inference"

    def get_all_phases(self):
        return {"inference": 0}


def _render_device():
    """GPU this slot renders on. molmo's EGL context asserts render_device is in CUDA_VISIBLE_DEVICES
    (renderer/opengl_context.py). The worker's MUJOCO_EGL_DEVICE_ID wins when it is one of those GPUs: the launcher's
    per-process pin (a launcher that sets it per run) and `env.worker_egl_devices` (make_vectorized_env pins slot i to
    egl_devices[i % n]) set it, which spreads the slots over the run's GPUs. Otherwise the first CUDA_VISIBLE_DEVICES
    entry (every slot on one GPU), as the LIBERO workers do."""
    visible = [d.strip() for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    pinned = os.environ.get("MUJOCO_EGL_DEVICE_ID", "").strip()
    if pinned.isdigit() and (not visible or pinned in visible):
        return int(pinned)
    return int(visible[0]) if visible and visible[0].isdigit() else 0


class MolmoPi05Env(gym.Env):
    """One fixed FrankaPickDroidMiniBench episode (house + objects + robot pose), init-randomised per reset."""

    metadata = {"render_modes": []}

    def __init__(self, episode_id, seed=0):
        super().__init__()
        self.episode_id = episode_id
        self._seed = seed
        img = Box(low=0, high=255, shape=(*IMAGE_HW, 3), dtype=np.uint8)
        state = Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.observation_space = Dict({"image": img, "wrist_image": img, "state": state, "full_state": state})
        self.action_space = Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.prompt = None  # set by the first reset (PromptSampler on the loaded task)
        self.t = 0
        self.episode_return = 0.0
        self._sampler = None
        self._task = None

    # -- lazy load (worker only) --------------------------------------------------------------------------------

    def _load(self):
        if self._sampler is not None:
            return
        _check_env_vars()
        # Before molmo_spaces imports jax: its CPU mjx kinematics must not open a GPU backend in every worker
        # (they inherit XLA_PYTHON_CLIENT_MEM_FRACTION ~0.9).
        import jax
        jax.config.update("jax_platforms", "cpu")
        if jax.default_backend() != "cpu":
            raise RuntimeError(f"molmo_pi05 worker: jax backend is {jax.default_backend()!r}, expected 'cpu' "
                               f"(the env must load in a fresh AsyncVectorEnv worker, before any jax use)")
        with _quiet():
            from molmo_spaces.evaluation.benchmark_schema import load_all_episodes
            from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
            from molmo_spaces.policy.learned_policy.utils import PromptSampler
            from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler
            episodes = load_all_episodes(Path(ASSETS_DIR) / BENCHMARK_REL)
        if not 0 <= self.episode_id < len(episodes):
            raise ValueError(f"molmo_pi05 episode id {self.episode_id} out of range 0..{len(episodes) - 1}")
        self._episode = episode = episodes[self.episode_id]
        if tuple(episode.img_resolution) != IMAGE_HW[::-1]:
            raise ValueError(f"molmo_pi05-{self.episode_id}: img_resolution {episode.img_resolution}, "
                             f"expected {IMAGE_HW[::-1]} (observation_space is static)")

        module, cls = EVAL_CONFIG.split(":")
        exp_config = getattr(importlib.import_module(module), cls)()
        exp_config.scene_dataset = episode.scene_dataset
        exp_config.data_split = episode.data_split
        ts = exp_config.task_sampler_config
        ts.render_device = _render_device()
        ts.randomize_init_object_positions = True
        ts.randomize_init_qpos = True
        ts.init_object_position_noise_xy = INIT_OBJECT_XY_NOISE
        ts.init_qpos_noise = INIT_QPOS_NOISE
        ts.init_position_randomization_max_attempts = INIT_MAX_ATTEMPTS
        self._prompt_sampler = PromptSampler(
            task_type=exp_config.task_type,
            prompt_templates=exp_config.policy_config.prompt_templates,
            prompt_object_word_num=exp_config.policy_config.prompt_object_word_num,
        )
        with _quiet():
            self._sampler = JsonEvalTaskSampler(exp_config, episode)

    def get_prompt(self):
        """Task language prompt; loads the task on first call (main_pi05 reads it via AsyncVectorEnv.call)."""
        if self.prompt is None:
            self.reset(seed=self._seed)
        return self.prompt

    # -- gym API --------------------------------------------------------------------------------------------------

    def _release_task(self):
        """Drop the current task: empty its caches and break the task <-> policy cycle so it is freed now."""
        task, self._task = self._task, None
        if task is None:
            return
        _drop_task_caches(task)
        policy = getattr(task, "_registered_policy", None)
        task._registered_policy = None
        if policy is not None:
            policy.task = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._load()
        self._release_task()
        self._prompt_sampler.next()
        with _quiet():
            self._task = self._sampler.sample_task(force_advance_scene=False, house_index=self._episode.house_index)
        if self._task is None:
            raise RuntimeError(f"molmo_pi05-{self.episode_id}: JsonEvalTaskSampler returned no task")
        sensors = getattr(getattr(self._task, "_sensor_suite", None), "sensors", None)
        if sensors is not None:
            for uuid in DROP_SENSOR_UUIDS:
                sensors.pop(uuid, None)
        if self._task.env.n_batch != 1:
            raise ValueError(f"molmo_pi05 needs n_batch=1, got {self._task.env.n_batch}")
        self._task.register_policy(_RegisteredPolicyAdapter())
        prompt = self._prompt_sampler.get_prompt(self._task).lower()
        if self.prompt is not None and prompt != self.prompt:
            # The agent tokenises one prompt per task at create(); a per-reset prompt would silently go stale.
            raise RuntimeError(f"molmo_pi05-{self.episode_id}: prompt changed across resets "
                               f"({self.prompt!r} -> {prompt!r})")
        self.prompt = prompt
        observations, infos = self._task.reset()
        if not observations or not infos:
            raise RuntimeError(f"molmo_pi05-{self.episode_id}: task reset returned empty observations / infos")
        self.t = 0
        self.episode_return = 0.0
        return self._make_obs(observations[0]), {"task_description": self.prompt}

    def step(self, action):
        if self._task is None:
            raise RuntimeError("molmo_pi05: step() before reset()")
        a = np.asarray(action, dtype=np.float32)
        gripper = 255.0 if a[7] > GRIPPER_CLOSE_THRESHOLD else 0.0
        observations, _, _, task_truncated, infos = self._task.step(
            {"arm": a[:7], "gripper": np.asarray([gripper], dtype=np.float32)})
        if not observations or not infos:
            raise RuntimeError(f"molmo_pi05-{self.episode_id}: task step returned empty observations / infos")
        _drop_task_caches(self._task)
        success = bool(infos[0]["success"])
        reward = 0.0 if success else -1.0  # TimeToSuccessAsRewardWrapper
        self.t += 1
        self.episode_return += reward
        truncated = (bool(task_truncated[0]) or self.t >= MAX_EPISODE_STEPS) and not success
        info = {"success": int(success), "task_description": self.prompt}
        if success or truncated:
            info["return"] = self.episode_return
            info["length"] = self.t
        return self._make_obs(observations[0]), reward, success, truncated, info

    def _make_obs(self, raw):
        qpos = raw["qpos"]
        grip = np.clip(float(np.asarray(qpos["gripper"])[0]) / GRIPPER_QPOS_CLOSED, 0.0, 1.0)
        state = np.concatenate([np.asarray(qpos["arm"][:7], dtype=np.float32), [grip]]).astype(np.float32)
        return {
            "image": np.ascontiguousarray(raw["exo_camera_1"], dtype=np.uint8),
            "wrist_image": np.ascontiguousarray(raw["wrist_camera"], dtype=np.uint8),
            "state": state,
            "full_state": state.copy(),
        }

    def close(self):
        if self._sampler is not None:
            self._sampler.close()
        self._sampler = None
        self._release_task()


def make_env(env_name, seed=0, post_success_steps=0, vector=False):
    """Vector slot only (`vector=True`, wrapped in libero_pi05's `VectorSlotWrapper`): see the module docstring."""
    episode_id = _parse_env_name(env_name)
    if not vector:
        raise ValueError(f"{env_name}: molmo_pi05 envs run only as AsyncVectorEnv slots (env.vector_envs=true); "
                         f"in-process, molmo_spaces would turn on jax x64 in the trainer")
    if post_success_steps != 0:
        raise ValueError(f"{env_name}: post_success_steps={post_success_steps}; molmo_pi05 terminates on success "
                         f"(vla-post-training), use 0")
    _check_env_vars()
    from envs.libero_pi05_utils import VectorSlotWrapper
    return VectorSlotWrapper(MolmoPi05Env(episode_id, seed=seed))
