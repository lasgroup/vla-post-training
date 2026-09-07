"""Gymnasium adapter for MolmoSpaces JSON benchmarks.

This module exposes a single-environment gym-like interface over MolmoSpaces
benchmark episodes.

Example:
    from src.molmo.molmospaces_gym_env import MolmoSpacesBenchmarkGymEnv
    import numpy as np

    env = MolmoSpacesBenchmarkGymEnv()
    obs, info = env.reset()
    action = {"arm": np.zeros(7), "gripper": np.zeros(1)}
    obs, reward, terminated, truncated, info = env.step(action)
    env.close()
"""

import dataclasses
import importlib
import logging
import os
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np

from src.envs.wrappers import ensure_gymnasium_env


logger = logging.getLogger(__name__)


def _silence_molmo_spaces_logs() -> None:
    # Keep errors visible, but suppress the INFO-level noise emitted by MolmoSpaces.
    logging.getLogger("molmo_spaces").setLevel(logging.ERROR)


@contextmanager
def _suppress_molmo_spaces_output():
    logging.getLogger("molmo_spaces").setLevel(logging.ERROR)
    with open(os.devnull, "w") as devnull:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Using objaverse data version .*",
                category=UserWarning,
            )
            with redirect_stdout(devnull), redirect_stderr(devnull):
                yield


with _suppress_molmo_spaces_output():
    from molmo_spaces.evaluation.benchmark_schema import load_all_episodes
    from molmo_spaces.policy.learned_policy.utils import PromptSampler
    from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler


class _RegisteredPolicyAdapter:
    """Minimal policy interface for MolmoSpaces policy-dependent sensors."""

    def __init__(self) -> None:
        self.target_poses = {"grasp": np.eye(4, dtype=np.float32)}
        self.task = None
        self.retry_count = 0

    def reset(self) -> None:
        return

    def get_phase(self) -> str:
        return "inference"

    def get_all_phases(self) -> dict[str, int]:
        return {"inference": 0}


@dataclasses.dataclass(frozen=True)
class MolmoSpacesGymConfig:
    # MLSPACES_BENCHMARK_DIR overrides the CSCS default off-cluster.
    benchmark_dir: str = os.environ.get(
        "MLSPACES_BENCHMARK_DIR",
        "/capstor/store/cscs/swissai/a143/molmospaces/assets/benchmarks/molmospaces-bench-v1/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231",
    )
    eval_config_cls: str = (
        "molmo_spaces.evaluation.configs.evaluation_configs:PiPolicyEvalConfig"
    )
    episode_sampling: Literal["sequential", "random"] = "sequential"
    seed: int = 0
    # sensor uuids to drop from the task sensor suite before stepping
    # i.e. segmentation masks that are ignored by pi0.5
    drop_sensor_uuids: tuple[str, ...] = ("object_image_points",)
    reduce_resolution: bool = False
    # initial-state randomization: perturb object XY and robot init_qpos with rejection sampling.
    randomize_init_positions: bool = True
    init_object_position_noise_xy: float = 0.01  # meters, +/- uniform per axis
    init_qpos_noise: float = 0.01  # radians, +/- uniform per joint
    randomization_max_attempts: int = 20


class MolmoSpacesBenchmarkGymEnv(gym.Env):
    """Single-env gym adapter for MolmoSpaces benchmark episodes."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        episode_id: int | None = None,
        render_device: int = 0,
        config: MolmoSpacesGymConfig = MolmoSpacesGymConfig(),
    ):
        super().__init__()
        self._episode_id = episode_id
        self._render_device = render_device
        self._config = config
        self._benchmark_dir = Path(config.benchmark_dir).expanduser().resolve()
        self._episodes = load_all_episodes(self._benchmark_dir)
        if not self._episodes:
            raise ValueError(
                f"No benchmark episodes found in {self._benchmark_dir}. "
                "Expected benchmark.json or house_*/episode_*.json files."
            )

        self._rng = np.random.default_rng(config.seed)
        self._next_episode_idx = 0
        self._sampler: JsonEvalTaskSampler | None = None
        self._task = None
        self._task_description: str | None = None
        self._closed = False
        self._loaded_episode_id: int | None = None
        self._prompt_sampler = None

        # Minimal placeholder spaces for v1.
        self.observation_space = gym.spaces.Dict({})
        self.action_space = gym.spaces.Dict({})

    def _make_eval_config(self):
        spec = self._config.eval_config_cls
        if ":" not in spec:
            raise ValueError(
                f"Invalid eval_config_cls '{spec}'. Expected format "
                "'module.path:ClassName'."
            )
        module_name, class_name = spec.split(":", maxsplit=1)
        module = importlib.import_module(module_name)
        try:
            eval_config_cls = getattr(module, class_name)
        except AttributeError as exc:
            raise ValueError(
                f"Could not resolve class '{class_name}' in module '{module_name}'."
            ) from exc
        exp_config = eval_config_cls()
        return exp_config

    def _make_prompt_sampler(self, exp_config: Any) -> PromptSampler:
        return PromptSampler(
            task_type=exp_config.task_type,
            prompt_templates=exp_config.policy_config.prompt_templates,
            prompt_object_word_num=exp_config.policy_config.prompt_object_word_num,
        )

    def _choose_episode(self):
        if self._episode_id is not None:
            if not 0 <= self._episode_id < len(self._episodes):
                raise ValueError(
                    f"episode_id {self._episode_id} out of range for available episodes."
                )
            idx = self._episode_id
        elif self._config.episode_sampling == "random":
            idx = int(self._rng.integers(len(self._episodes)))
        elif self._config.episode_sampling == "sequential":
            idx = self._next_episode_idx
            self._next_episode_idx = (self._next_episode_idx + 1) % len(self._episodes)
        else:
            raise ValueError(
                f"Unsupported episode_sampling='{self._config.episode_sampling}'. "
                "Expected one of {'sequential', 'random'}."
            )
        return self._episodes[idx]

    def _close_active_episode(self) -> None:
        if self._sampler is not None:
            self._sampler.close()
        self._sampler = None
        self._task = None
        self._task_description = None

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Environment is closed.")

    def _info_with_task_description(self, info: dict[str, Any]) -> dict[str, Any]:
        if self._task_description is None:
            raise RuntimeError("Task description is unavailable before reset().")
        return {**info, "task_description": self._task_description}

    def _prune_unused_sensors(self) -> None:
        if not self._config.drop_sensor_uuids:
            return
        sensor_suite = getattr(self._task, "_sensor_suite", None)
        if sensor_suite is None or not hasattr(sensor_suite, "sensors"):
            return
        for uuid in self._config.drop_sensor_uuids:
            sensor_suite.sensors.pop(uuid, None)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._ensure_open()
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        task_id = options["task_id"] if (options is not None and "task_id" in options) else "molmo_0"
        self._episode_id = int(task_id.split("_")[-1])
        episode = self._choose_episode()
        if self._config.reduce_resolution:
            episode = episode.model_copy(update={"img_resolution": tuple(int(e // 2) for e in episode.img_resolution)})
        reuse = self._sampler is not None and self._loaded_episode_id == self._episode_id

        if not reuse:
            self._close_active_episode()
            exp_config = self._make_eval_config()
            exp_config.scene_dataset = episode.scene_dataset
            exp_config.data_split = episode.data_split
            exp_config.task_sampler_config.render_device = self._render_device
            ts_cfg = exp_config.task_sampler_config
            ts_cfg.randomize_init_object_positions = self._config.randomize_init_positions
            ts_cfg.randomize_init_qpos = self._config.randomize_init_positions
            ts_cfg.init_object_position_noise_xy = self._config.init_object_position_noise_xy
            ts_cfg.init_qpos_noise = self._config.init_qpos_noise
            ts_cfg.init_position_randomization_max_attempts = self._config.randomization_max_attempts
            self._prompt_sampler = self._make_prompt_sampler(exp_config)
            with _suppress_molmo_spaces_output():
                self._sampler = JsonEvalTaskSampler(exp_config, episode)
            self._loaded_episode_id = self._episode_id

        self._prompt_sampler.next()
        with _suppress_molmo_spaces_output():
            self._task = self._sampler.sample_task(
                force_advance_scene=False,
                house_index=episode.house_index,
            )
        if self._task is None:
            raise RuntimeError("JsonEvalTaskSampler returned no task.")

        self._prune_unused_sensors()

        if self._task.env.n_batch != 1:
            raise ValueError(
                "MolmoSpacesBenchmarkGymEnv requires n_batch=1, got "
                f"n_batch={self._task.env.n_batch}."
            )

        self._task.register_policy(_RegisteredPolicyAdapter())
        self._task_description = self._prompt_sampler.get_prompt(self._task).lower()

        observations, infos = self._task.reset()
        if not observations:
            raise RuntimeError("Task reset returned empty observations.")
        if not infos:
            raise RuntimeError("Task reset returned empty infos.")
        return observations[0], self._info_with_task_description(infos[0])

    def step(self, action: dict[str, np.ndarray]):
        self._ensure_open()
        if self._task is None:
            raise RuntimeError("No active task. Call reset() before step().")

        observations, rewards, terminated, truncated, infos = self._task.step(action)
        if not observations:
            raise RuntimeError("Task step returned empty observations.")
        if not infos:
            raise RuntimeError("Task step returned empty infos.")

        return (
            observations[0],
            float(rewards[0]),
            bool(infos[0]["success"]),
            bool(truncated[0]),
            self._info_with_task_description(infos[0]),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._close_active_episode()
        self._closed = True


class MolmoActionAdapter(gym.ActionWrapper):
    """Converts OpenPI action vectors into Molmo env action dictionaries."""

    def __init__(
        self,
        env: gym.Env,
    ):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8,),
            dtype=np.float32,
        )

    def action(self, action):
        return {
            "arm": np.asarray(action[:7], dtype=np.float32),
            "gripper": np.asarray([255.0 if action[7] > 0.5 else 0.0], dtype=np.float32),
        }


def make_env_molmo(config, tasks, num_devices: int = 4):
    _silence_molmo_spaces_logs()
    env_config = MolmoSpacesGymConfig()
    benchmark_dir = Path(env_config.benchmark_dir).expanduser().resolve()

    with _suppress_molmo_spaces_output():
        episodes = load_all_episodes(benchmark_dir)
    max_episode_id = len(episodes) - 1
    task_ids = [int(task.split("_")[-1]) for task in tasks]

    if not task_ids:
        raise ValueError("Expected at least one Molmo task.")
    for task_id in task_ids:
        if not 0 <= task_id < len(episodes):
            raise ValueError(
                f"Task ID {task_id} out of range for available episodes 0..{max_episode_id}."
            )

    def env_fn(rank: int):
        env = MolmoSpacesBenchmarkGymEnv(
            episode_id=task_ids[0],
            render_device=rank % num_devices,
            config=env_config,
        )
        env = MolmoActionAdapter(env=env)
        # Converts gym envs to gymnasium style envs
        env = ensure_gymnasium_env(env)
        # Add timelimit wrapper
        env = TimeLimit(
            env,
            max_episode_steps=450,
        )
        return env

    return env_fn
