"""Gymnasium adapter for MolmoSpaces JSON benchmarks.

This module exposes a single-environment gym-like interface over MolmoSpaces
benchmark episodes.

Example:
    from src.molmo.molmospaces_gym_env import MolmoSpacesBenchmarkGymEnv, MolmoSpacesGymConfig
    import numpy as np

    cfg = MolmoSpacesGymConfig(
        benchmark_dir="/path/to/benchmark_dir",
    )
    env = MolmoSpacesBenchmarkGymEnv(cfg)
    obs, info = env.reset()
    action = {"arm": np.zeros(7), "gripper": np.zeros(1)}
    obs, reward, terminated, truncated, info = env.step(action)
    env.close()
"""

import dataclasses
import importlib
import logging
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
from molmo_spaces.evaluation.benchmark_schema import load_all_episodes
from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler
import numpy as np

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class MolmoSpacesGymConfig:
    benchmark_dir: str
    eval_config_cls: str = (
        "molmo_spaces.evaluation.configs.evaluation_configs:PiPolicyEvalConfig"
    )
    episode_sampling: Literal["sequential", "random"] = "sequential"
    seed: int = 0
    task_horizon_steps: int | None = None


class MolmoSpacesBenchmarkGymEnv(gym.Env):
    """Single-env gym adapter for MolmoSpaces benchmark episodes."""

    metadata = {"render_modes": []}

    def __init__(self, config: MolmoSpacesGymConfig):
        super().__init__()
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
        self._registered_policy: Any | None = None
        self._closed = False

        # Minimal placeholder spaces for v1.
        self.observation_space = gym.spaces.Dict({})
        self.action_space = gym.spaces.Dict({})

    def _resolve_eval_config_cls(self):
        spec = self._config.eval_config_cls
        if ":" not in spec:
            raise ValueError(
                f"Invalid eval_config_cls '{spec}'. Expected format "
                "'module.path:ClassName'."
            )
        module_name, class_name = spec.split(":", maxsplit=1)
        module = importlib.import_module(module_name)
        try:
            return getattr(module, class_name)
        except AttributeError as exc:
            raise ValueError(
                f"Could not resolve class '{class_name}' in module '{module_name}'."
            ) from exc

    def _make_eval_config(self):
        eval_config_cls = self._resolve_eval_config_cls()
        exp_config = eval_config_cls()
        if self._config.task_horizon_steps is not None:
            exp_config.task_horizon = self._config.task_horizon_steps
        return exp_config

    def _choose_episode(self):
        if self._config.episode_sampling == "random":
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

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Environment is closed.")

    def register_policy(self, policy: Any | None) -> None:
        """Register a policy object for policy-dependent sensors.

        The object should be task-registerable (i.e., compatible with
        BaseMujocoTask.register_policy) and ideally expose policy-sensor methods
        such as get_phase/get_all_phases/get_info plus a reset method.
        """
        self._registered_policy = policy
        if self._task is not None and policy is not None:
            self._task.register_policy(policy)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._ensure_open()
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if options and "registered_policy" in options:
            self.register_policy(options["registered_policy"])

        self._close_active_episode()

        episode = self._choose_episode()
        exp_config = self._make_eval_config()
        # Json benchmarks are authoritative; align config scene source with the selected episode.
        # This avoids loading a default scene dataset/split (e.g. procthor-10k/val)
        # for episodes that were generated from another source (e.g. ithor).
        exp_config.scene_dataset = episode.scene_dataset
        exp_config.data_split = episode.data_split

        self._sampler = JsonEvalTaskSampler(exp_config, episode)
        self._task = self._sampler.sample_task(
            force_advance_scene=False,
            house_index=episode.house_index,
        )
        if self._task is None:
            raise RuntimeError("JsonEvalTaskSampler returned no task.")

        if self._task.env.n_batch != 1:
            raise ValueError(
                "MolmoSpacesBenchmarkGymEnv requires n_batch=1, got "
                f"n_batch={self._task.env.n_batch}."
            )

        if self._registered_policy is not None:
            self._task.register_policy(self._registered_policy)

        observations, infos = self._task.reset()
        if not observations:
            raise RuntimeError("Task reset returned empty observations.")
        if not infos:
            raise RuntimeError("Task reset returned empty infos.")
        return observations[0], infos[0]

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
            bool(terminated[0]),
            bool(truncated[0]),
            infos[0],
        )

    def close(self) -> None:
        if self._closed:
            return
        self._close_active_episode()
        self._closed = True


class _NoOpRegisteredPolicy:
    """Minimal policy interface for policy-dependent Molmo sensors."""

    def __init__(self, prompt: str) -> None:
        self._prompt = prompt
        self.target_poses = {"grasp": np.eye(4, dtype=np.float32)}
        self.task = None

    def reset(self) -> None:
        return

    def get_prompt(self, default_prompt: str | None = None) -> str:
        return (default_prompt or self._prompt).lower()

    def get_phase(self) -> str:
        return "inference"

    def get_all_phases(self) -> dict[str, int]:
        return {"inference": 0}

    def get_info(self) -> dict[str, Any]:
        return {"policy_name": "openpi-online"}


class MolmoActionAdapter(gym.ActionWrapper):
    """Converts OpenPI action vectors into Molmo env action dictionaries."""

    def __init__(
        self,
        env: gym.Env,
        grasping_type: Literal["continuous", "binary"],
        gripper_threshold: float,
        gripper_scale: float,
    ):
        super().__init__(env)
        self._grasping_type = grasping_type
        self._gripper_threshold = float(gripper_threshold)
        self._gripper_scale = float(gripper_scale)
        self.action_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8,),
            dtype=np.float32,
        )

    def action(self, action):
        if isinstance(action, dict):
            return action

        action = np.asarray(action, dtype=np.float32)
        if action.ndim != 1 or action.shape[0] < 8:
            raise ValueError(
                "MolmoActionAdapter expected a 1D action vector with at least 8 dims, "
                f"got shape {action.shape}."
            )

        if self._grasping_type == "continuous":
            gripper = np.asarray(
                [np.clip(action[7] * self._gripper_scale, 0.0, self._gripper_scale)],
                dtype=np.float32,
            )
        elif self._grasping_type == "binary":
            gripper = np.asarray(
                [self._gripper_scale if action[7] > self._gripper_threshold else 0.0],
                dtype=np.float32,
            )
        else:
            raise ValueError(
                f"Unsupported grasping_type='{self._grasping_type}'. "
                "Expected one of {'continuous', 'binary'}."
            )

        return {
            "arm": np.asarray(action[:7], dtype=np.float32),
            "gripper": gripper,
        }


def _resolve_task_description(config: Any) -> str:
    if getattr(config.molmo, "task_description", None):
        return str(config.molmo.task_description)

    benchmark_dir = str(getattr(config.molmo, "benchmark_dir", "")).strip()
    if not benchmark_dir:
        raise ValueError(
            "Missing required Molmo benchmark directory. "
            "Set `--molmo.benchmark-dir` when `--domain molmo`."
        )

    try:
        episodes = load_all_episodes(Path(benchmark_dir).expanduser().resolve())
        if episodes and episodes[0].language.task_description:
            return str(episodes[0].language.task_description)
    except Exception as exc:
        logger.warning(
            "Could not infer Molmo task_description from benchmark_dir=%s: %s",
            benchmark_dir,
            exc,
        )
    return "do the task"


def make_env_molmo(config, num_devices: int = 4):
    """Build MolmoSpaces env factory for online training."""
    _ = num_devices
    task_description = _resolve_task_description(config)

    base_env_cfg = MolmoSpacesGymConfig(
        benchmark_dir=config.molmo.benchmark_dir,
        eval_config_cls=config.molmo.eval_config_cls,
        episode_sampling=config.molmo.episode_sampling,
        seed=config.seed,
        task_horizon_steps=config.molmo.task_horizon_steps,
    )

    def env_fn(rank: int):
        env_cfg = dataclasses.replace(base_env_cfg, seed=int(config.seed) + int(rank))
        env = MolmoSpacesBenchmarkGymEnv(env_cfg)
        env.register_policy(_NoOpRegisteredPolicy(prompt=task_description))
        env = MolmoActionAdapter(
            env=env,
            grasping_type=config.molmo.grasping_type,
            gripper_threshold=config.molmo.gripper_threshold,
            gripper_scale=config.molmo.gripper_scale,
        )
        return env

    return env_fn, task_description
