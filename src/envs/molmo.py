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
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
from molmo_spaces.evaluation.benchmark_schema import load_all_episodes
from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler
import numpy as np


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
