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
    from molmo_spaces.evaluation.configs.evaluation_configs import PiPolicyEvalConfig
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
    benchmark_dir: str = (
        "/capstor/store/cscs/swissai/a143/molmospaces/assets/benchmarks/molmospaces-bench-v1/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231"
    )
    eval_config_cls: str = (
        "molmo_spaces.evaluation.configs.evaluation_configs:PiPolicyEvalConfig"
    )
    episode_sampling: Literal["sequential", "random"] = "sequential"
    seed: int = 0


class MolmoSpacesBenchmarkGymEnv(gym.Env):
    """Single-env gym adapter for MolmoSpaces benchmark episodes."""

    metadata = {"render_modes": []}

    def __init__(self, episode_id: int | None = None, render_device: int = 0, config: MolmoSpacesGymConfig = MolmoSpacesGymConfig()):
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
        self._closed = False

        # Minimal placeholder spaces for v1.
        self.observation_space = gym.spaces.Dict({})
        self.action_space = gym.spaces.Dict({})

    def _make_eval_config(self):
        # TODO: simplify
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

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Environment is closed.")

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._ensure_open()
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._close_active_episode()

        episode = self._choose_episode()
        exp_config = self._make_eval_config()
        # Json benchmarks are authoritative; align config scene source with the selected episode.
        # This avoids loading a default scene dataset/split (e.g. procthor-10k/val)
        # for episodes that were generated from another source (e.g. ithor).
        exp_config.scene_dataset = episode.scene_dataset
        exp_config.data_split = episode.data_split
        exp_config.task_sampler_config.render_device = self._render_device

        with _suppress_molmo_spaces_output():
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

        self._task.register_policy(_RegisteredPolicyAdapter())

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
            bool(infos[0]["success"]),
            bool(truncated[0]),
            infos[0],
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

    task_descriptions = []
    task_ids = []

    with _suppress_molmo_spaces_output():
        eval_config = PiPolicyEvalConfig()
    prompt_sampler = PromptSampler(
        task_type=eval_config.task_type,
        prompt_templates=eval_config.policy_config.prompt_templates,
        prompt_object_word_num=eval_config.policy_config.prompt_object_word_num,
    )

    benchmark_dir = str(MolmoSpacesGymConfig().benchmark_dir).strip()
    with _suppress_molmo_spaces_output():
        episodes = load_all_episodes(Path(benchmark_dir).expanduser().resolve())

    for task in tasks:

        task_id = int(task.split("_")[-1])
        task_ids.append(task_id)
        assert task_id in range(len(episodes)), f"Task ID {task_id} out of range for available episodes."

        # TODO: is there a vleaner way to get prompts?
        episode = episodes[task_id]
        with _suppress_molmo_spaces_output():
            ep_config = PiPolicyEvalConfig()
        ep_config.scene_dataset = episode.scene_dataset
        ep_config.data_split = episode.data_split
        with _suppress_molmo_spaces_output():
            sampler = JsonEvalTaskSampler(ep_config, episode)
            _task = sampler.sample_task(force_advance_scene=False, house_index=episode.house_index)
        task_descriptions.append(prompt_sampler.get_prompt(_task).lower())
        sampler.close()

    def env_fn(rank: int):
        task_index = rank % len(task_ids)
        env = MolmoSpacesBenchmarkGymEnv(episode_id=task_ids[task_index], render_device=rank % num_devices)
        env = MolmoActionAdapter(env=env)        
        # Converts gym envs to gymnasium style envs
        env = ensure_gymnasium_env(env)
        # Add timelimit wrapper
        env = TimeLimit(
            env,
            max_episode_steps=450,
        )
        return env

    return env_fn, task_descriptions
