from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from typing import Any

import numpy as np

from src.training.config import OnlineTrainConfig


@dataclasses.dataclass(frozen=True)
class DSRLEnvConfig:
    env_num: int
    add_states: bool
    obs_prefix_key: str
    replan_steps: int
    discount: float
    add_per_step_data: bool
    seed: int
    action_deadzone: float


def resolve_dsrl_env_config(config: OnlineTrainConfig) -> DSRLEnvConfig:
    collect = config.collect
    return DSRLEnvConfig(
        env_num=int(collect.env_num),
        add_states=bool(collect.add_states),
        obs_prefix_key=str(collect.obs_prefix_key),
        replan_steps=int(collect.replan_steps),
        discount=float(config.discount),
        add_per_step_data=bool(collect.add_per_step_data),
        seed=int(config.seed),
        # Backwards-compatible fallback for older configs.
        action_deadzone=float(getattr(collect, "action_deadzone", 0.0011)),
    )


def _identity_filter(action: Any) -> Any:
    return action


def _deadzone_filter(action: Any, *, epsilon: float) -> Any:
    return np.where(np.abs(action) < epsilon, 0.0, action)


def build_pre_step_filter(env_config: DSRLEnvConfig) -> Callable[[Any], Any]:
    if env_config.action_deadzone <= 0.0:
        return _identity_filter
    return functools.partial(_deadzone_filter, epsilon=env_config.action_deadzone)
