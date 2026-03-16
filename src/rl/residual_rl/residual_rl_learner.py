"""Residual RL Learner: SAC policy that learns residual corrections on top of a frozen base policy.

Inherits from DSRLLearner, overriding only the observation normalization
(to concatenate ``base_action`` into the state vector) and action
post-processing (to scale by ``residual_action_scale``).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import jax.numpy as jnp

from src.rl.dsrl.dsrl_learner import DSRLLearner
from src.rl.dsrl.chunk_obs import normalize_observation_for_model
from src.rl.networks.rl_networks import ActionType, ObsType
from src.rl.dsrl.update_actor import PolicyDef
from src.rl.dsrl.update_critic import StateActionCriticDef
from src.training.config import OnlineTrainConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Residual-RL-specific observation normalization
# ---------------------------------------------------------------------------

def _normalize_residual_observation(observations: Any) -> Any:
    """Like normalize_observation_for_model but concatenates base_action into state."""
    normalized = normalize_observation_for_model(observations)
    if not isinstance(normalized, dict):
        return normalized

    if isinstance(observations, dict):
        base_action = observations.get("base_action")
        if base_action is None and isinstance(observations.get("observation"), dict):
            base_action = observations["observation"].get("base_action")
        if base_action is not None:
            ba = jnp.asarray(base_action, dtype=jnp.float32)
            if ba.ndim >= 3:
                ba = ba[:, -1, ...]
            if ba.ndim == 1:
                ba = ba[jnp.newaxis, ...]
            if "state" in normalized:
                normalized["state"] = jnp.concatenate([normalized["state"], ba], axis=-1)
            else:
                normalized["state"] = ba

    return normalized


def _normalize_replay_observation_for_model(obs_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a replay-buffer observation for residual RL.

    Replay observations already have flat ``state`` and ``base_action`` arrays,
    so we concatenate and collapse the temporal dim.
    """
    normalized: Dict[str, Any] = {}
    state = jnp.asarray(obs_dict["state"], dtype=jnp.float32)
    if state.ndim >= 3:
        state = state[:, -1, ...]
    if "base_action" in obs_dict:
        ba = jnp.asarray(obs_dict["base_action"], dtype=jnp.float32)
        if ba.ndim >= 3:
            ba = ba[:, -1, ...]
        state = jnp.concatenate([state, ba], axis=-1)
    normalized["state"] = state

    for img_key in ("image", "wrist_image"):
        if img_key in obs_dict:
            img = jnp.asarray(obs_dict[img_key], dtype=jnp.float32)
            if img.ndim >= 5:
                img = img[:, -1, ...]
            normalized[img_key] = img
    return normalized


def _unwrap_residual_observation(observations: Any) -> Any:
    """Extract model observations from ResidualRLVectorEnv outputs."""
    if (
        isinstance(observations, dict)
        and "observation" in observations
        and isinstance(observations["observation"], dict)
    ):
        unpacked = dict(observations["observation"])
        if "base_action" in observations:
            unpacked["base_action"] = observations["base_action"]
        return unpacked
    return observations


# ---------------------------------------------------------------------------
# ResidualRLLearner
# ---------------------------------------------------------------------------

class ResidualRLLearner(DSRLLearner):

    _ckpt_subdir: str = "residual_rl_state"

    def __init__(
        self,
        config: OnlineTrainConfig,
        dummy_obs: ObsType,
        dummy_act: ActionType,
        state_action_critic_def: StateActionCriticDef,
        policy_def: PolicyDef,
        task_description: str,
    ):
        self._residual_action_scale = float(getattr(config.rl, "residual_action_scale", 1.0))
        super().__init__(
            config=config,
            dummy_obs=dummy_obs,
            dummy_act=dummy_act,
            state_action_critic_def=state_action_critic_def,
            policy_def=policy_def,
            task_description=task_description,
        )
        logger.info("residual_action_scale=%.3f", self._residual_action_scale)

    def _normalize_obs(self, obs: Any) -> Any:
        return _normalize_residual_observation(obs)

    def _normalize_replay_obs(self, obs_dict: Dict[str, Any]) -> Any:
        return _normalize_replay_observation_for_model(obs_dict)

    def _post_process_actions(self, actions: np.ndarray) -> np.ndarray:
        return actions * self._residual_action_scale
