"""Compatibility import path for MolmoSpaces gym adapter.

Kept for older scripts that import `src.envs.molmo`.
"""

from src.molmo.molmospaces_gym_env import MolmoSpacesBenchmarkGymEnv
from src.molmo.molmospaces_gym_env import MolmoSpacesGymConfig

__all__ = ["MolmoSpacesBenchmarkGymEnv", "MolmoSpacesGymConfig"]
