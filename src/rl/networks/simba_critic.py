"""SimbaV2-based critic networks for AWR.

Drop-in replacements for StateActionCritic and StateValue in rl_networks.py.
Same __call__ interface, same inputs — just a different backbone.

Architecture per SimbaV2:
    input → linear embed → N x ResidualBlock(LayerNorm → Linear(expand) → GELU → Linear(contract)) → linear head
"""
from typing import Sequence

import flax.nnx as nnx
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from src.rl.networks.rl_networks import ObsType, ActionType
from src.rl.prefix_embedding import PREFIX_EMBEDDING_NAME


def _prepare_obs_vector(observation: ObsType) -> jnp.ndarray:
    """Concatenate prefix_embedding (if present) and state into a flat vector."""
    if isinstance(observation, (dict, FrozenDict)):
        parts = []
        if PREFIX_EMBEDDING_NAME in observation:
            parts.append(jnp.asarray(observation[PREFIX_EMBEDDING_NAME], dtype=jnp.float32))
        parts.append(jnp.asarray(observation["state"], dtype=jnp.float32))
        return jnp.concatenate(parts, axis=-1)
    return jnp.asarray(observation, dtype=jnp.float32)


class SimbaV2ResidualBlock(nnx.Module):
    """Single SimbaV2 residual block: LayerNorm → Linear(expand) → GELU → Linear(contract) → skip."""

    def __init__(self, hidden_dim: int, expansion_factor: int = 4, *, rngs: nnx.Rngs):
        self.norm = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.linear_expand = nnx.Linear(hidden_dim, hidden_dim * expansion_factor, rngs=rngs)
        self.linear_contract = nnx.Linear(hidden_dim * expansion_factor, hidden_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        x = self.norm(x)
        x = self.linear_expand(x)
        x = nnx.gelu(x)
        x = self.linear_contract(x)
        return x + residual


class SimbaV2Backbone(nnx.Module):
    """SimbaV2 backbone: embed → N residual blocks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        expansion_factor: int = 4,
        *,
        rngs: nnx.Rngs,
    ):
        self.embed = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.blocks = [
            SimbaV2ResidualBlock(hidden_dim, expansion_factor, rngs=rngs)
            for _ in range(num_blocks)
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return x


class SimbaV2StateActionCritic(nnx.Module):
    """SimbaV2 Q-critic.

    Inputs: observation dict (prefix_embedding + state) + action → scalar Q-value.
    Matches the __call__ interface of StateActionCritic.
    """

    def __init__(
        self,
        observation: ObsType,
        action: ActionType,
        hidden_dim: int,
        num_blocks: int,
        num_qs: int = 2,
        expansion_factor: int = 4,
        *,
        rngs: nnx.Rngs,
    ):
        obs_vec = _prepare_obs_vector(observation)
        act_vec = jnp.asarray(action, dtype=jnp.float32).reshape(action.shape[0], -1)
        input_dim = obs_vec.shape[-1] + act_vec.shape[-1]

        # TODO: replace with vmap ensemble (same pattern as StateActionEnsembleDecoder)
        # For now: single critic head as boilerplate
        self.backbone = SimbaV2Backbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            expansion_factor=expansion_factor,
            rngs=rngs,
        )
        self.head = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(
        self, observation: ObsType, action: ActionType, training: bool = False
    ) -> jnp.ndarray:
        obs_vec = _prepare_obs_vector(observation)
        act_vec = jnp.asarray(action, dtype=jnp.float32).reshape(action.shape[0], -1)
        x = jnp.concatenate([obs_vec, act_vec], axis=-1)
        x = self.backbone(x)
        return jnp.squeeze(self.head(x), -1)  # (B,)


class SimbaV2StateValue(nnx.Module):
    """SimbaV2 V-critic.

    Inputs: observation dict (prefix_embedding + state) → scalar V-value.
    Matches the __call__ interface of StateValue.
    """

    def __init__(
        self,
        observation: ObsType,
        hidden_dim: int,
        num_blocks: int,
        num_vs: int = 2,
        expansion_factor: int = 4,
        *,
        rngs: nnx.Rngs,
    ):
        obs_vec = _prepare_obs_vector(observation)
        input_dim = obs_vec.shape[-1]

        # TODO: replace with vmap ensemble
        self.backbone = SimbaV2Backbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            expansion_factor=expansion_factor,
            rngs=rngs,
        )
        self.head = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, observation: ObsType, training: bool = False) -> jnp.ndarray:
        obs_vec = _prepare_obs_vector(observation)
        x = self.backbone(obs_vec)
        return jnp.squeeze(self.head(x), -1)  # (B,)
