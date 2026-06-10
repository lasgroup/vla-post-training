# ruff: noqa: F722
from typing import Callable

import flax.nnx as nnx
import jax.numpy as jnp

from src.rl.networks.constants import default_init
from src.rl.networks.rl_networks import ObsType, ActionType
from src.rl.prefix_embedding import PREFIX_EMBEDDING_NAME


class _BroNetBlock(nnx.Module):
    """One residual block: Linear → LN → act → Linear → LN → skip-add."""

    def __init__(self, hidden_dim: int, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(hidden_dim, hidden_dim, kernel_init=default_init(), rngs=rngs)
        self.norm1 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, kernel_init=default_init(), rngs=rngs)
        self.norm2 = nnx.LayerNorm(hidden_dim, rngs=rngs)

    def __call__(
        self, x: jnp.ndarray, activation: Callable, training: bool = False
    ) -> jnp.ndarray:
        res = self.linear1(x)
        res = self.norm1(res)
        res = activation(res)
        res = self.linear2(res)
        res = self.norm2(res)
        return res + x


class BroNet(nnx.Module):
    """BRONet backbone: input projection followed by `depth` residual blocks.

    Args:
        input:           dummy input array (shape used for input_dim) or an int.
        hidden_dim:      width of every layer.
        depth:           number of residual blocks (1, 2, or 3).
        output_nodes:    output width when add_final_layer=True.
        add_final_layer: append a final Linear(output_nodes) head.
        activations:     element-wise activation applied after each norm.
        rngs:            NNX random-number state.
    """

    def __init__(
        self,
        input: jnp.ndarray | int,
        hidden_dim: int,
        depth: int,
        output_nodes: int = 1,
        add_final_layer: bool = False,
        activations: Callable[[jnp.ndarray], jnp.ndarray] = nnx.relu,
        *,
        rngs: nnx.Rngs,
    ):
        input_dim = input if isinstance(input, int) else int(input.shape[-1])

        self.proj = nnx.Linear(input_dim, hidden_dim, kernel_init=default_init(), rngs=rngs)
        self.proj_norm = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.blocks = [_BroNetBlock(hidden_dim, rngs=rngs) for _ in range(depth)]
        self.activations = activations
        self.add_final_layer = add_final_layer
        if add_final_layer:
            self.final = nnx.Linear(
                hidden_dim, output_nodes, kernel_init=default_init(), rngs=rngs
            )

    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = self.proj(x)
        x = self.proj_norm(x)
        x = self.activations(x)
        for block in self.blocks:
            x = block(x, self.activations, training=training)
        if self.add_final_layer:
            x = self.final(x)
        return x


def _obs_vector_keys(observation: ObsType) -> list[str]:
    """Returns observation keys in concat order: prefix embedding first, then state."""
    keys = []
    if isinstance(observation, dict) and PREFIX_EMBEDDING_NAME in observation:
        keys.append(PREFIX_EMBEDDING_NAME)
    keys.append("state")
    return keys


def _obs_input_dim(observation: ObsType, keys: list[str]) -> int:
    return sum(int(jnp.asarray(observation[k]).shape[-1]) for k in keys)


class BroNetStateActionCritic(nnx.Module):
    """Ensemble of BroNet Q-networks.

    Output mirrors the MLP decoder convention: scalar (num_qs, B) when
    ``num_bins == 1`` (Gaussian/MSE regression) and categorical logits
    (num_qs, B, num_bins) when ``num_bins > 1`` (distributional critic).
    """

    def __init__(
        self,
        observation: ObsType,
        action: ActionType,
        hidden_dim: int,
        depth: int,
        num_qs: int,
        num_bins: int = 1,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_bins = num_bins
        out_dim = max(1, num_bins)
        self._obs_keys = _obs_vector_keys(observation)
        input_dim = (
            _obs_input_dim(observation, self._obs_keys)
            + int(jnp.asarray(action).shape[-1])
        )
        self.nets = [
            BroNet(input_dim, hidden_dim, depth, output_nodes=out_dim, add_final_layer=True, rngs=rngs)
            for _ in range(num_qs)
        ]

    def __call__(
        self, observation: ObsType, action: ActionType, training: bool = False
    ) -> jnp.ndarray:
        parts = [jnp.asarray(observation[k]) for k in self._obs_keys] + [action]
        x = jnp.concatenate(parts, axis=-1)
        outs = [net(x, training=training) for net in self.nets]  # each (B, out_dim)
        if self.num_bins <= 1:
            outs = [o.squeeze(-1) for o in outs]  # (B,)
        return jnp.stack(outs, axis=0)  # (num_qs, B) or (num_qs, B, num_bins)


class BroNetStateValue(nnx.Module):
    """Ensemble of BroNet V-networks.

    Output mirrors the MLP decoder convention: scalar (num_vs, B) when
    ``num_bins == 1`` and categorical logits (num_vs, B, num_bins) when
    ``num_bins > 1``.
    """

    def __init__(
        self,
        observation: ObsType,
        hidden_dim: int,
        depth: int,
        num_vs: int,
        num_bins: int = 1,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_bins = num_bins
        out_dim = max(1, num_bins)
        self._obs_keys = _obs_vector_keys(observation)
        input_dim = _obs_input_dim(observation, self._obs_keys)
        self.nets = [
            BroNet(input_dim, hidden_dim, depth, output_nodes=out_dim, add_final_layer=True, rngs=rngs)
            for _ in range(num_vs)
        ]

    def __call__(self, observation: ObsType, training: bool = False) -> jnp.ndarray:
        parts = [jnp.asarray(observation[k]) for k in self._obs_keys]
        x = jnp.concatenate(parts, axis=-1)
        outs = [net(x, training=training) for net in self.nets]  # each (B, out_dim)
        if self.num_bins <= 1:
            outs = [o.squeeze(-1) for o in outs]  # (B,)
        return jnp.stack(outs, axis=0)  # (num_vs, B) or (num_vs, B, num_bins)
