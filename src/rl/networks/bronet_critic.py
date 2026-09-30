# ruff: noqa: F722
from typing import Callable

import flax.nnx as nnx
import jax.numpy as jnp

from src.rl.networks.constants import default_init
from src.rl.networks.rl_networks import ObsType, ActionType, PREFIX_EMBEDDING_NAME


_OBS_KEYS = (PREFIX_EMBEDDING_NAME, "state")


def _obs_vector(observation: ObsType) -> jnp.ndarray:
    return jnp.concatenate([jnp.asarray(observation[k]) for k in _OBS_KEYS], axis=-1)


class _BroNetBlock(nnx.Module):
    """One residual block: Linear → LN → act → Linear → LN → skip-add."""

    def __init__(self, hidden_dim: int, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(hidden_dim, hidden_dim, kernel_init=default_init(), rngs=rngs)
        self.norm1 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, kernel_init=default_init(), rngs=rngs)
        self.norm2 = nnx.LayerNorm(hidden_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray, activation: Callable) -> jnp.ndarray:
        res = self.linear1(x)
        res = self.norm1(res)
        res = activation(res)
        res = self.linear2(res)
        res = self.norm2(res)
        return res + x


class BroNet(nnx.Module):
    """BRONet: input projection, `depth` residual blocks and a final Linear(output_dim) head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        output_dim: int,
        activations: Callable[[jnp.ndarray], jnp.ndarray] = nnx.relu,
        *,
        rngs: nnx.Rngs,
    ):
        self.proj = nnx.Linear(input_dim, hidden_dim, kernel_init=default_init(), rngs=rngs)
        self.proj_norm = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.blocks = [_BroNetBlock(hidden_dim, rngs=rngs) for _ in range(depth)]
        self.activations = activations
        self.final = nnx.Linear(hidden_dim, output_dim, kernel_init=default_init(), rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.proj(x)
        x = self.proj_norm(x)
        x = self.activations(x)
        for block in self.blocks:
            x = block(x, self.activations)
        return self.final(x)


def _apply_ensemble(nets: list[BroNet], num_bins: int, x: jnp.ndarray) -> jnp.ndarray:
    outs = [net(x) for net in nets]  # each (B, out_dim)
    if num_bins <= 1:
        outs = [o.squeeze(-1) for o in outs]  # (B,)
    return jnp.stack(outs, axis=0)  # (n, B) or (n, B, num_bins)


class BroNetStateActionCritic(nnx.Module):
    """Ensemble of BroNet Q-networks.

    Outputs scalar (num_qs, B) when ``num_bins == 1`` (Gaussian/MSE regression)
    and categorical logits (num_qs, B, num_bins) when ``num_bins > 1``.
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
        input_dim = _obs_vector(observation).shape[-1] + action.shape[-1]
        self.nets = [BroNet(input_dim, hidden_dim, depth, max(1, num_bins), rngs=rngs) for _ in range(num_qs)]

    def __call__(self, observation: ObsType, action: ActionType) -> jnp.ndarray:
        x = jnp.concatenate([_obs_vector(observation), action], axis=-1)
        return _apply_ensemble(self.nets, self.num_bins, x)


class BroNetStateValue(nnx.Module):
    """Ensemble of BroNet V-networks; same output convention as the Q ensemble."""

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
        input_dim = _obs_vector(observation).shape[-1]
        self.nets = [BroNet(input_dim, hidden_dim, depth, max(1, num_bins), rngs=rngs) for _ in range(num_vs)]

    def __call__(self, observation: ObsType) -> jnp.ndarray:
        return _apply_ensemble(self.nets, self.num_bins, _obs_vector(observation))
