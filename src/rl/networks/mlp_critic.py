# ruff: noqa: F722
"""MLP critic backbone (alternative to BroNet).

An MLP encoder embeds the concatenated [prefix embedding, state]; an ensemble of
MLP heads maps the embedding (and the action, for Q) to the value output.
Outputs follow the BroNet convention: scalar (n, B) when ``num_bins == 1`` and
categorical logits (n, B, num_bins) when ``num_bins > 1``.
"""
from typing import Callable, Sequence

import flax.nnx as nnx
import jax.numpy as jnp

from src.rl.networks.constants import default_init
from src.rl.networks.rl_networks import ObsType, ActionType, PREFIX_EMBEDDING_NAME

_OBS_KEYS = (PREFIX_EMBEDDING_NAME, "state")


def _obs_vector(observation: ObsType) -> jnp.ndarray:
    return jnp.concatenate([jnp.asarray(observation[k]) for k in _OBS_KEYS], axis=-1)


class MLP(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        activations: Callable[[jnp.ndarray], jnp.ndarray] = nnx.relu,
        activate_final: bool = False,
        use_layer_norm: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.layers = []
        for i, hidden_dim in enumerate(hidden_dims):
            self.layers.append(nnx.Linear(input_dim, hidden_dim, kernel_init=default_init(1.0), rngs=rngs))
            if i < len(hidden_dims) - 1 or activate_final:
                if use_layer_norm:
                    self.layers.append(nnx.LayerNorm(hidden_dim, rngs=rngs))
                self.layers.append(activations)
            input_dim = hidden_dim

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x


class _Head(nnx.Module):
    """One ensemble member: LayerNorm MLP with a scalar or `num_bins`-logit output."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], num_bins: int, *, rngs: nnx.Rngs):
        self.num_bins = num_bins
        self.critic = MLP(input_dim, (*hidden_dims, max(1, num_bins)), use_layer_norm=True, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        out = self.critic(x)
        if self.num_bins <= 1:
            return jnp.squeeze(out, -1)  # (B,)
        return out  # (B, num_bins)


def _make_heads(num_heads: int, input_dim: int, hidden_dims: Sequence[int], num_bins: int, rngs: nnx.Rngs):
    @nnx.split_rngs(splits=num_heads)
    @nnx.vmap(in_axes=0, out_axes=0)
    def create(rngs):
        return _Head(input_dim, hidden_dims, num_bins, rngs=rngs)

    return create(rngs)


def _apply_heads(heads: _Head, x: jnp.ndarray) -> jnp.ndarray:
    @nnx.vmap(in_axes=(0, None), out_axes=0)  # head dim
    def call(head, x):
        return head(x)

    return call(heads, x)


class MLPStateActionCritic(nnx.Module):
    def __init__(
        self,
        observation: ObsType,
        action: ActionType,
        encoder_hidden_dims: Sequence[int],
        decoder_hidden_dims: Sequence[int],
        num_qs: int,
        num_bins: int = 1,
        *,
        rngs: nnx.Rngs,
    ):
        self.encoder = MLP(_obs_vector(observation).shape[-1], encoder_hidden_dims, activate_final=True, rngs=rngs)
        input_dim = encoder_hidden_dims[-1] + action.shape[-1]
        self.heads = _make_heads(num_qs, input_dim, decoder_hidden_dims, num_bins, rngs)

    def __call__(self, observation: ObsType, action: ActionType) -> jnp.ndarray:
        x = jnp.concatenate([self.encoder(_obs_vector(observation)), action], axis=-1)
        return _apply_heads(self.heads, x)  # (num_qs, B) or (num_qs, B, num_bins)


class MLPStateValue(nnx.Module):
    def __init__(
        self,
        observation: ObsType,
        encoder_hidden_dims: Sequence[int],
        decoder_hidden_dims: Sequence[int],
        num_vs: int,
        num_bins: int = 1,
        *,
        rngs: nnx.Rngs,
    ):
        self.encoder = MLP(_obs_vector(observation).shape[-1], encoder_hidden_dims, activate_final=True, rngs=rngs)
        self.heads = _make_heads(num_vs, encoder_hidden_dims[-1], decoder_hidden_dims, num_bins, rngs)

    def __call__(self, observation: ObsType) -> jnp.ndarray:
        return _apply_heads(self.heads, self.encoder(_obs_vector(observation)))  # (num_vs, B) or (num_vs, B, num_bins)
