"""SimbaV2 Hyper* building blocks (backbone components).

Ported from scale_rl with the following adaptations:
  - flax.linen → flax.nnx  (eager modules; explicit in_dim parameters required)
  - HyperNormalTanhPolicy omitted (policy head, not used for critic)
  - l2normalize defined locally (no scale_rl dependency)
  - bin_values computed in __call__ rather than stored as a linen buffer
"""
import math

import flax.nnx as nnx
import jax
import jax.numpy as jnp

EPS = 1e-8


def l2normalize(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    return x / jnp.maximum(jnp.linalg.norm(x, axis=axis, keepdims=True), EPS)


class Scaler(nnx.Module):
    """Learned per-element scale with a fixed forward multiplier for init control."""

    def __init__(self, dim: int, init: float = 1.0, scale: float = 1.0):
        self.scaler = nnx.Param(jnp.ones(dim) * scale)
        self.forward_scaler = init / scale

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.scaler.value * self.forward_scaler * x


class HyperDense(nnx.Module):
    """Bias-free linear layer with orthogonal kernel initialisation."""

    def __init__(self, in_dim: int, out_dim: int, *, rngs: nnx.Rngs):
        self.w = nnx.Linear(
            in_features=in_dim,
            out_features=out_dim,
            use_bias=False,
            kernel_init=nnx.initializers.orthogonal(scale=1.0, column_axis=0),
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.w(x)


class HyperMLP(nnx.Module):
    """Two-layer HyperDense MLP with a Scaler and L2-normalised output."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        scaler_init: float,
        scaler_scale: float,
        eps: float = EPS,
        *,
        rngs: nnx.Rngs,
    ):
        self.w1 = HyperDense(in_dim, hidden_dim, rngs=rngs)
        self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale)
        self.w2 = HyperDense(hidden_dim, out_dim, rngs=rngs)
        self.eps = eps

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.w1(x)
        x = self.scaler(x)
        x = nnx.relu(x) + self.eps  # eps prevents zero vector before l2normalize
        x = self.w2(x)
        x = l2normalize(x, axis=-1)
        return x


class HyperEmbedder(nnx.Module):
    """Input embedder: appends c_shift, l2-normalises, projects, scales, l2-normalises."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        scaler_init: float,
        scaler_scale: float,
        c_shift: float,
        *,
        rngs: nnx.Rngs,
    ):
        self.w = HyperDense(in_dim + 1, hidden_dim, rngs=rngs)
        self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale)
        self.c_shift = c_shift

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        new_axis = jnp.ones(x.shape[:-1] + (1,)) * self.c_shift
        x = jnp.concatenate([x, new_axis], axis=-1)
        x = l2normalize(x, axis=-1)
        x = self.w(x)
        x = self.scaler(x)
        x = l2normalize(x, axis=-1)
        return x


class HyperLERPBlock(nnx.Module):
    """Residual LERP block: mlp residual interpolated by a learned alpha Scaler."""

    def __init__(
        self,
        hidden_dim: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
        expansion: int = 4,
        *,
        rngs: nnx.Rngs,
    ):
        self.mlp = HyperMLP(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim * expansion,
            out_dim=hidden_dim,
            scaler_init=scaler_init / math.sqrt(expansion),
            scaler_scale=scaler_scale / math.sqrt(expansion),
            rngs=rngs,
        )
        self.alpha_scaler = Scaler(hidden_dim, alpha_init, alpha_scale)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        x = self.mlp(x)
        x = residual + self.alpha_scaler(x - residual)
        x = l2normalize(x, axis=-1)
        return x


class HyperCategoricalValue(nnx.Module):
    """Distributional value head: projects to num_bins logits, returns expected value."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_bins: int,
        min_v: float,
        max_v: float,
        scaler_init: float,
        scaler_scale: float,
        *,
        rngs: nnx.Rngs,
    ):
        self.w1 = HyperDense(in_dim, hidden_dim, rngs=rngs)
        self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale)
        self.w2 = HyperDense(hidden_dim, num_bins, rngs=rngs)
        self.bias = nnx.Param(jnp.zeros(num_bins))
        self.num_bins = num_bins
        self.min_v = min_v
        self.max_v = max_v

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, dict]:
        value = self.w1(x)
        value = self.scaler(value)
        value = self.w2(value) + self.bias.value

        log_prob = jax.nn.log_softmax(value, axis=-1)
        bin_values = jnp.linspace(self.min_v, self.max_v, self.num_bins)
        expected = jnp.sum(jnp.exp(log_prob) * bin_values, axis=-1)

        return expected, {"log_prob": log_prob}
