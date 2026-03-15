"""Value distributions for critic training.

Wraps raw network logits into a distribution with log_prob (for loss) and
mean (for inference/bootstrapping).
  - GaussianValueDistribution  (num_bins=1): MSE-equivalent regression
  - CategoricalValueDistribution (num_bins>1): classification over return bins
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import optax

import openpi.shared.array_typing as at


def _bin_centers(
    lower_bound: float, upper_bound: float, num_bins: int
) -> at.Float[at.Array, " k"]:
    return jnp.linspace(lower_bound, upper_bound, num_bins, dtype=jnp.float32)


def _discretize(
    values: at.Float[at.Array, "..."],
    bin_centers: at.Float[at.Array, " k"],
) -> at.Int[at.Array, "..."]:
    """Map continuous values to the nearest bin index in [0, num_bins-1]."""
    lower = bin_centers[0]
    upper = bin_centers[-1]
    num_bins = bin_centers.shape[0]
    values = jnp.clip(values, lower, upper)
    scaled = (values - lower) * (num_bins - 1) / (upper - lower)
    return jnp.clip(jnp.rint(scaled).astype(jnp.int32), 0, num_bins - 1)


class ValueDistribution:
    """Base class for critic output distributions."""

    def log_prob(self, targets: at.Float[at.Array, "..."]) -> at.Float[at.Array, "..."]:
        raise NotImplementedError

    def mean(self) -> at.Float[at.Array, "..."]:
        raise NotImplementedError


@dataclasses.dataclass
class GaussianValueDistribution(ValueDistribution):
    """Wraps scalar logits as a Gaussian with fixed scale.

    log_prob is gradient-equivalent to MSE. mean() is the identity.
    """

    logits: at.Float[at.Array, "..."]

    def log_prob(self, targets: at.Float[at.Array, "..."]) -> at.Float[at.Array, "..."]:
        return -jnp.square(self.logits - targets)

    def mean(self) -> at.Float[at.Array, "..."]:
        return self.logits


@dataclasses.dataclass
class CategoricalValueDistribution(ValueDistribution):
    """Categorical distribution over uniformly spaced value bins.

    log_prob discretizes continuous targets to the nearest bin and returns
    the negative cross-entropy. mean() returns the expected value under the
    predicted distribution.
    """

    logits: at.Float[at.Array, "... k"]
    bin_centers: at.Float[at.Array, " k"]

    def log_prob(self, targets: at.Float[at.Array, "..."]) -> at.Float[at.Array, "..."]:
        indices = _discretize(targets, self.bin_centers)
        # Broadcast indices to match logits' leading dims (all except the K dim).
        target_shape = self.logits.shape[:-1]
        indices = jnp.broadcast_to(indices, target_shape)
        return -optax.softmax_cross_entropy_with_integer_labels(self.logits, indices)

    def mean(self) -> at.Float[at.Array, "..."]:
        probs = jax.nn.softmax(self.logits, axis=-1)
        return jnp.sum(probs * self.bin_centers, axis=-1)


def get_value_bounds(config) -> tuple[float, float]:
    """Return (lower_bound, upper_bound) for the value function output range.

    Bounds are resolved once at OnlineTrainConfig construction time and stored
    as concrete floats in config.rl. This function simply reads them.
    """
    return float(config.rl.value_lower_bound), float(config.rl.value_upper_bound)


def make_value_distribution(
    logits: at.Float[at.Array, "..."],
    num_value_bins: int = 1,
    value_lower_bound: float = 0.0,
    value_upper_bound: float = 1.0,
) -> ValueDistribution:
    """Create a value distribution from raw network logits.

    Args:
        logits: Raw network outputs. Shape (..., batch) for Gaussian or
            (..., batch, K) for Categorical.
        num_value_bins: Number of bins. 1 = Gaussian (regression), >1 = Categorical.
        value_lower_bound: Lower end of the return range (Categorical only).
        value_upper_bound: Upper end of the return range (Categorical only).
    """
    logits = jnp.asarray(logits, dtype=jnp.float32)
    if num_value_bins <= 1:
        return GaussianValueDistribution(logits=logits)
    centers = _bin_centers(value_lower_bound, value_upper_bound, num_value_bins)
    return CategoricalValueDistribution(logits=logits, bin_centers=centers)
