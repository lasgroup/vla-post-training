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

    log_prob converts continuous targets to bin probabilities according to target_type:
      - "one_hot":  hard nearest-bin (prone to logit saturation)
      - "two_hot":  linear interpolation between the two bracketing bins

    mean() returns the expected value E[V] = sum(softmax(logits) * bin_centers).
    """

    logits: at.Float[at.Array, "... k"]
    bin_centers: at.Float[at.Array, " k"]
    target_type: str = "one_hot"

    def log_prob(self, targets: at.Float[at.Array, "..."]) -> at.Float[at.Array, "..."]:
        k = self.logits.shape[-1]
        target_shape = self.logits.shape[:-1]
        log_probs = jax.nn.log_softmax(self.logits, axis=-1)

        if self.target_type == "one_hot":
            indices = _discretize(targets, self.bin_centers)
            indices = jnp.broadcast_to(indices, target_shape)
            return -optax.softmax_cross_entropy_with_integer_labels(self.logits, indices)

        elif self.target_type == "two_hot":
            lower = self.bin_centers[0]
            upper = self.bin_centers[-1]
            bin_width = (upper - lower) / (k - 1)
            values = jnp.clip(targets, lower, upper)
            scaled = (values - lower) / bin_width          # float in [0, k-1]
            lower_idx = jnp.clip(jnp.floor(scaled).astype(jnp.int32), 0, k - 2)
            upper_idx = lower_idx + 1
            upper_weight = scaled - lower_idx.astype(jnp.float32)
            lower_weight = 1.0 - upper_weight
            lower_idx = jnp.broadcast_to(lower_idx, target_shape)
            upper_idx = jnp.broadcast_to(upper_idx, target_shape)
            lower_weight = jnp.broadcast_to(lower_weight, target_shape)
            upper_weight = jnp.broadcast_to(upper_weight, target_shape)
            soft = (jax.nn.one_hot(lower_idx, k) * lower_weight[..., jnp.newaxis] +
                    jax.nn.one_hot(upper_idx, k) * upper_weight[..., jnp.newaxis])
            return jnp.sum(soft * log_probs, axis=-1)

        else:
            raise ValueError(f"Unknown target_type: {self.target_type!r}")

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
    target_type: str = "one_hot",
) -> ValueDistribution:
    """Create a value distribution from raw network logits.

    Args:
        logits: Raw network outputs. Shape (..., batch) for Gaussian or
            (..., batch, K) for Categorical.
        num_value_bins: Number of bins. 1 = Gaussian (regression), >1 = Categorical.
        value_lower_bound: Lower end of the return range (Categorical only).
        value_upper_bound: Upper end of the return range (Categorical only).
        target_type: How to convert scalar targets to bin probabilities.
            "one_hot" | "two_hot" (Categorical only).
    """
    logits = jnp.asarray(logits, dtype=jnp.float32)
    if num_value_bins <= 1:
        return GaussianValueDistribution(logits=logits)
    centers = _bin_centers(value_lower_bound, value_upper_bound, num_value_bins)
    return CategoricalValueDistribution(
        logits=logits,
        bin_centers=centers,
        target_type=target_type,
    )
