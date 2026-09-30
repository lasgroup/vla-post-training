import math

import jax
import jax.numpy as jnp


def default_init(scale: float = math.sqrt(2.0)):
    """Orthogonal init, computed on CPU to avoid GPU cuSolver."""
    base = jax.nn.initializers.orthogonal(scale)

    def init(key, shape, dtype=jnp.float32):
        with jax.default_device(jax.devices("cpu")[0]):
            return base(key, shape, dtype)

    return init
