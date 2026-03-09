import math

import jax


import jax.numpy as jnp
### Change back after gpu fix
def _orthogonal_cpu(scale: float):
    base = jax.nn.initializers.orthogonal(scale)

    def init(key, shape, dtype=jnp.float32):
        # Keep orthogonal init exactly, but avoid GPU cuSolver.
        cpu = jax.devices("cpu")[0]
        with jax.default_device(cpu):
            return base(key, shape, dtype)

    return init


def default_init(scale: float = math.sqrt(2.0)):
    return _orthogonal_cpu(scale)

# def default_init(scale: float = math.sqrt(2.0)):
#     return jax.nn.initializers.orthogonal(scale)

def xavier_init():
    return jax.nn.initializers.xavier_normal()

def kaiming_init():
    return jax.nn.initializers.kaiming_normal()
