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


# The cuSolver `orgqr` fault on sm_120 (Blackwell) that motivated `_orthogonal_cpu`
# is fixed by the jax 0.6.0 / CUDA 12.9 upgrade -- direct GPU orthogonal init is
# verified orthogonal on sm_120. Keep `_orthogonal_cpu` above as a fallback for
# older jaxlib/GPUs.
def default_init(scale: float = math.sqrt(2.0)):
    return jax.nn.initializers.orthogonal(scale)

def xavier_init():
    return jax.nn.initializers.xavier_normal()

def kaiming_init():
    return jax.nn.initializers.kaiming_normal()
