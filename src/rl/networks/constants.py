import math

import jax


import jax.numpy as jnp
import numpy as np


def _orthogonal_cpu(scale: float):
    """Orthogonal init that always runs on the host CPU.

    Orthogonal init needs a QR factorization (cuSolver `orgqr` on GPU), which
    faults on some GPUs -- notably Blackwell / sm_120 (RTX PRO 6000), where it
    raises "cuSolver execution failed" and poisons the CUDA context. The obvious
    guard `with jax.default_device(cpu): base(...)` does NOT work here because
    these initializers run while tracing `jax.jit` (see update_critic.py), and
    `jax.default_device` is ignored under `jit` -- so the QR silently compiled
    onto the GPU and still crashed. `jax.pure_callback` runs the factorization
    eagerly on the host regardless of the surrounding trace, keeping the exact
    orthogonal semantics while never touching GPU cuSolver.
    """
    base = jax.nn.initializers.orthogonal(scale)
    cpu = jax.devices("cpu")[0]

    def init(key, shape, dtype=jnp.float32):
        result_shape = jax.ShapeDtypeStruct(shape, dtype)

        def _host_init(key_):
            with jax.default_device(cpu):
                return np.asarray(base(jnp.asarray(key_), shape, dtype))

        return jax.pure_callback(_host_init, result_shape, key)

    return init


def default_init(scale: float = math.sqrt(2.0)):
    return _orthogonal_cpu(scale)

# def default_init(scale: float = math.sqrt(2.0)):
#     return jax.nn.initializers.orthogonal(scale)

def xavier_init():
    return jax.nn.initializers.xavier_normal()

def kaiming_init():
    return jax.nn.initializers.kaiming_normal()
