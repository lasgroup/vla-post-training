import math

import jax


import jax.numpy as jnp

def default_init(scale: float = math.sqrt(2.0)):
    # lecun_normal avoids jnp.linalg.qr (cuSolver) used by orthogonal init,
    # which crashes on GH200/Hopper with JAX 0.5.3.
    return jax.nn.initializers.lecun_normal()

def xavier_init():
    return jax.nn.initializers.xavier_normal()

def kaiming_init():
    return jax.nn.initializers.kaiming_normal()
