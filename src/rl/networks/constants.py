import jax
import jax.numpy as jnp


def default_init(scale: float = jnp.sqrt(2)):
    return jax.nn.initializers.orthogonal(scale)

def xavier_init():
    return jax.nn.initializers.xavier_normal()

def kaiming_init():
    return jax.nn.initializers.kaiming_normal()