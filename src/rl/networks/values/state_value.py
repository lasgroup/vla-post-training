from typing import Callable, Sequence

import flax.nnx as nn
import jax.numpy as jnp

from src.rl.networks.mlp import MLP


class StateValue(nn.Module):
    def __init__(self, hidden_dims: Sequence[int],
                 activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
                 *, rngs: nn.Rngs):
        self.critic = MLP((*hidden_dims, 1),
                          activations=activations,
                          rngs=rngs)

    def __call__(self,
                 observations: jnp.ndarray,
                 training: bool = False) -> jnp.ndarray:
        critic = self.critic(observations,
                             training=training)
        return jnp.squeeze(critic, -1)


class StateValueEnsemble(nn.Module):
    def __init__(self, hidden_dims: Sequence[int],
                 activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
                 num_vs: int = 2,
                 *, rngs: nn.Rngs):
        
        self.num_vs = num_vs
        self.vmap_critic = nn.vmap(StateValue,
                                   variable_axes={'params': 0},
                                   split_rngs={'params': True},
                                   in_axes=None,
                                   out_axes=0,
                                   axis_size=num_vs)(hidden_dims, activations=activations, rngs=rngs)

    def __call__(self, observations, training: bool = False):
        qs = self.vmap_critic(observations, training=training)
        return qs
