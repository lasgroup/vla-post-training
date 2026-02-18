from typing import Callable, Sequence

import flax.nnx as nnx
import jax.numpy as jnp

from src.rl.networks.values.state_action_value import StateActionValue


class StateActionEnsemble(nnx.Module):
    def __init__(self, hidden_dims: Sequence[int],
                 activations: Callable[[jnp.ndarray], jnp.ndarray] = nnx.relu,
                 num_qs: int = 2,
                 use_action_sep: bool = False,
                 *, rngs: nnx.Rngs):
        self.num_qs = num_qs
        
        self.vmap_critic = nnx.vmap(StateActionValue,
                                   variable_axes={'params': 0},
                                   split_rngs={'params': True},
                                   in_axes=None,
                                   out_axes=0,
                                   axis_size=num_qs)(hidden_dims, activations=activations, use_action_sep=use_action_sep, rngs=rngs)

    def __call__(self, states, actions, training: bool = False):
        qs = self.vmap_critic(states, actions, training=training)
        return qs
