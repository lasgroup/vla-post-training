from typing import Callable, Sequence

import flax.nnx as nn
import jax.numpy as jnp

import jax

from src.rl.networks.mlp import MLP, MLPActionSep
from src.rl.networks.constants import default_init

from typing import (Any, Callable, Iterable, List, Optional, Sequence, Tuple,
                    Union)

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any 
Array = Any
PrecisionLike = Union[None, str, jax.lax.Precision, Tuple[str, str],
                      Tuple[jax.lax.Precision, jax.lax.Precision]]


class StateActionValue(nn.Module):
    def __init__(self, hidden_dims: Sequence[int],
                 activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
                 use_action_sep: bool = False,
                 *, rngs: nn.Rngs):
        self.use_action_sep = use_action_sep

        if use_action_sep:
            self.critic = MLPActionSep(
                (*hidden_dims, 1),
                activations=activations,
                use_layer_norm=True,
                rngs=rngs)
        else:
            self.critic = MLP((*hidden_dims, 1),
                        activations=activations,
                        use_layer_norm=True,
                        rngs=rngs)

    def __call__(self,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 training: bool = False):
        inputs = {'states': observations, 'actions': actions}
        critic = self.critic(inputs, training=training)
        return jnp.squeeze(critic, -1)
