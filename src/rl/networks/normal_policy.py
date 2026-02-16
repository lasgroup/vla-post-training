from typing import Optional, Sequence

import distrax
import flax.nnx as nn
import jax.numpy as jnp

from src.rl.networks import MLP
from src.rl.networks.constants import default_init, xavier_init


class NormalPolicy(nn.Module):
    def __init__(self, hidden_dims: Sequence[int],
                 action_dim: int,
                 dropout_rate: Optional[float] = None,
                 std: Optional[float] = 1.,
                 init_scale: Optional[float] = 1.,
                 output_scale: Optional[float] = 1.,
                 init_method: str = 'xavier',
                 *, rngs: nn.Rngs):
        self.std = std
        self.output_scale = output_scale
        
        self.mlp = MLP(hidden_dims,
                       activate_final=True,
                       dropout_rate=dropout_rate,
                       init_scale=init_scale,
                       rngs=rngs)

        if init_method == 'xavier':
            kernel_init = xavier_init()
        else:
            kernel_init = default_init(init_scale)
            
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim, kernel_init=kernel_init, rngs=rngs)

    def __call__(self,
                 observations: jnp.ndarray,
                 training: bool = False) -> distrax.Distribution:
        outputs = self.mlp(observations, training=training)

        means = self.mean_head(outputs)
        means *= self.output_scale

        return distrax.MultivariateNormalDiag(loc=means,
                                              scale_diag=jnp.ones_like(means)*self.std)
