from typing import Optional, Sequence

import flax.nnx as nn
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

# TFP aliases
tfd = tfp.distributions
tfb = tfp.bijectors

from src.rl.networks import MLP
from src.rl.networks.constants import default_init, xavier_init


class NormalPolicy(nn.Module):
    def __init__(self,
                 observation: jnp.ndarray | int,
                 action: jnp.ndarray | int,
                 hidden_dims: Sequence[int],
                 dropout_rate: Optional[float] = None,
                 std: Optional[float] = 1.,
                 init_scale: Optional[float] = 1.,
                 output_scale: Optional[float] = 1.,
                 init_method: str = 'xavier',
                 *, rngs: nn.Rngs):
        self.std = std
        self.output_scale = output_scale
        action_dim = action if isinstance(action, int) else action.shape[-1]

        self.mlp = MLP(input=observation,
                       hidden_dims=hidden_dims,
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
                 training: bool = False) -> tfd.Distribution:
        outputs = self.mlp(observations, training=training)

        means = self.mean_head(outputs)
        means *= self.output_scale

        # Replaced distrax.MultivariateNormalDiag with tfd.MultivariateNormalDiag
        return tfd.MultivariateNormalDiag(loc=means,
                                          scale_diag=jnp.ones_like(means) * self.std)