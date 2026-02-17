from typing import Optional, Sequence

import distrax
import flax.nnx as nn
import jax.numpy as jnp

from src.rl.networks import MLP
from src.rl.networks.constants import default_init, xavier_init

class LearnedStdNormalPolicy(nn.Module):
    def __init__(self, hidden_dims: Sequence[int],
                 action_dim: int,
                 dropout_rate: Optional[float] = None,
                 log_std_min: Optional[float] = -20,
                 log_std_max: Optional[float] = 2,
                 *, rngs: nn.Rngs):
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_dim = action_dim
        
        self.mlp = MLP(hidden_dims,
                       activate_final=True,
                       dropout_rate=dropout_rate,
                       rngs=rngs)
                       
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim, kernel_init=default_init(1e-2), rngs=rngs)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim, kernel_init=default_init(1e-2), rngs=rngs)

    def __call__(self,
                 observations: jnp.ndarray,
                 training: bool = False) -> distrax.Distribution:
        outputs = self.mlp(observations, training=training)

        means = self.mean_head(outputs)

        log_stds = self.log_std_head(outputs)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds))
        return distribution

class TanhMultivariateNormalDiag(distrax.Transformed):

    def __init__(self,
                 loc: jnp.ndarray,
                 scale_diag: jnp.ndarray,
                 low: Optional[jnp.ndarray] = None,
                 high: Optional[jnp.ndarray] = None):
        distribution = distrax.MultivariateNormalDiag(loc=loc,
                                                      scale_diag=scale_diag)

        layers = []

        if not (low is None or high is None):

            def rescale_from_tanh(x):
                x = (x + 1) / 2  # (-1, 1) => (0, 1)
                return x * (high - low) + low

            def forward_log_det_jacobian(x):
                high_ = jnp.broadcast_to(high, x.shape)
                low_ = jnp.broadcast_to(low, x.shape)
                return jnp.sum(jnp.log(0.5 * (high_ - low_)), -1)

            layers.append(
                distrax.Lambda(
                    rescale_from_tanh,
                    forward_log_det_jacobian=forward_log_det_jacobian,
                    event_ndims_in=1,
                    event_ndims_out=1))

        layers.append(distrax.Block(distrax.Tanh(), 1))

        bijector = distrax.Chain(layers)

        super().__init__(distribution=distribution, bijector=bijector)

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())

class LearnedStdTanhNormalPolicy(nn.Module):
    def __init__(self, hidden_dims: Sequence[int],
                 action_dim: int,
                 dropout_rate: Optional[float] = None,
                 log_std_min: Optional[float] = -20,
                 log_std_max: Optional[float] = 2,
                 low: Optional[float] = None,
                 high: Optional[float] = None,
                 *, rngs: nn.Rngs):
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.low = low
        self.high = high
        self.action_dim = action_dim
        
        self.mlp = MLP(hidden_dims,
                       activate_final=True,
                       dropout_rate=dropout_rate,
                       rngs=rngs)

        self.mean_head = nn.Linear(hidden_dims[-1], action_dim, kernel_init=default_init(1e-2), rngs=rngs)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim, kernel_init=default_init(1e-2), rngs=rngs)

    def __call__(self,
                 observations: jnp.ndarray,
                 training: bool = False) -> distrax.Distribution:
        outputs = self.mlp(observations, training=training)

        means = self.mean_head(outputs)

        log_stds = self.log_std_head(outputs)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = TanhMultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds), low=self.low, high=self.high)
        return distribution