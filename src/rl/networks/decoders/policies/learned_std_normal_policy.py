from typing import Optional, Sequence

import flax.nnx as nnx
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

# TFP aliases
tfd = tfp.distributions
tfb = tfp.bijectors

from src.rl.networks import MLP
from src.rl.networks.constants import default_init


class LearnedStdNormalPolicyDecoder(nnx.Module):
    def __init__(self,
                 observation: jnp.ndarray | int,
                 action: jnp.ndarray | int,
                 hidden_dims: Sequence[int],
                 dropout_rate: Optional[float] = None,
                 log_std_min: Optional[float] = -20,
                 log_std_max: Optional[float] = 2,
                 output_init_scale: Optional[float] = None,
                 log_std_init: Optional[float] = None,
                 *, rngs: nnx.Rngs):
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        action_dim = action if isinstance(action, int) else action.shape[-1]

        self.mlp = MLP(input=observation,
                       hidden_dims=hidden_dims,
                       activate_final=True,
                       dropout_rate=dropout_rate,
                       rngs=rngs)

        head_init = default_init(output_init_scale) if output_init_scale is not None else default_init(1e-2)
        self.mean_head = nnx.Linear(hidden_dims[-1], action_dim, kernel_init=head_init, rngs=rngs)
        self.log_std_head = nnx.Linear(hidden_dims[-1], action_dim, kernel_init=head_init, rngs=rngs)
        if log_std_init is not None:
            self.log_std_head.bias.value = jnp.full_like(self.log_std_head.bias.value, log_std_init)

    def __call__(self,
                 observations: jnp.ndarray,
                 training: bool = False) -> tfd.Distribution:
        outputs = self.mlp(observations, training=training)

        means = self.mean_head(outputs)

        log_stds = self.log_std_head(outputs)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        # Switched to tfd.MultivariateNormalDiag
        distribution = tfd.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds))
        return distribution


class TanhMultivariateNormalDiag(tfd.TransformedDistribution):
    def __init__(self,
                 loc: jnp.ndarray,
                 scale_diag: jnp.ndarray,
                 low: Optional[jnp.ndarray] = None,
                 high: Optional[jnp.ndarray] = None):
        # 1. Base Distribution
        distribution = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

        # 2. Build Bijector Chain
        # The chain applies operations from the list: y = b[0](b[1](x))
        # Logic: y = Shift(Scale(Tanh(x)))
        bijectors = []

        if low is not None and high is not None:
            # Rescale from (-1, 1) to (low, high)
            scale = (high - low) / 2.0
            shift = (high + low) / 2.0

            # Affine transformation (Applied last)
            bijectors.append(tfb.Shift(shift))
            bijectors.append(tfb.Scale(scale))

        # Tanh transformation (Applied first to the base normal distribution)
        bijectors.append(tfb.Tanh())

        bijector = tfb.Chain(bijectors)

        super().__init__(distribution=distribution, bijector=bijector)

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())


class LearnedStdTanhNormalPolicyDecoder(nnx.Module):
    def __init__(self,
                 observation: jnp.ndarray | int,
                 action: jnp.ndarray | int,
                 hidden_dims: Sequence[int],
                 dropout_rate: Optional[float] = None,
                 log_std_min: Optional[float] = -20,
                 log_std_max: Optional[float] = 2,
                 low: Optional[float] = None,
                 high: Optional[float] = None,
                 output_init_scale: Optional[float] = None,
                 log_std_init: Optional[float] = None,
                 *, rngs: nnx.Rngs):
        action_dim = action if isinstance(action, int) else action.shape[-1]
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.low = low
        self.high = high

        self.mlp = MLP(input=observation,
                       hidden_dims=hidden_dims,
                       activate_final=True,
                       dropout_rate=dropout_rate,
                       rngs=rngs)

        head_init = default_init(output_init_scale) if output_init_scale is not None else default_init(1e-2)
        self.mean_head = nnx.Linear(hidden_dims[-1], action_dim, kernel_init=head_init, rngs=rngs)
        self.log_std_head = nnx.Linear(hidden_dims[-1], action_dim, kernel_init=head_init, rngs=rngs)
        if log_std_init is not None:
            self.log_std_head.bias.value = jnp.full_like(self.log_std_head.bias.value, log_std_init)

    def __call__(self,
                 observations: jnp.ndarray,
                 training: bool = False) -> tfd.Distribution:
        outputs = self.mlp(observations, training=training)

        means = self.mean_head(outputs)

        log_stds = self.log_std_head(outputs)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = TanhMultivariateNormalDiag(loc=means,
                                                  scale_diag=jnp.exp(log_stds),
                                                  low=self.low,
                                                  high=self.high)
        return distribution