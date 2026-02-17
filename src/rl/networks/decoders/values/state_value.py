from typing import Callable, Sequence

import flax.nnx as nn
import jax.numpy as jnp

from src.rl.networks.mlp import MLP


class StateValue(nn.Module):
    def __init__(self,
                 observation: jnp.ndarray | int,
                 hidden_dims: Sequence[int],
                 activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
                 *, rngs: nn.Rngs):
        self.critic = MLP(input=observation,
                          hidden_dims=(*hidden_dims, 1),
                          activations=activations,
                          use_layer_norm=True,
                          rngs=rngs)

    def __call__(self,
                 observations: jnp.ndarray,
                 training: bool = False) -> jnp.ndarray:
        critic = self.critic(observations,
                             training=training)
        return jnp.squeeze(critic, -1)


class StateValueEnsemble(nn.Module):
    def __init__(self,
                 observation: jnp.ndarray | int,
                 hidden_dims: Sequence[int],
                 activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
                 num_vs: int = 2,
                 *, rngs: nn.Rngs):

        @nn.split_rngs(splits=num_vs)
        @nn.vmap(out_axes=0, in_axes=0)
        def create_critic(rgs):
            return StateValue(
                observation=observation,
                hidden_dims=hidden_dims,
                activations=activations,
                rngs=rgs,  # Wrap the key back into Rngs
            )
        self.vmap_critic = create_critic(rngs)

    def __call__(self, observations, training: bool = False):
        # Attempt to use the model
        @nn.vmap(in_axes=(0, None), out_axes=0)  # head dim
        def call_model(model, s):
            return model(s, training=training)
        return call_model(self.vmap_critic, observations)
