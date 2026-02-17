from typing import Callable, Sequence

import flax.nnx as nn
import jax.numpy as jnp

import jax

from src.rl.networks.mlp import MLP

from typing import (Any, Callable, Sequence, Tuple,
                    Union)

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any
PrecisionLike = Union[None, str, jax.lax.Precision, Tuple[str, str],
Tuple[jax.lax.Precision, jax.lax.Precision]]


class StateActionValue(nn.Module):
    def __init__(self,
                 observation: jnp.ndarray | int,
                 action: jnp.ndarray | int,
                 hidden_dims: Sequence[int],
                 activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
                 *,
                 rngs: nn.Rngs):
        self.critic = MLP(input=self._prepare_inputs(observations=observation, actions=action),
                          hidden_dims=(*hidden_dims, 1),
                          activations=activations,
                          use_layer_norm=True,
                          rngs=rngs)

    @staticmethod
    def _prepare_inputs(observations: jnp.ndarray, actions: jnp.ndarray):
        input = jnp.concatenate([observations, actions], axis=-1)
        return input

    def __call__(self,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 training: bool = False):
        input = self._prepare_inputs(observations=observations, actions=actions)
        critic = self.critic(input, training=training)
        return jnp.squeeze(critic, -1)


class StateActionEnsemble(nn.Module):
    def __init__(self,
                 observation: jnp.ndarray | int,
                 action: jnp.ndarray | int,
                 hidden_dims: Sequence[int],
                 activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
                 num_qs: int = 2,
                 *, rngs: nn.Rngs):

        @nn.split_rngs(splits=num_qs)
        @nn.vmap(out_axes=0, in_axes=0)
        def create_critic(rgs):
            return StateActionValue(
                observation=observation,
                action=action,
                hidden_dims=hidden_dims,
                activations=activations,
                rngs=rgs,  # Wrap the key back into Rngs
            )
        self.vmap_critic = create_critic(rngs)

    def __call__(self, observations, actions, training: bool = False):
        # Attempt to use the model
        @nn.vmap(in_axes=(0, None, None), out_axes=0)  # head dim
        def call_model(model, o, a):
            return model(o, a, training=training)
        return call_model(self.vmap_critic, observations, actions)