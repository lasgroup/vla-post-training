from typing import Callable, Optional, Sequence
import flax.nnx as nnx
import jax.numpy as jnp

from src.rl.networks.constants import default_init


class MLP(nnx.Module):
    def __init__(self,
                 input: jnp.ndarray | int,
                 hidden_dims: Sequence[int],
                 activations: Callable[[jnp.ndarray], jnp.ndarray] = nnx.relu,
                 activate_final: bool = False,  # Fixed type hint
                 dropout_rate: Optional[float] = None,
                 init_scale: Optional[float] = 1.,
                 use_layer_norm: bool = False,
                 *, rngs: nnx.Rngs):

        # Clean input parsing
        input_dim = input if isinstance(input, int) else input.shape[-1]

        self.layers = []

        for i, hidden_dim in enumerate(hidden_dims):
            # 1. Linear Layer
            self.layers.append(nnx.Linear(
                input_dim, hidden_dim,
                kernel_init=default_init(init_scale),
                rngs=rngs
            ))

            # 2. Logic for Non-Linearity Block
            is_last = (i == len(hidden_dims) - 1)
            if not is_last or activate_final:
                if dropout_rate is not None:
                    self.layers.append(nnx.Dropout(dropout_rate, rngs=rngs))
                if use_layer_norm:
                    self.layers.append(nnx.LayerNorm(hidden_dim, rngs=rngs))
                self.layers.append(activations)  # Store function directly

            input_dim = hidden_dim  # Update for next iteration

    def __call__(self, x, training: bool = False):
        assert x.ndim == 2, f"MLP expects (B, D), got shape {x.shape}"

        for layer in self.layers:
            # Handle layers that need the 'training' flag vs simple functions
            if isinstance(layer, nnx.Dropout):
                x = layer(x, deterministic=not training)
            x = layer(x)  # Activations
        return x
