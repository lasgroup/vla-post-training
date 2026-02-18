from typing import Dict, List, Union, Callable

import flax.nnx as nn
import jax.numpy as jnp
import jax
from flax.core.frozen_dict import FrozenDict

from src.rl.networks.constants import xavier_init
from src.rl.networks.mlp import MLP
from src.rl.networks.encoders.utils import extract_from_dict, EncoderType

EncoderDef = Callable[[Union[FrozenDict, Dict]], EncoderType]
MLPEncoderDef = Callable[[Union[FrozenDict, Dict]], EncoderType]
MLPDef = Callable[[jnp.ndarray], MLP]


class ImageEncoder(nn.Module):
    def __init__(self,
                 dummy_obs: Union[FrozenDict, Dict],
                 encoder_def: EncoderDef,
                 latent_dim: int,
                 use_bottleneck: bool = True,
                 *, rngs: nn.Rngs):
        # Initialize encoder
        self.encoder = encoder_def(dummy_obs)
        # Get a dummy embedding
        dummy_x = self.encoder(dummy_obs, False)
        self.use_bottleneck = use_bottleneck
        self.bottleneck_dense = None
        self.bottleneck_norm = None
        if self.use_bottleneck:
            self.bottleneck_dense = nn.Linear(dummy_x.shape[-1],
                                              latent_dim,
                                              kernel_init=xavier_init(),
                                              rngs=rngs)
            self.bottleneck_norm = nn.LayerNorm(latent_dim, rngs=rngs)

    def __call__(self,
                 observations: Union[FrozenDict, Dict],
                 training: bool = False):
        x = self.encoder(observations, training)

        if self.use_bottleneck:
            x = self.bottleneck_dense(x)
            x = self.bottleneck_norm(x)
            x = nn.tanh(x)
        return x


class MLPEncoder(nn.Module):
    def __init__(self,
                 dummy_obs: Union[FrozenDict, Dict],
                 encoder_def: MLPDef,
                 state_vector_keys: List[str] | None = None,):
        if state_vector_keys is None:
            state_vector_keys = ['state']
        self._state_vector_keys = state_vector_keys
        # Get a dummy state
        dummy_state = self._prepare_inputs(dummy_obs)
        # Initialize encoder
        self.encoder = encoder_def(dummy_state)

    def _prepare_inputs(self, observations: Union[FrozenDict, Dict]):
        observations = FrozenDict(observations)
        # 1. Stack the images
        # Resulting Shape: (Batch, Num_images, H, W, C)
        # We concatenate all vectors
        state_list = [extract_from_dict(observations, key) for key in self._state_vector_keys]
        state = jnp.concatenate(state_list, axis=-1)
        return state

    def __call__(self,
                 observations: Union[FrozenDict, Dict],
                 training: bool = False):
        state = self._prepare_inputs(observations)
        return self.encoder(state, training=training)


class BaseEncoder(nn.Module):
    def __init__(self,
                 dummy_obs: Union[FrozenDict, Dict, jnp.ndarray],
                 mlp_encoder_def: EncoderDef | None = None,
                 image_encoder_def: MLPEncoderDef | None = None,
                 ):
        if mlp_encoder_def is not None:
            self.mlp_encoder = mlp_encoder_def(dummy_obs)
        else:
            self.mlp_encoder = None
        if image_encoder_def is not None:
            self.image_encoder = image_encoder_def(dummy_obs)
        else:
            self.image_encoder = None

    def __call__(self,
                 observations: Union[FrozenDict, Dict, jnp.ndarray],
                 training: bool = False
                 ):
        x = []
        if self.mlp_encoder:
            x.append(self.mlp_encoder(observations=observations, training=training))
        if self.image_encoder:
            x.append(self.image_encoder(observations=observations, training=training))
        if len(x) > 0:
            x = jnp.concatenate(x, axis=-1)
            return x
        else:
            # If no mlp and image encoder is provided just concatenate all elements
            def flatten_and_concat(pytree):
                leaves = jax.tree_util.tree_leaves(pytree)
                flat_leaves = []
                for x in leaves:
                    # Assumes axis 0 is the batch dimension.
                    # Reshape to (Batch, -1) effectively flattens all other dimensions.
                    batch_size = x.shape[0]
                    flat_leaves.append(x.reshape(batch_size, -1))

                # Concatenate along the feature dimension (axis 1)
                return jnp.concatenate(flat_leaves, axis=-1)

            x = flatten_and_concat(observations)
            return x
