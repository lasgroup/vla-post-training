from typing import Dict, Optional, Sequence, Union

import flax.nnx as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from src.rl.networks.constants import default_init, xavier_init, kaiming_init

from functools import partial
from typing import Any, Callable, Sequence, Tuple
import distrax

ModuleDef = Any

class Encoder(nn.Module):
    def __init__(self,
                 features: Sequence[int] = (32, 32, 32, 32),
                 strides: Sequence[int] = (2, 1, 1, 1),
                 padding: str = 'VALID',
                 *, rngs: nn.Rngs):
        self.features = features
        self.strides = strides
        self.padding = padding
        
        assert len(features) == len(strides)
        
        self.layers = []
        self.rngs = rngs

    def __call__(self, observations: jnp.ndarray, training=False) -> jnp.ndarray:
        x = observations.astype(jnp.float32) / 255.0
        x = jnp.reshape(x, (*x.shape[:-2], -1))

        if not self.layers:
            in_ch = x.shape[-1]
            for features, stride in zip(self.features, self.strides):
                self.layers.append(nn.Conv(in_ch, features,
                            kernel_size=(3, 3),
                            strides=(stride, stride),
                            kernel_init=default_init(),
                            padding=self.padding,
                            rngs=self.rngs))
                in_ch = features

        for layer in self.layers:
            x = layer(x)
            x = nn.relu(x)

        return x.reshape((*x.shape[:-3], -1))
    

class PixelMultiplexer(nn.Module):
    def __init__(self,
                 encoder: Union[nn.Module, list],
                 network: nn.Module,
                 latent_dim: int,
                 use_bottleneck: bool=True,
                 *, rngs: nn.Rngs):
        self.encoder = encoder
        self.network = network
        self.latent_dim = latent_dim
        self.use_bottleneck = use_bottleneck
        
        self.rngs = rngs
        
        self.bottleneck_dense = None
        self.bottleneck_norm = None

    def __call__(self,
                 observations: Union[FrozenDict, Dict],
                 actions: Optional[jnp.ndarray] = None,
                 training: bool = False):
        observations = FrozenDict(observations)

        x = self.encoder(observations['pixels'], training=training)
        if self.use_bottleneck:
            if self.bottleneck_dense is None:
                self.bottleneck_dense = nn.Linear(x.shape[-1], self.latent_dim, kernel_init=xavier_init(), rngs=self.rngs)
                self.bottleneck_norm = nn.LayerNorm(self.latent_dim, rngs=self.rngs)
                
            x = self.bottleneck_dense(x)
            x = self.bottleneck_norm(x)
            x = nn.tanh(x)

        x = observations.copy(add_or_replace={'pixels': x})

        # print('fully connected keys', x.keys())
        if actions is None:
            return self.network(x, training=training)
        else:
            return self.network(x, actions, training=training)
