from typing import Callable, Optional, Sequence, Union
from flax.core import frozen_dict

import numpy as np
import flax.nnx as nnx
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from src.rl.networks.constants import default_init

def _flatten_dict(x: Union[FrozenDict, jnp.ndarray]):
    if hasattr(x, 'values'):
        obs = []
        for k, v in sorted(x.items()):
            # if k == "actions":
            #     v = v[:, 0:1, ...]
            if k == 'state': # flatten action chunk to 1D
                obs.append(jnp.reshape(v, [*v.shape[:-2], np.prod(v.shape[-2:])]))
                # v = jnp.reshape(v, [*v.shape[:-2], np.prod(v.shape[-2:])])
            elif k == 'prev_action' or k == 'actions':
                if v.ndim > 2:
                    # deal with action chunk
                    obs.append(jnp.reshape(v, [*v.shape[:-2], np.prod(v.shape[-2:])]))
                else:
                    obs.append(v)
            else:
                obs.append(_flatten_dict(v))
        return jnp.concatenate(obs, -1)
    else:
        return x

def _flatten_dict_special(x):
    if hasattr(x, 'values'):
        obs = []
        action = None
        for k, v in sorted(x.items()):
            if k == 'state' or k == 'prev_action':
                obs.append(jnp.reshape(v, [*v.shape[:-2], np.prod(v.shape[-2:])]))
            elif k == 'actions':
                print ('action shape: ', v.shape)
                action = v
            else:
                obs.append(_flatten_dict(v))
        return jnp.concatenate(obs, -1), action
    else:
        return x
        

class MLP(nnx.Module):
    def __init__(self, hidden_dims: Sequence[int],
                 activations: Callable[[jnp.ndarray], jnp.ndarray] = nnx.relu,
                 activate_final: int = False,
                 dropout_rate: Optional[float] = None,
                 init_scale: Optional[float] = 1.,
                 use_layer_norm: bool = False,
                 *, rngs: nnx.Rngs):
        self.hidden_dims = hidden_dims
        self.activations = activations
        self.activate_final = activate_final
        self.dropout_rate = dropout_rate
        self.init_scale = init_scale
        self.use_layer_norm = use_layer_norm

        self.layers = []
        self.rngs = rngs

    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = _flatten_dict(x)
        # print('mlp post flatten', x.shape)
        
        if not self.layers:
            input_dim = x.shape[-1]
            for i, size in enumerate(self.hidden_dims):
                self.layers.append(nnx.Linear(input_dim, size, kernel_init=default_init(self.init_scale), rngs=self.rngs))
                input_dim = size
                if i + 1 < len(self.hidden_dims) or self.activate_final:
                    if self.dropout_rate is not None:
                        self.layers.append(nnx.Dropout(rate=self.dropout_rate, rngs=self.rngs))
                    if self.use_layer_norm:
                        self.layers.append(nnx.LayerNorm(size, rngs=self.rngs))

        layer_idx = 0
        for i, size in enumerate(self.hidden_dims):
            x = self.layers[layer_idx](x)
            layer_idx += 1
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None:
                     x = self.layers[layer_idx](x, deterministic=not training)
                     layer_idx += 1
                if self.use_layer_norm:
                    x = self.layers[layer_idx](x)
                    layer_idx += 1
                x = self.activations(x)
        return x


class MLPActionSep(nnx.Module):
    def __init__(self, hidden_dims: Sequence[int],
                 activations: Callable[[jnp.ndarray], jnp.ndarray] = nnx.relu,
                 activate_final: int = False,
                 dropout_rate: Optional[float] = None,
                 init_scale: Optional[float] = 1.,
                 use_layer_norm: bool = False,
                 *, rngs: nnx.Rngs):
        self.hidden_dims = hidden_dims
        self.activations = activations
        self.activate_final = activate_final
        self.dropout_rate = dropout_rate
        self.init_scale = init_scale
        self.use_layer_norm = use_layer_norm

        self.layers = []
        self.rngs = rngs

    def __call__(self, x: jnp.ndarray, training: bool = False):
        x, action = _flatten_dict_special(x)
        print ('mlp action sep state post flatten', x.shape)
        print ('mlp action sep action post flatten', action.shape)
        
        if not self.layers:
            input_dim = x.shape[-1] + action.shape[-1]
            for i, size in enumerate(self.hidden_dims):
                self.layers.append(nnx.Linear(input_dim, size, kernel_init=default_init(), rngs=self.rngs))
                input_dim = size
                if i + 1 < len(self.hidden_dims) or self.activate_final:
                    if self.dropout_rate is not None:
                        self.layers.append(nnx.Dropout(rate=self.dropout_rate, rngs=self.rngs))
                    if self.use_layer_norm:
                        self.layers.append(nnx.LayerNorm(size, rngs=self.rngs))
        
        layer_idx = 0
        for i, size in enumerate(self.hidden_dims):
            x_used = jnp.concatenate([x, action], axis=-1)
            x = self.layers[layer_idx](x_used)
            layer_idx += 1
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None:
                    x = self.layers[layer_idx](x, deterministic=not training)
                    layer_idx += 1
                if self.use_layer_norm:
                    x = self.layers[layer_idx](x)
                    layer_idx += 1
                x = self.activations(x)
        return x