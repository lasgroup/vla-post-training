# Based on:
# https://github.com/google/flax/blob/main/examples/imagenet/models.py
# and
# https://github.com/google-research/big_transfer/blob/master/bit_jax/models.py
from functools import partial
from typing import Any, Callable, Sequence, Tuple

import flax.nnx as nnx
import jax.numpy as jnp

ModuleDef = Any


class MyGroupNorm(nnx.Module):
    def __init__(self, num_features, num_groups, epsilon=1e-5, dtype=jnp.float32, *, rngs: nnx.Rngs):
        self.gn = nnx.GroupNorm(num_groups=num_groups, epsilon=epsilon, dtype=dtype, rngs=rngs)

    def __call__(self, x, use_running_average: bool = False): # Ignoring use_running_average
        if x.ndim == 3:
            x = x[jnp.newaxis]
            x = self.gn(x)
            return x[0]
        else:
            return self.gn(x)

class ResNetV2Block(nnx.Module):
    """ResNet block."""
    def __init__(self, in_filters: int, filters: int, conv: ModuleDef, norm: ModuleDef, act: Callable, strides: Tuple[int, int] = (1, 1), *, rngs: nnx.Rngs):
        self.filters = filters
        self.act = act
        self.strides = strides
        
        self.norm1 = norm(in_filters, rngs=rngs)
        self.conv1 = conv(in_filters, filters, (3, 3), strides, rngs=rngs)
        self.norm2 = norm(filters, rngs=rngs)
        self.conv2 = conv(filters, filters, (3, 3), rngs=rngs)
        
        if strides != (1, 1) or in_filters != filters:
             self.proj_conv = conv(in_filters, filters, (1, 1), strides, rngs=rngs)
        else:
             self.proj_conv = None

    def __call__(self, x, train: bool = True):
        residual = x
        y = self.norm1(x, use_running_average=not train)
        y = self.act(y)
        y = self.conv1(y)
        y = self.norm2(y, use_running_average=not train)
        y = self.act(y)
        y = self.conv2(y)

        if self.proj_conv is not None:
             residual = self.proj_conv(residual)
        
        return residual + y


class ResNetV2Encoder(nnx.Module):
    """ResNetV2."""
    def __init__(self, stage_sizes: Sequence[int],
                 num_filters: int = 16,
                 dtype: Any = jnp.float32,
                 act: Callable = nnx.relu,
                 norm: str = 'batch',
                 *, rngs: nnx.Rngs):
        self.stage_sizes = stage_sizes
        self.num_filters = num_filters
        self.dtype = dtype
        self.act = act
        self.norm = norm
        
        def conv_factory(*args, **kwargs):
             kwargs.setdefault('use_bias', False)
             kwargs.setdefault('dtype', self.dtype)
             return nnx.Conv(*args, **kwargs)

        if self.norm == 'batch':
            def norm_factory(num_features, *args, **kwargs):
                 kwargs.setdefault('epsilon', 1e-5)
                 kwargs.setdefault('dtype', self.dtype)
                 kwargs.setdefault('momentum', 0.9)
                 return nnx.BatchNorm(num_features, *args, **kwargs)
        elif self.norm == 'groupnorm':
            def norm_factory(num_features, *args, **kwargs):
                 return MyGroupNorm(num_features, num_groups=4, epsilon=1e-5, dtype=self.dtype, *args, **kwargs)
        else:
            raise ValueError('norm not found')

        self.conv_in = None
        self.rngs = rngs
        
        self.blocks = []
        for i, block_size in enumerate(stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                
                if i == 0 and j == 0:
                     filters_in = self.num_filters
                
                filters_out = num_filters * 2**i

                self.blocks.append(
                    ResNetV2Block(filters_in,
                                  filters_out,
                                  strides=strides,
                                  conv=conv_factory,
                                  norm=norm_factory,
                                  act=act,
                                  rngs=rngs)
                )
                filters_in = filters_out
        
        self.norm_out = norm_factory(filters_out, rngs=rngs)

    def __call__(self, x, train: bool = True):
        x = x.astype(jnp.float32) / 255.0
        x = jnp.reshape(x, (*x.shape[:-2], -1))

        if self.conv_in is None:
             self.conv_in = nnx.Conv(x.shape[-1], self.num_filters, (3, 3), use_bias=False, dtype=self.dtype, rngs=self.rngs)

        x = self.conv_in(x)
        
        for block in self.blocks:
             x = block(x, train=train)

        if isinstance(self.norm_out, MyGroupNorm):
             x = self.norm_out(x)
        else:
             x = self.norm_out(x, use_running_average=not train)
             
        x = self.act(x)
        return x.reshape((*x.shape[:-3], -1))
