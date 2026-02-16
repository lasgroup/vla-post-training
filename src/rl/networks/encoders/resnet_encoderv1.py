import flax.nnx as nn
import jax.numpy as jnp

from functools import partial
from typing import Any, Callable, Sequence, Tuple
from src.rl.networks.constants import default_init, xavier_init, kaiming_init
from src.rl.networks.encoders.spatial_softmax import SpatialSoftmax
from src.rl.networks.encoders.cross_norm import CrossNorm

ModuleDef = Any


class MyGroupNorm(nn.Module):
    def __init__(self, num_groups, epsilon=1e-5, dtype=jnp.float32, *, rngs: nn.Rngs):
        self.gn = nn.GroupNorm(num_groups=num_groups, epsilon=epsilon, dtype=dtype, rngs=rngs)

    def __call__(self, x, use_running_average: bool = False):
        if x.ndim == 3:
            x = x[jnp.newaxis]
            x = self.gn(x)
            return x[0]
        else:
            return self.gn(x)

class ResNetBlock(nn.Module):
    """ResNet block."""
    def __init__(self, in_filters: int, filters: int, conv: ModuleDef, norm: ModuleDef, act: Callable, strides: Tuple[int, int] = (1, 1), *, rngs: nn.Rngs):
        self.filters = filters
        self.act = act
        self.strides = strides
        
        self.conv1 = conv(in_filters, filters, (3, 3), strides, rngs=rngs)
        self.norm1 = norm(rngs=rngs)
        self.conv2 = conv(filters, filters, (3, 3), rngs=rngs)
        self.norm2 = norm(rngs=rngs)

        if strides != (1, 1) or in_filters != filters:
             self.proj_conv = conv(in_filters, filters, (1, 1), strides, rngs=rngs)
             self.proj_norm = norm(rngs=rngs)
        else:
             self.proj_conv = None
             self.proj_norm = None

    def __call__(self, x, train: bool = True):
        residual = x
        y = self.conv1(x)
        y = self.norm1(y, use_running_average=not train)
        y = self.act(y)
        y = self.conv2(y)
        y = self.norm2(y, use_running_average=not train)

        if self.proj_conv is not None:
            residual = self.proj_conv(residual)
            residual = self.proj_norm(residual, use_running_average=not train)

        return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
    """Bottleneck ResNet block."""
    def __init__(self, in_filters: int, filters: int, conv: ModuleDef, norm: ModuleDef, act: Callable, strides: Tuple[int, int] = (1, 1), *, rngs: nn.Rngs):
        self.filters = filters
        self.act = act
        self.strides = strides
        
        self.conv1 = conv(in_filters, filters, (1, 1), rngs=rngs)
        self.norm1 = norm(rngs=rngs)
        self.conv2 = conv(filters, filters, (3, 3), strides, rngs=rngs)
        self.norm2 = norm(rngs=rngs)
        self.conv3 = conv(filters, filters * 4, (1, 1), rngs=rngs)
        self.norm3 = norm(scale_init=nn.initializers.zeros, rngs=rngs)

        if strides != (1, 1) or in_filters != filters * 4:
             self.proj_conv = conv(in_filters, filters * 4, (1, 1), strides, rngs=rngs)
             self.proj_norm = norm(rngs=rngs)
        else:
             self.proj_conv = None
             self.proj_norm = None

    def __call__(self, x, train: bool = True):
        residual = x
        y = self.conv1(x)
        y = self.norm1(y, use_running_average=not train)
        y = self.act(y)
        y = self.conv2(y)
        y = self.norm2(y, use_running_average=not train)
        y = self.act(y)
        y = self.conv3(y)
        y = self.norm3(y, use_running_average=not train)

        if self.proj_conv is not None:
             residual = self.proj_conv(residual)
             residual = self.proj_norm(residual, use_running_average=not train)

        return self.act(residual + y)


class ResNetEncoder(nn.Module):
    """ResNetV1."""
    def __init__(self, stage_sizes: Sequence[int],
                 block_cls: ModuleDef,
                 num_filters: int = 64,
                 dtype: Any = jnp.float32,
                 act: Callable = nn.relu,
                 conv: ModuleDef = nn.Conv,
                 norm: str = 'batch',
                 use_spatial_softmax: bool = True,
                 softmax_temperature: float = 1.0,
                 *, rngs: nn.Rngs):
        self.stage_sizes = stage_sizes
        self.block_cls = block_cls
        self.num_filters = num_filters
        self.dtype = dtype
        self.act = act
        self.conv = conv
        self.norm = norm
        self.use_spatial_softmax = use_spatial_softmax
        self.softmax_temperature = softmax_temperature

        def conv_factory(*args, **kwargs):
             kwargs.setdefault('use_bias', False)
             kwargs.setdefault('dtype', self.dtype)
             kwargs.setdefault('kernel_init', kaiming_init())
             return self.conv(*args, **kwargs)

        if self.norm == 'batch':
            def norm_factory(*args, **kwargs):
                 kwargs.setdefault('epsilon', 1e-5)
                 kwargs.setdefault('dtype', self.dtype)
                 kwargs.setdefault('momentum', 0.9)
                 return nn.BatchNorm(*args, **kwargs)
        elif self.norm == 'group':
            def norm_factory(*args, **kwargs):
                 return MyGroupNorm(num_groups=4, epsilon=1e-5, dtype=self.dtype, *args, **kwargs)
        elif self.norm == 'cross':
              def norm_factory(*args, **kwargs):
                   return CrossNorm(*args, **kwargs)
        elif self.norm == 'layer':
             def norm_factory(*args, **kwargs):
                  kwargs.setdefault('epsilon', 1e-5)
                  kwargs.setdefault('dtype', self.dtype)
                  return nn.LayerNorm(*args, **kwargs)
        else:
            raise ValueError('norm not found')
            
        
        self.conv_item = None
        self.rngs = rngs
        self.norm_item = norm_factory(rngs=rngs)
        
        self.blocks = []
        strides = (2, 2, 2, 1, 1)
        # block loop
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                stride = (strides[i + 1], strides[i + 1]) if i > 0 and j == 0 else (1, 1)
                
                # Determine in_filters.
                # If j==0, it's start of stage.
                # If i==0, in_filters = num_filters (output of conv_item)
                # If i>0, in_filters = output of previous stage.
                # Output of previous stage:
                # If block_cls is ResNetBlock: num_filters * 2**(i-1)
                # If block_cls is BottleneckResNetBlock: num_filters * 2**(i-1) * 4
                
                if i == 0 and j == 0:
                     filters_in = self.num_filters
                
                # Output filters of CURRENT block (before expansion if bottleneck)
                filters_out = self.num_filters * 2 ** i

                self.blocks.append(
                    self.block_cls(filters_in,
                                   filters_out,
                                   strides=stride,
                                   conv=conv_factory,
                                   norm=norm_factory,
                                   act=self.act,
                                   rngs=rngs)
                )
                
                # Update filters_in for next block
                if self.block_cls == BottleneckResNetBlock:
                     filters_in = filters_out * 4
                else:
                     filters_in = filters_out

        if self.use_spatial_softmax:
             self.spatial_softmax = None
             self.rngs = rngs
        else:
             self.spatial_softmax = None


    def __call__(self, observations: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        
        x = observations.astype(jnp.float32) / 255.0
        x = jnp.reshape(x, (*x.shape[:-2], -1))

        # Initial layers
        if self.conv_item is None:
             self.conv_item = self.conv(x.shape[-1], self.num_filters, (7, 7), (2, 2), padding=[(3, 3), (3, 3)], 
                                        use_bias=False, dtype=self.dtype, kernel_init=kaiming_init(), rngs=self.rngs)

        x = self.conv_item(x)
        
        if isinstance(self.norm_item, (nn.BatchNorm, CrossNorm)):
             x = self.norm_item(x, use_running_average=not train)
        else:
             x = self.norm_item(x)
             
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')

        for block in self.blocks:
             x = block(x, train=train)

        if self.use_spatial_softmax:
            if self.spatial_softmax is None:
                 height, width, channel = x.shape[len(x.shape) - 3:]
                 pos_x, pos_y = jnp.meshgrid(
                    jnp.linspace(-1., 1., height),
                    jnp.linspace(-1., 1., width)
                 )
                 pos_x = pos_x.reshape(height * width)
                 pos_y = pos_y.reshape(height * width)
                 self.spatial_softmax = SpatialSoftmax(height, width, channel, pos_x, pos_y, self.softmax_temperature, rngs=self.rngs)
            
            x = self.spatial_softmax(x)
        else:
            x = jnp.mean(x, axis=(len(x.shape) - 3,len(x.shape) - 2))
        return x

ResNetSmall = partial(ResNetEncoder, stage_sizes=(1, 1, 1, 1),
                   block_cls=ResNetBlock)
ResNet18 = partial(ResNetEncoder, stage_sizes=(2, 2, 2, 2),
                   block_cls=ResNetBlock)
ResNet34 = partial(ResNetEncoder, stage_sizes=(3, 4, 6, 3),
                   block_cls=ResNetBlock)
