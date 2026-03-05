# Based on:
# https://github.com/google/flax/blob/main/examples/imagenet/models.py
# and
# https://github.com/google-research/big_transfer/blob/master/bit_jax/models.py
from functools import partial
from typing import Any, Callable, Sequence, Tuple, List, Union, Dict
from src.rl.networks.encoders.cross_norm import ResNetGroupNorm
from src.rl.networks.encoders.utils import extract_from_dict

import flax.nnx as nnx
from flax.core.frozen_dict import FrozenDict
import jax.numpy as jnp
import jax

CNNDef = Callable[..., nnx.Conv]
NormDef = Callable[..., nnx.Module]
ActivationFn = Callable[[jax.Array], jax.Array]


class ResNetV2Block(nnx.Module):
    """ResNet block."""

    def __init__(self,
                 in_filters: int,
                 filters: int,
                 conv: CNNDef,
                 norm: NormDef,
                 act: ActivationFn,
                 strides: Tuple[int, int] = (1, 1), *, rngs: nnx.Rngs):
        self.filters = filters
        self.act = act
        self.strides = strides

        self.norm1 = norm(num_features=in_filters, rngs=rngs)
        self.conv1 = conv(in_filters, filters, (3, 3), strides, rngs=rngs)
        self.norm2 = norm(num_features=filters, rngs=rngs)
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

    def __init__(self,
                 input_example: Union[FrozenDict, Dict],
                 stage_sizes: Sequence[int],
                 num_filters: int = 16,
                 dtype: Any = jnp.float32,
                 act: ActivationFn = nnx.relu,
                 norm: str = 'batch',
                 *,
                 image_keys: List[str] | None = None,
                 rngs: nnx.Rngs):
        if image_keys is None:
            image_keys = ['pixels']
        self._image_keys = image_keys
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
                return nnx.BatchNorm(num_features=num_features, *args, **kwargs)
        elif self.norm == 'groupnorm':
            def norm_factory(*args, **kwargs):
                return ResNetGroupNorm(num_groups=4, epsilon=1e-5, dtype=self.dtype, *args, **kwargs)
        else:
            raise ValueError('norm not found')

        dummy_x = self._prepare_input(input_example)
        self.conv_in = nnx.Conv(dummy_x.shape[-1],
                               self.num_filters, (3, 3), use_bias=False, dtype=self.dtype,
                               rngs=rngs)
        self.rngs = rngs

        self.blocks = []
        filters_in = self.num_filters
        filters_out = self.num_filters
        for i, block_size in enumerate(stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)

                filters_out = num_filters * 2 ** i

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

    def _prepare_input(self, inputs: Union[FrozenDict, Dict]):
        img_list = [extract_from_dict(inputs, key) for key in self._image_keys]
        img = jnp.stack(img_list, axis=-1)
        img = img.astype(jnp.float32) / 255.0
        img = jnp.reshape(img, (*img.shape[:-2], -1))
        return img

    def __call__(self, x: Union[FrozenDict, Dict], train: bool = True):
        x = self._prepare_input(x)

        x = self.conv_in(x)

        for block in self.blocks:
            x = block(x, train=train)

        if isinstance(self.norm_out, ResNetGroupNorm):
            x = self.norm_out(x)
        else:
            x = self.norm_out(x, use_running_average=not train)

        x = self.act(x)
        return x.reshape((*x.shape[:-3], -1))

ResNetv2_Small = partial(ResNetV2Encoder, stage_sizes=(1, 1, 1, 1))
ResNetv2_18 = partial(ResNetV2Encoder, stage_sizes=(2, 2, 2, 2))
ResNetv2_34 = partial(ResNetV2Encoder, stage_sizes=(3, 4, 6, 3))