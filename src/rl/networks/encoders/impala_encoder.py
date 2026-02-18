import flax.nnx as nn
from flax.core.frozen_dict import FrozenDict
import jax.numpy as jnp
from src.rl.networks.constants import xavier_init
from src.rl.networks.encoders.utils import extract_from_dict
from typing import Sequence, List, Union, Dict


class ResnetBlock(nn.Module):
    def __init__(self,
                 num_ch,
                 *,
                 rngs: nn.Rngs):
        self.conv1 = nn.Conv(num_ch, num_ch,
                             kernel_size=(3, 3), strides=1, padding='SAME', kernel_init=xavier_init(), rngs=rngs)
        self.conv2 = nn.Conv(num_ch, num_ch,
                             kernel_size=(3, 3), strides=1, padding='SAME', kernel_init=xavier_init(), rngs=rngs)

    def __call__(self, x):
        inputs = x
        x = nn.relu(x)
        x = self.conv1(x)
        x = nn.relu(x)
        x = self.conv2(x)
        return x + inputs


class ResnetStack(nn.Module):
    def __init__(self, in_ch: int, num_ch: int, num_blocks: int, use_max_pooling: bool = True, *, rngs: nn.Rngs):
        self.in_ch = in_ch
        self.num_ch = num_ch
        self.num_blocks = num_blocks
        self.use_max_pooling = use_max_pooling

        initializer = xavier_init()
        self.conv_in = nn.Conv(
            in_features=in_ch,
            out_features=num_ch,
            kernel_size=(3, 3),
            strides=1,
            kernel_init=initializer,
            padding='SAME',
            rngs=rngs,
        )

        self.blocks = []
        for _ in range(num_blocks):
            self.blocks.append(ResnetBlock(num_ch, rngs=rngs))

    def __call__(self, x):
        x = self.conv_in(x)
        if self.use_max_pooling:
            x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')

        for block in self.blocks:
            x = block(x)
        return x


class ImpalaEncoder(nn.Module):
    def __init__(self,
                 input_example: Union[FrozenDict, Dict],
                 nn_scale: int = 1,
                 stack_sizes: Sequence[int] = (16, 32, 32),
                 num_blocks_list: Sequence[int] = (2, 2, 2),
                 image_keys: List[str] | None = None,
                 *, rngs: nn.Rngs):
        if image_keys is None:
            image_keys = ['pixels']
        self._image_keys = image_keys
        dummy_x = self._prepare_input(input_example)
        self.stack_blocks = []
        # Initialize stacks using inferred input shape
        in_ch = dummy_x.shape[-1]

        self.stack_blocks = []
        for i, sz in enumerate(stack_sizes):
            out_ch = sz * nn_scale
            self.stack_blocks.append(
                ResnetStack(in_ch=in_ch, num_ch=out_ch, num_blocks=num_blocks_list[i], rngs=rngs)
            )
            in_ch = out_ch

    def _prepare_input(self, inputs: Union[FrozenDict, Dict]):
        img_list = [extract_from_dict(inputs, key) for key in self._image_keys]
        img = jnp.stack(img_list, axis=-1)
        img = img.astype(jnp.float32) / 255.0
        img = jnp.reshape(img, (*img.shape[:-2], -1))
        return img

    def __call__(self, x: Union[FrozenDict, Dict], train=True):
        x = self._prepare_input(x)
        conv_out = x

        for block in self.stack_blocks:
            conv_out = block(conv_out)

        conv_out = nn.relu(conv_out)
        return conv_out.reshape((*x.shape[:-3], -1))


class SmallerImpalaEncoder(ImpalaEncoder):
    def __init__(self,
                 input_example: Union[FrozenDict, Dict],
                 nn_scale: int = 1, *, rngs: nn.Rngs,
                 image_keys: List[str] | None = None,
                 ):
        super().__init__(
            input_example=input_example,
            nn_scale=nn_scale,
            num_blocks_list=(2, 1, 1),
            rngs=rngs,
            image_keys=image_keys,
        )
