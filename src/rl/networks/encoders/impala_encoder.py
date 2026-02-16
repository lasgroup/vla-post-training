import flax.nnx as nn
import jax.numpy as jnp
from src.rl.networks.constants import xavier_init


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


class ImpalaEncoder(nn.Module):
    def __init__(self, nn_scale: int = 1, *, rngs: nn.Rngs):
        self.nn_scale = nn_scale
        self.stack_blocks = []
        self.rngs = rngs 

    def __call__(self, x, train=True):
        x = x.astype(jnp.float32) / 255.0
        x = jnp.reshape(x, (*x.shape[:-2], -1))

        if not self.stack_blocks:
             # Initialize stacks using inferred input shape
             
             stack_sizes = [16, 32, 32]
             num_blocks_list = [2, 2, 2]
             in_ch = x.shape[-1]
             
             self.stack_blocks = []
             for i, sz in enumerate(stack_sizes):
                 out_ch = sz * self.nn_scale
                 self.stack_blocks.append(
                     ResnetStack(in_ch=in_ch, num_ch=out_ch, num_blocks=num_blocks_list[i], rngs=self.rngs)
                 )
                 in_ch = out_ch

        conv_out = x

        for block in self.stack_blocks:
            conv_out = block(conv_out)

        conv_out = nn.relu(conv_out)
        return conv_out.reshape((*x.shape[:-3], -1))


class SmallerImpalaEncoder(nn.Module):
    def __init__(self, nn_scale: int = 1, *, rngs: nn.Rngs):
        self.nn_scale = nn_scale
        self.stack_blocks = []
        self.rngs = rngs

    def __call__(self, x, train=True):
        x = x.astype(jnp.float32) / 255.0
        x = jnp.reshape(x, (*x.shape[:-2], -1))

        if not self.stack_blocks:
             stack_sizes = [16, 32, 32]
             num_blocks_list = [2, 1, 1]
             in_ch = x.shape[-1]
             
             self.stack_blocks = []
             for i, sz in enumerate(stack_sizes):
                 out_ch = sz * self.nn_scale
                 self.stack_blocks.append(
                     ResnetStack(in_ch=in_ch, num_ch=out_ch, num_blocks=num_blocks_list[i], rngs=self.rngs)
                 )
                 in_ch = out_ch

        conv_out = x

        for block in self.stack_blocks:
            conv_out = block(conv_out)

        conv_out = nn.relu(conv_out)
        return conv_out.reshape((*x.shape[:-3], -1))


