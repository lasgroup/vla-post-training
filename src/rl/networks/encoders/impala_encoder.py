import flax.nnx as nnx
import jax.numpy as jnp
from src.rl.networks.constants import xavier_init


class ResnetBlock(nnx.Module):
    def __init__(self, num_ch, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(num_ch, num_ch, kernel_size=(3, 3), strides=1, padding='SAME', kernel_init=xavier_init(), rngs=rngs)
        self.conv2 = nnx.Conv(num_ch, num_ch, kernel_size=(3, 3), strides=1, padding='SAME', kernel_init=xavier_init(), rngs=rngs)

    def __call__(self, x):
        inputs = x
        x = nnx.relu(x)
        x = self.conv1(x)
        x = nnx.relu(x)
        x = self.conv2(x)
        return x + inputs


class ResnetStack(nnx.Module):
    def __init__(self, in_ch: int, num_ch: int, num_blocks: int, use_max_pooling: bool = True, *, rngs: nnx.Rngs):
        self.in_ch = in_ch
        self.num_ch = num_ch
        self.num_blocks = num_blocks
        self.use_max_pooling = use_max_pooling
        
        initializer = xavier_init()
        self.conv_in = nnx.Conv(
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
            x = nnx.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')
            
        for block in self.blocks:
            x = block(x)
        return x


class ImpalaEncoder(nnx.Module):
    def __init__(self, nn_scale: int = 1, *, rngs: nnx.Rngs):
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

        conv_out = nnx.relu(conv_out)
        return conv_out.reshape((*x.shape[:-3], -1))


class SmallerImpalaEncoder(nnx.Module):
    def __init__(self, nn_scale: int = 1, *, rngs: nnx.Rngs):
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

        conv_out = nnx.relu(conv_out)
        return conv_out.reshape((*x.shape[:-3], -1))
