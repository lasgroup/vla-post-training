from typing import Sequence, List, Union, Dict

import flax.nnx as nn
from flax.core.frozen_dict import FrozenDict
import jax.numpy as jnp

from src.rl.networks.constants import default_init
from src.rl.networks.encoders.utils import extract_from_dict


class CNNEncoder(nn.Module):
    def __init__(self,
                 input_example: Union[FrozenDict, Dict],
                 features: Sequence[int] = (32, 32, 32, 32),
                 strides: Sequence[int] = (2, 1, 1, 1),
                 padding: str = 'VALID',
                 image_keys: List[str] | None = None,
                 *,
                 rngs: nn.Rngs):
        if image_keys is None:
            image_keys = ['pixels']
        self._image_keys = image_keys
        self.layers = []
        inputs = self._prepare_input(inputs=input_example)
        in_ch = inputs.shape[-1]
        assert len(features) == len(strides)
        for features, stride in zip(features, strides):
            self.layers.append(nn.Conv(in_ch, features,
                                       kernel_size=(3, 3),
                                       strides=(stride, stride),
                                       kernel_init=default_init(),
                                       padding=padding,
                                       rngs=rngs))
            in_ch = features

    def _prepare_input(self, inputs: Union[FrozenDict, Dict]):
        img_list = [extract_from_dict(inputs, key) for key in self._image_keys]
        img = jnp.stack(img_list, axis=-1)
        img = img.astype(jnp.float32) / 255.0
        img = jnp.reshape(img, (*img.shape[:-2], -1))
        return img

    def __call__(self, observations: Union[FrozenDict, Dict], training=False) -> jnp.ndarray:
        x = self._prepare_input(observations)

        for layer in self.layers:
            x = layer(x)
            x = nn.relu(x)

        return x.reshape((*x.shape[:-3], -1))
