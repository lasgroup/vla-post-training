from typing import Dict, Optional, Sequence, Union

import flax.nnx as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict

from src.rl.networks.constants import default_init, xavier_init, kaiming_init

from functools import partial
from typing import Any, Callable, Sequence, Tuple
import distrax
import wandb

ModuleDef = Any

class SpatialSoftmax(nn.Module):
    def __init__(self, height: int, width: int, channel: int, pos_x: jnp.ndarray, pos_y: jnp.ndarray, temperature: Optional[float], log_heatmap: bool = False, *, rngs: nn.Rngs):
        self.height = height
        self.width = width
        self.channel = channel
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.temperature_val = temperature
        self.log_heatmap = log_heatmap

        if self.temperature_val == -1:
             self.temperature = nn.Param(jnp.ones((1,), dtype=jnp.float32))
        else:
             self.temperature = None

    def __call__(self, feature):
        if self.temperature_val == -1:
            temperature = self.temperature.value
        else:
            temperature = 1.

        # print(temperature)
        assert len(feature.shape) == 4
        batch_size, num_featuremaps = feature.shape[0], feature.shape[3]
        feature = feature.transpose(0, 3, 1, 2).reshape(batch_size, num_featuremaps, self.height * self.width)

        softmax_attention = nn.softmax(feature / temperature)
        expected_x = jnp.sum(self.pos_x * softmax_attention, axis=2, keepdims=True).reshape(batch_size, num_featuremaps)
        expected_y = jnp.sum(self.pos_y * softmax_attention, axis=2, keepdims=True).reshape(batch_size, num_featuremaps)
        expected_xy = jnp.concatenate([expected_x, expected_y], axis=1)

        expected_xy = jnp.reshape(expected_xy, [batch_size, 2*num_featuremaps])
        return expected_xy

