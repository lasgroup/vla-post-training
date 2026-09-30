from typing import Union
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

ObsType = Union[dict, FrozenDict, jnp.ndarray]
# Key of the cached VLA prefix embedding in critic observations and replay-buffer entries.
PREFIX_EMBEDDING_NAME = "prefix_embedding"
ActionType = jnp.ndarray
