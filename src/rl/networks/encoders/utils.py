from typing import Dict, Union, Callable
from flax.core.frozen_dict import FrozenDict
import jax.numpy as jnp


def extract_from_dict(data: FrozenDict | Dict, key: str):
    """
    Extracts values from a dictionary using a list of slash-separated keys.
    """
    data = FrozenDict(data)
    # 1. Start at the top of the dictionary
    current_value = data

    # 2. Split the key by '|' (e.g. 'obs|image|right' -> ['obs', 'image', 'right'])
    steps = key.split('|')

    # 3. Walk down the dictionary tree
    for step in steps:
        current_value = current_value[step]
    return current_value


EncoderType = Callable[[Union[FrozenDict, Dict], bool], jnp.ndarray]
NormType = Callable[[jnp.ndarray], bool]