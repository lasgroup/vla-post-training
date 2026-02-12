from typing import Any, Dict, Union
import flax
import numpy as np
from collections import namedtuple

DataType = Union[np.ndarray, Dict[str, "DataType"]]
PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
StepData = namedtuple(
    "StepData", ["obs", "action", "next_obs", "reward", "terminate", "truncate"]
)