import gc
from typing import Dict, Any, Tuple
from src.rl.mpo_weighted_sft.mpo_weighted_sft_learner import MPOWeightedSFTLearner
import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.training.utils as training_utils
import jax


class FlowGRPOLearner(MPOWeightedSFTLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Delete super class methods
        del self._update_critics_jitted
        del self._update_policy_jitted
        gc.collect()
