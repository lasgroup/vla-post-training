"""AWR with flow-GRPO-style log-prob loss.

The smallest possible deviation from `AdvantageWeightedSFTLearner`: 
only swaps the actor `_train_step` to one that replaces AWR's FM regression
with `clip(r, 1-eps, 1+eps) * exp(A/beta/scale)` against actions sampled
from the (EMA) policy. See `update_actor.train_step` for the loss.
"""
import functools

from src.rl.advantage_weighted_sft.advantage_weighted_sft_learner import (
    AdvantageWeightedSFTLearner,
)
from src.rl.awr_logprob.update_actor import train_step as awr_logprob_train_step


class AWRLogProbLearner(AdvantageWeightedSFTLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Swap only the actor train_step. Keep AWR's update(), JIT wrappers,
        # normalizer, mc_return path, EMA-resume logic.
        self._train_step = functools.partial(awr_logprob_train_step, self._config)
        self._refresh_update_functions()
