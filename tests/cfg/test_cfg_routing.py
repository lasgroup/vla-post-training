"""Which sampling phases classifier-free guidance is armed for."""

import pytest

from src.rl.filtered_sft_agent.filtered_sft_learner import FilteredSFTLearner
from src.training.collect import _tag_metrics
from src.training.config import FilteredSFTLearnerConfig


class _Learner(FilteredSFTLearner):
    def __init__(self, cfg_scale, guide_collection: bool):
        self._config = type("C", (), {"rl": FilteredSFTLearnerConfig(cfg_scale=cfg_scale, cfg_guide_collection=guide_collection)})
        self._cfg_scales = list(self._config.rl.cfg_scale)
        self._cfg_scale = self._cfg_scales[0]
        self._cfg_active = False

    def start_data_collection(self, step=None, *, evaluation=False):
        self._cfg_active = self._cfg_scale != 1.0 and (evaluation or self._config.rl.cfg_guide_collection)


def _armed(learner, *, evaluation: bool):
    learner.start_data_collection(evaluation=evaluation)
    return learner._cfg_active


def test_guidance_is_evaluation_only_by_default():
    learner = _Learner(cfg_scale=(1.5,), guide_collection=False)
    assert _armed(learner, evaluation=True)
    assert not _armed(learner, evaluation=False)


def test_guide_collection_arms_both_phases():
    learner = _Learner(cfg_scale=(1.5,), guide_collection=True)
    assert _armed(learner, evaluation=True)
    assert _armed(learner, evaluation=False)


@pytest.mark.parametrize("guide_collection", [True, False])
def test_unit_scale_never_arms_the_cfg_path(guide_collection):
    learner = _Learner(cfg_scale=(1.0,), guide_collection=guide_collection)
    assert not _armed(learner, evaluation=True)
    assert not _armed(learner, evaluation=False)


def test_set_cfg_scale_selects_the_armed_scale():
    learner = _Learner(cfg_scale=(1.0, 1.5, 3.0), guide_collection=False)
    assert learner.cfg_scales == [1.0, 1.5, 3.0]
    assert not _armed(learner, evaluation=True)
    learner.set_cfg_scale(3.0)
    assert _armed(learner, evaluation=True)
    assert learner._cfg_scale == 3.0


def test_sweep_tags_metrics_per_scale():
    tagged = _tag_metrics({"eval/success_rate": 0.5, "eval/success_rate/libero_90_79": 0.25}, scale=1.5)
    assert tagged == {"eval/cfg1.5/success_rate": 0.5, "eval/cfg1.5/success_rate/libero_90_79": 0.25}


def test_sweep_rejects_guided_collection_at_many_scales():
    with pytest.raises(ValueError, match="single rl.cfg_scale"):
        FilteredSFTLearnerConfig(cfg_scale=(1.0, 2.0), cfg_guide_collection=True)


def test_default_is_unguided():
    assert FilteredSFTLearnerConfig().cfg_scale == (1.0,)
    assert FilteredSFTLearnerConfig().cfg_dropout_prob == 0.0
