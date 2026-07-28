"""Which sampling phases classifier-free guidance is armed for."""

import types

import pytest

from src.rl.filtered_sft_agent.filtered_sft_learner import FilteredSFTLearner
from src.training.collect import _tag_metrics
from src.training.config import FilteredSFTLearnerConfig


def _learner(cfg_scale, guide_collection: bool):
    # The routing decision needs none of the model/checkpoint machinery.
    scales = [cfg_scale] if isinstance(cfg_scale, float) else list(cfg_scale)
    learner = object.__new__(FilteredSFTLearner)
    learner._cfg_scales = scales
    learner._cfg_scale = scales[0]
    learner._cfg_guide_collection = guide_collection
    learner._policy = types.SimpleNamespace(_sample_kwargs={})
    return learner


def _armed(learner, *, evaluation: bool):
    learner._set_guidance(evaluation=evaluation)
    return learner._policy._sample_kwargs.get("cfg_scale")


def test_guidance_is_evaluation_only_by_default():
    learner = _learner(cfg_scale=1.5, guide_collection=False)
    assert _armed(learner, evaluation=True) == 1.5
    assert _armed(learner, evaluation=False) is None


def test_guide_collection_arms_both_phases():
    learner = _learner(cfg_scale=1.5, guide_collection=True)
    assert _armed(learner, evaluation=True) == 1.5
    assert _armed(learner, evaluation=False) == 1.5


@pytest.mark.parametrize("guide_collection", [True, False])
def test_unit_scale_never_arms_the_cfg_path(guide_collection):
    learner = _learner(cfg_scale=1.0, guide_collection=guide_collection)
    assert _armed(learner, evaluation=True) is None
    assert _armed(learner, evaluation=False) is None


def test_toggling_does_not_accumulate_kwargs():
    learner = _learner(cfg_scale=2.0, guide_collection=False)
    learner._policy._sample_kwargs = {"noise_level": 0.3}
    _armed(learner, evaluation=True)
    _armed(learner, evaluation=False)
    assert learner._policy._sample_kwargs == {"noise_level": 0.3}


def test_set_cfg_scale_selects_the_armed_scale():
    learner = _learner(cfg_scale=[1.0, 1.5, 3.0], guide_collection=False)
    assert learner.cfg_scales == [1.0, 1.5, 3.0]
    assert _armed(learner, evaluation=True) is None  # first scale is the 1.0 baseline
    learner.set_cfg_scale(3.0)
    assert _armed(learner, evaluation=True) == 3.0


def test_sweep_tags_metrics_per_scale():
    tagged = _tag_metrics(
        {"eval/success_rate": 0.5, "eval/success_rate/libero_90_79": 0.25}, scale=1.5
    )
    assert tagged == {
        "eval/cfg1.5/success_rate": 0.5,
        "eval/cfg1.5/success_rate/libero_90_79": 0.25,
    }


def test_sweep_rejects_guided_collection_at_many_scales():
    with pytest.raises(ValueError, match="single rl.cfg_scale"):
        FilteredSFTLearnerConfig(cfg_scale=[1.0, 2.0], cfg_guide_collection=True)


def test_scalar_cfg_scale_is_normalized_to_a_list():
    assert FilteredSFTLearnerConfig(cfg_scale=2.0).cfg_scale == [2.0]
    assert FilteredSFTLearnerConfig().cfg_scale == [1.0]
