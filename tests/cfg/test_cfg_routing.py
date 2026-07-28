"""Which sampling phases classifier-free guidance is armed for."""

import types

import pytest

from src.rl.filtered_sft_agent.filtered_sft_learner import FilteredSFTLearner


def _learner(cfg_scale: float, guide_collection: bool):
    # The routing decision needs none of the model/checkpoint machinery.
    learner = object.__new__(FilteredSFTLearner)
    learner._cfg_scale = cfg_scale
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
