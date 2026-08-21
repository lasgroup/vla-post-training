"""Reward shaping (A1) and value-bound resolution for the reference-alignment change.

Covers `TimeToSuccessAsRewardWrapper`'s new `success_bonus` and the
`get_value_bounds` range it implies. Both are pure Python / pure arithmetic —
no model, no XLA compile — so these run in milliseconds and are unaffected by
the CPU-thread pressure that makes the model-level tests in this directory
flaky on a busy node.

The regression guard matters more than the new behaviour: the wrapper is built
by `FilteredSFTLearner._make_env` for ALL six learners (filtered SFT, AWR, MPO,
FlowGRPO, Best-of-N, OGPO), so a nonzero default would silently change the
reward for every one of them and invalidate every completed run.
"""

import dataclasses

import gymnasium as gym
import numpy as np

from src.envs.wrappers import TimeToSuccessAsRewardWrapper
from src.rl.value_distribution import get_value_bounds
from src.training.config import get_config


class _StubEnv(gym.Env):
    """Terminates on a scripted step; the reward it returns is always discarded."""

    def __init__(self, terminate_on: int):
        self._terminate_on = terminate_on
        self._t = 0

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self._t += 1
        terminate = self._t == self._terminate_on
        # A deliberately wrong reward: the wrapper must overwrite it entirely.
        return np.zeros(1, dtype=np.float32), 999.0, terminate, False, {}


def _rollout(bonus, terminate_on=4, n_steps=4):
    env = TimeToSuccessAsRewardWrapper(_StubEnv(terminate_on), success_bonus=bonus)
    env.reset()
    return [env.step(None)[1] for _ in range(n_steps)]


def test_zero_bonus_reproduces_the_original_reward_exactly():
    # THE regression guard for the five learners that do not opt in.
    assert _rollout(0.0) == [-1.0, -1.0, -1.0, 0.0]


def test_default_bonus_is_zero():
    # dsrl_env.py:486-487 constructs the wrapper positionally with no bonus;
    # the default must keep that call site behaviourally unchanged.
    env = TimeToSuccessAsRewardWrapper(_StubEnv(2))
    env.reset()
    assert [env.step(None)[1] for _ in range(2)] == [-1.0, 0.0]


def test_bonus_is_paid_only_on_the_terminating_step():
    assert _rollout(90.0) == [-1.0, -1.0, -1.0, 90.0]


def test_truncated_episode_never_receives_the_bonus():
    # terminate_on beyond the rollout: every step is a -1, no bonus anywhere.
    assert _rollout(90.0, terminate_on=99) == [-1.0] * 4


def test_bonus_does_not_change_termination():
    for bonus in (0.0, 90.0):
        env = TimeToSuccessAsRewardWrapper(_StubEnv(3), success_bonus=bonus)
        env.reset()
        assert [env.step(None)[2] for _ in range(3)] == [False, False, True]


def _bounds(*, bonus, time_to_success=True, discount=0.995):
    cfg = get_config("pi05_libero_online_ogpo_sft")
    cfg = dataclasses.replace(
        cfg,
        rl=dataclasses.replace(cfg.rl, discount=discount),
        collect=dataclasses.replace(
            cfg.collect,
            success_reward_bonus=bonus,
            use_time_to_success_as_reward=time_to_success,
        ),
    )
    return get_value_bounds(cfg)


def test_lower_bound_is_the_bellman_fixed_point_not_the_truncated_sum():
    # -1/(1-0.995) = -200 exactly. `fix_mc_returns` overwrites every failed
    # episode's MC return with precisely this value (filtered_sft_learner.py:
    # 749-751), so -200 -- not the -(1-g^400)/(1-g) = -173.07 the 400-step horizon
    # implies -- is what the critic actually regresses onto.
    lower, _ = _bounds(bonus=0.0)
    # Tolerance, not equality: (1 - 0.995) is 0.0050000000000000044 in binary
    # floating point, so the exact quotient is -199.99999999999983. 1e-9 is far
    # tighter than the ~26.6 that separates this bound from the truncated-sum
    # value it replaced, which is the distinction the test exists to pin.
    assert abs(lower - (-200.0)) < 1e-9
    # Guard the value this replaced, so a revert is caught rather than silently
    # re-narrowing the range.
    horizon_truncated = -(1.0 - 0.995**400) / 0.005
    assert abs(horizon_truncated - (-173.07)) < 0.01
    assert lower < horizon_truncated


def test_upper_bound_tracks_the_success_bonus():
    assert _bounds(bonus=0.0)[1] == 0.0
    assert _bounds(bonus=90.0)[1] == 90.0


def test_sparse_positive_reward_bounds_are_unaffected():
    # The bonus only applies to the time-to-success reward; the [0, 1] branch
    # must be untouched by it.
    assert _bounds(bonus=0.0, time_to_success=False) == (0.0, 1.0)
    assert _bounds(bonus=90.0, time_to_success=False) == (0.0, 1.0)


def test_fix_mc_returns_still_fires_for_failures_and_not_for_successes():
    # The constancy test at filtered_sft_learner.py:750 is what pins failures to
    # -200; a bonus must not accidentally make failed episodes non-constant, nor
    # leave successful ones constant. Replicated here rather than imported
    # because the surrounding function needs a full episode/buffer fixture.
    failure = np.array([-1.0] * 10)
    assert np.all(failure == failure[0])            # -> overwritten to -200

    success_no_bonus = np.array([-1.0] * 9 + [0.0])
    assert not np.all(success_no_bonus == success_no_bonus[0])

    success_with_bonus = np.array([-1.0] * 9 + [90.0])
    assert not np.all(success_with_bonus == success_with_bonus[0])


def test_reference_config_carries_the_aligned_values():
    ref = get_config("pi05_libero_online_ogpo_ref")
    base = get_config("pi05_libero_online_ogpo_sft")

    assert ref.rl.critic.num_qs == 10 and ref.rl.critic.num_vs == 10
    assert ref.rl.critic.reduction == "mean"
    assert ref.rl.critic.td_weight_schedule.init_value == 0.95
    assert ref.rl.advantage_combination == "grpo_conservative"
    assert ref.rl.normalize_group_advantage is False
    assert ref.rl.normalize_advantage_per_task is False
    assert ref.rl.adv_clip_sym is None
    assert ref.rl.critic_success_oversample is True
    assert ref.rl.use_success_buffer is True
    assert ref.rl.n_samples == 8
    # Deliberately NOT the reference's values -- each was dropped on measurement.
    assert ref.rl.clip_epsilon == 0.1
    assert ref.rl.discount == 0.995
    assert ref.rl.group_num_samples == 8

    # The baseline config must be untouched by all of the above.
    assert base.rl.critic.num_qs == 2
    assert base.rl.critic.reduction == "min"
    assert base.rl.critic_success_oversample is False
    assert base.rl.n_samples == 1
    assert base.collect.success_reward_bonus == 0.0
