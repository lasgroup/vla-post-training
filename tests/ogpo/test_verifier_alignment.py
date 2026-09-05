# ruff: noqa: F722
"""Independent (adversarial) verification of the 2026-08-20 reference-alignment change.

Written by a verifier that did NOT implement the change. Every claim under test is
taken from the change spec, not from the implementation's reasoning:

  * A1 backward compatibility is checked **differentially**, against a verbatim copy
    of the pre-change wrapper body, over a randomized termination/truncation script —
    not by re-asserting the constants the implementation chose.
  * `get_value_bounds` is likewise checked differentially, against a verbatim copy of
    the pre-change function, over **every registered config** — which is how the
    "inert because num_value_bins == 1 everywhere" claim is falsified for the
    Best-of-N distributional sweeps.
  * A4's RNG-stream claim is checked structurally (AST), because the alternative
    needs real pi05 weights and a GPU.

All pure Python / numpy: no model, no XLA compile.
"""

import ast
import dataclasses
import inspect
import pathlib
import subprocess

import gymnasium as gym
import numpy as np
import pytest

from src.envs.wrappers import TimeToSuccessAsRewardWrapper
from src.rl.ogpo.ogpo_learner import OGPOAgentLearner
from src.rl.replay_buffer import ShardedReplayBuffer
from src.rl.value_distribution import get_value_bounds
import src.training.config as _config


_ROOT = pathlib.Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------------------
# Verbatim pre-change implementations. These are the differential baselines; comparing a
# refactor to itself proves nothing (CLAUDE.md step 3).
# --------------------------------------------------------------------------------------


class _PreChangeTimeToSuccessWrapper(gym.Wrapper):
    """Verbatim body of `TimeToSuccessAsRewardWrapper.step` before the change."""

    def __init__(self, env: gym.Env):
        super().__init__(env=env)

    def step(self, action):
        obs, _, terminate, truncate, info = self.env.step(action)
        time_to_success_reward = 0.0 if terminate else -1.0
        return obs, time_to_success_reward, terminate, truncate, info


def _pre_change_get_value_bounds(config) -> tuple[float, float]:
    """Verbatim `get_value_bounds` before the change (value_distribution.py:116-145)."""
    crit = config.rl.critic
    if crit.value_lower_bound is not None and crit.value_upper_bound is not None:
        return float(crit.value_lower_bound), float(crit.value_upper_bound)

    discount = float(config.rl.discount)
    T = int(config.collect.max_episode_steps)
    if config.collect.use_time_to_success_as_reward:
        lower = -(1.0 - discount**T) / (1.0 - discount) if discount < 1.0 else -float(T)
        upper = 0.0
    else:
        lower = 0.0
        upper = 1.0
    if crit.num_value_bins > 1:
        half_bw = (upper - lower) / (2 * (crit.num_value_bins - 1))
        lower -= half_bw
        upper += half_bw
    return float(lower), float(upper)


# --------------------------------------------------------------------------------------
# 1. A1 backward compatibility (the #1 risk: six learners share this wrapper)
# --------------------------------------------------------------------------------------


class _ScriptedEnv(gym.Env):
    """Replays a scripted list of (terminate, truncate) flags, with junk rewards."""

    def __init__(self, script):
        self._script = list(script)
        self._t = 0

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        terminate, truncate = self._script[self._t]
        self._t += 1
        # A junk reward the wrapper is contractually obliged to discard.
        return np.zeros(1, dtype=np.float32), 3.7 * self._t, terminate, truncate, {"i": self._t}


def _scripts(seed=0, n=40, length=12):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        flags = rng.random((length, 2)) < 0.25
        out.append([(bool(a), bool(b)) for a, b in flags])
    # Plus the three structurally interesting scripts, explicitly.
    out.append([(False, False)] * length)                       # neither
    out.append([(False, False)] * (length - 1) + [(False, True)])  # pure truncation
    out.append([(False, False)] * (length - 1) + [(True, False)])  # pure success
    return out


@pytest.mark.parametrize("script", _scripts())
def test_default_bonus_is_bit_identical_to_the_pre_change_wrapper(script):
    """Differential: default `success_bonus` must reproduce the old reward EXACTLY.

    Bit-identical, not approximately: this wrapper is built by
    `filtered_sft_wrap_env` for filtered SFT, AWR, MPO, FlowGRPO, Best-of-N AND
    OGPO, so any drift silently rewrites the reward of five learners that never
    opted in. `==` on Python floats, and the full 5-tuple, not just the reward.
    """
    new = TimeToSuccessAsRewardWrapper(_ScriptedEnv(script))
    old = _PreChangeTimeToSuccessWrapper(_ScriptedEnv(script))
    new.reset()
    old.reset()
    for _ in range(len(script)):
        o_new = new.step(None)
        o_old = old.step(None)
        assert o_new[1] == o_old[1]
        assert type(o_new[1]) is type(o_old[1])
        assert (o_new[2], o_new[3], o_new[4]) == (o_old[2], o_old[3], o_old[4])


def test_explicit_zero_bonus_also_bit_identical():
    """`success_bonus=0.0` passed explicitly (what every non-ref recipe emits)."""
    script = [(False, False), (False, False), (True, False)]
    new = TimeToSuccessAsRewardWrapper(_ScriptedEnv(script), success_bonus=0.0)
    old = _PreChangeTimeToSuccessWrapper(_ScriptedEnv(script))
    new.reset()
    old.reset()
    assert [new.step(None)[1] for _ in script] == [old.step(None)[1] for _ in script]


def test_dsrl_positional_construction_still_works():
    """`dsrl_env.py:487` builds the wrapper positionally with no bonus (quarantined tree)."""
    sig = inspect.signature(TimeToSuccessAsRewardWrapper.__init__)
    assert list(sig.parameters)[1] == "env", "env must stay the first positional parameter"
    assert sig.parameters["success_bonus"].default == 0.0
    env = TimeToSuccessAsRewardWrapper(_ScriptedEnv([(False, False), (True, False)]))
    env.reset()
    assert [env.step(None)[1] for _ in range(2)] == [-1.0, 0.0]


def test_truncation_never_pays_the_bonus():
    """A time-limit truncation is NOT a success; the bonus must not leak into it."""
    script = [(False, False), (False, False), (False, True)]
    env = TimeToSuccessAsRewardWrapper(_ScriptedEnv(script), success_bonus=72.0)
    env.reset()
    assert [env.step(None)[1] for _ in script] == [-1.0, -1.0, -1.0]


def test_bonus_is_paid_on_every_terminating_step_not_just_the_first():
    """Documents actual semantics: the wrapper is stateless, so if the inner env
    reports `terminate` on more than one step (LIBERO's QueryFrequencyWrapper pads a
    terminated chunk with terminate=True), EVERY such step is paid the bonus."""
    script = [(False, False), (True, False), (True, False)]
    env = TimeToSuccessAsRewardWrapper(_ScriptedEnv(script), success_bonus=72.0)
    env.reset()
    assert [env.step(None)[1] for _ in script] == [-1.0, 72.0, 72.0]


# --------------------------------------------------------------------------------------
# 2. get_value_bounds — differential over EVERY registered config
# --------------------------------------------------------------------------------------


# Only the ONLINE configs have a `collect` block / an RL learner; the rest of
# `_CONFIGS` is openpi's offline training set.
_REGISTERED = [
    c.name for c in _config._CONFIGS if isinstance(c, _config.OnlineTrainConfig)
]


def test_registered_config_names_cover_the_new_one():
    assert "pi05_libero_online_ogpo_ref" in _REGISTERED
    assert "pi05_libero_online_ogpo_sft" in _REGISTERED
    assert len(_REGISTERED) == len(set(_REGISTERED))


@pytest.mark.parametrize("name", _REGISTERED)
def test_value_bounds_unchanged_for_every_registered_config_as_shipped(name):
    """As registered (num_value_bins == 1 everywhere), the Gaussian critic ignores the
    bounds — but the *returned numbers* still change for the time-to-success configs.

    This test pins WHICH configs see a different tuple, so the "inert" claim is
    measured rather than asserted.
    """
    cfg = _config.get_config(name)
    if not hasattr(cfg.rl, "critic"):
        pytest.skip(f"{name} has no critic config")
    before = _pre_change_get_value_bounds(cfg)
    after = get_value_bounds(cfg)
    if cfg.collect.use_time_to_success_as_reward:
        # lower moves from the horizon-truncated sum to the Bellman fixed point.
        assert after[0] < before[0]
        assert after[0] == pytest.approx(-1.0 / (1.0 - cfg.rl.discount), rel=1e-12)
        assert after[1] == 0.0 == before[1]  # bonus defaults to 0 in every registered config
    else:
        assert after == before
    # ...and it is unused, because the distribution is Gaussian.
    assert cfg.rl.critic.num_value_bins == 1


def test_bounds_change_is_NOT_inert_for_the_distributional_best_of_n_sweeps():
    """The change spec claims the bound fix is inert because `num_value_bins == 1`
    everywhere. It is not: two committed sweep YAMLs set 201 bins on a Best-of-N
    config, and Best-of-N is the one learner that actually builds a categorical head
    (`best_of_n_learner.py:57-67` forwards `num_bins`).

    Those arms discretize the target onto `linspace(lower, upper, 201)`, so moving
    `lower` moves every bin center — a real behavioural change, and a break in
    comparability with the already-completed runs of that sweep.
    """
    cfg = _config.get_config("pi05_libero_online_best_of_n")
    cfg = dataclasses.replace(
        cfg,
        rl=dataclasses.replace(
            cfg.rl,
            critic=dataclasses.replace(
                cfg.rl.critic, num_value_bins=201, use_distributional_critic=True
            ),
        ),
    )
    before = _pre_change_get_value_bounds(cfg)
    after = get_value_bounds(cfg)
    assert before != after
    # Half-bin padding is applied on top of the new, wider range.
    assert after[0] == pytest.approx(-200.5, abs=1e-6)
    assert after[1] == pytest.approx(0.5, abs=1e-6)
    assert before[0] == pytest.approx(-173.50, abs=0.01)
    # The bin width itself changes, i.e. every one of the 201 centers moves.
    width_before = (before[1] - before[0]) / 200
    width_after = (after[1] - after[0]) / 200
    assert width_before != pytest.approx(width_after, rel=1e-6)


def test_half_bin_padding_composes_with_a_positive_bonus():
    cfg = _config.get_config("pi05_libero_online_best_of_n")
    cfg = dataclasses.replace(
        cfg,
        collect=dataclasses.replace(cfg.collect, success_reward_bonus=72.0),
        rl=dataclasses.replace(
            cfg.rl,
            critic=dataclasses.replace(cfg.rl.critic, num_value_bins=3),
        ),
    )
    lower, upper = get_value_bounds(cfg)
    # raw range [-200, 72]; half bin = 272 / (2*2) = 68
    assert lower == pytest.approx(-268.0, abs=1e-6)
    assert upper == pytest.approx(140.0, abs=1e-6)


def test_explicit_user_override_still_wins_over_the_new_lower_bound():
    cfg = _config.get_config("pi05_libero_online_ogpo_ref")
    cfg = dataclasses.replace(
        cfg,
        collect=dataclasses.replace(cfg.collect, success_reward_bonus=72.0),
        rl=dataclasses.replace(
            cfg.rl,
            critic=dataclasses.replace(
                cfg.rl.critic, value_lower_bound=-5.0, value_upper_bound=5.0
            ),
        ),
    )
    assert get_value_bounds(cfg) == (-5.0, 5.0)


def test_new_lower_bound_actually_bounds_the_data_the_old_one_did_not():
    """`fix_mc_returns` writes exactly reward/(1-gamma) for failed episodes
    (filtered_sft_learner.py:750-751). Reproduce that arithmetic and check the old
    bound truncated it while the new one does not."""
    discount = 0.995
    pinned_mc_return = -1.0 / (1.0 - discount)
    cfg = _config.get_config("pi05_libero_online_ogpo_sft")
    cfg = dataclasses.replace(cfg, rl=dataclasses.replace(cfg.rl, discount=discount))
    old_lower, _ = _pre_change_get_value_bounds(cfg)
    new_lower, _ = get_value_bounds(cfg)
    assert pinned_mc_return < old_lower          # the old bound was violated by real data
    assert pinned_mc_return >= new_lower - 1e-9  # the new one is not


# --------------------------------------------------------------------------------------
# 3. A4 — success oversampling
# --------------------------------------------------------------------------------------


def _update_fn_ast():
    src = (_ROOT / "src" / "rl" / "ogpo" / "ogpo_learner.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "update":
            return node
    raise AssertionError("OGPOAgentLearner.update not found")


def _count_rng_splits(node):
    n = 0
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
            if sub.func.attr == "split" and ast.unparse(sub.func).endswith("random.split"):
                n += 1
    return n


def test_the_extra_rng_split_lives_entirely_inside_the_oversample_guard():
    """If the new `jax.random.split` sat outside the flag guard, every existing OGPO
    run's RNG stream would shift by one split per critic step — silently changing
    sampled batches and SDE noise for arms that never enabled the flag."""
    update = _update_fn_ast()
    guards = [
        n for n in ast.walk(update)
        if isinstance(n, ast.If) and "critic_success_oversample" in ast.unparse(n.test)
    ]
    assert len(guards) == 1, "expected exactly one critic_success_oversample guard"
    assert _count_rng_splits(guards[0]) == 1
    # Total splits in update(): UTD loop + oversample + policy loop.
    assert _count_rng_splits(update) == 3


def test_the_oversample_guard_also_checks_the_buffer_is_full():
    update = _update_fn_ast()
    guard = [
        n for n in ast.walk(update)
        if isinstance(n, ast.If) and "critic_success_oversample" in ast.unparse(n.test)
    ][0]
    test_src = ast.unparse(guard.test)
    assert "_success_data_buffer is not None" in test_src
    assert "critic_batch_size" in test_src


def test_oversample_metrics_use_a_distinct_prefix_and_distinct_locals():
    """Reusing `q_info`/`value_info` would overwrite the `critic/` series with
    success-only numbers; reusing the `critic/` prefix would break comparability."""
    src = (_ROOT / "src" / "rl" / "ogpo" / "ogpo_learner.py").read_text()
    assert "critic_sb/q_" in src and "critic_sb/value_" in src
    update = _update_fn_ast()
    guard = [
        n for n in ast.walk(update)
        if isinstance(n, ast.If) and "critic_success_oversample" in ast.unparse(n.test)
    ][0]
    guard_src = ast.unparse(guard)
    # The pre-existing names must not be rebound inside the guard.
    assert "q_info" not in guard_src.replace("q_sb_info", "")
    assert "value_info" not in guard_src.replace("value_sb_info", "")


def _dummy_buffer_payload(n_windows=4, obs_dim=3, act_h=2):
    n_obs = n_windows + act_h
    obs = {
        "state": np.random.rand(n_obs, obs_dim).astype(np.float32),
        "image": {"base": np.zeros((n_obs, 4, 4, 3), dtype=np.uint8)},
        "image_mask": {"base": np.ones((n_obs,), dtype=bool)},
        "prefix_embedding": np.random.rand(n_obs, 5).astype(np.float32),
    }
    return {
        "observations": obs,
        "obs_index": np.arange(n_windows, dtype=np.int64),
        "next_obs_index": np.arange(n_windows, dtype=np.int64) + act_h,
        "actions": np.random.rand(n_windows, act_h, 7).astype(np.float32),
        "reward": np.random.rand(n_windows).astype(np.float32),
        "mc_return": np.random.rand(n_windows).astype(np.float32),
        "discount": np.zeros((n_windows,), dtype=np.float32),
        "is_success": np.ones((n_windows,), dtype=np.float32),
    }


def test_success_buffer_and_online_buffer_share_one_schema_and_one_sample_contract():
    """A4 samples the success buffer with the SAME kwargs the online critic batch uses.
    `ogpo_learner.py:79-91` builds it from `_get_online_replay_buffer()` with only
    `buffer_capacity` replaced, so the schemas must be identical and `sample()` must
    accept `batch_size=` + `drop_obs_keys=` on both.
    """
    payload = _dummy_buffer_payload()
    online = ShardedReplayBuffer(dummy_data=payload, max_capacity=64, seed=0, freeze_dict=False)
    success = ShardedReplayBuffer(dummy_data=payload, max_capacity=8, seed=0, freeze_dict=False)
    online.insert(payload)
    success.insert(payload)

    drop = ("image", "image_mask")  # _OGPO_CRITIC_DROP_OBS_KEYS
    b_on = online.sample(batch_size=3, drop_obs_keys=drop)
    b_sb = success.sample(batch_size=3, drop_obs_keys=drop)

    assert set(b_on) == set(b_sb)
    assert set(b_on["observation"]) == set(b_sb["observation"]) == {"state", "prefix_embedding"}
    assert set(b_on["next_observation"]) == set(b_sb["next_observation"])
    for k in b_on:
        if k in ("observation", "next_observation"):
            for kk in b_on[k]:
                assert b_on[k][kk].shape == b_sb[k][kk].shape
        else:
            assert np.asarray(b_on[k]).shape == np.asarray(b_sb[k]).shape

    # Only a different capacity distinguishes them.
    assert success.max_capacity != online.max_capacity


def test_replay_buffer_sample_signature_matches_the_a4_call_site():
    sig = inspect.signature(ShardedReplayBuffer.sample)
    assert "batch_size" in sig.parameters
    assert sig.parameters["drop_obs_keys"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["drop_obs_keys"].default == ()


def test_oversample_flag_is_read_by_direct_attribute_access_not_getattr():
    """best_practices.md forbids `getattr(cfg, 'x', default)` for config reads."""
    src = (_ROOT / "src" / "rl" / "ogpo" / "ogpo_learner.py").read_text()
    assert 'getattr(self._config.rl, "critic_success_oversample"' not in src
    assert "rl_config.critic_success_oversample" in src


# --------------------------------------------------------------------------------------
# 4/5. The new registered config and the recipe parameterization
# --------------------------------------------------------------------------------------


def _learner_class_for(cfg):
    """Replicates the isinstance dispatch in scripts/exp.py:74-85, order included."""
    if isinstance(cfg.rl, _config.OGPOPrivilegedLearnerConfig):
        return "OGPOPrivilegedLearner"
    if isinstance(cfg.rl, _config.OGPOSFTLearnerConfig):
        return "OGPOAgentLearner"
    if isinstance(cfg.rl, _config.AdvantageWeightedSFTLearnerConfig):
        return "AdvantageWeightedSFTLearner"
    if isinstance(cfg.rl, _config.BestofNLearnerConfig):
        return "BestofNLearner"
    if isinstance(cfg.rl, _config.FilteredSFTLearnerConfig):
        return "FilteredSFTLearner"
    return "UNSUPPORTED"


def test_dispatch_for_every_registered_config_is_unchanged_by_the_new_entry():
    expected = {
        "pi05_libero_online_filtered_sft": "FilteredSFTLearner",
        "pi05_molmo_online_filtered_sft": "FilteredSFTLearner",
        "pi05_libero_online_aw_sft": "AdvantageWeightedSFTLearner",
        "pi05_libero_online_mpo_sft": "AdvantageWeightedSFTLearner",
        "pi05_libero_online_flow_grpo_sft": "AdvantageWeightedSFTLearner",
        "pi05_libero_online_best_of_n": "BestofNLearner",
        "pi05_molmo_online_best_of_n": "BestofNLearner",
        "pi05_libero_online_dsrl": "UNSUPPORTED",
        "pi05_libero_online_ogpo_sft": "OGPOAgentLearner",
        "pi05_libero_online_ogpo_ref": "OGPOAgentLearner",
        "pi05_libero_online_ogpo_privileged": "OGPOPrivilegedLearner",
    }
    got = {name: _learner_class_for(_config.get_config(name)) for name in _REGISTERED}
    assert got == expected


def test_ref_config_satisfies_the_grpo_conservative_preconditions():
    """`update_actor.py:269-273` asserts G > 1 and adv_strategy == 'vanilla'."""
    ref = _config.get_config("pi05_libero_online_ogpo_ref").rl
    assert ref.advantage_combination == "grpo_conservative"
    assert ref.group_num_samples > 1
    assert ref.adv_strategy == "vanilla"
    # grpo_conservative reads Q heads only, so num_vs need not equal num_qs -- but
    # per_critic_value_target (update_critic.py:325) would require it. Both hold here.
    assert ref.critic.num_qs == ref.critic.num_vs
    assert ref.critic.per_critic_value_target is False


def test_ref_config_satisfies_the_ogpo_learner_init_guards():
    """`ogpo_learner.py:53-70`: online_ratio must be 1.0, and
    critic_success_oversample requires use_success_buffer."""
    ref = _config.get_config("pi05_libero_online_ogpo_ref").rl
    assert ref.online_ratio == 1.0
    assert not (ref.critic_success_oversample and not ref.use_success_buffer)


def test_baseline_ogpo_config_is_untouched_in_every_field_the_change_could_reach():
    base = _config.get_config("pi05_libero_online_ogpo_sft")
    assert base.collect.success_reward_bonus == 0.0
    assert base.rl.critic_success_oversample is False
    assert base.rl.critic.num_qs == 2 and base.rl.critic.num_vs == 2
    assert base.rl.critic.reduction == "min"
    assert base.rl.n_samples == 1
    assert base.rl.advantage_combination == "reduced"
    assert base.rl.use_success_buffer is False
    assert base.rl.adv_clip_sym is None
    assert base.rl.clip_epsilon == 0.01
    assert base.rl.group_num_samples == 8


def test_success_reward_bonus_defaults_to_zero_on_every_registered_config():
    for name in _REGISTERED:
        assert _config.get_config(name).collect.success_reward_bonus == 0.0, name


# --- recipe: DRY-mode inspection (echoes the command, never runs it) -------------------


def _dry(script, tmp_path, **env_overrides):
    import os

    env = dict(os.environ)
    env.update(
        {
            "DRY": "1",
            "GPU": "0",
            "PROJECT_DIR": str(_ROOT),
            "CKPT_BASE_DIR": str(tmp_path / "ckpt"),
            "STORE_ROOT": str(_ROOT / "run_store"),
        }
    )
    env.update({k: str(v) for k, v in env_overrides.items()})
    out = subprocess.run(
        ["bash", str(_ROOT / "scripts" / script)],
        capture_output=True, text=True, env=env, timeout=180,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip().splitlines()[-1]



def _resolve(cmd: str):
    """Parse an emitted `uv run scripts/exp.py ...` line into a real OnlineTrainConfig.

    The alignment flags are emitted UNCONDITIONALLY (ogpo_multitask_4task.sh), so the
    command TEXT is no longer identical to the pre-alignment script. What must hold is
    that the RESOLVED config is unchanged at the defaults, which is what this checks.
    """
    import shlex, sys
    import tyro
    import src.training.config as _C

    argv0, sys.argv = sys.argv, ["pytest"]
    try:
        return tyro.extras.overridable_config_cli(
            {n: (n, c) for n, c in _C._CONFIGS_DICT.items()},
            args=shlex.split(cmd)[3:],  # drop "uv run scripts/exp.py"
        )
    finally:
        sys.argv = argv0


def _alignment_fields(c):
    r, cr, co = c.rl, c.rl.critic, c.collect
    return {
        "config": c.name,
        "success_reward_bonus": co.success_reward_bonus,
        "num_qs": cr.num_qs,
        "num_vs": cr.num_vs,
        "reduction": cr.reduction,
        "td_weight": cr.td_weight_schedule.init_value,
        "n_samples": r.n_samples,
        "critic_success_oversample": r.critic_success_oversample,
        "advantage_combination": r.advantage_combination,
    }


_BASELINE_RESOLVED = {
    "config": "pi05_libero_online_ogpo_sft",
    "success_reward_bonus": 0.0,
    "num_qs": 2,
    "num_vs": 2,
    "reduction": "min",
    "td_weight": 1.0,
    "n_samples": 1,
    "critic_success_oversample": False,
    "advantage_combination": "grpo_conservative",
}

@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"NUM_QS": ""},
        {"CRITIC_RED": ""},
        {"BON_N": ""},
        {"SUCC_BONUS": ""},
        {"TD_W": ""},
        {"SB_Q": "0"},
        {"NUM_QS": "2", "CRITIC_RED": "min", "BON_N": "1", "SUCC_BONUS": "0", "TD_W": "1"},
    ],
)
def test_baseline_recipe_resolves_to_the_baseline_config_at_its_defaults(tmp_path, overrides):
    """The parameterization must be INERT ON THE RESOLVED CONFIG for the baseline recipe.

    Supersedes an earlier version of this test which asserted that no alignment flag
    appeared in the command TEXT. That property was deliberately given up: the guards now
    emit unconditionally so an env var is authoritative for ANY config (without that,
    NUM_QS=2 on the ref recipe silently left num_qs at 10). Config equivalence is the
    stronger replacement and is what the contract actually needs.
    """
    cmd = _dry("ogpo_multitask_4task.sh", tmp_path, **overrides)
    assert _alignment_fields(_resolve(cmd)) == _BASELINE_RESOLVED, overrides


def test_succ_bonus_0_and_0_point_0_now_resolve_identically(tmp_path):
    """`SUCC_BONUS` used to be string-compared against "0", so `0.0` emitted the flag and
    `0` did not -- harmless but asymmetric. The flag is now always emitted, so both spellings
    resolve to the same bonus. Pinned so the asymmetry cannot creep back."""
    for spelling in ("0", "0.0", "00"):
        cmd = _dry("ogpo_multitask_4task.sh", tmp_path, SUCC_BONUS=spelling)
        assert _resolve(cmd).collect.success_reward_bonus == 0.0, spelling


def test_ref_recipe_emits_the_documented_aligned_flag_set(tmp_path):
    cmd = _dry("ogpo_multitask_4task_ref.sh", tmp_path)
    for expected in (
        " pi05_libero_online_ogpo_ref ",
        "--rl.advantage_combination grpo_conservative",
        "--rl.critic.num_qs 10",
        "--rl.critic.num_vs 10",
        "--rl.critic.reduction mean",
        "--rl.n_samples 8",
        "--collect.success_reward_bonus 90",
        "--rl.critic_success_oversample",
        "--rl.critic.td_weight_schedule.init_value 0.95",
        "--rl.use_success_buffer",
    ):
        assert expected in cmd, expected
    for absent in (
        "--rl.normalize_group_advantage",
        "--rl.normalize_advantage_per_task",
        "--rl.adv_clip_sym",
    ):
        assert absent not in cmd, absent


def test_ref_recipe_documented_baseline_override_string_actually_reproduces_the_baseline(
    tmp_path,
):
    """`ogpo_multitask_4task_ref.sh:41-42` documents an env-var string that is claimed
    to reproduce the baseline stack (D10: "a reviewer should be able to reproduce the
    current stack from the new recipe by setting env vars alone").

    It does not. Each guard only APPENDS a flag when the var is set away from the
    baseline value, but the ref *config* already carries the aligned value as its
    dataclass default -- so the baseline value emits nothing and the config default
    survives.
    """
    cmd = _dry(
        "ogpo_multitask_4task_ref.sh",
        tmp_path,
        NUM_QS="2", CRITIC_RED="min", TD_W="1", BON_N="1", SB_Q="0",
        SUCC_BONUS="0", NORM="1", MT_ADV="1", CLIP_SYM="4.0", CONS="0",
    )
    assert "--rl.critic.num_qs 2" in cmd, "num_qs stays at the ref config's 10"
    assert "--rl.critic.reduction min" in cmd, "reduction stays at the ref config's mean"
    assert "--rl.n_samples 1" in cmd, "n_samples stays at the ref config's 8"
    assert "--rl.no-critic_success_oversample" in cmd, "oversampling cannot be turned off"
    assert "--rl.advantage_combination reduced" in cmd, "combination stays grpo_conservative"


# ======================================================================================
# SECOND INDEPENDENT VERIFICATION PASS (different verifier, same change).
#
# Nothing above is trusted: everything below re-derives its claim from source. The
# focus is on ground this file did not already cover --
#   * the reward series as it ACTUALLY reaches `_save_episode_in_buffer` (i.e. after
#     `QueryFrequencyWrapper`'s post-termination zero padding, wrappers.py:194-206),
#     which is what `fix_mc_returns`' constancy gate really sees;
#   * the `num_qs = 10` numeric paths the ref config newly activates;
#   * the collection-time best-of-N aggregation the ref config newly activates;
#   * the structural placement of the A4 block (inside `if update_critic:`).
# ======================================================================================

import jax.numpy as _jnp
import flax.nnx as _nnx

from src.rl.advantage_weighted_sft.update_critic import (
    critic_values_per_head as _critic_values_per_head,
    summarize_critic_values as _summarize_critic_values,
)
from src.rl.ogpo.update_actor import _grpo_conservative_advantage
from src.rl.networks.bronet_critic import BroNetStateActionCritic, BroNetStateValue


# --------------------------------------------------------------------------------------
# V2-1. The reward series as the buffer actually sees it.
# --------------------------------------------------------------------------------------


def _episode_reward_series(*, success: bool, bonus: float, replan=5, horizon=400,
                           terminate_substep=2):
    """Replicate exactly what lands in `episode_data["reward"]`.

    `TimeToSuccessAsRewardWrapper` (wrappers.py:230-253) sits INSIDE
    `QueryFrequencyWrapper` (filtered_sft_learner.py:62-76), which steps the inner env
    `replan` times per chunk and, on early termination, pads the rest of the chunk with a
    hardcoded `reward: 0.0` (wrappers.py:194-206). `collect.py:157-172` then stores the
    whole (env, replan) block, and `_save_episode_in_buffer` concatenates the chunks.
    """
    rewards, terminates = [], []
    if not success:
        # TimeLimit truncation (libero.py:116-119). horizon % replan == 0 for every
        # entry in `get_max_steps_libero`, so truncation lands on a chunk boundary and
        # NO padding is emitted.
        assert horizon % replan == 0
        rewards = [-1.0] * horizon
        terminates = [False] * horizon
        return np.asarray(rewards), np.asarray(terminates)
    n_full_chunks = 3
    rewards += [-1.0] * (n_full_chunks * replan)
    terminates += [False] * (n_full_chunks * replan)
    # Terminating chunk: -1 up to the terminating sub-step, the bonus on it, then the
    # wrapper's 0.0 padding for the remainder of the chunk.
    rewards += [-1.0] * terminate_substep + [bonus]
    terminates += [False] * terminate_substep + [True]
    pad = replan - terminate_substep - 1
    rewards += [0.0] * pad
    terminates += [True] * pad
    return np.asarray(rewards), np.asarray(terminates)


def _fix_mc_returns_fires(reward_series):
    """`filtered_sft_learner.py:752` -- verbatim gate, on the FULL array."""
    return bool(np.all(reward_series == reward_series[0]))


@pytest.mark.parametrize("bonus", [0.0, 72.0])
def test_failed_episodes_still_hit_the_constancy_gate_with_any_bonus(bonus):
    """The whole -200 story depends on this firing. A bonus must not perturb it: a
    failure never terminates, so no bonus is ever paid and the series stays all -1."""
    r, term = _episode_reward_series(success=False, bonus=bonus)
    assert not term.any()
    assert set(np.unique(r)) == {-1.0}
    assert _fix_mc_returns_fires(r)
    assert r[0] / (1 - 0.995) == pytest.approx(-200.0, abs=1e-9)


@pytest.mark.parametrize("substep", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("bonus", [0.0, 72.0])
def test_successful_episodes_never_hit_the_constancy_gate(bonus, substep):
    """Regardless of where in the chunk the episode terminates, and with or without a
    bonus, a success is non-constant -- so `fix_mc_returns` leaves its MC return alone.

    This is the assumption README.md flags as "verify, don't assume". With bonus=0 the
    terminal reward (0.0) is indistinguishable from the padding; the -1 prefix is what
    saves it, and that holds for every substep because n_full_chunks >= 1.
    """
    r, _ = _episode_reward_series(success=True, bonus=bonus, terminate_substep=substep)
    assert not _fix_mc_returns_fires(r)


def test_padding_zeros_are_excluded_from_every_stored_quantity():
    """`n_steps = np.where(done)[0][0] + 1` (filtered_sft_learner.py:724) cuts at the
    FIRST done, and the padded entries carry terminated=True, so they are the tail --
    every stored array is sliced before them. If they were included, the +72 chunk's
    reward window would be diluted by zeros."""
    act_h = 10
    r, term = _episode_reward_series(success=True, bonus=72.0, terminate_substep=2)
    done = term
    n_steps = int(np.where(done)[0][0] + 1)
    n_windows = n_steps - act_h + 1
    assert n_windows > 0
    # The last index any window or MC return touches.
    assert (n_windows - 1) + act_h - 1 == n_steps - 1
    assert r[n_steps - 1] == 72.0
    assert set(np.unique(r[n_steps:])) <= {0.0}


def test_a_terminating_window_bootstraps_zero_so_its_target_is_the_bonus():
    """README.md's stated intent: `_discount = 0.0` when any step in the window
    terminates (filtered_sft_learner.py:747), so the success chunk's Q target is the
    discounted reward window alone -- no V(s') term."""
    discount, act_h = 0.995, 10
    r, term = _episode_reward_series(success=True, bonus=72.0, terminate_substep=2)
    n_steps = int(np.where(term)[0][0] + 1)
    n_windows = n_steps - act_h + 1
    w_gammas = np.array([discount**i for i in range(act_h)])
    rewards = np.asarray(
        [(r[s:s + act_h] * w_gammas).sum() for s in range(n_windows)]
    )
    discounts = np.asarray(
        [0.0 if np.any(term[s:s + act_h]) else discount**act_h for s in range(n_windows)]
    )
    # Only the last window reaches the terminal step here.
    assert discounts[-1] == 0.0
    # ...and its reward is dominated by the bonus rather than by the -1 stream.
    assert rewards[-1] > 0.0
    # Without the bonus that same window is negative -- the sign flip IS the change.
    r0, _ = _episode_reward_series(success=True, bonus=0.0, terminate_substep=2)
    rewards0 = np.asarray(
        [(r0[s:s + act_h] * w_gammas).sum() for s in range(n_windows)]
    )
    assert rewards0[-1] < 0.0


def test_short_episode_guard_makes_the_degenerate_constant_success_unreachable():
    """A one-step success would be reward == [bonus], i.e. CONSTANT, and
    `fix_mc_returns` would then write bonus/(1-gamma) = +14400 as an MC return.
    `n_windows <= 0 -> return` (filtered_sft_learner.py:730-731) makes that unreachable
    for any act_h > 1: the shortest stored episode already carries -1 prefixes."""
    act_h = 10
    degenerate = np.asarray([72.0])
    assert _fix_mc_returns_fires(degenerate)
    assert degenerate[0] / (1 - 0.995) == pytest.approx(14400.0, rel=1e-9)
    n_windows = len(degenerate) - act_h + 1
    assert n_windows <= 0, "such an episode is dropped before fix_mc_returns runs"
    # Anything long enough to be stored has at least act_h-1 leading -1s.
    shortest_stored = np.asarray([-1.0] * (act_h - 1) + [72.0])
    assert len(shortest_stored) - act_h + 1 == 1
    assert not _fix_mc_returns_fires(shortest_stored)


# --------------------------------------------------------------------------------------
# V2-2. num_qs = 10: the numeric paths the ref config newly activates.
# --------------------------------------------------------------------------------------


def _ref_cfg(**collect_over):
    cfg = _config.get_config("pi05_libero_online_ogpo_ref")
    if collect_over:
        cfg = dataclasses.replace(
            cfg, collect=dataclasses.replace(cfg.collect, **collect_over)
        )
    return cfg


def test_summarize_critic_values_honours_mean_reduction_over_ten_heads():
    cfg = _ref_cfg()
    assert cfg.rl.critic.reduction == "mean" and cfg.rl.critic.num_qs == 10
    logits = _jnp.asarray(np.arange(10 * 4, dtype=np.float32).reshape(10, 4))
    got = np.asarray(_summarize_critic_values(logits, cfg, critic_reduction="mean"))
    assert got.shape == (4,)
    np.testing.assert_allclose(got, np.asarray(logits).mean(axis=0), rtol=1e-6)
    # `min` must still be a different number, i.e. the knob is live at n=10.
    got_min = np.asarray(_summarize_critic_values(logits, cfg, critic_reduction="min"))
    assert not np.allclose(got, got_min)


def test_critic_values_per_head_returns_all_ten_heads():
    cfg = _ref_cfg()
    logits = _jnp.asarray(np.random.RandomState(0).randn(10, 6).astype(np.float32))
    heads = np.asarray(_critic_values_per_head(logits, cfg))
    assert heads.shape == (10, 6)


def test_grpo_conservative_is_head_count_agnostic_and_gates_harder_at_ten_heads():
    """`_grpo_conservative_advantage` (update_actor.py:63-84) is written for any n, but
    sign-unanimity over 10 heads zeroes strictly more samples than over 2 -- the
    interaction between C3 and C7 that the change spec does not quantify."""
    rng = np.random.RandomState(7)
    B, G = 16, 8
    q10 = _jnp.asarray(rng.randn(10, B * G).astype(np.float32))
    q2 = q10[:2]
    a10 = np.asarray(_grpo_conservative_advantage(q10, B, G))
    a2 = np.asarray(_grpo_conservative_advantage(q2, B, G))
    assert a10.shape == a2.shape == (B * G,)
    zero10 = float((a10 == 0.0).mean())
    zero2 = float((a2 == 0.0).mean())
    assert zero10 > zero2, (zero10, zero2)
    # Sanity: a sample surviving the 10-head gate must survive the 2-head gate too.
    assert np.all((a10 != 0.0) <= (a2 != 0.0))


def test_grpo_conservative_matches_the_reference_safe_max_formula():
    """Reference `_safe_max` (OGPO_public ogpo/agents/modules/pg_helper.py:455-461):
        x_min * (x_min > 0) + x_max * (x_max < 0)
    applied to per-head, group-centred Q. Differential against a verbatim transcription,
    since C3's justification is that this mode IS the reference's adv_strategy."""
    def _reference(q_heads, B, G):
        heads = np.asarray(q_heads).reshape(-1, B, G)
        heads = heads - heads.mean(axis=-1, keepdims=True)
        adv = heads.reshape(heads.shape[0], B * G)
        x_min, x_max = adv.min(axis=0), adv.max(axis=0)
        return x_min * (x_min > 0) + x_max * (x_max < 0)

    rng = np.random.RandomState(11)
    B, G = 8, 8
    for n in (2, 10):
        q = _jnp.asarray(rng.randn(n, B * G).astype(np.float32))
        np.testing.assert_allclose(
            np.asarray(_grpo_conservative_advantage(q, B, G)),
            _reference(q, B, G),
            rtol=1e-5, atol=1e-6,
        )


def test_bronet_ensembles_build_and_emit_ten_heads():
    """`BroNetStateActionCritic` builds `num_qs` independent towers in a Python list
    (bronet_critic.py:118-121). Nothing there assumes 2, but the ref config is the first
    thing in the tree to ask for 10, so build it once and check the output rank."""
    obs = {"state": _jnp.zeros((3, 8), dtype=_jnp.float32)}
    act = _jnp.zeros((3, 14), dtype=_jnp.float32)
    rngs = _nnx.Rngs(0)
    q = BroNetStateActionCritic(
        observation=obs, action=act, hidden_dim=16, depth=1, num_qs=10, rngs=rngs
    )
    v = BroNetStateValue(observation=obs, hidden_dim=16, depth=1, num_vs=10, rngs=rngs)
    assert np.asarray(q(obs, act)).shape == (10, 3)
    assert np.asarray(v(obs)).shape == (10, 3)
    assert len(q.nets) == 10 and len(v.nets) == 10


# --------------------------------------------------------------------------------------
# V2-3. Best-of-N collection (A3) -- activated by `n_samples = 8` on the ref config.
# --------------------------------------------------------------------------------------


def test_best_of_n_collection_scoring_ignores_critic_reduction():
    """`AdvantageWeightedSFTLearner.sample_actions` hardcodes `scores.min(axis=0)` and
    never consults `rl.critic.reduction`. Harmless while `n_samples == 1` (the path is
    skipped) or `reduction == "min"`; the ref config sets n_samples=8 AND reduction=mean,
    so collection now selects on min-of-10 while every other consumer of the same Q
    ensemble uses mean-of-10. Pinned, not asserted-away."""
    src = (_ROOT / "src" / "rl" / "advantage_weighted_sft"
           / "advantage_weighted_sft_learner.py").read_text()
    tree = ast.parse(src)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "sample_actions"
    )
    body = ast.unparse(fn)
    assert "scores.min(axis=0)" in body
    assert "critic.reduction" not in body

    ref = _config.get_config("pi05_libero_online_ogpo_ref").rl
    assert ref.n_samples > 1 and ref.critic.reduction == "mean"
    # The two aggregates genuinely differ at n=10.
    heads = np.random.RandomState(3).randn(10, 5)
    assert not np.allclose(heads.min(axis=0), heads.mean(axis=0))


def test_best_of_n_path_is_gated_on_n_samples_so_the_baseline_never_enters_it():
    base = _config.get_config("pi05_libero_online_ogpo_sft").rl
    assert base.n_samples == 1
    for name in _REGISTERED:
        rl = _config.get_config(name).rl
        if name in ("pi05_libero_online_best_of_n", "pi05_molmo_online_best_of_n",
                    "pi05_libero_online_ogpo_ref"):
            continue
        assert getattr(rl, "n_samples", 1) == 1, name


# --------------------------------------------------------------------------------------
# V2-4. A4 structural placement (beyond "the rng split is inside the guard").
# --------------------------------------------------------------------------------------


def _ogpo_update_drop_key_call_sites():
    """Every `drop_obs_keys=` argument passed to a buffer sample in `update()`."""
    return [
        ast.unparse(kw.value)
        for node in ast.walk(_ogpo_update_ast())
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == "drop_obs_keys"
    ]


def _ogpo_update_ast():
    src = (_ROOT / "src" / "rl" / "ogpo" / "ogpo_learner.py").read_text()
    tree = ast.parse(src)
    return next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "update"
    )


def test_the_oversample_block_is_nested_inside_the_update_critic_gate():
    """If it sat outside `if update_critic:` it would fire on critic-frozen steps
    (critic.training_start_step / update_interval), silently training the critic when
    the surrounding code says it must not."""
    update = _ogpo_update_ast()
    update_critic_ifs = [
        n for n in ast.walk(update)
        if isinstance(n, ast.If) and ast.unparse(n.test).strip() == "update_critic"
    ]
    assert len(update_critic_ifs) == 1
    nested = any(
        isinstance(sub, ast.If) and "critic_success_oversample" in ast.unparse(sub.test)
        for sub in ast.walk(update_critic_ifs[0])
    )
    assert nested, "A4 block must live under `if update_critic:`"


def test_the_oversample_block_commits_both_critic_states():
    """A donated jit (`donate_argnums=(1, 2)`, AWR:159) invalidates the inputs, so
    failing to rebind BOTH returned states would leave a donated-away buffer on the
    learner -- a use-after-donate on the next step, not a lost update."""
    update = _ogpo_update_ast()
    guard = next(
        n for n in ast.walk(update)
        if isinstance(n, ast.If) and "critic_success_oversample" in ast.unparse(n.test)
    )
    body = ast.unparse(guard)
    assert "self._state_action_critic_state = q_state" in body
    assert "self._value_state = value_state" in body


def test_the_oversample_block_draws_from_the_success_buffer_not_the_online_one():
    update = _ogpo_update_ast()
    guard = next(
        n for n in ast.walk(update)
        if isinstance(n, ast.If) and "critic_success_oversample" in ast.unparse(n.test)
    )
    body = ast.unparse(guard)
    assert "self._success_data_buffer.sample(" in body
    assert "self._online_data_buffer.sample(" not in body
    # Same drop-key gate as the online critic batch, so the jit sees one input
    # shape. The gate itself lives in the `_critic_drop_obs_keys` hook (which
    # the privileged learner overrides); what matters here is that this block
    # calls the SAME hook every other critic sample in update() calls.
    assert "self._critic_drop_obs_keys()" in body
    hook_src = inspect.getsource(OGPOAgentLearner._critic_drop_obs_keys)
    assert "_OGPO_CRITIC_DROP_OBS_KEYS" in hook_src
    assert "store_prefix_rep" in hook_src
    for sample_call in _ogpo_update_drop_key_call_sites():
        assert sample_call == "self._critic_drop_obs_keys()"


def test_no_registered_config_trips_the_new_fail_fast():
    """`ogpo_learner.py:61-70` raises for oversample-without-success-buffer. Cheap to
    get wrong in a future config; assert the invariant over everything registered."""
    for name in _REGISTERED:
        rl = _config.get_config(name).rl
        if getattr(rl, "critic_success_oversample", False):
            assert rl.use_success_buffer, name


# --------------------------------------------------------------------------------------
# V2-5. Value bounds: composition with the bonus the ref RECIPE (not the config) sets.
# --------------------------------------------------------------------------------------


def test_the_ref_config_alone_carries_no_success_bonus():
    """The bonus is the single largest intervention in the change, and it is the ONLY
    aligned knob that lives in the recipe rather than in the registered config. Running
    `uv run scripts/exp.py pi05_libero_online_ogpo_ref` directly therefore reproduces
    none of A1. Pinned so it is a decision, not an accident."""
    assert _config.get_config("pi05_libero_online_ogpo_ref").collect.success_reward_bonus == 0.0
    assert get_value_bounds(_ref_cfg()) == (pytest.approx(-200.0, abs=1e-9), 0.0)


def test_bounds_with_the_recipe_bonus_bracket_the_data_both_ways():
    cfg = _ref_cfg(success_reward_bonus=72.0)
    lower, upper = get_value_bounds(cfg)
    assert lower == pytest.approx(-200.0, abs=1e-9)
    assert upper == 72.0
    # The two extremes the data can actually take: the pinned failure MC return and a
    # success whose terminal chunk bootstraps zero.
    assert lower <= -1.0 / (1.0 - cfg.rl.discount) + 1e-9
    assert upper >= 72.0 * cfg.rl.discount**0


def test_bounds_are_inert_at_one_bin_which_is_what_the_ref_run_uses():
    """Gaussian critic ignores the bounds entirely (value_distribution.py:238-239), and
    the ref recipe pins `--rl.critic.num_value_bins 1`. So for THIS change the bound fix
    is defensive only; the behavioural exposure is the 201-bin Best-of-N sweeps."""
    cfg = _ref_cfg(success_reward_bonus=72.0)
    assert cfg.rl.critic.num_value_bins == 1
    from src.rl.value_distribution import GaussianValueDistribution, make_value_distribution
    lower, upper = get_value_bounds(cfg)
    d1 = make_value_distribution(_jnp.zeros((10, 4)), 1, lower, upper)
    d2 = make_value_distribution(_jnp.zeros((10, 4)), 1, -1.0, 1.0)
    assert isinstance(d1, GaussianValueDistribution)
    np.testing.assert_array_equal(np.asarray(d1.mean()), np.asarray(d2.mean()))


def test_negative_bonus_cannot_invert_the_range():
    cfg = _ref_cfg(success_reward_bonus=-5.0)
    lower, upper = get_value_bounds(cfg)
    assert upper == 0.0 and lower < upper


# --------------------------------------------------------------------------------------
# V2-6. The reference constant the +72 sizing is derived from.
# --------------------------------------------------------------------------------------


_REFERENCE_ROOT = pathlib.Path("/home/pchellap/Projects/SafeVADAR/OGPO_public")


def test_reference_success_bonus_constant_is_five_not_four():
    """`README.md` (D1) and `config.py:381-383` both state the reference "pays +4 per
    success step", and the +72 magnitude is derived from it as 4 x 9 / 100 x 200 = 72.

    The reference source says `reward += 5.0`, in BOTH robomimic wrappers. With 5.0 the
    same derivation gives 5 x 9 / 100 x 200 = 90.
    """
    src = _REFERENCE_ROOT / "envs" / "robomimic_utils.py"
    if not src.exists():
        pytest.skip("reference checkout not present on this machine")
    text = src.read_text()
    assert "reward += 5.0" in text
    assert "reward += 4.0" not in text
    recipe = (_REFERENCE_ROOT / "scripts" / "ogpo" / "square_image_paligemma.sh").read_text()
    assert "post_success_steps=8" in recipe
    # The derivation the change actually used, with the constant the source actually has.
    assert 4.0 * 9 / 100 * 200 == pytest.approx(72.0)
    assert 5.0 * 9 / 100 * 200 == pytest.approx(90.0)


# --------------------------------------------------------------------------------------
# V2-7. Differential against the ACTUAL committed pre-change files (git show HEAD:...),
#       not against a hand transcription. A transcription can be wrong in the same way
#       the implementation is; `git show` cannot.
# --------------------------------------------------------------------------------------


import importlib.util as _ilu
import tempfile as _tempfile


def _module_at_head(relpath, modname):
    """Import the HEAD (pre-change) version of a source file as a separate module."""
    blob = subprocess.run(
        ["git", "show", f"HEAD:{relpath}"],
        cwd=_ROOT, capture_output=True, text=True,
    )
    if blob.returncode != 0:
        pytest.skip(f"cannot read HEAD:{relpath}: {blob.stderr.strip()}")
    tmpdir = _tempfile.mkdtemp()
    path = pathlib.Path(tmpdir) / f"{modname}.py"
    path.write_text(blob.stdout)
    spec = _ilu.spec_from_file_location(modname, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _head_get_value_bounds():
    """Extract the committed pre-change `get_value_bounds` and exec it standalone.

    The whole module cannot be imported out of tree (`from __future__ import annotations`
    plus dataclasses that resolve their annotations against the real package), and the
    function only needs builtins -- so lift exactly that function definition out of the
    HEAD blob with `ast` and exec it. Still the committed text, not a transcription.
    """
    blob = subprocess.run(
        ["git", "show", "HEAD:src/rl/value_distribution.py"],
        cwd=_ROOT, capture_output=True, text=True,
    )
    if blob.returncode != 0:
        pytest.skip(f"cannot read HEAD blob: {blob.stderr.strip()}")
    tree = ast.parse(blob.stdout)
    fn = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "get_value_bounds"
    )
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<head>", "exec"), ns)
    return ns["get_value_bounds"]


def test_head_value_distribution_differential_over_every_registered_config():
    """Compare the committed pre-change `get_value_bounds` tuple-for-tuple."""
    head_fn = _head_get_value_bounds()
    head = type("H", (), {"get_value_bounds": staticmethod(head_fn)})
    changed, unchanged = [], []
    for name in _REGISTERED:
        cfg = _config.get_config(name)
        if not hasattr(cfg.rl, "critic"):
            continue
        before = head.get_value_bounds(cfg)
        after = get_value_bounds(cfg)
        (changed if before != after else unchanged).append(name)
        if cfg.collect.use_time_to_success_as_reward:
            assert after[0] == pytest.approx(-1.0 / (1.0 - cfg.rl.discount), rel=1e-12)
            assert after[1] == before[1] == 0.0
            assert after[0] < before[0]
        else:
            assert after == before
    # Every time-to-success config sees a different LOWER bound; none sees a different
    # upper bound, because no registered config sets a bonus.
    assert changed, "expected the lower bound to move for the time-to-success configs"


def test_head_value_distribution_differential_is_a_real_change_at_201_bins():
    """The committed sweep `scripts/configs/tuning/bofn/1_.../*.yaml` sets
    `rl.critic.num_value_bins: 201` on `pi05_libero_online_best_of_n` in 2 of its 4 arms.
    That is a reachable, checked-in configuration -- so the change is NOT inert there."""
    head_fn = _head_get_value_bounds()
    head = type("H", (), {"get_value_bounds": staticmethod(head_fn)})
    cfg = _config.get_config("pi05_libero_online_best_of_n")
    cfg = dataclasses.replace(
        cfg,
        rl=dataclasses.replace(
            cfg.rl,
            critic=dataclasses.replace(cfg.rl.critic, num_value_bins=201),
        ),
    )
    before = head.get_value_bounds(cfg)
    after = get_value_bounds(cfg)
    assert before != after
    # Every one of the 201 bin centers moves, and the pinned failure return (-200) goes
    # from OUT of range (clipped by `_discretize`) to IN range.
    from src.rl.value_distribution import make_bin_centers
    c_before = np.asarray(make_bin_centers(before[0], before[1], 201))
    c_after = np.asarray(make_bin_centers(after[0], after[1], 201))
    assert not np.allclose(c_before, c_after)
    pinned = -1.0 / (1.0 - cfg.rl.discount)
    assert pinned < c_before.min(), "the old range could not represent the pinned return"
    assert pinned >= c_after.min() - 1e-6


def test_the_sweep_yaml_that_makes_the_bounds_non_inert_is_actually_checked_in():
    """Named explicitly so the claim is falsifiable from the repo, not from memory."""
    hits = []
    for p in (_ROOT / "scripts" / "configs").rglob("*.yaml"):
        text = p.read_text()
        if "num_value_bins" in text and "201" in text:
            hits.append(p.relative_to(_ROOT))
    assert hits, "expected at least one sweep YAML setting num_value_bins > 1"


def test_head_wrapper_differential_over_a_randomized_flag_script():
    """Differential of the shipped wrapper's default against the committed one."""
    head = _module_at_head("src/envs/wrappers.py", "_head_wrappers")
    OldWrapper = head.TimeToSuccessAsRewardWrapper
    rng = np.random.default_rng(1234)
    for _ in range(60):
        script = [
            (bool(a), bool(b)) for a, b in (rng.random((15, 2)) < 0.3)
        ]
        new = TimeToSuccessAsRewardWrapper(_ScriptedEnv(script))
        old = OldWrapper(_ScriptedEnv(script))
        new.reset(); old.reset()
        for _ in script:
            rn, ro = new.step(None), old.step(None)
            assert rn[1] == ro[1] and type(rn[1]) is type(ro[1])
            assert (rn[2], rn[3]) == (ro[2], ro[3])
    # And the old wrapper genuinely cannot take the new kwarg -- i.e. the shipped tests
    # for bonus behaviour would fail on a revert, they are not tautological.
    with pytest.raises(TypeError):
        OldWrapper(_ScriptedEnv([(True, False)]), success_bonus=72.0)
