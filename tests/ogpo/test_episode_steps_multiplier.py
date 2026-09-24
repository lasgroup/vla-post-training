"""`CollectionConfig.episode_steps_multiplier` — the EP_MULT knob.

Adversarial verification of docs/changes/2026-08-27-episode-steps-multiplier/.
The claim under test is a behavior-preservation claim: at the default value 1
NOTHING moves, and the only thing that moves at 2 is the env TimeLimit (plus
the T in the discount>=1 value-bound fallback, which is dead code for every
registered config).

Everything here is pure Python arithmetic or closure inspection — no XLA
compile, no model, no simulator step — so tolerances are EXACT (`==`): both
sides of every comparison run the identical float expression, so any
difference at all is a real behavior change, not accumulation.

What these tests cannot cover, and why:
  * `src/envs/molmo.py:339` — `import src.envs.molmo` raises
    `ModuleNotFoundError: No module named 'molmo_spaces'` on CPU, so the molmo
    TimeLimit edit is read-verified only.
  * A real 800-step LIBERO rollout truncating at 800 needs a GPU + EGL.
"""

import dataclasses

import pytest

from src.rl.value_distribution import get_value_bounds
from src.training.config import CollectionConfig, _CONFIGS, get_config


# Only the online configs carry a `collect` block (11 of the 42 registered).
ONLINE_CONFIGS = [c for c in _CONFIGS if hasattr(c, "collect")]


def _with_mult(cfg, mult: int):
    return dataclasses.replace(
        cfg, collect=dataclasses.replace(cfg.collect, episode_steps_multiplier=mult)
    )


# --------------------------------------------------------------------------- #
# 1. The config field itself
# --------------------------------------------------------------------------- #

def test_default_is_one():
    assert CollectionConfig().episode_steps_multiplier == 1


def test_replace_to_two_round_trips_and_touches_nothing_else():
    base = CollectionConfig()
    two = dataclasses.replace(base, episode_steps_multiplier=2)
    assert two.episode_steps_multiplier == 2
    assert base.episode_steps_multiplier == 1, "frozen dataclass: the original must not mutate"
    # Every other field is byte-identical -- the field is additive, not a rename.
    changed = {
        f.name
        for f in dataclasses.fields(CollectionConfig)
        if getattr(base, f.name) != getattr(two, f.name)
    }
    assert changed == {"episode_steps_multiplier"}


def test_every_registered_online_config_leaves_the_field_at_one():
    # THE behavior-preservation guard: any registered config shipping a value
    # other than 1 silently changes episode length for that whole campaign.
    assert ONLINE_CONFIGS, "no online configs found -- the filter is wrong, not the repo"
    offenders = {
        c.name: c.collect.episode_steps_multiplier
        for c in ONLINE_CONFIGS
        if c.collect.episode_steps_multiplier != 1
    }
    assert offenders == {}


# --------------------------------------------------------------------------- #
# 2. get_value_bounds — the invariant the spec claims
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("cfg", ONLINE_CONFIGS, ids=[c.name for c in ONLINE_CONFIGS])
def test_value_bounds_identical_at_multiplier_1_and_2(cfg):
    # Exact equality, not a tolerance: every registered config has discount < 1,
    # where lower = -1/(1-gamma) and upper = success_reward_bonus -- neither
    # reads T -- so the two calls evaluate the SAME expression. Any drift here
    # means the change leaked into a live code path.
    if not hasattr(cfg.rl, "critic"):
        # filtered-SFT and DSRL learner configs carry no critic block, so
        # get_value_bounds is never called for them (same skip as
        # test_verifier_alignment.py:200-201).
        pytest.skip(f"{cfg.name} has no critic config")
    assert cfg.rl.discount < 1.0, "premise of this test: no registered config uses discount >= 1"
    assert get_value_bounds(_with_mult(cfg, 1)) == get_value_bounds(cfg)
    assert get_value_bounds(_with_mult(cfg, 2)) == get_value_bounds(cfg)


def test_value_bounds_do_scale_with_the_multiplier_at_discount_one():
    # The only live effect of the value_distribution.py:128 edit. Synthetic:
    # discount=1.0 is not reachable from any registered config, so this is the
    # one place the T-dependence is observable at all.
    cfg = dataclasses.replace(
        get_config("pi05_libero_online_ogpo_ref"),
        rl=dataclasses.replace(get_config("pi05_libero_online_ogpo_ref").rl, discount=1.0),
    )
    assert cfg.collect.max_episode_steps == 400
    assert cfg.collect.use_time_to_success_as_reward is True
    assert cfg.rl.critic.num_value_bins == 1, "half-bin padding would make these bounds non-integral"
    assert get_value_bounds(_with_mult(cfg, 1)) == (-400.0, 0.0)
    assert get_value_bounds(_with_mult(cfg, 2)) == (-800.0, 0.0)
    assert get_value_bounds(_with_mult(cfg, 3)) == (-1200.0, 0.0)


def test_value_bounds_sparse_reward_path_stays_T_independent_at_any_multiplier():
    # The `else` branch (0, 1) never reads T; guard that the edit did not move
    # the T computation above the branch in a way that changes it.
    base = get_config("pi05_libero_online_ogpo_ref")
    cfg = dataclasses.replace(
        base, collect=dataclasses.replace(base.collect, use_time_to_success_as_reward=False)
    )
    assert get_value_bounds(_with_mult(cfg, 1)) == get_value_bounds(_with_mult(cfg, 7)) == (0.0, 1.0)


def test_explicit_bound_override_still_short_circuits_the_multiplier():
    # An explicit (lower, upper) wins before T is ever computed.
    base = get_config("pi05_libero_online_ogpo_ref")
    cfg = dataclasses.replace(
        base,
        rl=dataclasses.replace(
            base.rl,
            discount=1.0,
            critic=dataclasses.replace(
                base.rl.critic, value_lower_bound=-5.0, value_upper_bound=5.0
            ),
        ),
    )
    assert get_value_bounds(_with_mult(cfg, 1)) == (-5.0, 5.0)
    assert get_value_bounds(_with_mult(cfg, 9)) == (-5.0, 5.0)


# --------------------------------------------------------------------------- #
# 3. tyro CLI — the flag the recipe emits must actually land
# --------------------------------------------------------------------------- #

def _parse(name, *args):
    import tyro

    from src.training.config import OnlineTrainConfig

    return tyro.cli(OnlineTrainConfig, default=get_config(name), args=["--exp-name", "t", *args])


def test_cli_flag_lands_and_default_is_one():
    # `scripts/ogpo_multitask_4task.sh:283` emits this flag UNCONDITIONALLY,
    # including `... 1` when EP_MULT is unset -- so both legs are on the hot path.
    assert _parse("pi05_libero_online_ogpo_sft").collect.episode_steps_multiplier == 1
    assert (
        _parse(
            "pi05_libero_online_ogpo_sft", "--collect.episode_steps_multiplier", "1"
        ).collect.episode_steps_multiplier
        == 1
    )
    assert (
        _parse(
            "pi05_libero_online_ogpo_sft", "--collect.episode_steps_multiplier", "2"
        ).collect.episode_steps_multiplier
        == 2
    )


def test_cli_flag_at_one_leaves_the_resolved_config_untouched():
    # The recipe's `--collect.episode_steps_multiplier 1` must be a true no-op.
    plain = _parse("pi05_libero_online_ogpo_sft")
    explicit = _parse("pi05_libero_online_ogpo_sft", "--collect.episode_steps_multiplier", "1")
    assert plain.collect == explicit.collect


# --------------------------------------------------------------------------- #
# 4. LIBERO env construction (CPU-reachable: the TimeLimit arg is computed
#    before any MuJoCo context exists, so the closure can be inspected without
#    a GPU).
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def libero():
    return pytest.importorskip(
        "src.envs.libero", reason="LIBERO/robosuite not installed in this env"
    )


def test_suite_max_steps_map_is_unchanged(libero):
    # The multiplier must not have been folded into the map itself.
    assert libero.get_max_steps_libero("libero_90") == 400
    assert libero.get_max_steps_libero("libero_spatial") == 220
    assert libero.get_max_steps_libero("libero_object") == 280
    assert libero.get_max_steps_libero("libero_goal") == 300
    assert libero.get_max_steps_libero("libero_10") == 520


def _closure_max_steps(libero, cfg, task):
    import inspect

    env_fn = libero.make_env_libero(cfg, tasks=[task], num_devices=1)
    return inspect.getclosurevars(env_fn).nonlocals["max_steps"]


@pytest.mark.parametrize(
    ("task", "base"), [("libero_90_79", 400), ("libero_10_0", 520)]
)
def test_make_env_libero_scales_the_timelimit_by_the_multiplier(libero, task, base):
    # `max_steps` is what is passed straight into TimeLimit(max_episode_steps=...)
    # at libero.py:117-120; nothing else touches it between the two lines.
    cfg = get_config("pi05_libero_online_ogpo_sft")
    assert _closure_max_steps(libero, _with_mult(cfg, 1), task) == base
    assert _closure_max_steps(libero, _with_mult(cfg, 2), task) == 2 * base
    assert _closure_max_steps(libero, _with_mult(cfg, 3), task) == 3 * base


def test_make_env_libero_now_requires_the_field_on_its_config(libero):
    """Contract pin: `make_env_libero` reads
    `config.collect.episode_steps_multiplier` unconditionally, so any stub
    config must carry the field. The three ad-hoc `SimpleNamespace` stubs that
    call it (scripts/repro_egl_drain.py, scripts/egl_safe_probe.py x2) were
    updated to pass episode_steps_multiplier=1 when the verifier flagged them
    (docs/changes/2026-08-27-episode-steps-multiplier/VERIFICATION.md, F-A).
    """
    import types

    stub = types.SimpleNamespace(
        collect=types.SimpleNamespace(env_resolution=224, num_steps_wait=10)
    )
    with pytest.raises(AttributeError, match="episode_steps_multiplier"):
        libero.make_env_libero(stub, tasks=["libero_90_79"], num_devices=1)


def test_multiplier_below_one_is_rejected_at_config_construction():
    """F-B fix: gymnasium's TimeLimit truncates on elapsed >= max_episode_steps,
    so a 0 or negative multiplier truncates every episode at step 0 -- a run
    that trains and means nothing. OnlineTrainConfig.__post_init__ must reject
    it (dataclasses.replace on the frozen config re-runs __post_init__)."""
    cfg = get_config("pi05_libero_online_ogpo_sft")
    for bad in (0, -1):
        with pytest.raises(ValueError, match="episode_steps_multiplier"):
            _with_mult(cfg, bad)
    # And >= 1 still constructs.
    assert _with_mult(cfg, 2).collect.episode_steps_multiplier == 2
