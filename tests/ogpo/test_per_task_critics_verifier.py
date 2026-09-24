# ruff: noqa: F722
"""Independent (step-3 verifier) tests for per-task critics.

Written from the change spec + diff only, to cover claims that
``tests/ogpo/test_per_task_critics.py`` states but does not exercise, and to pin
behaviour the verifier found unguarded. Kept in a separate file so the additions
are distinguishable from the implementing session's tests.

What is here and why:

* ``test_unbound_online_batch_to_sft_batch_...`` — REGRESSION. The 4-tuple
  override reads ``self._task_registry`` unconditionally, which breaks the
  existing unbound call in ``tests/ogpo/test_split_equivalence.py:610``.
* ``test_missing_task_index_in_critic_batch_raises`` — the "find it before a GPU
  run does" case: a critic step under ``num_tasks`` with no ``task_index``.
* ``test_q_td_target_bootstraps_next_obs_under_the_same_task`` — D1's actual
  claim (Q's TD target must use the SAME task's V), untested upstream.
* ``test_critic_optimizer_per_task_clip_decouples_tasks_end_to_end`` — the clip
  decision (D5) at the ``tx`` the TrainState actually holds, not at the exposed
  ``per_task_clip_chain`` helper.
* ``test_jit1_g_expansion_routes_each_state`` — the docstring claim "routing
  survives the G-expansion". The upstream jit-1 tests run at
  ``group_num_samples=1``, where the expansion is a no-op.
* ``test_per_task_mean_matches_numpy_reference_3d`` — ``*heads`` with two
  leading axes, and a balanced batch reducing to the plain mean.
* Two ``KNOWN_GAP`` tests pin unguarded behaviour discovered by the verifier
  (out-of-range slot routing; ``TaskRegistry.from_json`` not bounding slots by
  ``num_tasks``). They assert what the code does today so a future change that
  tightens either one fails loudly here.

Tolerances: ``0.0`` where the graph is literally identical (routing/gather,
absent-task no-op), ``1e-6`` where float32 reassociation is possible.
"""
import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import openpi.training.sharding as sharding
from src.rl.advantage_weighted_sft.advantage_weighted_sft_learner import (
    AdvantageWeightedSFTLearner,
)
from src.rl.advantage_weighted_sft.update_critic import (
    _critic_optimizer,
    per_task_mean,
    train_q_step,
)
from src.rl.filtered_sft_agent.filtered_sft_learner import FilteredSFTLearner
from src.rl.networks.bronet_critic import BroNetStateActionCritic, BroNetStateValue
from src.rl.networks.per_task_critic import (
    TASK_INDEX_NAME,
    PerTaskStateActionCritic,
    _gather_task,
)
from src.rl.ogpo.ogpo_learner import OGPOAgentLearner
from src.rl.prefix_embedding import PREFIX_EMBEDDING_NAME
from src.rl.task_registry import TaskRegistry
from src.training.config import get_config

_ATOL_EXACT = 0.0
_ATOL_F32 = 1e-6
_S, _E, _AH, _AD = 8, 16, 2, 4


# --------------------------------------------------------------------------- #
# helpers (kept independent of the implementing session's fixtures)
# --------------------------------------------------------------------------- #

def _cfg(num_tasks, **critic_kw):
    base = get_config("pi05_libero_online_ogpo_sft")
    critic = dataclasses.replace(
        base.rl.critic, use_bronet=True, bronet_hidden_dim=16, bronet_depth=1,
        num_qs=2, num_vs=2, num_tasks=num_tasks, **critic_kw,
    )
    return dataclasses.replace(base, rl=dataclasses.replace(base.rl, critic=critic))


def _defs(config):
    hd, dep = config.rl.critic.bronet_hidden_dim, config.rl.critic.bronet_depth
    nq, nv = config.rl.critic.num_qs, config.rl.critic.num_vs

    def sa(o, a, rngs):
        return BroNetStateActionCritic(
            observation=o, action=a, hidden_dim=hd, depth=dep, num_qs=nq, rngs=rngs
        )

    def sv(o, rngs):
        return BroNetStateValue(observation=o, hidden_dim=hd, depth=dep, num_vs=nv, rngs=rngs)

    return sa, sv


def _obs(rng, b, task_index=None):
    k1, k2 = jax.random.split(rng)
    o = {
        "state": jax.random.normal(k1, (b, _S)),
        PREFIX_EMBEDDING_NAME: jax.random.normal(k2, (b, _E)),
    }
    if task_index is not None:
        o[TASK_INDEX_NAME] = jnp.asarray(task_index, jnp.int32)
    return o


def _batch(rng, task_index):
    b = len(task_index)
    k1, k2, k3, k4 = jax.random.split(rng, 4)
    return (
        _obs(k1, b, task_index),
        jax.random.normal(k3, (b, _AH, _AD)),
        _obs(k2, b, task_index),
        -jnp.ones((b,)),
        jnp.full((b,), 0.99),
        -100.0 + 50.0 * jax.random.normal(k4, (b,)),
    )


def _states(config, rng):
    from src.rl.advantage_weighted_sft.update_critic import (
        init_state_action_critic_train_state,
        init_state_value_train_state,
    )
    mesh = sharding.make_mesh(1)
    sa, sv = _defs(config)
    qk, vk = jax.random.split(rng)
    dummy = {"state": jnp.zeros((1, _S)), PREFIX_EMBEDDING_NAME: jnp.zeros((1, _E))}
    if config.rl.critic.num_tasks is not None:
        dummy[TASK_INDEX_NAME] = jnp.zeros((1,), jnp.int32)
    q, _ = init_state_action_critic_train_state(
        config, qk, mesh, critic_def=sa, dummy_obs=dummy, dummy_act=jnp.zeros((1, _AH, _AD))
    )
    v, _ = init_state_value_train_state(config, vk, mesh, critic_def=sv, dummy_obs=dummy)
    return q, v


def _scale_task(state, t, factor):
    """Multiply every Param leaf of slot ``t`` (params AND ema_params) by ``factor``."""
    def _m(tree):
        return tree.map(lambda p, v: v.replace(v.value * factor) if p[1] == t else v)
    return dataclasses.replace(
        state,
        params=_m(state.params),
        ema_params=None if state.ema_params is None else _m(state.ema_params),
    )


def _observation_dict(b=2):
    return {
        "state": jnp.zeros((b, _S)),
        "image": {"base_0_rgb": jnp.zeros((b, 4, 4, 3), jnp.uint8)},
        "image_mask": {"base_0_rgb": jnp.ones((b,), bool)},
        "tokenized_prompt": jnp.zeros((b, 8), jnp.int32),
        "tokenized_prompt_mask": jnp.ones((b, 8), bool),
    }


# --------------------------------------------------------------------------- #
# A. REGRESSION in the existing suite
# --------------------------------------------------------------------------- #

def test_unbound_online_batch_to_sft_batch_still_works_with_stub_self():
    """``tests/ogpo/test_split_equivalence.py::test_g_stored_prefix_threading``
    calls this override UNBOUND with ``self=None`` (the override "never touches
    self"). The 4-tuple version reads ``self._task_registry`` unconditionally, so
    the existing leg now dies with AttributeError before it can unpack.

    Gating on a class attribute default would keep the old call working; reading
    it off ``self`` does not.
    """
    obs = _observation_dict()
    obs[PREFIX_EMBEDDING_NAME] = jnp.ones((2, _E))
    batch = {"observation": obs, "actions": jnp.zeros((2, _AH, _AD))}
    out = OGPOAgentLearner._online_batch_to_sft_batch(None, batch)
    assert len(out) == 4
    assert out[3] is None, "no registry => no task_index sidecar"


def test_filtered_sft_gates_are_class_level_defaults():
    # Both gates must exist as class attributes so an unbound/partially built
    # learner reads None rather than raising (see the regression above).
    assert FilteredSFTLearner._num_critic_tasks is None
    assert FilteredSFTLearner._task_registry is None
    assert OGPOAgentLearner._task_registry is None


# --------------------------------------------------------------------------- #
# B. A critic call under num_tasks with no task_index must raise, not default
# --------------------------------------------------------------------------- #

def test_missing_task_index_in_critic_batch_raises():
    config = _cfg(2)
    q_state, v_state = _states(config, jax.random.key(0))
    obs, act, nobs, r, d, mc = _batch(jax.random.key(1), [0, 1, 0, 1])
    stripped = (
        {k: v for k, v in obs.items() if k != TASK_INDEX_NAME},
        act,
        {k: v for k, v in nobs.items() if k != TASK_INDEX_NAME},
        r, d, mc,
    )
    with pytest.raises(KeyError, match=TASK_INDEX_NAME):
        train_q_step(config, jax.random.key(2), q_state, v_state, stripped)


# --------------------------------------------------------------------------- #
# C. D1: Q's TD target bootstraps V(next_obs) under the SAME task
# --------------------------------------------------------------------------- #

def test_q_td_target_bootstraps_next_obs_under_the_same_task():
    config = _cfg(2)  # td_weight schedule of the base config is TD-only at step 0
    q_state, v_state = _states(config, jax.random.key(0))
    batch = _batch(jax.random.key(1), [0, 0, 0, 0])  # every sample is task 0
    rng = jax.random.key(2)

    base_q, base_info = train_q_step(config, rng, q_state, v_state, batch)
    other_v = _scale_task(v_state, 1, 3.0)      # perturb the ABSENT task's V
    same_v = _scale_task(v_state, 0, 3.0)       # perturb the PRESENT task's V

    q_other, info_other = train_q_step(config, rng, q_state, other_v, batch)
    q_same, info_same = train_q_step(config, rng, q_state, same_v, batch)

    for a, b in zip(jax.tree.leaves(base_q.params), jax.tree.leaves(q_other.params)):
        np.testing.assert_allclose(
            np.asarray(a), np.asarray(b), atol=_ATOL_EXACT,
            err_msg="another task's V must not reach a task-0 batch's TD target",
        )
    assert float(base_info["td_loss"]) == float(info_other["td_loss"])
    assert float(base_info["td_loss"]) != float(info_same["td_loss"]), (
        "the OWN task's V must reach the TD target (otherwise the test is vacuous)"
    )


# --------------------------------------------------------------------------- #
# D. D5: the per-task clip at the tx the TrainState actually holds
# --------------------------------------------------------------------------- #

def test_task_masks_partition_the_whole_param_tree():
    """``_task_subtree_mask`` returns False for any keypath it does not recognise
    (``len(keypath) < 2`` / no ``DictKey('tasks')``), so a future change to how the
    wrapper stores its sub-critics would make every mask all-False and silently
    disable critic gradient clipping entirely. Nothing in the code asserts that
    the T masks cover the tree; assert it here.
    """
    from src.rl.advantage_weighted_sft.update_critic import _task_subtree_mask
    T = 3
    config = _cfg(T)
    sa, _ = _defs(config)
    dummy = {"state": jnp.zeros((1, _S)), PREFIX_EMBEDDING_NAME: jnp.zeros((1, _E)),
             TASK_INDEX_NAME: jnp.zeros((1,), jnp.int32)}
    q = PerTaskStateActionCritic(sa, T, dummy, jnp.zeros((1, _AH * _AD)), rngs=nnx.Rngs(jax.random.key(0)))
    params = nnx.filter_state(nnx.state(q), nnx.Param)
    n_leaves = len(jax.tree.leaves(params))
    assert n_leaves > 0
    covered = np.zeros(n_leaves, dtype=int)
    for t in range(T):
        covered += np.array([int(bool(m)) for m in jax.tree.leaves(_task_subtree_mask(T, t)(params))])
    assert (covered == 1).all(), (
        f"the {T} task masks must partition the Param tree exactly once each; got {set(covered.tolist())}"
    )


def test_critic_optimizer_per_task_clip_decouples_tasks_end_to_end():
    """Task 1's Adam update must be invariant to task 0's gradient magnitude.

    Tested on ``_critic_optimizer(config)`` (what ``init_*_train_state`` installs),
    not on the exposed clip helper. The ``num_tasks=None`` optimizer is included as
    the positive control: one global clip DOES couple them.
    """
    config_pt, config_none = _cfg(2), _cfg(None)
    sa, _ = _defs(config_pt)
    dummy = {"state": jnp.zeros((1, _S)), PREFIX_EMBEDDING_NAME: jnp.zeros((1, _E)),
             TASK_INDEX_NAME: jnp.zeros((1,), jnp.int32)}
    q = PerTaskStateActionCritic(sa, 2, dummy, jnp.zeros((1, _AH * _AD)), rngs=nnx.Rngs(jax.random.key(0)))
    params = nnx.filter_state(nnx.state(q), nnx.Param)

    def grads(task0_scale):
        return params.map(
            lambda p, v: v.replace(jnp.full_like(v.value, task0_scale if p[1] == 0 else 1e-3))
        )

    def task1_update(tx, g):
        u, _ = tx.update(g, tx.init(params), params)
        return [np.asarray(v.value) for p, v in u.flat_state() if p[1] == 1]

    tx_pt = _critic_optimizer(config_pt)
    small, big = task1_update(tx_pt, grads(1e-3)), task1_update(tx_pt, grads(1e3))
    for a, b in zip(small, big):
        np.testing.assert_allclose(a, b, atol=_ATOL_EXACT,
                                   err_msg="per-task clip: task 1 must not feel task 0")

    tx_gl = _critic_optimizer(config_none)
    small_g, big_g = task1_update(tx_gl, grads(1e-3)), task1_update(tx_gl, grads(1e3))
    assert any(not np.allclose(a, b) for a, b in zip(small_g, big_g)), (
        "control: a single global clip must couple the tasks (else the test is vacuous)"
    )


# --------------------------------------------------------------------------- #
# E. per_task_mean: numpy reference with two leading axes + balanced batch
# --------------------------------------------------------------------------- #

def test_per_task_mean_matches_numpy_reference_3d():
    rng = np.random.default_rng(0)
    per_sample = jnp.asarray(rng.normal(size=(2, 3, 7)).astype(np.float32))  # heads (2,3), b=7
    idx = np.array([0, 3, 0, 3, 3, 0, 0], np.int32)  # tasks 1 and 2 absent
    T = 4
    got = float(per_task_mean(per_sample, jnp.asarray(idx), T))

    ps = np.asarray(per_sample).reshape(-1, 7)  # (6, 7)
    per_task = []
    for t in range(T):
        m = idx == t
        if not m.any():
            continue
        per_task.append(ps[:, m].mean(axis=1).mean())  # mean over b, then over heads
    expected = float(np.mean(per_task))
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=_ATOL_F32)


def test_per_task_mean_balanced_batch_equals_plain_mean():
    # Equal counts per task => the per-task reduction collapses to jnp.mean.
    per_sample = jax.random.normal(jax.random.key(1), (3, 8))
    idx = jnp.asarray([0, 0, 1, 1, 2, 2, 3, 3], jnp.int32)
    np.testing.assert_allclose(
        float(per_task_mean(per_sample, idx, 4)), float(jnp.mean(per_sample)),
        rtol=1e-6, atol=_ATOL_F32,
    )


# --------------------------------------------------------------------------- #
# F. jit-1 with G > 1 (the untested half of the sidecar claim)
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def g2_fx():
    """jit-1 fixture at group_num_samples=2, so ``_expand`` on task_index is live."""
    import functools

    # NOT ``from tests.ogpo.test_split_equivalence import ...``: a site-packages
    # package named ``tests`` shadows the repo's ``tests/`` directory (which has no
    # __init__.py), so that spelling raises ModuleNotFoundError. It is what
    # tests/ogpo/test_per_task_critics.py::actor_fx uses, which is why both of its
    # jit-1 tests ERROR at setup. pytest imports this file's siblings under the
    # ``ogpo`` package (tests/ogpo/__init__.py exists, tests/ does not), so try
    # that first and fall back to the bare module name.
    import importlib
    _se = None
    for _name in ("ogpo.test_split_equivalence", "test_split_equivalence"):
        try:
            _se = importlib.import_module(_name)
            break
        except ImportError:
            continue
    assert _se is not None, "could not import test_split_equivalence's builders"
    _B = _se._B
    _build_config, _build_critics = _se._build_config, _se._build_critics
    _build_policy_state, _make_params = _se._build_policy_state, _se._make_params
    _original_recompute = _se._original_recompute
    from src.rl.ogpo.update_actor import sample_and_advantage

    config = _build_config()
    config = dataclasses.replace(
        config, rl=dataclasses.replace(config.rl, group_num_samples=2)
    )
    assert _B == 2 and config.rl.group_num_samples == 2
    model = config.model.create(jax.random.key(0))
    params = _make_params(config, model)
    mesh = sharding.make_mesh(1)
    policy_observation = config.model.fake_obs(batch_size=_B)
    model_cast = nnx.merge(nnx.graphdef(model), params)
    model_cast.eval()
    prefix = _original_recompute(model_cast, policy_observation)
    policy_state = _build_policy_state(config, model, params, ema_params=None)
    ema = nnx.filter_state(params, config.trainable_filter)

    config_pt = dataclasses.replace(
        config,
        rl=dataclasses.replace(
            config.rl, critic=dataclasses.replace(config.rl.critic, num_tasks=2)
        ),
    )
    q_shared, v_shared = _build_critics(config, model, mesh, jax.random.key(1))
    q_pt, v_pt = _build_critics(config_pt, model, mesh, jax.random.key(1))

    def _fill(pt_state, shared_state):
        flat = dict(pt_state.params.flat_state())
        for p, v in shared_state.params.flat_state():
            for t in range(2):
                flat[("tasks", t) + tuple(p)] = v
        new_params = nnx.State.from_flat_path(flat)
        return dataclasses.replace(pt_state, params=new_params, ema_params=new_params)

    return dict(
        sa_jit=jax.jit(functools.partial(sample_and_advantage, config_pt)),
        sa_jit_shared=jax.jit(functools.partial(sample_and_advantage, config)),
        policy_state=policy_state, q_shared=q_shared, v_shared=v_shared,
        q_pt=_fill(q_pt, q_shared), v_pt=_fill(v_pt, v_shared),
        policy_observation=policy_observation, prefix=prefix, ema=ema, G=2,
    )


def test_jit1_task_index_none_is_bit_identical_to_the_7_arg_call(g2_fx):
    """The upstream test making this claim never runs (its fixture ERRORs), so it
    is re-made here on shared critics: passing ``task_index=None`` must reproduce
    the 7-positional call exactly, at G=2."""
    fx = g2_fx
    args = (jax.random.key(7), fx["policy_state"], fx["q_shared"], fx["v_shared"],
            fx["policy_observation"], fx["prefix"], fx["ema"])
    ref = fx["sa_jit_shared"](*args)
    got = fx["sa_jit_shared"](*args, None)
    for a, b in zip(jax.tree.leaves(ref), jax.tree.leaves(got)):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_jit1_g_expansion_routes_each_state(g2_fx):
    """B=2 states, G=2 samples each. State 0 is task 0, state 1 is task 1.
    Perturbing slot 1's Q must move BOTH of state 1's samples and NEITHER of
    state 0's — i.e. task_index is repeated (not tiled) alongside the states.
    """
    fx = g2_fx
    task_index = jnp.asarray([0, 1], jnp.int32)

    def call(jit_fn, q, v, *extra):
        return jit_fn(jax.random.key(7), fx["policy_state"], q, v,
                      fx["policy_observation"], fx["prefix"], fx["ema"], *extra)

    ref = call(fx["sa_jit"], fx["q_pt"], fx["v_pt"], task_index)
    shared = call(fx["sa_jit_shared"], fx["q_shared"], fx["v_shared"])
    adv_ref, adv_shared = np.asarray(ref[5]), np.asarray(shared[5])
    assert adv_ref.shape == (2 * fx["G"],)
    np.testing.assert_allclose(adv_ref, adv_shared, atol=1e-5,
                               err_msg="cloned slots must reproduce the shared critic at G=2")

    bumped = fx["q_pt"].params.map(lambda p, v: v.replace(v.value * 1.5) if p[1] == 1 else v)
    q_bumped = dataclasses.replace(fx["q_pt"], params=bumped, ema_params=bumped)
    out = np.asarray(call(fx["sa_jit"], q_bumped, fx["v_pt"], task_index)[5])

    # reshape(B, G): rows are states, columns are that state's G samples.
    ref_bg, out_bg = adv_ref.reshape(2, fx["G"]), out.reshape(2, fx["G"])
    np.testing.assert_allclose(out_bg[0], ref_bg[0], atol=1e-6,
                               err_msg="state 0 (task 0) must not see slot 1's Q")
    assert np.all(np.abs(out_bg[1] - ref_bg[1]) > 1e-6), (
        "every one of state 1's G samples must be scored by slot 1"
    )


# --------------------------------------------------------------------------- #
# G. Learner-construction guards (all raise before config.model.create)
# --------------------------------------------------------------------------- #

def _mt_config(name="pi05_libero_online_ogpo_sft", *, num_tasks, tasks, eval_tasks=None, n_samples=None):
    base = get_config(name)
    rl = dataclasses.replace(
        base.rl, critic=dataclasses.replace(base.rl.critic, num_tasks=num_tasks)
    )
    if n_samples is not None:
        rl = dataclasses.replace(rl, n_samples=n_samples)
    collect = dataclasses.replace(
        base.collect, tasks=list(tasks),
        eval_tasks=list(eval_tasks if eval_tasks is not None else tasks),
    )
    return dataclasses.replace(base, rl=rl, collect=collect)


def test_awr_subclass_without_support_raises():
    cfg = _mt_config(num_tasks=2, tasks=["libero_90_79", "libero_90_31"])
    with pytest.raises(ValueError, match="OGPOAgentLearner"):
        AdvantageWeightedSFTLearner.__init__(
            object.__new__(AdvantageWeightedSFTLearner), cfg
        )


def test_num_tasks_must_equal_distinct_train_tasks():
    cfg = _mt_config(num_tasks=4, tasks=["libero_90_79", "libero_90_31"])
    with pytest.raises(ValueError, match="distinct task"):
        OGPOAgentLearner.__init__(object.__new__(OGPOAgentLearner), cfg)


def test_bofn_scoring_with_heldout_eval_tasks_raises():
    cfg = _mt_config(
        num_tasks=2, tasks=["libero_90_79", "libero_90_31"],
        eval_tasks=["libero_90_79", "libero_90_31", "libero_90_82"], n_samples=8,
    )
    with pytest.raises(ValueError, match="no critic"):
        OGPOAgentLearner.__init__(object.__new__(OGPOAgentLearner), cfg)


def test_registered_pertask_config_is_self_consistent():
    """Verifier finding F7 (fixed): ``pi05_libero_online_ogpo_sft_pertask`` pins
    num_tasks=4, so it must ALSO carry a 4-distinct-task collect set or the R3
    equality check rejects the registered name standalone. It now bakes in the
    mt4 set (the same four ids ``scripts/ogpo_multitask_4task.sh`` uses)."""
    cfg = get_config("pi05_libero_online_ogpo_sft_pertask")
    assert cfg.rl.critic.num_tasks == 4
    assert len(set(cfg.collect.tasks)) == 4
    assert set(cfg.collect.eval_tasks) <= set(cfg.collect.tasks), (
        "eval on held-out tasks would have no critic under best-of-N"
    )
    assert cfg.collect.tasks == ["libero_90_79", "libero_90_31", "libero_90_82", "libero_90_38"]


# --------------------------------------------------------------------------- #
# G1. Buffer schema (D3 / D8) — the transition field, exercised on the real buffer
# --------------------------------------------------------------------------- #

def _buffer_dummy(with_task_index: bool):
    d = {
        "observations": {
            "state": np.zeros((1, _S), np.float32),
            "image": np.zeros((1, 4, 4, 3), np.uint8),
        },
        "actions": np.zeros((1, _AH, _AD), np.float32),
        "reward": np.zeros((1,), np.float32),
        "mc_return": np.zeros((1,), np.float32),
        "discount": np.zeros((1,), np.float32),
        "is_success": np.zeros((1,), np.float32),
    }
    if with_task_index:
        d["task_index"] = np.zeros((1,), np.int32)
    return d


def _insert_episode(buf, task_index, n, obs_scale):
    data = {
        "observations": {
            "state": np.full((n + 1, _S), obs_scale, np.float32),
            "image": np.zeros((n + 1, 4, 4, 3), np.uint8),
        },
        "obs_index": np.arange(n, dtype=np.int64),
        "next_obs_index": np.arange(n, dtype=np.int64) + 1,
        "actions": np.zeros((n, _AH, _AD), np.float32),
        "reward": -np.ones((n,), np.float32),
        "mc_return": -np.ones((n,), np.float32),
        "discount": np.full((n,), 0.99, np.float32),
        "is_success": np.zeros((n,), np.float32),
    }
    if task_index is not None:
        data["task_index"] = np.full((n,), task_index, np.int32)
    buf.insert(data)


def test_task_index_is_a_transition_field_that_survives_sample_and_shards(tmp_path):
    """D3's plumbing claim, on the real buffer: ``task_index`` is a transition
    field, so it lands TOP-LEVEL in every sampled batch (not under
    ``observation``), survives ``drop_obs_keys``, and round-trips through
    ``save_shard`` / ``restore_shards``. Also D8: inserting a pre-change (no
    task_index) transition into a per-task buffer raises.
    """
    from src.rl.replay_buffer import ShardedReplayBuffer

    buf = ShardedReplayBuffer(dummy_data=_buffer_dummy(True), max_capacity=64, seed=0, freeze_dict=False)
    _insert_episode(buf, 0, 4, 1.0)
    _insert_episode(buf, 2, 4, 2.0)

    batch = buf.sample(batch_size=8, drop_obs_keys=("image",), ordinals=np.arange(8))
    assert "task_index" in batch and "task_index" not in batch["observation"]
    assert np.asarray(batch["task_index"]).dtype == np.int32
    np.testing.assert_array_equal(np.asarray(batch["task_index"]), [0] * 4 + [2] * 4)
    assert "image" not in batch["observation"], "drop_obs_keys must not touch the transition field"

    shard_dir = tmp_path / "shards"
    buf.save_shard(shard_dir / "step_000000.h5")
    buf2 = ShardedReplayBuffer(dummy_data=_buffer_dummy(True), max_capacity=64, seed=0, freeze_dict=False)
    buf2.restore_shards(shard_dir)
    back = buf2.sample(batch_size=8, ordinals=np.arange(8))
    np.testing.assert_array_equal(np.asarray(back["task_index"]), [0] * 4 + [2] * 4)

    # D8: a transition without the field cannot enter a per-task buffer.
    with pytest.raises(ValueError, match="structure"):
        _insert_episode(buf, None, 2, 3.0)


def test_num_tasks_none_buffer_has_no_task_index_key():
    """The gate keeps the pre-change schema byte-for-byte for the other five
    learners: no ``task_index`` key at all, so pre-change shards still restore."""
    from src.rl.replay_buffer import ShardedReplayBuffer

    buf = ShardedReplayBuffer(dummy_data=_buffer_dummy(False), max_capacity=32, seed=0, freeze_dict=False)
    _insert_episode(buf, None, 4, 1.0)
    batch = buf.sample(batch_size=4, ordinals=np.arange(4))
    assert "task_index" not in batch
    with pytest.raises(ValueError, match="structure"):
        _insert_episode(buf, 1, 2, 2.0)


# --------------------------------------------------------------------------- #
# G2. Checkpoint layout (Tier-2 trigger, untested upstream)
# --------------------------------------------------------------------------- #

def test_per_task_critic_state_orbax_round_trip_and_shared_mismatch_raises(tmp_path):
    """The critic param tree gains a task level, so (a) it must survive the
    ``StandardCheckpointer`` round trip ``_restore_rl_checkpoint`` uses, and
    (b) a shared-critic checkpoint must NOT structure-match a per-task template
    (D8: new-run only, fail fast). Both exercised at the orbax layer with the
    same ``{"state_action_critic_state": ..., "value_state": ...}`` shape the
    learner saves.
    """
    import orbax.checkpoint as ocp

    cfg_pt, cfg_none = _cfg(2), _cfg(None)
    q_pt, v_pt = _states(cfg_pt, jax.random.key(0))
    q_sh, v_sh = _states(cfg_none, jax.random.key(0))
    ckptr = ocp.StandardCheckpointer()

    path = tmp_path / "rl_state" / "100"
    ckptr.save(path, {"state_action_critic_state": q_pt, "value_state": v_pt})
    restored = ckptr.restore(path, {"state_action_critic_state": q_pt, "value_state": v_pt})
    for a, b in zip(jax.tree.leaves(q_pt.params),
                    jax.tree.leaves(restored["state_action_critic_state"].params)):
        assert np.array_equal(np.asarray(a), np.asarray(b))

    shared_path = tmp_path / "rl_state" / "200"
    ckptr.save(shared_path, {"state_action_critic_state": q_sh, "value_state": v_sh})
    with pytest.raises(Exception):  # orbax raises a structure/shape error, not a typed one
        ckptr.restore(shared_path, {"state_action_critic_state": q_pt, "value_state": v_pt})


# --------------------------------------------------------------------------- #
# H. KNOWN GAPS — pinned as-is so a future tightening fails here on purpose
# --------------------------------------------------------------------------- #

def test_gather_out_of_range_slot_yields_nan_not_a_raise_KNOWN_GAP():
    """``_gather_task`` does not bounds-check; ``jnp.take_along_axis``'s default
    out-of-bounds mode fills with NaN rather than raising or clamping.

    Good news: an out-of-range slot poisons the value with NaN (which then
    propagates through ``per_task_mean``'s einsum, since NaN*0 is NaN), so it
    surfaces as a NaN loss rather than as a silently-wrong-task score. Bad news:
    it surfaces only at runtime, with no message naming the slot. Nothing in the
    change can produce one (the registry bounds every index); a corrupted or
    foreign buffer shard could.
    """
    outs = jnp.asarray(np.arange(2 * 1 * 3, dtype=np.float32).reshape(2, 1, 3))  # (T=2, n=1, b=3)
    got = np.asarray(_gather_task(outs, jnp.asarray([0, 1, 7], jnp.int32)))
    assert got.shape == (1, 3)
    assert got[0, 0] == outs[0, 0, 0] and got[0, 1] == outs[1, 0, 1]
    assert np.isnan(got[0, 2]), "out-of-range slot must be NaN-filled, not clamped to a real task"


def test_registry_keyed_on_task_id_separates_ids_that_share_a_description():
    """Verifier finding F1 (fixed): slots are keyed on the task ID, so two ids with
    the same LIBERO language string get distinct critics and the registry fills
    all ``num_tasks`` slots. (The pre-fix registry keyed on the description and
    collapsed 79/82 onto one slot, leaving a fourth critic never trained.)
    """
    reg = TaskRegistry(4)
    ids_to_desc = {
        "libero_90_79": "pick up the book and place it in the left compartment of the caddy",
        "libero_90_31": "put the black bowl on top of the cabinet",
        "libero_90_82": "pick up the book and place it in the left compartment of the caddy",
        "libero_90_38": "put the right moka pot on the stove",
    }
    slots = {i: reg.index_for(i) for i in ids_to_desc}  # keyed on the ID, description unused
    assert slots["libero_90_79"] != slots["libero_90_82"], "same description, distinct ids => distinct slots"
    assert len(reg) == reg.num_tasks == 4


def test_libero_90_mt4_task_descriptions_actually_collide():
    """The description collision is not hypothetical: it is the mt4 recipe's own
    task list (``scripts/ogpo_multitask_4task.sh:70``) -- which is WHY the registry
    keys on the task id. Skipped when LIBERO's
    first-run config is absent (importing it would block on input())."""
    import os
    if not os.path.exists(os.path.join(os.path.expanduser("~"), ".libero", "config.yaml")):
        pytest.skip("LIBERO first-run config absent; import would prompt interactively")
    benchmark = pytest.importorskip("libero.libero.benchmark")
    suite = benchmark.get_benchmark_dict()["libero_90"]()
    langs = {i: suite.get_task(i).language for i in (79, 31, 82, 38)}
    assert langs[79] == langs[82], langs
    assert len(set(langs.values())) == 3, (
        f"mt4 runs 4 task ids but only {len(set(langs.values()))} distinct descriptions: {langs}"
    )


def test_from_json_bounds_slots_by_num_tasks(tmp_path):
    """Verifier finding F10 (fixed): a sidecar with more registered tasks than
    slots must be rejected at load, not hand out an out-of-range slot (which
    ``_gather_task`` would NaN-fill rather than raise on)."""
    import json
    path = tmp_path / "task_registry_0.json"
    path.write_text(json.dumps({"num_tasks": 2, "tasks": {"a": 0, "b": 1, "c": 2}}))
    with pytest.raises(ValueError, match="holds 3 tasks"):
        TaskRegistry.from_json(path, 2)
