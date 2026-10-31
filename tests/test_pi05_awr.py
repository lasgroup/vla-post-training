"""
Verifier tests for the pi05 paper-AWR change (docs/changes/2026-10-22-pi05-awr/).

  a. q_helper.compute_lambda_returns vs an independent forward-view (weighted n-step) reference.
  b. ReplayBuffer.insertion_order / write_field, sample_sequence's lambda_returns pass-through.
  c. pi05_online_runner._recompute_lambda_returns on a wrapped buffer (chunk < N, not dividing N).
  d. pg_helper.compute_awr_weights + differential tests of both refactored call sites vs VERBATIM pre-change code.
  e. OGPOPi05Agent AWR surface on CPU (small Value nets, stubbed `_bc_loss_per_sample`, no VLM):
     value_loss (lambda_return), value_update, _awr_terms, actor_update, grad_diagnostics(None, ...),
     and the PPO `_update_impl` vs a VERBATIM copy of the pre-change `_update_impl`.
  f. create() validation (stops at the checkpoint load) and main_pi05 run-level validation / wiring.
  g. The new script: every forwarded flag exists, and its resolved config passes create() / main_pi05 validation.
  h. run_pi05_round_rl under awr with a stub agent / envs, including a resume at a round boundary and mid-round.

Run:  JAX_PLATFORMS=cpu uv run --no-sync pytest tests/test_pi05_awr.py -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import copy
import re
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: F401  (used by the verbatim old _update_impl)

import ogpo.runners.pi05_online_runner as R
from ogpo.agents.modules.flax_utils import ModuleDict, TrainState
from ogpo.agents.modules.pg_helper import compute_awr_cfm_loss, compute_awr_weights, compute_ogpo_advantages
from ogpo.agents.modules.q_helper import compute_lambda_returns
from ogpo.agents.ogpo_pi05 import OGPOPi05Agent, PREFIX_DIM, STATE_DIM, T
from ogpo.networks import Value
from ogpo.utils.config_loader import AttrDict
from ogpo.utils.datasets import ReplayBuffer
from ogpo.utils.sharding import zero_shardings  # noqa: F401  (used by the verbatim old _update_impl)

ROOT = os.path.join(os.path.dirname(__file__), '..')


# =====================================================================================================================
# a. compute_lambda_returns
# =====================================================================================================================

def _ref_lambda_returns(rewards, masks, terminals, v_next, gamma, lam):
    """Independent forward view: R^λ_t = (1-λ) Σ_{n<L} λ^{n-1} G^(n)_t + λ^{L-1} G^(L)_t per episode, where L is the
    number of steps left in t's episode (ends at a terminal or at the segment end) and
    G^(n)_t = Σ_{k<n} γ^k r_{t+k} + γ^n m_{t+n-1} V(s_{t+n})."""
    n = len(rewards)
    ends = [i for i in range(n) if terminals[i] > 0.5 or i == n - 1]
    out = np.zeros(n, np.float64)
    start = 0
    for e in ends:
        for t in range(start, e + 1):
            L = e - t + 1
            G = []
            for m in range(1, L + 1):
                g = sum(gamma ** k * rewards[t + k] for k in range(m))
                g += gamma ** m * masks[t + m - 1] * v_next[t + m - 1]
                G.append(g)
            if L == 1:
                out[t] = G[0]
            else:
                out[t] = (1 - lam) * sum(lam ** (m - 1) * G[m - 1] for m in range(1, L)) + lam ** (L - 1) * G[L - 1]
        start = e + 1
    return out


def _segment(seed=0):
    """Episodes: partial (truncated tail of an older episode, terminated), terminated, truncated, terminated of
    length 1, then an open tail (last element not terminal)."""
    rs = np.random.RandomState(seed)
    lens_kinds = [(3, 'term'), (5, 'term'), (4, 'trunc'), (1, 'term'), (2, 'trunc'), (3, 'open')]
    r, m, d = [], [], []
    for L, kind in lens_kinds:
        for i in range(L):
            last = i == L - 1
            r.append(rs.randn())
            m.append(0.0 if (last and kind == 'term') else 1.0)
            d.append(1.0 if (last and kind in ('term', 'trunc')) else 0.0)
    n = len(r)
    return (np.array(r, np.float32), np.array(m, np.float32), np.array(d, np.float32),
            rs.randn(n).astype(np.float32))


@pytest.mark.parametrize('lam', [0.0, 0.3, 0.95, 1.0])
@pytest.mark.parametrize('gamma', [0.99, 0.8])
def test_lambda_returns_match_forward_view(lam, gamma):
    r, m, d, vn = _segment()
    got = compute_lambda_returns(r, m, d, vn, gamma, lam)
    ref = _ref_lambda_returns(r, m, d, vn, gamma, lam)
    assert got.dtype == np.float32 and got.shape == r.shape
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)


def test_lambda_returns_lambda0_is_one_step_target():
    r, m, d, vn = _segment(1)
    np.testing.assert_allclose(compute_lambda_returns(r, m, d, vn, 0.97, 0.0), r + 0.97 * m * vn, rtol=1e-6, atol=1e-6)


def test_lambda_returns_lambda1_terminated_is_mc_return():
    rs = np.random.RandomState(2)
    r = rs.randn(6).astype(np.float32)
    m = np.ones(6, np.float32); m[-1] = 0.0
    d = np.zeros(6, np.float32); d[-1] = 1.0
    vn = rs.randn(6).astype(np.float32) * 100  # V must not matter at λ=1 on a terminated episode
    mc = np.array([sum(0.9 ** k * r[t + k] for k in range(6 - t)) for t in range(6)])
    np.testing.assert_allclose(compute_lambda_returns(r, m, d, vn, 0.9, 1.0), mc, rtol=1e-5, atol=1e-5)
    # truncated instead: MC plus the discounted bootstrap at the last step
    m_tr = np.ones(6, np.float32)
    boot = np.array([0.9 ** (6 - t) * vn[-1] for t in range(6)])
    np.testing.assert_allclose(compute_lambda_returns(r, m_tr, d, vn, 0.9, 1.0), mc + boot, rtol=1e-5, atol=1e-3)


def test_lambda_returns_restart_at_episode_ends():
    """The value after a terminal does not leak backwards into the earlier episode."""
    r, m, d, vn = _segment(3)
    base = compute_lambda_returns(r, m, d, vn, 0.99, 0.95)
    r2 = r.copy(); r2[3:] += 50.0  # change only episodes 2+ (episode 1 = indices 0..2)
    out = compute_lambda_returns(r2, m, d, vn, 0.99, 0.95)
    np.testing.assert_array_equal(out[:3], base[:3])
    assert np.all(np.abs(out[3:] - base[3:]) > 1.0)


# =====================================================================================================================
# b. ReplayBuffer
# =====================================================================================================================

def _small_rb(max_size, lam_field=True):
    schema = dict(observations=np.zeros(3, np.float32), next_observations=np.zeros(3, np.float32),
                  actions=np.zeros(2, np.float32), rewards=np.array(0.0, np.float32),
                  terminals=np.array(0.0, np.float32), masks=np.array(1.0, np.float32),
                  mc_returns=np.array(0.0, np.float32), task_ids=np.array(0, np.int32))
    if lam_field:
        schema['lambda_returns'] = np.array(0.0, np.float32)
    return ReplayBuffer.create(schema, max_size, numpy_rng=np.random.default_rng(0))


def _tr(i, lam_field=True, term=0.0, mask=1.0, reward=0.0):
    t = dict(observations=np.array([i, 0, 0], np.float32), next_observations=np.array([i + 1, 0, 0], np.float32),
             actions=np.zeros(2, np.float32), rewards=np.float32(reward), terminals=np.float32(term),
             masks=np.float32(mask), mc_returns=np.float32(0), task_ids=np.int32(i % 2),
             is_success=0.0, score=1e-9)
    if lam_field:
        t['lambda_returns'] = 0.0
    return t


def test_insertion_order_before_and_after_wrap():
    rb = _small_rb(7)
    assert rb.insertion_order().shape == (0,)
    for i in range(5):
        rb.add_transition(_tr(i))
    np.testing.assert_array_equal(rb.insertion_order(), np.arange(5))
    for i in range(5, 7):  # exactly full, pointer back at 0
        rb.add_transition(_tr(i))
    np.testing.assert_array_equal(rb.insertion_order(), np.arange(7))
    for i in range(7, 9):
        rb.add_transition(_tr(i))
    order = rb.insertion_order()
    np.testing.assert_array_equal(order, [2, 3, 4, 5, 6, 0, 1])
    assert order.dtype == np.int64
    np.testing.assert_array_equal(rb['observations'][order, 0], np.arange(2, 9))  # oldest -> newest
    for i in range(9, 25):  # several wraps
        rb.add_transition(_tr(i))
    np.testing.assert_array_equal(rb['observations'][rb.insertion_order(), 0], np.arange(18, 25))


def test_write_field():
    rb = _small_rb(5)
    for i in range(5):
        rb.add_transition(_tr(i))
    rb.write_field('lambda_returns', np.array([4, 0, 2]), np.array([1.5, 2.5, 3.5], np.float32))
    np.testing.assert_array_equal(rb['lambda_returns'], [2.5, 0, 3.5, 0, 1.5])
    with pytest.raises(KeyError, match='lambda_return'):
        _small_rb(5, lam_field=False).write_field('lambda_returns', np.array([0]), np.array([1.0]))


def test_sample_sequence_lambda_returns_at_window_start():
    rb = _small_rb(40)
    for i in range(40):
        rb.add_transition(_tr(i, term=float(i % 7 == 6)))
    rb.write_field('lambda_returns', np.arange(40), 100.0 + np.arange(40, dtype=np.float32))
    for per_task in (None, 2):
        b = rb.sample_sequence(64, sequence_length=4, discount=0.9, per_task=per_task)
        start = b['observations'][:, 0].astype(int)
        assert b['lambda_returns'].shape == (64,)
        np.testing.assert_array_equal(b['lambda_returns'], 100.0 + start)
        np.testing.assert_array_equal(b['mc_returns'].shape, (64,))
    # buffers without the field: key absent, rest unchanged
    rb2 = _small_rb(40, lam_field=False)
    for i in range(40):
        rb2.add_transition(_tr(i, lam_field=False))
    assert 'lambda_returns' not in rb2.sample_sequence(8, sequence_length=3, discount=0.9)


def test_flush_episode_tags_lambda_field_only_when_present():
    for lam_field in (False, True):
        rb = _small_rb(10, lam_field=lam_field)
        traj = [{k: v for k, v in _tr(i, lam_field=False, term=float(i == 2), reward=-1.0).items()
                 if k not in ('is_success', 'score', 'mc_returns')} for i in range(3)]
        R._flush_episode(rb, traj, {'success': 0.0}, 0.9)  # tree_map over the schema must match exactly
        assert rb.size == 3 and ('lambda_returns' in rb) == lam_field
        if lam_field:
            np.testing.assert_array_equal(rb['lambda_returns'][:3], 0.0)


# =====================================================================================================================
# c. _recompute_lambda_returns
# =====================================================================================================================

class _VStub:
    """state_values(x) = deterministic function of the cached features; records every call's input shape."""

    def __init__(self):
        self.shapes = []

    @staticmethod
    def f(x):
        return 0.3 * x[:, 0] - 0.05 * x[:, 1] + 1.0

    def state_values(self, x):
        self.shapes.append(np.asarray(x).shape)
        return jnp.asarray(self.f(np.asarray(x)))


def _fill_episodes(rb, n_eps, seed=0):
    """Contiguous episodes of random length 1..6, terminated or truncated at the end; obs[0] = global index."""
    rs = np.random.RandomState(seed)
    i = 0
    for _ in range(n_eps):
        L, kind = rs.randint(1, 7), rs.choice(['term', 'trunc'])
        for s in range(L):
            last = s == L - 1
            t = _tr(i, term=float(last), mask=0.0 if (last and kind == 'term') else 1.0, reward=rs.randn())
            t['observations'] = np.array([i, rs.randn(), 0], np.float32)
            t['next_observations'] = np.array([i + 1, rs.randn(), 0], np.float32)
            rb.add_transition(t)
            i += 1
    return i


@pytest.mark.parametrize('max_size,n_eps,chunk', [(23, 12, 5), (200, 8, 7), (23, 12, 64)])
def test_recompute_lambda_returns_insertion_order(max_size, n_eps, chunk):
    rb = _small_rb(max_size)
    total = _fill_episodes(rb, n_eps)
    wrapped = total > max_size
    if max_size == 23:
        assert wrapped and rb.pointer != 0  # the case that matters
    agent = _VStub()
    gamma, lam = 0.97, 0.9
    stats = R._recompute_lambda_returns(agent, rb, gamma, lam, chunk=chunk)
    order = rb.insertion_order()
    obs, nobs = rb['observations'][order], rb['next_observations'][order]
    ref = _ref_lambda_returns(rb['rewards'][order], rb['masks'][order], rb['terminals'][order], _VStub.f(nobs), gamma, lam)
    np.testing.assert_allclose(rb['lambda_returns'][order], ref, rtol=1e-5, atol=1e-5)
    # every V call is one fixed-size chunk (no recompiles), 2 x ceil(N / chunk) calls
    assert all(s == (chunk, 3) for s in agent.shapes)
    assert len(agent.shapes) == 2 * -(-rb.size // chunk)
    v = _VStub.f(obs)
    np.testing.assert_allclose(stats['lambda_return_mean'], ref.mean(), rtol=1e-5)
    np.testing.assert_allclose(stats['v_pre_fit_mean'], v.mean(), rtol=1e-5)
    np.testing.assert_allclose(stats['adv_pre_fit_mean'], (ref - v).mean(), rtol=1e-4, atol=1e-5)
    # teeth: the physical index order gives a different answer once the buffer has wrapped
    if wrapped:
        phys = np.arange(rb.size)
        wrong = _ref_lambda_returns(rb['rewards'][phys], rb['masks'][phys], rb['terminals'][phys],
                                    _VStub.f(rb['next_observations'][phys]), gamma, lam)
        assert not np.allclose(rb['lambda_returns'][phys], wrong, atol=1e-4)


def test_recompute_lambda_returns_v_mc_stats():
    """V-vs-MC diagnostics: whole buffer and the newest n_new transitions, against numpy on insertion order."""
    rb = _small_rb(23)
    _fill_episodes(rb, 12)
    order = rb.insertion_order()
    n = order.shape[0]
    rs = np.random.RandomState(1)
    rb.write_field('mc_returns', order, rs.randn(n).astype(np.float32) * 3 - 5)
    rb._dict['is_success'][order] = (rs.rand(n) > 0.6).astype(np.float32)
    stats = R._recompute_lambda_returns(_VStub(), rb, 0.97, 0.9, chunk=5, n_new=9)
    v = _VStub.f(rb['observations'][order]).astype(np.float64)
    mc = rb['mc_returns'][order].astype(np.float64)
    succ = rb['is_success'][order] > 0.5
    rank = lambda x: np.argsort(np.argsort(x))
    for pre, sl in (('vmc_', slice(0, n)), ('vmc_new_', slice(n - 9, n))):
        vv, mm, ss = v[sl], mc[sl], succ[sl]
        assert stats[pre + 'corr'] == pytest.approx(np.corrcoef(vv, mm)[0, 1], abs=1e-6)
        assert stats[pre + 'spearman'] == pytest.approx(np.corrcoef(rank(vv), rank(mm))[0, 1], abs=1e-6)
        assert stats[pre + 'bias'] == pytest.approx((vv - mm).mean(), abs=1e-5)
        assert stats[pre + 'rmse'] == pytest.approx(np.sqrt(((vv - mm) ** 2).mean()), abs=1e-5)
        assert stats[pre + 'explained_var'] == pytest.approx(1 - (vv - mm).var() / mm.var(), abs=1e-6)
        assert stats[pre + 'v_success'] == pytest.approx(vv[ss].mean(), abs=1e-5)
        assert stats[pre + 'mc_fail'] == pytest.approx(mm[~ss].mean(), abs=1e-5)
    lam = rb['lambda_returns'][order].astype(np.float64)
    assert stats['target_mc_corr'] == pytest.approx(np.corrcoef(lam, mc)[0, 1], abs=1e-5)
    assert stats['adv_success_mean'] == pytest.approx((lam - v)[succ].mean(), abs=1e-4)
    tasks = rb['task_ids'][order][n - 9:]
    for k in np.unique(tasks):
        m = tasks == k
        exp = np.corrcoef(v[n - 9:][m], mc[n - 9:][m])[0, 1] if m.sum() > 1 else np.nan
        np.testing.assert_allclose(stats[f'vmc_new_corr/task{k}'], exp, atol=1e-6)
    # V == MC exactly -> corr 1, EV 1, bias 0
    rb.write_field('mc_returns', order, _VStub.f(rb['observations'][order]).astype(np.float32))
    s2 = R._recompute_lambda_returns(_VStub(), rb, 0.97, 0.9, chunk=5)
    assert s2['vmc_corr'] == pytest.approx(1.0) and s2['vmc_explained_var'] == pytest.approx(1.0, abs=1e-6)
    assert abs(s2['vmc_bias']) < 1e-5 and 'vmc_new_corr' not in s2  # n_new=0: no held-out block


# =====================================================================================================================
# d. compute_awr_weights + differential tests vs the pre-change inline code
# =====================================================================================================================

A_VALS = jnp.array([-1.0, -0.1, 0.0, 0.05, 0.1, 0.149, 0.2, 1.0, 20.0])


def test_awr_weights_clip_after_exp():
    beta, wmax = 0.05, 20.0
    w = np.asarray(compute_awr_weights(A_VALS, beta, wmax))
    np.testing.assert_allclose(w, np.minimum(np.exp(np.asarray(A_VALS, np.float64) / beta), wmax), rtol=1e-5)
    assert w[2] == 1.0                                  # A = 0 -> 1
    assert w.max() == 20.0 and (w == 20.0).sum() == 3   # capped exactly at ω_max (A = 0.2, 1, 50)
    assert w[5] < 20.0                                  # exp(2.98) = 19.7: just below the cap, not clipped
    # clipping A/β at 20 before the exp would give e^20, not 20
    assert w.max() < np.exp(20.0) / 1e6


def test_awr_weights_modes_and_unknown():
    beta, wmax = 0.5, 3.0
    a = np.asarray(A_VALS, np.float64)
    sym = np.minimum(np.exp(a / beta), wmax)
    asym = np.minimum(np.exp(np.maximum(a, 0) / beta), wmax)
    pos = np.minimum(np.exp(a / beta) * (a > 0), wmax)
    for mode, ref in (('symmetric', sym), ('asymmetric', asym), ('positive_only', pos)):
        np.testing.assert_allclose(np.asarray(compute_awr_weights(A_VALS, beta, wmax, mode)), ref, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(compute_awr_weights(A_VALS, beta, wmax)), sym, rtol=1e-5)  # default
    with pytest.raises(ValueError, match='symmetric'):
        compute_awr_weights(A_VALS, beta, wmax, 'symetric')


def _old_ogpo_awr_branch(q_agg, q_full, config):
    """VERBATIM pre-change `compute_ogpo_advantages` awr branch (git HEAD 369c8a0)."""
    if config.adv_strategy == 'awr':
        advantages = q_agg - q_agg.mean(axis=0, keepdims=True)
        awr_mode = config.get('awr_mode', 'symmetric')
        if awr_mode == 'asymmetric':
            weights = jnp.exp(jnp.maximum(0.0, advantages) / config.awr_beta)
        elif awr_mode == 'positive_only':
            weights = jnp.exp(advantages / config.awr_beta) * (advantages > 0).astype(jnp.float32)
        else:  # symmetric (default, matches original behavior)
            weights = jnp.exp(advantages / config.awr_beta)
        awr_weight_max = config.get('awr_weight_max', 20.0)
        weights = jnp.clip(weights, 0.0, awr_weight_max)
        return weights, q_agg
    else:
        raise ValueError(f"Unknown adv_strategy: {config.adv_strategy}")


def _old_compute_awr_cfm_loss(actor_fn, observations, actions, advantages, n_mc, rng, awr_beta, awr_weight_max=20.0,
                              awr_mode='symmetric'):
    """VERBATIM pre-change `pg_helper.compute_awr_cfm_loss` body (git HEAD 369c8a0)."""
    batch_size = observations.shape[0]
    act_dim = actions.shape[-1]

    # 1. Compute AWR weights based on mode
    if awr_mode == 'asymmetric':
        weights = jnp.exp(jnp.maximum(0.0, advantages) / awr_beta)
    elif awr_mode == 'positive_only':
        weights = jnp.exp(advantages / awr_beta) * (advantages > 0).astype(jnp.float32)
    else:  # symmetric (default)
        weights = jnp.exp(advantages / awr_beta)

    weights = jnp.clip(weights, 0.0, awr_weight_max)
    weight_sum = weights.mean() + 1e-8
    weights = weights / weight_sum  # normalize so mean weight ≈ 1

    # 2. Draw MC samples for CFM loss
    rng, tau_key, eps_key = jax.random.split(rng, 3)
    taus = jax.random.uniform(tau_key, (n_mc,))
    eps = jax.random.normal(eps_key, (n_mc, batch_size, act_dim))

    # 3. Compute noised actions and velocity targets (OT flow matching)
    taus_expanded = taus[:, None, None]
    x_tau = taus_expanded * actions[None] + (1.0 - taus_expanded) * eps
    vel_target = actions[None] - eps

    # 4. Compute CFM loss for each MC sample
    def single_mc_cfm_loss(x_tau_i, vel_target_i, tau_i):
        t = jnp.full((batch_size, 1), tau_i)
        v_pred = actor_fn(observations, x_tau_i, t)
        return jnp.sum(jnp.square(v_pred - vel_target_i), axis=-1)  # [batch]

    cfm_losses = jax.vmap(single_mc_cfm_loss)(x_tau, vel_target, taus)  # [n_mc, batch]
    cfm_loss_per_sample = cfm_losses.mean(axis=0)  # [batch]

    # 5. Weighted loss
    loss = jnp.mean(weights * cfm_loss_per_sample)

    stats = {
        'awr_weight_mean': weights.mean(),
        'awr_weight_std': weights.std(),
        'awr_weight_max': weights.max(),
        'awr_weight_min': weights.min(),
        'awr_cfm_loss_mean': cfm_loss_per_sample.mean(),
        'awr_cfm_loss_std': cfm_loss_per_sample.std(),
        'awr_frac_positive': (advantages > 0).astype(jnp.float32).mean(),
    }

    return loss, stats


@pytest.mark.parametrize('mode', ['symmetric', 'asymmetric', 'positive_only', None])
@pytest.mark.parametrize('wmax', [20.0, None])
def test_compute_ogpo_advantages_awr_matches_old(mode, wmax):
    k = jax.random.split(jax.random.PRNGKey(5), 2)
    q_agg = jax.random.normal(k[0], (8, 6)) * 0.4
    q_full = jax.random.normal(k[1], (8, 3, 6))
    cfg = {'adv_strategy': 'awr', 'awr_beta': 0.1}
    if mode is not None:
        cfg['awr_mode'] = mode
    if wmax is not None:
        cfg['awr_weight_max'] = wmax
    cfg = AttrDict(cfg)
    new_w, new_q = compute_ogpo_advantages(q_agg, q_full, cfg)
    old_w, old_q = _old_ogpo_awr_branch(q_agg, q_full, cfg)
    np.testing.assert_array_equal(np.asarray(new_w), np.asarray(old_w))
    np.testing.assert_array_equal(np.asarray(new_q), np.asarray(old_q))
    assert float(new_w.max()) <= 20.0 and float(new_w.max()) > 1.0


@pytest.mark.parametrize('mode', ['symmetric', 'asymmetric', 'positive_only'])
def test_compute_awr_cfm_loss_matches_old(mode):
    k = jax.random.split(jax.random.PRNGKey(9), 4)
    B, D, A = 7, 4, 3
    obs = jax.random.normal(k[0], (B, D))
    acts = jax.random.normal(k[1], (B, A))
    adv = jax.random.normal(k[2], (B,)) * 0.3
    W = jax.random.normal(k[3], (D + A + 1, A))

    def actor_fn(o, x, t):
        return jnp.tanh(jnp.concatenate([o, x, t], -1) @ W)

    new = compute_awr_cfm_loss(actor_fn, obs, acts, adv, 5, jax.random.PRNGKey(1), 0.1, 4.0, mode)
    old = _old_compute_awr_cfm_loss(actor_fn, obs, acts, adv, 5, jax.random.PRNGKey(1), 0.1, 4.0, mode)
    np.testing.assert_array_equal(np.asarray(new[0]), np.asarray(old[0]))
    assert new[1].keys() == old[1].keys()
    for key in old[1]:
        np.testing.assert_array_equal(np.asarray(new[1][key]), np.asarray(old[1][key]))
    # gradient through the loss also identical
    g_new = jax.grad(lambda w: compute_awr_cfm_loss(lambda o, x, t: jnp.tanh(jnp.concatenate([o, x, t], -1) @ w),
                                                    obs, acts, adv, 5, jax.random.PRNGKey(1), 0.1, 4.0, mode)[0])(W)
    g_old = jax.grad(lambda w: _old_compute_awr_cfm_loss(lambda o, x, t: jnp.tanh(jnp.concatenate([o, x, t], -1) @ w),
                                                         obs, acts, adv, 5, jax.random.PRNGKey(1), 0.1, 4.0, mode)[0])(W)
    np.testing.assert_array_equal(np.asarray(g_new), np.asarray(g_old))


def test_old_paths_typo_mode_now_raises():
    """Documented behaviour change: an unknown awr_mode used to fall through to symmetric; now it raises."""
    cfg = AttrDict({'adv_strategy': 'awr', 'awr_beta': 0.1, 'awr_mode': 'symetric'})
    q = jnp.ones((4, 2))
    _old_ogpo_awr_branch(q, None, cfg)  # silently symmetric
    with pytest.raises(ValueError):
        compute_ogpo_advantages(q, None, cfg)


# =====================================================================================================================
# e. Agent (CPU stub: small Value nets, no VLM)
# =====================================================================================================================

B, NUM_QS, NUM_VS, HL, HB, NT = 8, 3, 3, 5, 4, 2
OBS = PREFIX_DIM + STATE_DIM
GAMMA, TAU, ATAU, VLR, ALR = 0.99, 0.3, 0.2, 0.05, 0.1
BETA, WMAX = 0.05, 20.0


class _AWRStub(OGPOPi05Agent):
    """Real AWR / update code; `_bc_loss_per_sample` is a cheap param-dependent per-window loss (optionally RNG
    dependent via config['stub_rng']); the VLM prefix and SDE sampler raise."""

    def _prefix(self, images, task_ids=None, kv_only=False):  # kv_only: 2026-10-26 pi05_drop_empty_camera signature
        raise AssertionError("the AWR path must not run the VLM prefix")

    def _sample_sde(self, *a, **k):
        raise AssertionError("the AWR path must not sample SDE chains")

    def _bc_loss_per_sample(self, bc_batch, grad_params, rng):
        x = bc_batch['observations'][:, -STATE_DIM:]                       # (B, 8)
        pred = jnp.tanh(x @ grad_params['w'])                              # (B, 7)
        err = jnp.square(pred[:, None, :] - bc_batch['actions']).sum(-1)    # (B, HB)
        cfm = (err * bc_batch['valid']).mean(axis=1)
        return cfm + self.config.get('stub_rng', 0.0) * jax.random.normal(rng, cfm.shape) * jnp.sum(grad_params['w'] ** 2)


def _value_def(n):
    return Value(hidden_dims=(16, 16), layer_norm=True, num_ensembles=n, critic_loss_type='mse',
                 q_min=-100.0, q_max=0.0, obs_two_tier=True, two_tier_img_dim=PREFIX_DIM,
                 two_tier_proprio_dim=STATE_DIM, two_tier_fused_dim=0)


def _nets(seed=0):
    k = jax.random.split(jax.random.PRNGKey(seed), 4)
    qdef, vdef = _value_def(NUM_QS), _value_def(NUM_VS)
    qnet = ModuleDict({'critic': qdef, 'target_critic': copy.deepcopy(qdef)})
    vnet = ModuleDict({'value': vdef, 'target_value': copy.deepcopy(vdef)})
    exq = dict(observations=jnp.zeros((1, OBS)), actions=jnp.zeros((1, HL * T.ENV_ACTION_DIM)))
    exv = dict(observations=jnp.zeros((1, OBS)))
    qp = qnet.init(k[0], critic=exq, target_critic=exq)['params']
    qp['modules_target_critic'] = qnet.init(k[1], critic=exq, target_critic=exq)['params']['modules_critic']
    vp = vnet.init(k[2], value=exv, target_value=exv)['params']
    vp['modules_target_value'] = vnet.init(k[3], value=exv, target_value=exv)['params']['modules_value']
    return TrainState.create(qnet, qp, optax.sgd(VLR)), TrainState.create(vnet, vp, optax.sgd(VLR))


def _agent(cls=_AWRStub, K=1, v_agg='mean', objective='awr', donate=False, stub_rng=0.0, **kw):
    c = dict(num_qs=NUM_QS, v_ensemble_size=NUM_VS, v_agg=v_agg, q_agg='mean', discount=GAMMA, horizon_length=HL,
             tau=TAU, actor_tau=ATAU, use_success_buffer_q=False, q_variance_reduction=False, donate_agent=donate,
             grad_accum_steps=K, awr_beta=BETA, awr_weight_max=WMAX, awr_mode='symmetric', entropy_coeff=0.37,
             bc_coeff=0.7, stub_rng=stub_rng, use_bc_regularization=False,
             group_num_samples=2, ppo_batch_size=B, batch_size=B)
    if objective is not None:
        c['pi05_actor_objective'] = objective
        c['v_training_mode'] = 'lambda_return' if objective == 'awr' else 'sarsa'
    c.update(kw)
    consts = {'tokens': jnp.zeros((NT, 4), jnp.int32),
              'action_q01': -jnp.ones(T.MODEL_ACTION_DIM) * 0.8, 'action_q99': jnp.ones(T.MODEL_ACTION_DIM) * 1.3}
    q, v = _nets()
    w = jax.random.normal(jax.random.PRNGKey(42), (STATE_DIM, 7)) * 0.5
    tx = optax.sgd(ALR)
    return cls(rng=jax.random.PRNGKey(0), actor_params={'w': w}, target_actor_params={'w': w * 0.5 + 0.1},
               actor_opt_state=tx.init({'w': w}), init_actor_params={'w': jnp.copy(w)}, frozen_params={},
               critic_network=q, consts=consts, config=c, model=None, actor_tx=tx, value_network=v)


def _v_heads(agent, obs, params=None):
    p = agent.value_network.params if params is None else params
    return agent.value_network.select('value')(obs, params=p, is_encoded=True)


def _bc_batch(agent, seed=0, spread=0.3, v_agg='mean'):
    k = jax.random.split(jax.random.PRNGKey(200 + seed), 4)
    obs = jax.random.normal(k[0], (B, OBS))
    valid = np.ones((B, HB), np.float32)
    valid[0, 2:] = 0.0
    heads = np.asarray(_v_heads(agent, obs))
    v = heads.mean(axis=0) if v_agg == 'mean' else heads.min(axis=0)
    adv = np.linspace(-spread, spread, B).astype(np.float32)  # some weights hit ω_max, some do not
    return dict(images=jnp.zeros((B, 1)), observations=obs,
                actions=jax.random.uniform(k[1], (B, HB, 7), minval=-1, maxval=1), valid=jnp.asarray(valid),
                task_ids=jnp.asarray(np.arange(B) % NT, jnp.int32), lambda_returns=jnp.asarray(v + adv))


def _v_batch(seed=0):
    k = jax.random.split(jax.random.PRNGKey(300 + seed), 2)
    valid = np.ones((B, 1), np.float32)
    valid[3, 0] = 0.0
    return dict(observations=jax.random.normal(k[0], (B, OBS)), lambda_returns=jax.random.normal(k[1], (B,)) * 5,
                valid=jnp.asarray(valid), task_ids=jnp.asarray(np.arange(B) % NT, jnp.int32))


def _close(a, b, atol=1e-5, rtol=1e-5):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), atol=atol, rtol=rtol)


def _equal(a, b):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    return all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))


# ---- value loss / value_update ---------------------------------------------------------------------------------------

def test_value_loss_lambda_return_mode():
    agent, b = _agent(), _v_batch()  # no 'actions' in the batch: the critic path would KeyError
    loss, info = agent.value_loss(b, agent.value_network.params, jax.random.PRNGKey(1))
    v = _v_heads(agent, b['observations'])
    assert v.shape == (NUM_VS, B)
    ref = float((jnp.square(v - b['lambda_returns'][None]) * b['valid'][:, 0]).mean())
    np.testing.assert_allclose(float(loss), ref, rtol=1e-6)
    np.testing.assert_allclose(float(info['v_target_mean']), float(b['lambda_returns'].mean()), rtol=1e-6)
    assert 'v_target/task0' in info and 'v_mean/task1' in info


def _ref_value_step(agent, b):
    vs = agent.value_network
    w = b['valid'][:, -1]

    def loss(p):
        v = vs.select('value')(b['observations'], params=p, is_encoded=True)
        return (jnp.square(v - b['lambda_returns']) * w).mean()

    p = jax.tree.map(lambda x, g: x - VLR * g, vs.params, jax.grad(loss)(vs.params))
    p = dict(p)
    p['modules_target_value'] = jax.tree.map(lambda o, t: o * TAU + t * (1 - TAU), p['modules_value'],
                                             p['modules_target_value'])
    return p


@pytest.mark.parametrize('donate', [False, True])
def test_value_update_matches_hand_rolled(donate):
    agent, b = _agent(donate=donate), _v_batch()
    ref = _ref_value_step(agent, b)
    host = lambda t: jax.tree.map(np.array, t)  # noqa: E731  (donation deletes the old agent's buffers)
    old_q, old_actor, old_rng = host(agent.critic_network), host((agent.actor_params, agent.target_actor_params,
                                                                  agent.actor_opt_state)), host(agent.rng)
    old_step = int(agent.value_network.step)
    new, info = agent.value_update(b)
    _close(new.value_network.params, ref)
    assert int(new.value_network.step) == old_step + 1
    assert _equal(new.rng, jax.random.split(old_rng)[0])
    assert _equal(new.critic_network, old_q)
    assert _equal((new.actor_params, new.target_actor_params, new.actor_opt_state), old_actor)
    assert {'value/v_loss', 'value/v_grad/norm', 'value/v_target_mean'} <= set(info)
    assert all(k.startswith('value/') for k in info)


@pytest.mark.parametrize('K', [2, 4])
def test_value_update_grad_accum_matches_full(K):
    a1, aK, b = _agent(K=1), _agent(K=K), _v_batch()
    n1, i1 = a1.value_update(b)
    nK, iK = aK.value_update(b)
    _close(n1.value_network.params, nK.value_network.params)
    np.testing.assert_allclose(float(i1['value/v_loss']), float(iK['value/v_loss']), rtol=1e-5)


# ---- _awr_terms ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('v_agg', ['mean', 'min'])
def test_awr_terms_weights_and_loss(v_agg):
    agent = _agent(v_agg=v_agg)
    bc = _bc_batch(agent, v_agg=v_agg)
    rng = jax.random.PRNGKey(4)
    loss, ent, bcl, info = agent._awr_terms(bc, agent.actor_params, rng)
    heads = np.asarray(_v_heads(agent, bc['observations']))
    v = heads.mean(0) if v_agg == 'mean' else heads.min(0)
    adv = np.asarray(bc['lambda_returns']) - v
    w = np.minimum(np.exp(adv.astype(np.float64) / BETA), WMAX)
    assert 0 < (w >= WMAX).sum() < B and (w < 1).any()  # teeth: clipped, unclipped and down-weighted windows
    cfm = np.asarray(agent._bc_loss_per_sample(bc, agent.actor_params, rng))
    np.testing.assert_allclose(float(loss), (w * cfm).mean(), rtol=1e-4)
    assert float(ent) == 0.0 and float(bcl) == 0.0
    np.testing.assert_allclose(float(info['awr_ess']), w.sum() ** 2 / (B * (w ** 2).sum()), rtol=1e-4)
    np.testing.assert_allclose(float(info['awr_w_clip_frac']), (w >= WMAX).mean(), rtol=1e-6)
    np.testing.assert_allclose(float(info['awr_w_max']), WMAX, rtol=0)
    np.testing.assert_allclose(float(info['awr_adv_mean']), adv.mean(), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(float(info['awr_v_mean']), v.mean(), rtol=1e-5)
    np.testing.assert_allclose(float(info['cfm_loss']), cfm.mean(), rtol=1e-5)
    tids = np.asarray(bc['task_ids'])
    for k in range(NT):
        np.testing.assert_allclose(float(info[f'awr_w/task{k}']), w[tids == k].mean(), rtol=1e-4)
    # actor_loss / _actor_terms dispatch: entropy and BC coefficients (non-zero here) contribute nothing
    total, info2 = agent.actor_loss(None, bc, agent.actor_params, rng)
    np.testing.assert_allclose(float(total), float(loss), rtol=1e-6)
    assert agent._actor_terms(None, bc, agent.actor_params, rng)[0] == loss


def test_awr_terms_normalised_advantage():
    """awr_normalize_adv: w = min(exp(((R^λ - V) - m) / (s + 1e-5) / β), ω_max) with the batch's m / s fields."""
    agent = _agent(awr_normalize_adv=True, awr_beta=1.0)
    bc = _bc_batch(agent, spread=6.0)
    m, s = 0.7, 1.2
    bc = {**bc, 'awr_adv_mean': jnp.full((B,), m, jnp.float32), 'awr_adv_std': jnp.full((B,), s, jnp.float32)}
    rng = jax.random.PRNGKey(4)
    loss, _, _, info = agent._awr_terms(bc, agent.actor_params, rng)
    v = np.asarray(_v_heads(agent, bc['observations'])).mean(0)
    raw = np.asarray(bc['lambda_returns']) - v
    adv = (raw - m) / (s + 1e-5)
    w = np.minimum(np.exp(adv.astype(np.float64) / 1.0), WMAX)
    assert 0 < (w >= WMAX).sum() < B  # teeth: some clipped, some not
    cfm = np.asarray(agent._bc_loss_per_sample(bc, agent.actor_params, rng))
    np.testing.assert_allclose(float(loss), (w * cfm).mean(), rtol=1e-4)
    np.testing.assert_allclose(float(info['awr_adv_mean']), adv.mean(), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(float(info['awr_adv_raw_mean']), raw.mean(), rtol=1e-4, atol=1e-5)
    # off: the same batch ignores the fields
    off = _agent(awr_beta=1.0)
    loss_off, _, _, _ = off._awr_terms(bc, off.actor_params, rng)
    raw_off = np.asarray(bc['lambda_returns']) - np.asarray(_v_heads(off, bc['observations'])).mean(0)
    w_off = np.minimum(np.exp(raw_off.astype(np.float64)), WMAX)
    cfm_off = np.asarray(off._bc_loss_per_sample(bc, off.actor_params, rng))
    np.testing.assert_allclose(float(loss_off), (w_off * cfm_off).mean(), rtol=1e-4)
    assert not np.allclose(w_off, w)


def test_awr_terms_stop_gradient():
    agent = _agent()
    bc = _bc_batch(agent)
    rng = jax.random.PRNGKey(4)

    def via_v(vp):
        ag = agent.replace(value_network=agent.value_network.replace(params=vp))
        return ag._awr_terms(bc, agent.actor_params, rng)[0]

    g_v = jax.grad(via_v)(agent.value_network.params)
    assert all(float(jnp.abs(x).max()) == 0.0 for x in jax.tree.leaves(g_v))
    g_r = jax.grad(lambda r: agent._awr_terms({**bc, 'lambda_returns': r}, agent.actor_params, rng)[0])(bc['lambda_returns'])
    assert float(jnp.abs(g_r).max()) == 0.0
    # actor gradient = gradient of mean(w_const * cfm(p))
    _, _, _, info = agent._awr_terms(bc, agent.actor_params, rng)
    v = _v_heads(agent, bc['observations']).mean(0)
    w = jnp.minimum(jnp.exp((bc['lambda_returns'] - v) / BETA), WMAX)
    g_ref = jax.grad(lambda p: jnp.mean(w * agent._bc_loss_per_sample(bc, p, rng)))(agent.actor_params)
    g = jax.grad(lambda p: agent._awr_terms(bc, p, rng)[0])(agent.actor_params)
    _close(g, g_ref, atol=1e-6)
    assert float(jnp.abs(g['w']).max()) > 0


def test_ppo_objective_dispatches_to_ppo_machinery():
    """With pi05_actor_objective=ppo (or the key absent) `_actor_terms` enters the PPO path (here: the prefix pass)."""
    for objective in ('ppo', None):
        agent = _agent(objective=objective)
        bc = _bc_batch(agent)
        batch = {'images': jnp.zeros((B, 1)), 'observations': bc['observations']}
        with pytest.raises(AssertionError, match='VLM prefix'):
            agent._actor_terms(batch, bc, agent.actor_params, jax.random.PRNGKey(0),
                               ppo_batch={'images': jnp.zeros((B, 1)), 'observations': bc['observations']})


# ---- actor_update ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('donate', [False, True])
def test_actor_update_matches_hand_rolled(donate):
    agent = _agent(stub_rng=0.05, donate=donate)  # RNG-dependent loss: checks the key the step uses
    bc = _bc_batch(agent)
    new_rng, actor_rng = jax.random.split(agent.rng)
    loss_fn = lambda p: agent._awr_terms(bc, p, actor_rng)[0]  # noqa: E731
    g = jax.grad(loss_fn)(agent.actor_params)
    p_new = jax.tree.map(lambda p, gg: p - ALR * gg, agent.actor_params, g)
    t_new = jax.tree.map(lambda p, t: p * ATAU + t * (1 - ATAU), p_new, agent.target_actor_params)
    host = lambda t: jax.tree.map(np.array, t)  # noqa: E731  (donation deletes the old agent's buffers)
    old_v, old_q = host(agent.value_network), host(agent.critic_network)
    old_actor = host(agent.actor_params)
    ref_loss = float(loss_fn(old_actor))
    new_rng = np.array(new_rng)
    new, info = agent.actor_update(bc)
    _close(new.actor_params, p_new, atol=1e-6)
    _close(new.target_actor_params, t_new, atol=1e-6)
    assert _equal(new.rng, new_rng)
    assert _equal(new.value_network, old_v) and _equal(new.critic_network, old_q)
    np.testing.assert_allclose(float(info['actor/grad_norm']), float(optax.global_norm(g)), rtol=1e-5)
    np.testing.assert_allclose(float(info['actor/awr_loss']), ref_loss, rtol=1e-5)
    assert all(k.startswith('actor/') for k in info)
    assert not any(k.startswith(('critic/', 'value/')) for k in info)
    # teeth: a different key gives a different step
    g_other = jax.grad(lambda p: new._awr_terms(bc, p, jax.random.PRNGKey(123))[0])(old_actor)
    g = jax.grad(lambda p: new._awr_terms(bc, p, actor_rng)[0])(old_actor)
    assert not np.allclose(np.asarray(g_other['w']), np.asarray(g['w']), atol=1e-6)


@pytest.mark.parametrize('K', [2, 4])
def test_actor_update_grad_accum_matches_full(K):
    a1, aK = _agent(K=1), _agent(K=K)
    bc = _bc_batch(a1)
    n1, i1 = a1.actor_update(bc)
    nK, iK = aK.actor_update(bc)
    _close((n1.actor_params, n1.target_actor_params), (nK.actor_params, nK.target_actor_params), atol=1e-6)
    for k in ('actor/awr_loss', 'actor/grad_norm', 'actor/cfm_loss', 'actor/awr_w_max', 'actor/awr_adv_min'):
        np.testing.assert_allclose(float(i1[k]), float(iK[k]), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('K', [1, 2])
def test_grad_diagnostics_with_none_td_batch(K):
    agent = _agent(K=K)
    bc = _bc_batch(agent)
    d = agent.grad_diagnostics(None, bc)
    rng = jax.random.fold_in(agent.rng, 7)
    if K == 1:
        g = jax.grad(lambda p: agent._awr_terms(bc, p, rng)[0])(agent.actor_params)
        np.testing.assert_allclose(float(d['grad_pg_norm']), float(optax.global_norm(g)), rtol=1e-5)
    assert float(d['grad_bc_norm']) == 0.0 and float(d['grad_pg_norm']) > 0
    assert np.isfinite(float(d['grad_cos_pg_bc']))


# ---- PPO path: _update_impl after the _actor_step extraction == verbatim pre-change _update_impl --------------------

def _old_update_impl(self, batch, batch_success, bc_batch, success_flag, ppo_batch=None):
    """VERBATIM `OGPOPi05Agent._update_impl` at git HEAD 369c8a0."""
    new_rng, actor_rng, critic_rng, sb_rng = jax.random.split(self.rng, 4)

    if self._accum_steps == 1:
        (_, actor_info), grads = jax.value_and_grad(
            lambda p: self.actor_loss(batch, bc_batch, p, actor_rng, ppo_batch), has_aux=True)(self.actor_params)
    else:
        grads, actor_info = self._accum_grad(
            lambda p, b, k: self.actor_loss(b[0], b[1], p, k, b[2]), self.actor_params,
            (batch, bc_batch, ppo_batch), actor_rng)
    updates, new_opt_state = self.actor_tx.update(grads, self.actor_opt_state, self.actor_params)
    new_actor_params = optax.apply_updates(self.actor_params, updates)
    actor_tau = self.config['actor_tau']
    new_target = jax.tree.map(lambda p, tp: p * actor_tau + tp * (1 - actor_tau),
                              new_actor_params, self.target_actor_params)
    if self.zero_actor_state and self.mesh is not None:
        # ZeRO-1: keep the input layout so donation reuses the buffers and nothing recompiles.
        rep = NamedSharding(self.mesh, P())
        new_actor_params = jax.lax.with_sharding_constraint(new_actor_params, rep)
        new_target = jax.lax.with_sharding_constraint(new_target, rep)
        new_opt_state = jax.lax.with_sharding_constraint(new_opt_state, zero_shardings(new_opt_state, self.mesh))
    actor_info['grad_norm'] = optax.global_norm(grads)

    # Critic sees the pre-update target actor, as in OGPOAgent._update.
    new_critic, new_value, critic_info = self._critic_updates(batch, batch_success, success_flag, critic_rng, sb_rng)

    info = {**{f'actor/{k}': v for k, v in actor_info.items()}, **critic_info}
    return self.replace(
        rng=new_rng, actor_params=new_actor_params, target_actor_params=new_target,
        actor_opt_state=new_opt_state, critic_network=new_critic, value_network=new_value), info


_old_update_jit = jax.jit(_old_update_impl)


class _PPOStub(_AWRStub):
    """PPO-shaped `_actor_terms` computable on CPU: RNG-dependent pg term reading batch / ppo_batch, plus BC."""

    def _actor_terms(self, batch, bc_batch, grad_params, rng, ppo_batch=None):
        src = ppo_batch if ppo_batch is not None else batch
        x = src['observations'][:, -STATE_DIM:]
        noise = jax.random.normal(rng, (x.shape[0], 7))
        pg = jnp.mean(jnp.tanh(x @ grad_params['w']) * noise)
        ent = jnp.mean(jnp.square(grad_params['w'])) * 0.1
        bcl = self._bc_loss_per_sample(bc_batch, grad_params, jax.random.fold_in(rng, 1)).mean()
        return pg, ent, bcl, {'pg_loss': pg, 'bc_loss': bcl, 'adv_max': pg, 'adv_min': -pg}


def _td_batch(seed=0):
    k = jax.random.split(jax.random.PRNGKey(400 + seed), 5)
    valid = np.ones((B, HL), np.float32); valid[0, -1] = 0.0
    masks = np.ones((B, HL), np.float32); masks[1, -1] = 0.0
    return dict(observations=jax.random.normal(k[0], (B, OBS)), next_observations=jax.random.normal(k[1], (B, OBS)),
                actions=jax.random.uniform(k[2], (B, HL, T.ENV_ACTION_DIM), minval=-1, maxval=1),
                rewards=jax.random.normal(k[3], (B, HL)), masks=jnp.asarray(masks), valid=jnp.asarray(valid),
                images=jnp.zeros((B, 1)), task_ids=jnp.asarray(np.arange(B) % NT, jnp.int32))


@pytest.mark.parametrize('K', [1, 2])
@pytest.mark.parametrize('use_sb,flag', [(False, False), (True, True), (True, False)])
@pytest.mark.parametrize('with_ppo_batch', [False, True])
def test_update_impl_ppo_matches_verbatim_old(K, use_sb, flag, with_ppo_batch):
    agent = _agent(cls=_PPOStub, K=K, objective='ppo', stub_rng=0.05, use_success_buffer_q=use_sb, num_qs=NUM_QS)
    batch, bs = _td_batch(0), _td_batch(1)
    bc = _bc_batch(agent)
    ppo = {'images': jnp.zeros((B, 1)), 'observations': _td_batch(2)['observations']} if with_ppo_batch else None
    f = jnp.asarray(flag)
    new_a, new_i = agent.update(batch, bs, bc, f, ppo)
    old_a, old_i = _old_update_jit(agent, batch, bs, bc, f, ppo)
    assert _equal(new_a, old_a)
    assert new_i.keys() == old_i.keys() and _equal(new_i, old_i)
    a2, _ = new_a.update(batch, bs, bc, f, ppo)
    b2, _ = _old_update_jit(old_a, batch, bs, bc, f, ppo)
    assert _equal(a2, b2)
    # teeth: the actor actually moved, and the SARSA-V critic / V stepped
    assert not _equal(new_a.actor_params, agent.actor_params)
    assert not _equal(new_a.value_network.params, agent.value_network.params)


# =====================================================================================================================
# f. create() validation, main_pi05 validation / wiring
# =====================================================================================================================

class _Reached(Exception):
    pass


def _yaml_agent_config(**over):
    from ogpo.utils.config_loader import config_to_agent_dict, load_config, parse_cli_args
    ac = dict(config_to_agent_dict(load_config(*parse_cli_args(['--algo=ogpo']))))
    ac.update(pi05_action_horizon=10, pi05_checkpoint='/nonexistent', pi05_robot='libero', pi05_critic_pooling='all')
    ac.update(over)
    return ac


AWR_OK = dict(pi05_actor_objective='awr', use_value_fn=True, v_training_mode='lambda_return', v_ensemble_size=2,
              v_agg='mean', awr_beta=0.05, awr_weight_max=20.0, awr_lambda=0.95, awr_mode='symmetric',
              use_bc_regularization=False, q_variance_reduction=False)


@pytest.fixture
def stop_at_load(monkeypatch):
    import ogpo.agents.ogpo_pi05 as Pm

    def _load(path):
        raise _Reached(path)
    monkeypatch.setattr(Pm, 'load_pi05_params', _load)


def test_create_defaults_and_awr_reach_checkpoint_load(stop_at_load):
    with pytest.raises(_Reached):
        OGPOPi05Agent.create(0, 'p', _yaml_agent_config())  # YAML default: ppo
    with pytest.raises(_Reached):
        OGPOPi05Agent.create(0, 'p', _yaml_agent_config(**AWR_OK))
    with pytest.raises(_Reached):
        OGPOPi05Agent.create(0, 'p', _yaml_agent_config(**{**AWR_OK, 'v_agg': 'min', 'awr_lambda': 0.0}))
    with pytest.raises(_Reached):  # ppo + SARSA-V still allowed
        OGPOPi05Agent.create(0, 'p', _yaml_agent_config(use_value_fn=True, v_training_mode='sarsa', v_ensemble_size=2,
                                                        q_variance_reduction=False))


@pytest.mark.parametrize('over,match', [
    ({'pi05_actor_objective': 'awrr'}, 'ppo \\| awr'),
    ({'pi05_actor_objective': 'ppo'}, 'lambda_return'),            # lambda_return without awr
    ({'use_value_fn': False}, 'use_value_fn=true'),
    ({'v_training_mode': 'sarsa'}, 'v_training_mode=lambda_return'),
    ({'awr_beta': 0.0}, 'awr_beta'),
    ({'awr_beta': -0.1}, 'awr_beta'),
    ({'awr_weight_max': 0.0}, 'awr_weight_max'),
    ({'awr_mode': 'asymmetric'}, 'symmetric'),
    ({'awr_lambda': 1.5}, 'awr_lambda'),
    ({'awr_lambda': -0.01}, 'awr_lambda'),
    ({'v_agg': 'subsample'}, 'v_agg'),
    ({'use_bc_regularization': True}, 'use_bc_regularization'),
    ({'q_variance_reduction': True}, 'q_variance_reduction'),
    ({'v_ensemble_size': 1}, 'v_ensemble_size'),
])
def test_create_rejects_bad_awr_configs(stop_at_load, over, match):
    with pytest.raises(ValueError, match=match):
        OGPOPi05Agent.create(0, 'p', _yaml_agent_config(**{**AWR_OK, **over}))


def _main_harness(monkeypatch, *extra):
    from tests.test_pi05_multitask import _Built, _main, _patch_main
    M = _main(monkeypatch, *extra)
    built = _patch_main(monkeypatch, M)

    def logging(**kw):
        built['prefixes'] = kw['prefixes']
        return None, SimpleNamespace(close=lambda: None), None
    monkeypatch.setattr(M, 'setup_experiment_logging', logging)
    return M, built, _Built


GOOD = 'libero_pi05-libero_90-1'
AWR_FLAGS = ['--agent.pi05_actor_objective=awr', '--agent.use_value_fn=true', '--agent.v_training_mode=lambda_return',
             '--agent.v_ensemble_size=2']


@pytest.mark.parametrize('extra,match', [
    ([], 'collection_mode=rounds'),
    (['--training.collection_mode=online'], 'collection_mode=rounds'),
    (['--training.collection_mode=rounds', '--training.first_round_policy=ode'], 'first_round_policy'),
    (['--training.collection_mode=rounds', '--training.freeze_actor=true'], 'freeze_actor=false'),
    (['--training.collection_mode=rounds', '--training.actor_update_interval=2'], 'actor_update_interval=1'),
    (['--training.collection_mode=rounds', '--training.awr_value_steps=-1'], 'awr_value_steps'),
])
def test_main_pi05_awr_validation(monkeypatch, extra, match):
    M, built, _ = _main_harness(monkeypatch, f'--env.name={GOOD}', *AWR_FLAGS, *extra)
    with pytest.raises(ValueError, match=match):
        M.main()
    assert 'envs' not in built  # fires before any env / agent is built


@pytest.mark.parametrize('awr', [False, True])
def test_main_pi05_awr_wiring(monkeypatch, awr):
    flags = (AWR_FLAGS + ['--training.awr_value_steps=7']) if awr else []
    M, built, Built = _main_harness(monkeypatch, f'--env.name={GOOD}', '--training.collection_mode=rounds',
                                    '--training.num_rounds=1', '--training.rollouts_per_round=1',
                                    '--training.updates_per_round=1', *flags)
    with pytest.raises(Built):
        M.main()
    assert built['runner'] == 'rounds'
    assert built['kw']['awr_value_steps'] == (7 if awr else 200)
    assert ('awr' in built['prefixes']) == awr
    assert built['agent_config']['pi05_actor_objective'] == ('awr' if awr else 'ppo')


# =====================================================================================================================
# h. run_pi05_round_rl under awr (stub agent / envs)
# =====================================================================================================================

from tests._pi05_stubs import Logger, StubAgent, TaskEnv, episodes_in_buffer, reg  # noqa: E402


def _copy(d):
    return None if d is None else {k: np.array(v) for k, v in d.items()}


class _AWRStubAgent(StubAgent):
    def _next(self):
        new = _AWRStubAgent(self.log, self.strict, self.gen + 1)
        if self.strict:
            self.dead = True
        return new

    @staticmethod
    def vfun(x, gen):
        x = np.asarray(x)
        return 0.01 * x[:, 2] - 0.1 * x[:, 0] + 0.001 * gen

    def sample_actions_explore(self, imgs, key, task_ids=None, critic_obs=None):
        self._use('sample_actions_explore')
        self.log.append(('explore', self._key(key), None if task_ids is None else np.asarray(task_ids).copy()))
        return self._chunk(critic_obs, key)

    def sample_actions_BON(self, *a, **k):
        raise AssertionError('awr must not use best-of-N')

    def update(self, *a, **k):
        raise AssertionError('awr must not call update')

    def critic_update(self, *a, **k):
        raise AssertionError('awr must not call critic_update')

    def state_values(self, x):
        self._use('state_values')
        self.log.append(('state_values', self.gen, np.asarray(x).shape))
        return jnp.asarray(self.vfun(x, self.gen))

    def value_update(self, batch):
        self._use('value_update')
        assert set(batch) == {'observations', 'lambda_returns', 'valid', 'task_ids'}, set(batch)
        self.log.append(('value_update', _copy(batch)))
        return self._next(), {'value/v_loss': float(np.sum(batch['lambda_returns']))}

    def actor_update(self, bc_batch):
        self._use('actor_update')
        assert set(bc_batch) - {'awr_adv_mean', 'awr_adv_std'} == {'images', 'observations', 'actions', 'valid',
                                                                    'task_ids', 'lambda_returns'}
        self.log.append(('actor_update', _copy(bc_batch)))
        return self._next(), {'actor/awr_loss': float(np.sum(bc_batch['lambda_returns']))}

    def grad_diagnostics(self, batch, bc_batch):
        self._use('grad_diagnostics')
        assert batch is None and bc_batch is not None
        self.log.append(('diag', None, _copy(bc_batch)))
        return {'grad_pg_norm': 1.0, 'grad_bc_norm': 0.0}


AWR_CFG = {'grad_diagnostics': True, 'ppo_batch_size': 2, 'pi05_actor_objective': 'awr', 'awr_lambda': 0.9}
AKW = dict(batch_size=2, success_buffer_batch_size=2, horizon_length=3, bc_horizon=4, discount=0.99, buffer_size=2000,
           seed=3)


@pytest.fixture
def tiny_images(monkeypatch):
    monkeypatch.setattr(R.T, 'resize_images', lambda x: x)


def _ref_rb_returns(rb, gen, lam=0.9, gamma=0.99):
    o = rb.insertion_order()
    return o, _ref_lambda_returns(rb['rewards'][o], rb['masks'][o], rb['terminals'][o],
                                  _AWRStubAgent.vfun(rb['next_observations'][o], gen), gamma, lam)


def test_round_runner_awr_schedule(tiny_images):
    log, logger = [], Logger()
    agent = _AWRStubAgent(log, strict=True)
    NR, UPR, VS = 3, 4, 3
    _, rb = R.run_pi05_round_rl(agent, TaskEnv(0, reg()), TaskEnv(0, reg()), AWR_CFG, logger, num_rounds=NR,
                                rollouts_per_round=2, updates_per_round=UPR, eval_interval=4, eval_episodes=2,
                                log_interval=2, awr_value_steps=VS, **AKW)
    kinds = [c[0] for c in log]
    assert kinds.count('value_update') == NR * VS and kinds.count('actor_update') == NR * UPR
    assert 'bon' not in kinds and 'update' not in kinds and 'critic_update' not in kinds
    assert 'explore' in kinds and 'diag' in kinds
    # every round: collection (explore) -> λ pass (state_values) -> V fit -> λ pass with the fitted V -> policy
    # steps, in that order (reference awr_agent._update)
    seq = [k for k in kinds if k in ('explore', 'state_values', 'value_update', 'actor_update')]
    collapsed = [k for i, k in enumerate(seq) if i == 0 or seq[i - 1] != k]
    assert collapsed == ['explore', 'state_values', 'value_update', 'state_values', 'actor_update'] * NR, collapsed
    # round r: the pre-fit pass bootstraps with the agent after round r-1 (gen r * (VS + UPR)), the post-fit pass
    # with the fitted V (gen r * (VS + UPR) + VS)
    gens = sorted({c[1] for c in log if c[0] == 'state_values'})
    assert gens == sorted([r * (VS + UPR) for r in range(NR)] + [r * (VS + UPR) + VS for r in range(NR)])
    # buffer: λ-returns from the last round's post-fit pass, in insertion order; episodes intact
    episodes_in_buffer(rb)
    o, ref = _ref_rb_returns(rb, (NR - 1) * (VS + UPR) + VS)
    np.testing.assert_allclose(rb['lambda_returns'][o], ref, rtol=1e-5, atol=1e-6)
    # last round's V and policy batches carry the buffer's λ-returns at the window start (obs = (task, seed, t))
    key = {tuple(rb['observations'][i, :3]): rb['lambda_returns'][i] for i in range(rb.size)}
    last_v = [c[1] for c in log if c[0] == 'value_update'][-VS:]
    last_a = [c[1] for c in log if c[0] == 'actor_update'][-UPR:]
    for b in last_a:  # policy batches read the post-fit returns
        for obs, lr in zip(b['observations'], b['lambda_returns']):
            assert key[tuple(obs[:3])] == pytest.approx(lr)
    _, ref_pre = _ref_rb_returns(rb, (NR - 1) * (VS + UPR))  # V batches read the pre-fit (frozen) targets
    key_pre = {tuple(rb['observations'][i, :3]): ref_pre[j] for j, i in enumerate(o)}
    for b in last_v:
        for obs, lr in zip(b['observations'], b['lambda_returns']):
            assert key_pre[tuple(obs[:3])] == pytest.approx(lr, rel=1e-5, abs=1e-6)
    # teeth: pre- and post-fit returns differ, so the two checks above are not the same check
    assert not np.allclose(ref_pre, ref)
    assert all(b['actions'].shape[1] == AKW['bc_horizon'] for b in last_a)
    # logs: one awr row per round at its first step, ODE eval only
    awr_rows = [(s, d) for p, s, d in logger.logs if p == 'awr']
    assert [s for s, _ in awr_rows] == [r * UPR for r in range(NR)]
    assert all(d['value_fit_steps'] == VS and 'value/v_loss' in d for _, d in awr_rows)
    np.testing.assert_allclose(awr_rows[-1][1]['awr/lambda_return_mean'], ref_pre.mean(), rtol=1e-5)  # pre-fit pass
    np.testing.assert_allclose(awr_rows[-1][1]['awr/post_lambda_return_mean'], ref.mean(), rtol=1e-5)  # post-fit
    # V-vs-MC diagnostics on every row; this round's collection is scored as held-out data
    assert all('awr/vmc_corr' in d and 'awr/vmc_new_corr' in d and 'awr/target_mc_corr' in d for _, d in awr_rows)
    assert not any(p == 'eval_sde' for p, _, _ in logger.logs)
    assert sorted({s for p, s, _ in logger.logs if p == 'eval'}) == [0, 4, 8, 12]
    assert any(p == 'online_agent' and 'actor/awr_loss' in d for p, _, d in logger.logs)


def test_round_runner_awr_refreshes_and_normalised_adv(tiny_images):
    """K target refreshes: K x (λ pass -> VS frozen-target V steps), then the post-fit pass; the policy batches carry
    the post-fit pass's buffer-wide mean / std of R^λ - V when awr_normalize_adv is on."""
    log, logger = [], Logger()
    NR, UPR, VS, K = 2, 3, 2, 3
    cfg = {**AWR_CFG, 'awr_normalize_adv': True}
    _, rb = R.run_pi05_round_rl(_AWRStubAgent(log), TaskEnv(0, reg()), TaskEnv(0, reg()), cfg, logger, num_rounds=NR,
                                rollouts_per_round=2, updates_per_round=UPR, eval_interval=0, eval_episodes=1,
                                log_interval=1, awr_value_steps=VS, awr_value_refreshes=K, **AKW)
    kinds = [c[0] for c in log]
    assert kinds.count('value_update') == NR * K * VS and kinds.count('actor_update') == NR * UPR
    seq = [k for k in kinds if k in ('explore', 'state_values', 'value_update', 'actor_update')]
    collapsed = [k for i, k in enumerate(seq) if i == 0 or seq[i - 1] != k]
    assert collapsed == (['explore'] + ['state_values', 'value_update'] * K + ['state_values', 'actor_update']) * NR
    # each refresh's V batches regress onto targets bootstrapped with the V at the start of that refresh
    v_batches = [c[1] for c in log if c[0] == 'value_update']
    last_round = v_batches[-K * VS:]
    gen0 = (NR - 1) * (K * VS + UPR)
    o = rb.insertion_order()
    for k in range(K):
        _, ref_k = _ref_rb_returns(rb, gen0 + k * VS)
        key_k = {tuple(rb['observations'][i, :3]): ref_k[j] for j, i in enumerate(o)}
        for b in last_round[k * VS:(k + 1) * VS]:
            for obs, lr in zip(b['observations'], b['lambda_returns']):
                assert key_k[tuple(obs[:3])] == pytest.approx(lr, rel=1e-5, abs=1e-6)
    # post-fit pass: buffer returns and the actor's normalisation stats use the fully fitted V
    _, ref_post = _ref_rb_returns(rb, gen0 + K * VS)
    np.testing.assert_allclose(rb['lambda_returns'][o], ref_post, rtol=1e-5, atol=1e-6)
    adv = ref_post - _AWRStubAgent.vfun(rb['observations'][o], gen0 + K * VS)
    for b in [c[1] for c in log if c[0] == 'actor_update'][-UPR:]:
        np.testing.assert_allclose(b['awr_adv_mean'], adv.mean(), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(b['awr_adv_std'], adv.std(), rtol=1e-4, atol=1e-5)
    rows = [d for p, _, d in logger.logs if p == 'awr']
    assert [d['value_fit_steps'] for d in rows] == [K * VS] * NR and [d['value_refreshes'] for d in rows] == [K] * NR
    assert all('awr/vmc_new_corr' in d and 'awr/post_vmc_corr' in d and 'awr/post_adv_std' in d for d in rows)
    np.testing.assert_allclose(rows[-1]['awr/post_adv_std'], adv.std(), rtol=1e-4)
    # off by default: no normalisation fields in the policy batches
    log2 = []
    R.run_pi05_round_rl(_AWRStubAgent(log2), TaskEnv(0, reg()), TaskEnv(0, reg()), AWR_CFG, Logger(), num_rounds=1,
                        rollouts_per_round=2, updates_per_round=1, eval_interval=0, eval_episodes=1, log_interval=1,
                        awr_value_steps=1, **AKW)
    assert all('awr_adv_mean' not in c[1] for c in log2 if c[0] == 'actor_update')


def test_round_runner_awr_zero_value_steps(tiny_images):
    log, logger = [], Logger()
    R.run_pi05_round_rl(_AWRStubAgent(log), TaskEnv(0, reg()), TaskEnv(0, reg()), AWR_CFG, logger, num_rounds=2,
                        rollouts_per_round=2, updates_per_round=2, eval_interval=0, eval_episodes=1, log_interval=1,
                        awr_value_steps=0, **AKW)
    kinds = [c[0] for c in log]
    assert kinds.count('value_update') == 0 and kinds.count('actor_update') == 4
    assert [d['value_fit_steps'] for p, _, d in logger.logs if p == 'awr'] == [0, 0]


def test_round_runner_ppo_has_no_lambda_field(tiny_images):
    """Default objective: the buffer schema is unchanged (no lambda_returns field)."""
    log, logger = [], Logger()
    _, rb = R.run_pi05_round_rl(StubAgent(log), TaskEnv(0, reg()), TaskEnv(0, reg()), {'ppo_batch_size': 2,
                                'grad_diagnostics': False}, logger, num_rounds=1, rollouts_per_round=2,
                                updates_per_round=1, eval_interval=0, eval_episodes=1, log_interval=1, **AKW)
    assert 'lambda_returns' not in rb._dict
    assert not any(p == 'awr' for p, _, _ in logger.logs)


class _Crash(Exception):
    pass


def _awr_rounds(monkeypatch, store, crash_at=None, save_dir=None):
    from ogpo.utils.resume import truncate_csvs
    monkeypatch.setattr(R.T, 'resize_images', lambda x: x)
    monkeypatch.setattr(R, '_agent_state', lambda agent, step: {'step': step, 'gen': agent.gen})

    def fake_resume(resume, man, agent, rb, save_dir, upr, seed):
        state, csv_dir, dirs = resume.fetch(man)
        meta = resume.load_buffer(rb, dirs)
        truncate_csvs(csv_dir, save_dir, state['step'], keep_at_step=state['step'] // upr == man['round'])
        return _AWRStubAgent(agent.log, gen=state['gen']), state['step'], meta['episodes'], meta['env_steps']

    monkeypatch.setattr(R, '_resume_state', fake_resume)
    log, logger = [], Logger()
    base = logger.log

    def lg(data, prefix, step):
        if crash_at == (prefix, step):
            raise _Crash
        base(data, prefix, step)
    logger.log = lg
    try:
        _, rb = R.run_pi05_round_rl(_AWRStubAgent(log), TaskEnv(0, reg()), TaskEnv(0, reg()), AWR_CFG, logger,
                                    num_rounds=3, rollouts_per_round=2, updates_per_round=4, eval_interval=0,
                                    eval_episodes=1, log_interval=2, awr_value_steps=3, save_dir=save_dir,
                                    resume=store, **AKW)
    except _Crash:
        rb = None
    return rb, logger.logs, log


@pytest.mark.parametrize('crash_at,man_expected,fresh', [(('online_agent', 6), (1, 4), True),
                                                         (('online_agent', 8), (1, 6), False)])
def test_round_runner_awr_resume(tmp_path, monkeypatch, crash_at, man_expected, fresh):
    from ogpo.utils.resume import ResumeStore
    mk = lambda name: ResumeStore(tmp_path / f'{name}_local', tmp_path / 'remote', 4, ckpt_interval=2)  # noqa: E731
    ref_rb, _, _ = _awr_rounds(monkeypatch, None)
    store = mk('a')
    assert _awr_rounds(monkeypatch, store, crash_at, save_dir=str(tmp_path / 'sa'))[0] is None
    assert store.wait() is None
    man = mk('peek').manifest()
    assert (man['round'], man['step']) == man_expected
    rb, logs, log = _awr_rounds(monkeypatch, mk('b'), save_dir=str(tmp_path / 'sb'))
    assert rb.size == ref_rb.size and 'lambda_returns' in rb._dict
    step0 = man_expected[1]
    awr_rows = [(s, d['value_fit_steps']) for p, s, d in logs if p == 'awr']
    assert awr_rows[0] == (step0, 3 if fresh else 0)
    assert awr_rows[1:] == [(8, 3)]
    kinds = [c[0] for c in log]
    assert kinds.count('value_update') == 3 * (1 + int(fresh))
    assert kinds.count('actor_update') == 12 - step0
    assert max(s for p, s, _ in logs if p == 'online_agent') == 12
    # the restored buffer's λ-field was recomputed (non-zero) before the first resumed policy step
    first_actor = next(c for c in log if c[0] == 'actor_update')
    assert np.any(first_actor[1]['lambda_returns'] != 0.0)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
