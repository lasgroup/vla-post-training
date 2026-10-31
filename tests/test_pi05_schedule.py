"""
CPU tests for the pi05 update schedule after the merge of manan/libero-vla-fix
(docs/changes/2026-10-06-merge-manan-libero-vla-fix; originally docs/changes/2026-09-28-pi05-epoch-schedule).

  Runners (stub agent / envs / logger, tests/_pi05_stubs.py):
    - run_pi05_online_rl, actor_update_interval=1 vs a VERBATIM copy of Manan's runner
      (tests/_pi05_runner_manan.py): identical buffer, reset seeds, actions, agent calls (RNG keys, update and
      grad-diagnostics batches) and logged metrics.
    - actor_update_interval=N in both runners: update / critic_update pattern, no BC batch sampled on critic-only
      steps, grad diagnostics always get the last actor step's (batch, bc_batch), never bc_batch=None.
    - donate safety: a stub agent that raises when a replaced (pre-update) instance is used; both runners through
      several updates, logs and evals (and Manan's runner, which holds its encoders across updates, fails it).
  Agent (subclass with tiny fake losses, no pi05 checkpoint):
    - update vs a verbatim copy of the pre-change update; critic_update vs update's critic half;
      grad accumulation == full batch (incl. an explicit ppo_batch, and grad_diagnostics); donate_agent == plain.
  Config: load_config with the merged keys (old epoch keys gone).

Run:  JAX_PLATFORMS=cpu uv run pytest tests/test_pi05_schedule.py -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from types import SimpleNamespace

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import ogpo.runners.pi05_online_runner as R
import tests._pi05_runner_manan as MAN
from ogpo.utils.datasets import ReplayBuffer
from tests._pi05_stubs import (Logger, StaleAgentError, StubAgent, TaskEnv, assert_buffers_equal,
                               assert_call_logs_equal, assert_logs_equal, episodes_in_buffer, reg)


@pytest.fixture(autouse=True)
def tiny_images(monkeypatch):
    monkeypatch.setattr(R.T, 'resize_images', lambda x: x)  # same module object as MAN.T


COMMON = dict(batch_size=2, success_buffer_batch_size=2, horizon_length=3, bc_horizon=4, discount=0.99,
              buffer_size=2000, eval_episodes=2, seed=3)
PER_STEP = dict(online_steps=90, start_training=10, eval_interval=40, log_interval=5)
ROUNDS = dict(num_rounds=3, rollouts_per_round=2, updates_per_round=5, eval_interval=4, log_interval=3)


def _online(fn, strict=False, diag=True, **extra):
    tr, ev, log = reg(), reg(), []
    agent, logger = StubAgent(log, strict=strict), Logger()
    _, rb = fn(agent, TaskEnv(0, tr), TaskEnv(0, ev), {'grad_diagnostics': diag}, logger, **COMMON, **PER_STEP,
               **extra)
    return SimpleNamespace(rb=rb, log=log, logs=logger.logs, tr=tr, ev=ev)


def _rounds(fn, strict=False, diag=True, m=1, **extra):
    tr, ev, log = reg(), reg(), []
    agent, logger = StubAgent(log, strict=strict), Logger()
    if m == 1:
        env, eval_env = TaskEnv(0, tr), TaskEnv(0, ev)
    else:
        env = [('a', [TaskEnv(0, tr) for _ in range(m)], 0), ('b', [TaskEnv(1, tr) for _ in range(m)], 1)]
        eval_env = [('a', [TaskEnv(0, ev) for _ in range(m)], 0), ('b', [TaskEnv(1, ev) for _ in range(m)], 1)]
    _, rb = fn(agent, env, eval_env, {'grad_diagnostics': diag, 'ppo_batch_size': 2}, logger, **COMMON,
               **{**ROUNDS, **extra})
    return SimpleNamespace(rb=rb, log=log, logs=logger.logs, tr=tr, ev=ev)


# ---------------------------------------------------------------------------
# Per-step runner: differential vs Manan's verbatim runner
# ---------------------------------------------------------------------------

def test_online_interval1_matches_manan():
    old = _online(MAN.run_pi05_online_rl)
    new = _online(R.run_pi05_online_rl, actor_update_interval=1)
    assert_buffers_equal(old.rb, new.rb)
    assert old.tr['resets'] == new.tr['resets'] and old.ev['resets'] == new.ev['resets']
    for x, y in ((old.tr, new.tr), (old.ev, new.ev)):
        assert len(x['actions']) == len(y['actions'])
        for p, q in zip(x['actions'], y['actions']):
            assert p[:3] == q[:3] and np.array_equal(p[3], q[3])
    assert_call_logs_equal(old.log, new.log)
    assert_logs_equal(old.logs, new.logs)
    kinds = [c[0] for c in new.log]
    assert kinds.count('update') == PER_STEP['online_steps'] - PER_STEP['start_training'] + 1
    assert {'diag', 'ode', 'bon', 'encode'} <= set(kinds)
    assert any(c[4] for c in new.log if c[0] == 'update') and not all(c[4] for c in new.log if c[0] == 'update')
    episodes_in_buffer(new.rb)


def test_online_rejects_task_list():
    with pytest.raises(AssertionError, match='collection_mode=rounds'):
        R.run_pi05_online_rl(StubAgent([]), [('a', TaskEnv(0, reg()))], TaskEnv(0, reg()), {}, Logger(),
                             **COMMON, **PER_STEP)


# ---------------------------------------------------------------------------
# actor_update_interval in both runners
# ---------------------------------------------------------------------------

def _count_bc_samples(monkeypatch):
    """Counts sample_sequence calls of BC-window length (bc_horizon), i.e. BC batches drawn."""
    n = {'bc': 0}
    orig = ReplayBuffer.sample_sequence

    def spy(self, batch_size, sequence_length, *a, **k):
        if sequence_length == COMMON['bc_horizon']:
            n['bc'] += 1
        return orig(self, batch_size, sequence_length, *a, **k)
    monkeypatch.setattr(ReplayBuffer, 'sample_sequence', spy)
    return n


def _check_interval(run, N):
    upd = [c for c in run.log if c[0] in ('update', 'critic_update')]
    kinds = [c[0] for c in upd]
    assert len(kinds) > 2 * N
    assert [i for i, k in enumerate(kinds) if k == 'update'] == list(range(0, len(kinds), N))
    for c in upd:
        assert (c[3] is None) == (c[0] == 'critic_update')  # BC batch only on actor steps
    # grad diagnostics: the most recent actor step's batches, never a critic-only step's
    last = None
    n_diag = 0
    for c in run.log:
        if c[0] == 'update':
            last = c
        elif c[0] == 'diag':
            n_diag += 1
            assert last is not None
            for k in c[1]:
                assert np.array_equal(c[1][k], last[1][k]), k
            assert c[2].keys() == last[3].keys()
            for k in c[2]:
                assert np.array_equal(c[2][k], last[3][k]), k
    assert n_diag > 0
    # window logs: actor keys only from actor steps, critic keys from all
    for p, _, d in run.logs:
        if p == 'online_agent':
            assert 'critic/td_loss' in d and 'actor/pg_loss' in d and 'diag/grad_pg_norm' in d
    return kinds


@pytest.mark.parametrize('N', [2, 4])
def test_online_actor_update_interval(monkeypatch, N):
    n = _count_bc_samples(monkeypatch)
    run = _online(R.run_pi05_online_rl, actor_update_interval=N)
    kinds = _check_interval(run, N)
    assert n['bc'] == kinds.count('update')
    assert len(kinds) == PER_STEP['online_steps'] - PER_STEP['start_training'] + 1
    episodes_in_buffer(run.rb)


@pytest.mark.parametrize('N,m', [(2, 1), (3, 1), (2, 2)])
def test_round_actor_update_interval(monkeypatch, N, m):
    n = _count_bc_samples(monkeypatch)
    run = _rounds(R.run_pi05_round_rl, actor_update_interval=N, m=m)
    kinds = _check_interval(run, N)
    assert n['bc'] == kinds.count('update')
    assert len(kinds) == ROUNDS['num_rounds'] * ROUNDS['updates_per_round']
    # the step counter (not the per-round index) drives the pattern across rounds: 15 updates, N=2 -> 8 actor steps
    assert kinds.count('update') == -(-len(kinds) // N)


def test_round_interval1_all_full():
    run = _rounds(R.run_pi05_round_rl, actor_update_interval=1)
    assert {c[0] for c in run.log if 'update' in c[0]} == {'update'}


# ---------------------------------------------------------------------------
# donate_agent safety: nothing may use an agent after it was replaced by an update
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('N', [1, 2])
def test_online_never_uses_replaced_agent(N):
    run = _online(R.run_pi05_online_rl, strict=True, actor_update_interval=N)
    kinds = [c[0] for c in run.log]
    assert kinds.count('update') + kinds.count('critic_update') > 10 and 'diag' in kinds
    assert sum(1 for p, _, _ in run.logs if p == 'eval') >= 3


@pytest.mark.parametrize('N,m', [(1, 1), (2, 1), (2, 3)])
def test_round_never_uses_replaced_agent(N, m):
    run = _rounds(R.run_pi05_round_rl, strict=True, actor_update_interval=N, m=m)
    kinds = [c[0] for c in run.log]
    assert kinds.count('update') + kinds.count('critic_update') == 15 and 'diag' in kinds
    assert sum(1 for p, _, _ in run.logs if p == 'eval') >= 4
    # collection in rounds 2 and 3 ran on post-update agents
    assert kinds.index('bon') < kinds.index('update') < len(kinds) - 1 - kinds[::-1].index('bon')


@pytest.mark.parametrize('fn', ['online', 'rounds'])
def test_strict_stub_has_teeth(fn):
    # Manan's runners keep their encoders across updates; with a donated agent that is a use-after-free.
    with pytest.raises(StaleAgentError):
        if fn == 'online':
            _online(MAN.run_pi05_online_rl, strict=True)
        else:
            _rounds(MAN.run_pi05_round_rl, strict=True)


# ---------------------------------------------------------------------------
# Agent: update vs pre-change update; critic_update vs update's critic half
# ---------------------------------------------------------------------------

from ogpo.agents.modules.flax_utils import ModuleDict, TrainState  # noqa: E402
from ogpo.agents.ogpo_pi05 import OGPOPi05Agent  # noqa: E402


class _TinyQ(nn.Module):
    @nn.compact
    def __call__(self, obs, actions):
        return nn.Dense(1)(jnp.concatenate([obs, actions], -1))[..., 0]


class _FakePi05(OGPOPi05Agent):
    """Real update / critic_update / _critic_step / _critic_updates; fake losses."""

    def critic_loss(self, batch, grad_params, rng):
        q = self.critic_network.select('critic')(batch['observations'], batch['actions'], params=grad_params)
        tq = self.critic_network.select('target_critic')(batch['observations'], batch['actions'])
        ta = jnp.sum(self.target_actor_params['w'])  # depends on the target actor
        target = batch['rewards'] + 0.9 * tq + ta + 0.1 * jax.random.normal(rng, q.shape)
        loss = jnp.mean((q - jax.lax.stop_gradient(target)) ** 2)
        return loss, {'td_loss': loss, 'q_mean': q.mean()}

    def actor_loss(self, batch, bc_batch, grad_params, rng, ppo_batch=None):
        pred = batch['observations'] @ grad_params['w']
        loss = (jnp.mean((pred - batch['rewards']) ** 2)
                + 0.1 * jnp.sum(jax.random.normal(rng, grad_params['w'].shape) * grad_params['w'])
                + jnp.mean(bc_batch['actions']) * jnp.sum(grad_params['w']))
        return loss, {'pg_loss': loss}


@jax.jit
def _old_update(self, batch, batch_success, bc_batch, success_flag):
    """Verbatim body of OGPOPi05Agent.update at git HEAD 955ae22."""
    new_rng, actor_rng, critic_rng, sb_rng = jax.random.split(self.rng, 4)

    (_, actor_info), grads = jax.value_and_grad(
        lambda p: self.actor_loss(batch, bc_batch, p, actor_rng), has_aux=True)(self.actor_params)
    updates, new_opt_state = self.actor_tx.update(grads, self.actor_opt_state, self.actor_params)
    new_actor_params = optax.apply_updates(self.actor_params, updates)
    actor_tau = self.config['actor_tau']
    new_target = jax.tree.map(lambda p, tp: p * actor_tau + tp * (1 - actor_tau),
                              new_actor_params, self.target_actor_params)
    actor_info['grad_norm'] = optax.global_norm(grads)

    # Critic sees the pre-update target actor, as in OGPOAgent._update.
    new_critic, critic_info = self._critic_step(self.critic_network, batch, critic_rng)
    if self.config.get('use_success_buffer_q', False):
        new_critic = jax.lax.cond(
            success_flag,
            lambda cn: self._critic_step(cn, batch_success, sb_rng)[0],
            lambda cn: cn,
            new_critic)

    info = {**{f'actor/{k}': v for k, v in actor_info.items()},
            **{f'critic/{k}': v for k, v in critic_info.items()}}
    return self.replace(
        rng=new_rng, actor_params=new_actor_params, target_actor_params=new_target,
        actor_opt_state=new_opt_state, critic_network=new_critic), info


def _make_fake(use_sb):
    obs_dim, act_dim = 5, 3
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    ex = (jnp.zeros((1, obs_dim)), jnp.zeros((1, act_dim)))
    net = ModuleDict({'critic': _TinyQ(), 'target_critic': _TinyQ()})
    params = net.init(k1, critic=ex, target_critic=ex)['params']
    critic = TrainState.create(net, params, optax.adam(1e-2))
    actor_tx = optax.adam(1e-2)
    w = jax.random.normal(k2, (obs_dim,))
    agent = _FakePi05(
        rng=jax.random.PRNGKey(7), actor_params={'w': w}, target_actor_params={'w': w * 0.5 + 1.0},
        actor_opt_state=actor_tx.init({'w': w}), init_actor_params={'w': jnp.copy(w)}, frozen_params={}, critic_network=critic,
        consts={},
        config={'tau': 0.05, 'actor_tau': 0.1, 'use_success_buffer_q': use_sb}, model=None, actor_tx=actor_tx)

    def mk(k):
        a, b, c = jax.random.split(k, 3)
        return {'observations': jax.random.normal(a, (4, obs_dim)), 'actions': jax.random.normal(b, (4, act_dim)),
                'rewards': jax.random.normal(c, (4,))}
    batch, batch_success = mk(k3), mk(jax.random.fold_in(k3, 1))
    bc = {'actions': jnp.ones((4, 2, 7)), 'images': jnp.zeros((4, 1)), 'valid': jnp.ones((4, 2))}
    return agent, batch, batch_success, bc


def _tree_equal(a, b):
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    assert jax.tree.structure(a) == jax.tree.structure(b)
    return all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb))


@pytest.mark.parametrize('use_sb', [False, True])
@pytest.mark.parametrize('flag', [False, True])
def test_update_matches_pre_change(use_sb, flag):
    agent, batch, bs, bc = _make_fake(use_sb)
    new_agent, new_info = agent.update(batch, bs, bc, jnp.asarray(flag))
    old_agent, old_info = _old_update(agent, batch, bs, bc, jnp.asarray(flag))
    assert _tree_equal(new_agent, old_agent)
    assert new_info.keys() == old_info.keys()
    assert _tree_equal(new_info, old_info)
    # two steps in a row still match (rng advance identical)
    a2, _ = new_agent.update(batch, bs, bc, jnp.asarray(flag))
    b2, _ = _old_update(old_agent, batch, bs, bc, jnp.asarray(flag))
    assert _tree_equal(a2, b2)


@pytest.mark.parametrize('use_sb', [False, True])
@pytest.mark.parametrize('flag', [False, True])
def test_critic_update_matches_update_critic_half(use_sb, flag):
    agent, batch, bs, bc = _make_fake(use_sb)
    full, full_info = agent.update(batch, bs, bc, jnp.asarray(flag))
    crit, crit_info = agent.critic_update(batch, bs, jnp.asarray(flag))
    # critic identical => critic_update also uses the (pre-update) target actor and same keys
    assert _tree_equal(full.critic_network, crit.critic_network)
    assert _tree_equal(full.rng, crit.rng)
    assert _tree_equal(crit.actor_params, agent.actor_params)
    assert _tree_equal(crit.target_actor_params, agent.target_actor_params)
    assert _tree_equal(crit.actor_opt_state, agent.actor_opt_state)
    assert set(crit_info) == {k for k in full_info if k.startswith('critic/')}
    assert not _tree_equal(crit.critic_network.params, agent.critic_network.params)
    # SB branch actually changes the result when enabled and flagged
    if use_sb:
        other, _ = agent.critic_update(batch, bs, jnp.asarray(not flag))
        assert not _tree_equal(other.critic_network.params, crit.critic_network.params)


# ---------------------------------------------------------------------------
# Gradient accumulation: K micro-batches == one full batch for mean losses
# ---------------------------------------------------------------------------

class _DetFakePi05(_FakePi05):
    """Fake losses without RNG terms, so K micro-batches must reproduce the full-batch update."""

    def critic_loss(self, batch, grad_params, rng):
        q = self.critic_network.select('critic')(batch['observations'], batch['actions'], params=grad_params)
        tq = self.critic_network.select('target_critic')(batch['observations'], batch['actions'])
        target = batch['rewards'] + 0.9 * tq + jnp.sum(self.target_actor_params['w'])
        loss = jnp.mean((q - jax.lax.stop_gradient(target)) ** 2)
        return loss, {'td_loss': loss, 'q_mean': q.mean(), 'q_max': q.max(), 'q_min': q.min()}

    def actor_loss(self, batch, bc_batch, grad_params, rng, ppo_batch=None):
        pred = batch['observations'] @ grad_params['w']
        loss = jnp.mean((pred - batch['rewards']) ** 2) + jnp.mean(bc_batch['actions'] * pred[:, None, None])
        return loss, {'pg_loss': loss}


def _det_agent(use_sb, K):
    agent, batch, bs, bc = _make_fake(use_sb)
    bc = {**bc, 'actions': jax.random.normal(jax.random.PRNGKey(3), bc['actions'].shape)}
    fields = {f: getattr(agent, f) for f in ('rng', 'actor_params', 'target_actor_params', 'actor_opt_state',
                                             'init_actor_params', 'frozen_params', 'critic_network', 'consts',
                                             'model', 'actor_tx')}
    return _DetFakePi05(**fields, config={**agent.config, 'grad_accum_steps': K}), batch, bs, bc


def _tree_close(a, b):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('K', [2, 4])
@pytest.mark.parametrize('use_sb', [False, True])
def test_grad_accum_matches_full_batch(K, use_sb):
    a1, batch, bs, bc = _det_agent(use_sb, 1)
    aK, *_ = _det_agent(use_sb, K)
    flag = jnp.asarray(True)
    n1, i1 = a1.update(batch, bs, bc, flag)
    nK, iK = aK.update(batch, bs, bc, flag)
    for f in ('actor_params', 'target_actor_params', 'actor_opt_state'):
        _tree_close(getattr(n1, f), getattr(nK, f))
    _tree_close((n1.critic_network.params, n1.critic_network.opt_state),
                (nK.critic_network.params, nK.critic_network.opt_state))
    assert i1.keys() == iK.keys()
    for k in ('actor/pg_loss', 'actor/grad_norm', 'critic/td_loss', 'critic/q_mean', 'critic/q_max',
              'critic/q_min', 'critic/grad/norm', 'critic/grad/max', 'critic/grad/min'):
        np.testing.assert_allclose(float(i1[k]), float(iK[k]), rtol=1e-5, atol=1e-6)
    c1, _ = a1.critic_update(batch, bs, flag)
    cK, _ = aK.critic_update(batch, bs, flag)
    _tree_close(c1.critic_network.params, cK.critic_network.params)


def test_grad_accum_micro_batches_interleave_rows():
    agent, *_ = _det_agent(False, 2)
    x = jnp.arange(8.0)
    seen = []

    def loss(p, b, key):
        seen.append(b)
        return jnp.sum(p * b), {}

    g, _ = agent._accum_grad(loss, jnp.zeros(4), x, jax.random.PRNGKey(0))
    # micro-batch k = rows k, k+K, ... ; mean of the two micro-batch gradients
    np.testing.assert_array_equal(np.asarray(g), (np.array([0, 2, 4, 6]) + np.array([1, 3, 5, 7])) / 2)


class _PPOFakePi05(_DetFakePi05):
    """Deterministic actor loss that also reads ppo_batch when given (as _actor_terms does)."""

    def actor_loss(self, batch, bc_batch, grad_params, rng, ppo_batch=None):
        loss, info = _DetFakePi05.actor_loss(self, batch, bc_batch, grad_params, rng)
        if ppo_batch is not None:
            loss = loss + jnp.mean(jnp.tanh(ppo_batch['observations'] @ grad_params['w']))
        return loss, {**info, 'pg_loss': loss}


@pytest.mark.parametrize('K', [2, 4])
def test_grad_accum_with_ppo_batch_matches_full_batch(K):
    a1, batch, bs, bc = _det_agent(False, 1)
    aK, *_ = _det_agent(False, K)
    a1 = _PPOFakePi05(**{f: getattr(a1, f) for f in a1.__dataclass_fields__})
    aK = _PPOFakePi05(**{f: getattr(aK, f) for f in aK.__dataclass_fields__})
    ppo = {'observations': jax.random.normal(jax.random.PRNGKey(11), (8, 5)), 'images': jnp.zeros((8, 1)),
           'task_ids': jnp.zeros(8, jnp.int32)}
    flag = jnp.asarray(True)
    n1, i1 = a1.update(batch, bs, bc, flag, ppo)
    nK, iK = aK.update(batch, bs, bc, flag, ppo)
    _tree_close((n1.actor_params, n1.actor_opt_state), (nK.actor_params, nK.actor_opt_state))
    for k in ('actor/grad_norm', 'actor/pg_loss'):
        np.testing.assert_allclose(float(i1[k]), float(iK[k]), rtol=1e-5, atol=1e-6)
    # ppo_batch actually reaches the loss (Adam's first step is ~sign(g), so compare the gradient norm)
    _, i0 = a1.update(batch, bs, bc, flag)
    assert abs(float(i0['actor/grad_norm']) - float(i1['actor/grad_norm'])) > 1e-4


class _TermFakePi05(_DetFakePi05):
    """Deterministic _actor_terms for grad_diagnostics."""

    def _actor_terms(self, batch, bc_batch, grad_params, rng, ppo_batch=None):
        pred = batch['observations'] @ grad_params['w']
        pg = jnp.mean((pred - batch['rewards']) ** 2)
        bcl = jnp.mean(bc_batch['actions'] * pred[:, None, None] ** 2)
        return pg, 0.0, bcl, {}


@pytest.mark.parametrize('K', [2, 4])
def test_grad_diagnostics_accum_matches_full_batch(K):
    a1, batch, _, bc = _det_agent(False, 1)
    aK, *_ = _det_agent(False, K)
    mk = lambda a: _TermFakePi05(**{f: getattr(a, f) for f in a.__dataclass_fields__})  # noqa: E731
    a1, aK = mk(a1), mk(aK)
    a1 = a1.replace(config={**a1.config, 'bc_coeff': 0.5})
    aK = aK.replace(config={**aK.config, 'bc_coeff': 0.5})
    d1, dK = a1.grad_diagnostics(batch, bc), aK.grad_diagnostics(batch, bc)
    assert d1.keys() == dK.keys()
    for k in d1:
        np.testing.assert_allclose(float(d1[k]), float(dK[k]), rtol=1e-5, atol=1e-6)
    assert float(d1['grad_bc_norm']) > 0 and float(d1['grad_pg_norm']) > 0


@pytest.mark.parametrize('use_sb', [False, True])
def test_donate_agent_matches_plain_update(use_sb):
    """donate_agent=True reuses the old agent's buffers; results equal the plain jit, across two chained steps."""
    plain, batch, bs, bc = _make_fake(use_sb)
    don, *_ = _make_fake(use_sb)
    don = don.replace(config={**don.config, 'donate_agent': True})
    flag = jnp.asarray(True)
    p1, pi1 = plain.update(batch, bs, bc, flag)
    d1, di1 = don.update(batch, bs, bc, flag)
    p2, _ = p1.critic_update(batch, bs, flag)
    d2, _ = d1.critic_update(batch, bs, flag)
    p3, pi3 = p2.update(batch, bs, bc, flag)
    d3, di3 = d2.update(batch, bs, bc, flag)
    leaves_equal = lambda a, b: all(np.array_equal(np.asarray(x), np.asarray(y))  # noqa: E731  (configs differ)
                                    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))
    assert leaves_equal(p3, d3) and _tree_equal(pi3, di3) and _tree_equal(pi1, di1)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_load_config_merged_keys():
    from ogpo.utils.config_loader import config_to_agent_dict, load_config, parse_cli_args
    cfg = load_config(*parse_cli_args(['--algo=ogpo']))
    tr = cfg['training']
    assert tr['actor_update_interval'] == 1 and cfg['env']['parallel_envs'] == 1
    assert cfg['agent']['grad_accum_steps'] == 1 and cfg['agent']['pi05_critic_pooling'] == 'all'
    for k in ('online_schedule', 'num_epochs', 'epoch_episodes', 'epoch_grad_steps'):
        assert k not in tr, k
    assert tr.get('collection_mode', 'online') == 'online'  # main_pi05's default schedule
    cfg = load_config(*parse_cli_args(['--algo=ogpo', '--training.actor_update_interval=100', '--env.parallel_envs=4',
                                       '--agent.grad_accum_steps=2', '--training.collection_mode=rounds',
                                       '--training.num_rounds=5', '--training.rollouts_per_round=20',
                                       '--training.updates_per_round=10000']))
    tr = cfg['training']
    assert tr['actor_update_interval'] == 100 and isinstance(tr['actor_update_interval'], int)
    assert cfg['env']['parallel_envs'] == 4 and isinstance(cfg['env']['parallel_envs'], int)
    assert tr['collection_mode'] == 'rounds'
    assert (tr['num_rounds'], tr['rollouts_per_round'], tr['updates_per_round']) == (5, 20, 10000)
    ac = config_to_agent_dict(cfg)
    assert ac['grad_accum_steps'] == 2 and isinstance(ac['grad_accum_steps'], int)

