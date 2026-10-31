"""
CPU tests for OGPOPi05Agent.critic_loss q_variance_reduction
(docs/changes/2026-09-28-pi05-q-variance-reduction).

The pi05 model is replaced by fakes (`_prefix`, `_sample_sde`); the critic is a real
two-tier `Value` ModuleDict built as in `OGPOPi05Agent.create`, with tiny dims.

  - tiling: SDE row g*B+b carries next state b's KV / prefix mask; KV batch axis is 1.
  - VR on: matches an independent numpy reimplementation of the reference OGPOAgent
    semantics (reduce over G per ensemble member, then aggregate) for every
    q_vr_reduction x q_agg; a mis-tiled mutant fails it.
  - VR off: bit-identical to the pre-change critic_loss (git HEAD 955ae22).
  - VR on with G identical samples == VR off (mean / median / trimmed_mean).
  - gradients reach only modules_critic, and equal the grad against a constant target.
  - create(): the two ValueError conditions, with the checkpoint loaders faked.

Run:  JAX_PLATFORMS=cpu uv run pytest tests/test_pi05_q_vr.py -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ogpo.agents.ogpo_pi05 as P
from ogpo.agents.modules.flax_utils import ModuleDict, TrainState
from ogpo.agents.modules.q_helper import aggregate_q_values, compute_td_loss, compute_td_target
from ogpo.agents.ogpo_pi05 import OGPOPi05Agent, PREFIX_DIM, STATE_DIM, T, _tile
from ogpo.networks import Value

import optax

B, NUM_QS, H_PI, HL = 5, 4, 10, 2          # batch, ensemble, pi05 chunk len, executed steps
L, S, K, HD = 2, 3, 1, 2                    # fake KV cache: layers, seq, kv heads, head dim
FLAT = H_PI * T.MODEL_ACTION_DIM
SAMPLER_LOG = []                            # rows' b (read from the KV cache) per _sample_sde call


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def _row_noise(rng, n):
    return jax.random.normal(rng, (n, FLAT))


class _FakePi05(OGPOPi05Agent):
    """Real critic_loss / _critic_actions / _normalize_env_actions; fake pi05."""

    def _prefix(self, images, task_ids=None, kv_only=False):  # kv_only: 2026-10-26 pi05_drop_empty_camera signature  # task_ids optional: the verbatim pre-change copy omits it
        # images[:, 0] carries the state index b.
        b_idx = images[:, 0, 0, 0, 0].astype(jnp.float32)
        n = images.shape[0]
        k = jnp.broadcast_to(b_idx[None, :, None, None, None], (L, n, S, K, HD))
        kv = (k, k + 100.0)
        pmask = jnp.arange(S)[None, :] <= (b_idx[:, None] % S)     # row-specific mask pattern
        return None, kv, pmask

    def _sample_sde(self, params, kv_cache, prefix_mask, rng):
        k, v = kv_cache
        assert k.shape[1] == v.shape[1] == prefix_mask.shape[0]
        b_kv = np.asarray(k[0, :, 0, 0, 0])
        # KV rows (batch axis 1) and mask rows (axis 0) must describe the same state.
        np.testing.assert_array_equal(np.asarray(v[1, :, 2, 0, 1]) - 100.0, b_kv)
        np.testing.assert_array_equal(np.asarray(prefix_mask).sum(-1) - 1, b_kv % S)
        SAMPLER_LOG.append(b_kv)
        n = prefix_mask.shape[0]
        if self.config.get('_fake_ignore_g', False):
            noise = _row_noise(rng, B)[b_kv.astype(int)]           # same sample for every g
        else:
            noise = _row_noise(rng, n)
        actions = 0.3 * noise + 0.5 * jnp.asarray(b_kv)[:, None] - 1.0
        return actions, None, None, None


def _config(**kw):
    c = dict(pi05_action_horizon=H_PI, horizon_length=HL, num_qs=NUM_QS, q_agg='min',
             discount=0.99, q_variance_reduction=False, q_vr_num_samples=6, q_vr_reduction='mean')
    c.update(kw)
    return c


def _critic_state(seed=0):
    critic_def = Value(
        hidden_dims=(16, 16), layer_norm=True, num_ensembles=NUM_QS, critic_loss_type='mse',
        q_min=-100.0, q_max=0.0, obs_two_tier=True, two_tier_img_dim=PREFIX_DIM,
        two_tier_proprio_dim=STATE_DIM, two_tier_fused_dim=0)
    net = ModuleDict({'critic': critic_def, 'target_critic': copy.deepcopy(critic_def)})
    ex = dict(observations=jnp.zeros((1, PREFIX_DIM + STATE_DIM)), actions=jnp.zeros((1, HL * T.ENV_ACTION_DIM)))
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    params = net.init(k1, critic=ex, target_critic=ex)['params']
    # Distinct target params so critic-vs-target mixups are visible.
    params['modules_target_critic'] = net.init(k2, critic=ex, target_critic=ex)['params']['modules_critic']
    return TrainState.create(net, params, optax.adam(1e-3))


def _agent(**kw):
    consts = {'tokens': jnp.zeros((1, 4), jnp.int32), 'action_q01': -jnp.ones(T.MODEL_ACTION_DIM) * 0.8, 'action_q99': jnp.ones(T.MODEL_ACTION_DIM) * 1.3}
    return _FakePi05(rng=jax.random.PRNGKey(0), actor_params={}, target_actor_params={}, actor_opt_state=None,
                     init_actor_params={},
                     frozen_params={}, critic_network=_critic_state(), consts=consts, config=_config(**kw),
                     model=None, actor_tx=None)


def _batch(seed=1):
    r = np.random.RandomState(seed)
    img = np.zeros((B, 2, 2, 2, 3), np.float32)
    img[:, 0] = np.arange(B)[:, None, None, None]
    return {
        'next_images': jnp.asarray(img),
        'next_task_ids': jnp.zeros(B, jnp.int32),
        'observations': jnp.asarray(r.randn(B, PREFIX_DIM + STATE_DIM), jnp.float32),
        'next_observations': jnp.asarray(r.randn(B, PREFIX_DIM + STATE_DIM), jnp.float32),
        'actions': jnp.asarray(r.uniform(-1, 1, (B, HL, T.ENV_ACTION_DIM)), jnp.float32),
        'rewards': jnp.asarray(-r.rand(B, HL), jnp.float32),
        'masks': jnp.asarray((r.rand(B, HL) > 0.3).astype(np.float32)),
        'valid': jnp.asarray((r.rand(B, HL) > 0.2).astype(np.float32)),
    }


def _pre_change_critic_loss(self, batch, grad_params, rng):
    """Verbatim OGPOPi05Agent.critic_loss at git HEAD 955ae22."""
    c = self.config
    sample_rng, agg_rng = jax.random.split(rng)
    batch_actions = self._normalize_env_actions(batch['actions']).reshape(batch['actions'].shape[0], -1)

    _, kv, pmask = self._prefix(batch['next_images'])
    next_actions, _, _, _ = self._sample_sde(self.target_actor_params, kv, pmask, sample_rng)
    next_actions = self._critic_actions(next_actions)
    next_qs = self.critic_network.select('target_critic')(batch['next_observations'], next_actions, is_encoded=True)
    next_q = aggregate_q_values(next_qs, method=c['q_agg'], rng=agg_rng, num_qs=c['num_qs'])

    target_q = compute_td_target(batch['rewards'], batch['masks'], next_q, c['discount'], c['horizon_length'])
    q = self.critic_network.select('critic')(batch['observations'], batch_actions, params=grad_params, is_encoded=True)
    td_loss, _ = compute_td_loss(q, target_q, valid_mask=batch['valid'][..., -1], loss_type='mse')
    return td_loss, {
        'td_loss': td_loss, 'q_mean': q.mean(), 'q_max': q.max(), 'q_min': q.min(),
        'next_q_mean': next_q.mean(), 'next_actions_std': next_actions.std(),
    }


# ---------------------------------------------------------------------------
# Independent numpy reference (OGPOAgent VR semantics)
# ---------------------------------------------------------------------------

def _np_reduce(x, method):                   # x: (G, num_qs, B)
    if method == 'mean':
        return x.mean(0)
    if method == 'median':
        return np.median(x, 0)
    if method == 'trimmed_mean':
        G = x.shape[0]
        lo = G // 4
        return np.sort(x, 0)[lo:G - lo].mean(0)
    raise AssertionError(method)


def _np_agg(x, method, agg_rng):             # x: (num_qs, B)
    if method == 'min':
        return x.min(0)
    if method == 'mean':
        return x.mean(0)
    if method == 'subsample':
        idx = np.asarray(jax.random.randint(agg_rng, (2,), 0, NUM_QS))
        return x[idx].min(0)
    raise AssertionError(method)


def _reference(agent, batch, rng, G, reduction, q_agg, grad_params):
    sample_rng, agg_rng = jax.random.split(rng)
    noise = np.asarray(_row_noise(sample_rng, G * B)).reshape(G, B, FLAT)
    b = np.arange(B, dtype=np.float32)
    act = 0.3 * noise + 0.5 * b[None, :, None] - 1.0                      # (G, B, FLAT), paired with state b
    act = act.reshape(G, B, H_PI, T.MODEL_ACTION_DIM)[:, :, :HL, :T.ENV_ACTION_DIM].reshape(G, B, -1)
    tc = agent.critic_network.select('target_critic')
    qs = np.stack([np.asarray(tc(batch['next_observations'], jnp.asarray(act[g]), is_encoded=True))
                   for g in range(G)])                                    # (G, num_qs, B), per-g calls
    next_q = _np_agg(_np_reduce(qs, reduction), q_agg, agg_rng)
    target = np.asarray(batch['rewards'])[:, -1] + 0.99 ** HL * np.asarray(batch['masks'])[:, -1] * next_q
    q01, q99 = -0.8, 1.3
    ba = ((np.asarray(batch['actions']) - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0).reshape(B, -1)
    q = np.asarray(agent.critic_network.select('critic')(batch['observations'], jnp.asarray(ba),
                                                       params=grad_params, is_encoded=True))
    loss = (np.square(q - target) * np.asarray(batch['valid'])[:, -1]).mean()
    std = qs.std(0)
    return loss, next_q, std


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_tile_kv_axis1_matches_pmask_axis0():
    n, G = 3, 4
    kv = (jnp.arange(L * n * S).reshape(L, n, S, 1, 1).astype(jnp.float32),) * 2
    pm = jnp.arange(n * S).reshape(n, S)
    kv_g, pm_g = _tile(kv, G, axis=1), _tile(pm, G, axis=0)
    assert kv_g[0].shape == (L, G * n, S, 1, 1) and pm_g.shape == (G * n, S)
    for g in range(G):
        for b in range(n):
            np.testing.assert_array_equal(kv_g[0][:, g * n + b], kv[0][:, b])
            np.testing.assert_array_equal(pm_g[g * n + b], pm[b])


def test_vr_pairing_rows_are_g_major():
    SAMPLER_LOG.clear()
    G = 6
    agent = _agent(q_variance_reduction=True, q_vr_num_samples=G)
    agent.critic_loss(_batch(), agent.critic_network.params, jax.random.PRNGKey(3))
    assert len(SAMPLER_LOG) == 1                                   # one batched SDE call
    np.testing.assert_array_equal(SAMPLER_LOG[0], np.tile(np.arange(B), G))


@pytest.mark.parametrize('reduction', ['mean', 'median', 'trimmed_mean'])
@pytest.mark.parametrize('q_agg', ['min', 'mean', 'subsample'])
@pytest.mark.parametrize('G', [2, 6])
def test_vr_matches_numpy_reference(reduction, q_agg, G):
    agent = _agent(q_variance_reduction=True, q_vr_num_samples=G, q_vr_reduction=reduction, q_agg=q_agg)
    batch, rng = _batch(), jax.random.PRNGKey(11)
    gp = agent.critic_network.params
    loss, info = agent.critic_loss(batch, gp, rng)
    ref_loss, ref_next_q, ref_std = _reference(agent, batch, rng, G, reduction, q_agg, gp)
    np.testing.assert_allclose(float(info['next_q_mean']), ref_next_q.mean(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(float(loss), ref_loss, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(float(info['q_vr_sample_std']), ref_std.mean(), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(float(info['q_vr_sample_std_max']), ref_std.max(), rtol=1e-4, atol=1e-6)
    assert set(info) == {'td_loss', 'q_mean', 'q_max', 'q_min', 'next_q_mean', 'next_actions_std',
                         'target_q_mean', 'q_head_std', 'q_vr_sample_std', 'q_vr_sample_std_max'}


def test_vr_reduce_then_aggregate_order_matters():
    """Guard that the reference check can tell reduce->agg from agg->reduce."""
    G = 6
    agent = _agent(q_variance_reduction=True, q_vr_num_samples=G, q_vr_reduction='median', q_agg='min')
    batch, rng = _batch(), jax.random.PRNGKey(11)
    _, info = agent.critic_loss(batch, agent.critic_network.params, rng)
    sample_rng, _ = jax.random.split(rng)
    noise = np.asarray(_row_noise(sample_rng, G * B)).reshape(G, B, FLAT)
    act = (0.3 * noise + 0.5 * np.arange(B)[None, :, None] - 1.0).reshape(G, B, H_PI, -1)[:, :, :HL, :7].reshape(G, B, -1)
    tc = agent.critic_network.select('target_critic')
    qs = np.stack([np.asarray(tc(batch['next_observations'], jnp.asarray(act[g]), is_encoded=True)) for g in range(G)])
    swapped = np.median(qs.min(1), 0).mean()                       # aggregate first, then reduce
    assert abs(float(info['next_q_mean']) - swapped) > 1e-4


def test_mis_tiled_mutant_fails_reference(monkeypatch):
    """jnp.repeat tiling (b*G+g) must be caught by the reference comparison."""
    monkeypatch.setattr(P, '_tile', lambda tree, n, axis: jax.tree.map(lambda x: jnp.repeat(x, n, axis=axis), tree))
    G = 3
    agent = _agent(q_variance_reduction=True, q_vr_num_samples=G, q_vr_reduction='mean', q_agg='mean')
    batch, rng = _batch(), jax.random.PRNGKey(11)
    loss, _ = agent.critic_loss(batch, agent.critic_network.params, rng)
    ref_loss, _, _ = _reference(agent, batch, rng, G, 'mean', 'mean', agent.critic_network.params)
    assert not np.allclose(float(loss), ref_loss, rtol=1e-5)


@pytest.mark.parametrize('q_agg', ['min', 'mean', 'subsample'])
def test_vr_off_bit_identical_to_pre_change(q_agg):
    agent = _agent(q_variance_reduction=False, q_agg=q_agg)
    batch = _batch()
    for seed in range(3):
        rng = jax.random.PRNGKey(seed)
        new_loss, new_info = agent.critic_loss(batch, agent.critic_network.params, rng)
        old = _pre_change_critic_loss(agent, batch, agent.critic_network.params, rng)
        assert set(new_info) == set(old[1]) | {'target_q_mean', 'q_head_std'}  # critic diagnostics added since
        new = (new_loss, {k: new_info[k] for k in old[1]})
        assert jax.tree.structure(new) == jax.tree.structure(old)
        for a, b in zip(jax.tree.leaves(new), jax.tree.leaves(old)):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    gn = jax.grad(lambda p: agent.critic_loss(batch, p, rng)[0])(agent.critic_network.params)
    go = jax.grad(lambda p: _pre_change_critic_loss(agent, batch, p, rng)[0])(agent.critic_network.params)
    for a, b in zip(jax.tree.leaves(gn), jax.tree.leaves(go)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize('reduction', ['mean', 'median', 'trimmed_mean'])
@pytest.mark.parametrize('q_agg', ['min', 'mean', 'subsample'])
def test_vr_identical_samples_equals_non_vr(reduction, q_agg):
    batch, rng = _batch(), jax.random.PRNGKey(5)
    on = _agent(q_variance_reduction=True, q_vr_num_samples=8, q_vr_reduction=reduction, q_agg=q_agg,
                _fake_ignore_g=True)
    off = _agent(q_variance_reduction=False, q_agg=q_agg, _fake_ignore_g=True)
    l_on, i_on = on.critic_loss(batch, on.critic_network.params, rng)
    l_off, i_off = off.critic_loss(batch, off.critic_network.params, rng)
    np.testing.assert_allclose(float(l_on), float(l_off), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(float(i_on['next_q_mean']), float(i_off['next_q_mean']), rtol=1e-6, atol=1e-7)
    assert float(i_on['q_vr_sample_std_max']) < 1e-5


def test_vr_gradients_only_through_online_critic():
    agent = _agent(q_variance_reduction=True, q_vr_num_samples=4, q_vr_reduction='median', q_agg='subsample')
    batch, rng = _batch(), jax.random.PRNGKey(2)
    gp = agent.critic_network.params
    grads = jax.grad(lambda p: agent.critic_loss(batch, p, rng)[0])(gp)
    tgt = jax.tree.leaves(grads['modules_target_critic'])
    assert all(float(jnp.abs(g).max()) == 0.0 for g in tgt)
    assert sum(float(jnp.abs(g).sum()) for g in jax.tree.leaves(grads['modules_critic'])) > 0

    # Equal to the grad of MSE against a constant target built by the reference.
    _, next_q, _ = _reference(agent, batch, rng, 4, 'median', 'subsample', gp)
    target = jnp.asarray(np.asarray(batch['rewards'])[:, -1] + 0.99 ** HL * np.asarray(batch['masks'])[:, -1] * next_q)
    ba = agent._normalize_env_actions(batch['actions']).reshape(B, -1)

    def const_loss(p):
        q = agent.critic_network.select('critic')(batch['observations'], ba, params=p, is_encoded=True)
        return (jnp.square(q - target) * batch['valid'][:, -1]).mean()

    ref_grads = jax.grad(const_loss)(gp)
    for a, b in zip(jax.tree.leaves(grads['modules_critic']), jax.tree.leaves(ref_grads['modules_critic'])):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-4, atol=1e-6)


def test_vr_jit_matches_eager():
    agent = _agent(q_variance_reduction=True, q_vr_num_samples=4, q_vr_reduction='trimmed_mean', q_agg='min')
    batch, rng = _batch(), jax.random.PRNGKey(9)
    eager = agent.critic_loss(batch, agent.critic_network.params, rng)

    # _sample_sde's numpy checks need concrete arrays; jit a variant with a pure fake.
    class _Pure(_FakePi05):
        def _sample_sde(self, params, kv_cache, prefix_mask, rng):
            b_kv = kv_cache[0][0, :, 0, 0, 0]
            return 0.3 * _row_noise(rng, prefix_mask.shape[0]) + 0.5 * b_kv[:, None] - 1.0, None, None, None

    pure = _Pure(**{f: getattr(agent, f) for f in ('rng', 'actor_params', 'target_actor_params', 'actor_opt_state',
                                                    'init_actor_params', 'frozen_params', 'critic_network', 'consts', 'config',
                                                    'model', 'actor_tx')})
    jitted = jax.jit(lambda a, p: a.critic_loss(batch, p, rng))(pure, pure.critic_network.params)
    np.testing.assert_allclose(float(jitted[0]), float(eager[0]), rtol=1e-5)
    np.testing.assert_allclose(float(jitted[1]['next_q_mean']), float(eager[1]['next_q_mean']), rtol=1e-5)


# ---------------------------------------------------------------------------
# create() validation (checkpoint loaders faked)
# ---------------------------------------------------------------------------

class _Stop(Exception):
    pass


@pytest.fixture
def fake_ckpt(monkeypatch):
    monkeypatch.setattr(P, 'Pi05', lambda **kw: None)
    monkeypatch.setattr(P, 'load_pi05_params', lambda path: {})
    monkeypatch.setattr(P, 'split_params', lambda p: ({}, {'w': np.zeros(2)}))
    ns = {'actions': {'q01': np.zeros(32), 'q99': np.ones(32)}, 'state': {'q01': np.zeros(32), 'q99': np.ones(32)}}
    monkeypatch.setattr(P.T, 'load_norm_stats', lambda path, asset_id: ns)

    class _Tok:
        def __init__(self, *a):
            pass

        def tokenize(self, prompt):
            return np.zeros(4, np.int32), np.ones(4, bool)

    monkeypatch.setattr(P.T, 'PromptTokenizer', _Tok)

    def _stop(*a, **k):                      # reached only after validation passed
        raise _Stop()

    monkeypatch.setattr(P, 'Value', _stop)
    monkeypatch.setattr(P.optax, 'adamw', lambda *a, **k: optax.identity())


def _create_cfg(**kw):
    c = dict(pi05_checkpoint='/nonexistent', pi05_robot='libero', pi05_tokenizer='x', pi05_action_horizon=10, horizon_length=5,
             pi05_clip_grad_norm=1.0, pi05_lr=1e-5, pi05_weight_decay=0.0,
             q_variance_reduction=True, q_vr_num_samples=16, q_vr_reduction='median',
             pi05_critic_pooling='all', pi05_critic_arch='mlp',
             subsample_bon=True, pi05_bon_sampler='sde', pi05_bon_agg='subsample', use_value_fn=False)
    c.update(kw)
    return _StopCfg(c)


class _StopCfg(dict):
    """Raises _Stop on the first critic key, i.e. only after validation passed."""

    def __getitem__(self, k):
        if k == 'value_hidden_dims':
            raise _Stop()
        return super().__getitem__(k)


@pytest.mark.parametrize('kw,match', [
    (dict(q_vr_num_samples=1), 'q_vr_num_samples=1'),
    (dict(q_vr_num_samples=0), 'q_vr_num_samples=0'),
    (dict(q_vr_reduction='max'), 'mean \\| median \\| trimmed_mean'),
    (dict(q_variance_reduction=False, q_vr_reduction='avg'), 'q_vr_reduction'),
])
def test_create_rejects(fake_ckpt, kw, match):
    with pytest.raises(ValueError, match=match):
        OGPOPi05Agent.create(0, ['task'], _create_cfg(**kw))


@pytest.mark.parametrize('kw', [dict(), dict(q_vr_num_samples=2, q_vr_reduction='trimmed_mean'),
                                dict(q_variance_reduction=False, q_vr_num_samples=1, q_vr_reduction='mean')])
def test_create_accepts(fake_ckpt, kw):
    with pytest.raises(_Stop):
        OGPOPi05Agent.create(0, ['task'], _create_cfg(**kw))


def test_yaml_defaults_have_vr_keys():
    from ogpo.utils.config_loader import load_config
    cfg = load_config('ogpo', {})
    a = cfg['agent']
    assert a['q_variance_reduction'] is False and a['q_vr_num_samples'] == 8 and a['q_vr_reduction'] == 'mean'
