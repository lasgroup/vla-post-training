"""
Independent verifier tests for the pi05 SARSA-V critic (docs/changes/2026-10-09-sarsa-v-critic/).

Stub agent: OGPOPi05Agent with a fake `_prefix` / `_actor_fn` (as tests/test_pi05_bon_only.py) and the REAL
`_sample_sde`, `_log_prob`, `_actor_terms` (PPO + BC), critic and value networks, on the YAML agent config.

  1. Default path (value_network=None) == VERBATIM 8891663 `critic_loss` / `_critic_step` / `_critic_updates` /
     `_update_impl` / `_critic_update_impl`, jitted, bit-exact (params, opt states, rng, info values and key set),
     over q_variance_reduction x use_success_buffer_q x success_flag x grad_accum_steps, plus donate_agent=True.
  2. SARSA-V: full `update` (actor + critic) — the actor half is bit-identical to the default agent's (PPO advantages
     read the Q target critic, never V); perturbing V changes nothing in the actor; update's critic/value half ==
     critic_update; donate path == plain path; v_agg='subsample' draws its head indices from the documented keys
     (Q bootstrap over v_ensemble_size heads, V target over num_qs heads); grad_accum_steps=2 for `_value_step`
     == full batch; per-task info for the V step; `_prefix` / `_sample_sde` never called on the critic path.
  3. create(): use_value_fn validation (raised before any checkpoint load), V architecture (critic kwargs minus the
     action input, v_ensemble_size heads) for mlp / bronet / bronet_vla, separate optimiser instance, distinct target
     buffers, default rng / critic params unchanged.

Run:  JAX_PLATFORMS=cpu python -m pytest tests/test_pi05_sarsa_v_verify.py -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import copy

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import ogpo.agents.ogpo_pi05 as P
from ogpo.agents.modules.flax_utils import ModuleDict, TrainState
from ogpo.agents.modules.q_helper import (aggregate_q_values, compute_td_loss, compute_td_target,
                                          reduce_q_over_samples)
from ogpo.agents.ogpo_pi05 import OGPOPi05Agent, PREFIX_DIM, STATE_DIM, T, _per_task, _tile
from ogpo.networks import Value
from ogpo.utils.config_loader import config_to_agent_dict, load_config

B, NUM_QS, NUM_VS, H_PI, HL = 4, 4, 3, 10, 5
OBS = PREFIX_DIM + STATE_DIM
L, S, KV, HD = 2, 3, 1, 2

CALLS = []


class _Fake(OGPOPi05Agent):
    """Fake VLM prefix / velocity; real SDE sampler, log-prob, actor terms, critic, value."""

    def _prefix(self, images, task_ids=None, kv_only=False):  # kv_only: 2026-10-26 pi05_drop_empty_camera signature
        CALLS.append('prefix')
        b_idx = images[:, 0, 0, 0, 0].astype(jnp.float32)
        n = images.shape[0]
        k = jnp.broadcast_to(b_idx[None, :, None, None, None], (L, n, S, KV, HD))
        return None, (k, k + 100.0), jnp.ones((n, S), bool)

    def _actor_fn(self, params, kv_cache, prefix_mask):
        k, _ = kv_cache
        b = k[0, :, 0, 0, 0]

        def actor_fn(obs, x, t, params_=None, is_encoded=True):
            return params['w'] * x + 0.5 * b[:, None] + 2.0 * t

        return actor_fn


class _NoPolicy(_Fake):
    def _prefix(self, images, task_ids=None, kv_only=False):  # kv_only: 2026-10-26 pi05_drop_empty_camera signature
        raise AssertionError("SARSA-V critic path must not run the VLM prefix")

    def _sample_sde(self, *a, **k):
        raise AssertionError("SARSA-V critic path must not sample the policy")


def _config(**kw):
    c = config_to_agent_dict(load_config('ogpo', {}))
    c.update(pi05_action_horizon=H_PI, horizon_length=HL, num_qs=NUM_QS, v_ensemble_size=NUM_VS,
             batch_size=B, ppo_batch_size=B, group_num_samples=3, flow_steps=4, tau=0.3, actor_tau=0.2,
             q_vr_num_samples=3, donate_agent=False, grad_accum_steps=1, use_success_buffer_q=False,
             q_min=-100.0, q_max=0.0, adv_clip_min=None, use_bc_regularization=True,
             use_tapered_noise=True, use_constant_noise=False, constant_noise_std=0.1)
    c.update(kw)
    return c


def _vdef(n):
    return Value(hidden_dims=(16, 16), layer_norm=True, num_ensembles=n, critic_loss_type='mse',
                 q_min=-100.0, q_max=0.0, obs_two_tier=True, two_tier_img_dim=PREFIX_DIM,
                 two_tier_proprio_dim=STATE_DIM, two_tier_fused_dim=0)


def _nets(tx_q, tx_v, seed=0):
    k = jax.random.split(jax.random.PRNGKey(seed), 4)
    qdef, vdef = _vdef(NUM_QS), _vdef(NUM_VS)
    qnet = ModuleDict({'critic': qdef, 'target_critic': copy.deepcopy(qdef)})
    vnet = ModuleDict({'value': vdef, 'target_value': copy.deepcopy(vdef)})
    exq = dict(observations=jnp.zeros((1, OBS)), actions=jnp.zeros((1, HL * T.ENV_ACTION_DIM)))
    exv = dict(observations=jnp.zeros((1, OBS)))
    qp = qnet.init(k[0], critic=exq, target_critic=exq)['params']
    qp['modules_target_critic'] = qnet.init(k[1], critic=exq, target_critic=exq)['params']['modules_critic']
    vp = vnet.init(k[2], value=exv, target_value=exv)['params']
    vp['modules_target_value'] = vnet.init(k[3], value=exv, target_value=exv)['params']['modules_value']
    return TrainState.create(qnet, qp, tx_q), TrainState.create(vnet, vp, tx_v)


def _agent(cls=_Fake, with_v=False, n_tasks=1, tx_q=None, tx_v=None, **kw):
    dim_mask = np.zeros((H_PI, T.MODEL_ACTION_DIM), np.float32)
    dim_mask[:HL, :T.ENV_ACTION_DIM] = 1.0
    consts = {'tokens': jnp.zeros((n_tasks, 4), jnp.int32), 'dim_mask': jnp.asarray(dim_mask.reshape(-1)),
              'action_q01': -jnp.ones(T.MODEL_ACTION_DIM) * 0.8, 'action_q99': jnp.ones(T.MODEL_ACTION_DIM) * 1.3}
    q, v = _nets(tx_q or optax.adam(1e-2), tx_v or optax.adam(1e-2))
    actor_tx = optax.adam(1e-2)
    w = {'w': jnp.float32(0.3)}
    return cls(rng=jax.random.PRNGKey(3), actor_params=w, target_actor_params={'w': jnp.float32(0.7)},
               actor_opt_state=actor_tx.init(w), init_actor_params={'w': jnp.float32(0.3)}, frozen_params={},
               critic_network=q, consts=consts, config=_config(**kw), model=None, actor_tx=actor_tx,
               value_network=v if with_v else None)


def _batch(seed=0, n_tasks=1):
    r = np.random.RandomState(seed)
    img = np.zeros((B, 2, 2, 2, 3), np.float32)
    img[:, 0] = (np.arange(B) + 10 * seed)[:, None, None, None]
    valid = np.ones((B, HL), np.float32)
    valid[0, -1] = 0.0
    masks = np.ones((B, HL), np.float32)
    masks[1, -1] = 0.0
    b = {
        'images': jnp.asarray(img), 'next_images': jnp.asarray(img + 1.0),
        'observations': jnp.asarray(r.randn(B, OBS), jnp.float32),
        'next_observations': jnp.asarray(r.randn(B, OBS), jnp.float32),
        'actions': jnp.asarray(r.uniform(-1, 1, (B, HL, T.ENV_ACTION_DIM)), jnp.float32),
        'rewards': jnp.asarray(-r.rand(B, HL), jnp.float32), 'masks': jnp.asarray(masks), 'valid': jnp.asarray(valid),
        'mc_returns': jnp.asarray(-r.rand(B) * 10, jnp.float32),
        'is_success': jnp.asarray((r.rand(B, HL) > 0.5).astype(np.float32)),
    }
    if n_tasks > 1:
        b['task_ids'] = jnp.asarray(np.arange(B) % n_tasks, jnp.int32)
    return b


def _bc(seed=0):
    r = np.random.RandomState(50 + seed)
    img = np.zeros((B, 2, 2, 2, 3), np.float32)
    img[:, 0] = np.arange(B)[:, None, None, None]
    return {'images': jnp.asarray(img), 'actions': jnp.asarray(r.uniform(-1, 1, (B, H_PI, T.ENV_ACTION_DIM)), jnp.float32),
            'valid': jnp.ones((B, H_PI), jnp.float32)}


def _fields(agent):
    return {f: getattr(agent, f) for f in agent.__dataclass_fields__}


def _assert_tree_equal(a, b, leaves_only=False):
    if leaves_only:  # separately built agents hold distinct (static) optax instances
        assert len(jax.tree.leaves(a)) == len(jax.tree.leaves(b))
    else:
        assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def _assert_info_equal(a, b):
    assert set(a) == set(b), (set(a) ^ set(b))
    for k in a:
        np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]), err_msg=k)


def _close(a, b, rtol=1e-5, atol=1e-6):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=rtol, atol=atol)


def _differs(a, b):
    return any(not np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))


# --- VERBATIM ogpo/agents/ogpo_pi05.py @ 8891663 ----------------------------------------------------------------

def _head_critic_loss(self, batch, grad_params, rng):
    c = self.config
    sample_rng, agg_rng = jax.random.split(rng)
    batch_actions = self._normalize_env_actions(batch['actions']).reshape(batch['actions'].shape[0], -1)

    _, kv, pmask = self._prefix(batch['next_images'], batch.get('task_ids'))
    target_critic = self.critic_network.select('target_critic')
    vr_info = {}
    if c['q_variance_reduction']:
        # Mirrors OGPOAgent's q_variance_reduction path with one batched SDE call:
        # reduce over G samples per ensemble member, then aggregate over the ensemble.
        G, B = c['q_vr_num_samples'], pmask.shape[0]
        kv_g, pmask_g = _tile(kv, G, axis=1), _tile(pmask, G, axis=0)  # index g * B + b
        next_actions, _, _, _ = self._sample_sde(self.target_actor_params, kv_g, pmask_g, sample_rng)
        next_actions = self._critic_actions(next_actions).reshape(G, B, -1)
        next_qs_g = jax.vmap(
            lambda a: target_critic(batch['next_observations'], a, is_encoded=True))(next_actions)  # (G, num_qs, B)
        next_qs = reduce_q_over_samples(next_qs_g, method=c['q_vr_reduction'])
        q_std = next_qs_g.std(axis=0)
        vr_info = {'q_vr_sample_std': q_std.mean(), 'q_vr_sample_std_max': q_std.max(),
                   'next_actions_std': next_actions.std(axis=0).mean()}  # across-sample, as the reference
    else:
        next_actions, _, _, _ = self._sample_sde(self.target_actor_params, kv, pmask, sample_rng)
        next_actions = self._critic_actions(next_actions)
        next_qs = target_critic(batch['next_observations'], next_actions, is_encoded=True)
    next_q = aggregate_q_values(next_qs, method=c['q_agg'], rng=agg_rng, num_qs=c['num_qs'])

    target_q = compute_td_target(batch['rewards'], batch['masks'], next_q, c['discount'], c['horizon_length'])
    q = self.critic_network.select('critic')(batch['observations'], batch_actions, params=grad_params, is_encoded=True)
    td_loss, _ = compute_td_loss(q, target_q, valid_mask=batch['valid'][..., -1], loss_type='mse')
    K = self.consts['tokens'].shape[0]
    per_task = _per_task({
        'q_mean': q.mean(axis=0), 'td_err_sq': jnp.square(q.mean(axis=0) - target_q),
        'target_q': target_q,
    }, batch.get('task_ids'), K)
    return td_loss, {
        **per_task,
        'td_loss': td_loss, 'q_mean': q.mean(), 'q_max': q.max(), 'q_min': q.min(),
        'next_q_mean': next_q.mean(), 'next_actions_std': next_actions.std(), **vr_info,
        'target_q_mean': target_q.mean(), 'q_head_std': q.std(axis=0).mean(),
        **self._mc_stats(jax.lax.stop_gradient(q).mean(axis=0), batch),
    }


def _head_critic_step(self, critic_network, batch, rng):
    agent = self.replace(critic_network=critic_network)
    if self._accum_steps == 1:
        new_critic, info = critic_network.apply_loss_fn(lambda p: agent.critic_loss(batch, p, rng))
    else:
        grads, info = self._accum_grad(lambda p, b, k: agent.critic_loss(b, p, k), critic_network.params, batch, rng)
        leaves = jax.tree.leaves(grads)  # same grad/* stats as TrainState.apply_loss_fn
        info.update({'grad/max': jnp.max(jnp.stack([x.max() for x in leaves])),
                     'grad/min': jnp.min(jnp.stack([x.min() for x in leaves])),
                     'grad/norm': jnp.sum(jnp.stack([jnp.linalg.norm(x.reshape(-1)) for x in leaves]))})
        new_critic = critic_network.apply_gradients(grads=grads)
    tau = self.config['tau']
    new_critic.params['modules_target_critic'] = jax.tree.map(
        lambda p, tp: p * tau + tp * (1 - tau),
        new_critic.params['modules_critic'], new_critic.params['modules_target_critic'])
    return new_critic, info


def _head_critic_updates(self, batch, batch_success, success_flag, critic_rng, sb_rng):
    """RB TD step, then the success-buffer TD step if enabled and available."""
    new_critic, critic_info = self._critic_step(self.critic_network, batch, critic_rng)
    if self.config.get('use_success_buffer_q', False):
        new_critic = jax.lax.cond(
            success_flag,
            lambda cn: self._critic_step(cn, batch_success, sb_rng)[0],
            lambda cn: cn,
            new_critic)
    return new_critic, {f'critic/{k}': v for k, v in critic_info.items()}


def _head_update_impl(self, batch, batch_success, bc_batch, success_flag, ppo_batch=None):
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
    actor_info['grad_norm'] = optax.global_norm(grads)

    # Critic sees the pre-update target actor, as in OGPOAgent._update.
    new_critic, critic_info = self._critic_updates(batch, batch_success, success_flag, critic_rng, sb_rng)

    info = {**{f'actor/{k}': v for k, v in actor_info.items()}, **critic_info}
    return self.replace(
        rng=new_rng, actor_params=new_actor_params, target_actor_params=new_target,
        actor_opt_state=new_opt_state, critic_network=new_critic), info


def _head_critic_update_impl(self, batch, batch_success, success_flag):
    new_rng, _, critic_rng, sb_rng = jax.random.split(self.rng, 4)
    new_critic, critic_info = self._critic_updates(batch, batch_success, success_flag, critic_rng, sb_rng)
    return self.replace(rng=new_rng, critic_network=new_critic), critic_info


class _Head(_Fake):
    critic_loss = _head_critic_loss
    _critic_step = _head_critic_step
    _critic_updates = _head_critic_updates


_head_update = jax.jit(_head_update_impl)
_head_critic_update = jax.jit(_head_critic_update_impl)


# --- 1. default path bit-identity --------------------------------------------------------------------------------

DEFAULT_GRID = [(vr, sb, flag, acc) for vr in (False, True) for sb in (False, True)
                for flag in (False, True) for acc in (1, 2) if sb or not flag]


@pytest.mark.parametrize('vr,sb,flag,acc', DEFAULT_GRID)
def test_default_update_bit_identical_to_head(vr, sb, flag, acc):
    kw = dict(q_variance_reduction=vr, use_success_buffer_q=sb, grad_accum_steps=acc)
    agent = _agent(**kw)
    head = _Head(**_fields(agent))
    assert agent.value_network is None
    batch, bs, bc, f = _batch(0), _batch(1), _bc(), jnp.asarray(flag)
    new, new_info = agent.update(batch, bs, bc, f)
    old, old_info = _head_update(head, batch, bs, bc, f)
    assert new.value_network is None
    _assert_tree_equal(new, _Fake(**_fields(old)))
    _assert_info_equal(new_info, old_info)
    # second step: rng advance identical
    new2, i2 = new.update(batch, bs, bc, f)
    old2, j2 = _head_update(_Head(**_fields(old)), batch, bs, bc, f)
    _assert_tree_equal(new2, _Fake(**_fields(old2)))
    _assert_info_equal(i2, j2)


@pytest.mark.parametrize('vr,sb,flag,acc', DEFAULT_GRID)
def test_default_critic_update_bit_identical_to_head(vr, sb, flag, acc):
    kw = dict(q_variance_reduction=vr, use_success_buffer_q=sb, grad_accum_steps=acc)
    agent = _agent(**kw)
    head = _Head(**_fields(agent))
    batch, bs, f = _batch(0), _batch(1), jnp.asarray(flag)
    new, new_info = agent.critic_update(batch, bs, f)
    old, old_info = _head_critic_update(head, batch, bs, f)
    _assert_tree_equal(new, _Fake(**_fields(old)))
    _assert_info_equal(new_info, old_info)
    if sb and acc == 1 and not vr:  # teeth: the SB branch changes the critic when flagged
        other, _ = agent.critic_update(batch, bs, jnp.asarray(not flag))
        assert _differs(other.critic_network.params, new.critic_network.params)


@pytest.mark.parametrize('vr', [False, True])
def test_default_critic_loss_and_step_unjitted_equal_head(vr):
    agent = _agent(q_variance_reduction=vr, n_tasks=2)
    head = _Head(**_fields(agent))
    batch, rng = _batch(0, n_tasks=2), jax.random.PRNGKey(11)
    with jax.disable_jit():
        l1, i1 = agent.critic_loss(batch, agent.critic_network.params, rng)
        l0, i0 = _head_critic_loss(head, batch, head.critic_network.params, rng)
    np.testing.assert_array_equal(np.asarray(l1), np.asarray(l0))
    _assert_info_equal(i1, i0)
    assert list(i1) == list(i0)  # same key order too
    c1, ci1 = jax.jit(lambda a: a._critic_step(a.critic_network, batch, rng))(agent)
    c0, ci0 = jax.jit(lambda a: _head_critic_step(a, a.critic_network, batch, rng))(head)
    _assert_tree_equal(c1, c0)
    _assert_info_equal(ci1, ci0)


@pytest.mark.parametrize('sb', [False, True])
def test_default_donate_path_bit_identical_to_head(sb):
    kw = dict(use_success_buffer_q=sb, donate_agent=True)
    batch, bs, bc, f = _batch(0), _batch(1), _bc(), jnp.asarray(True)
    new, ni = _agent(**kw).update(batch, bs, bc, f)
    old, oi = _head_update(_Head(**_fields(_agent(**kw))), batch, bs, bc, f)
    _assert_tree_equal(new, old, leaves_only=True)
    _assert_info_equal(ni, oi)
    newc, nci = _agent(**kw).critic_update(batch, bs, f)
    oldc, oci = _head_critic_update(_Head(**_fields(_agent(**kw))), batch, bs, f)
    _assert_tree_equal(newc, oldc, leaves_only=True)
    _assert_info_equal(nci, oci)


def test_default_pytree_has_no_value_leaves():
    a = _agent()
    head_like = _Head(**_fields(a))
    assert len(jax.tree.leaves(a)) == len(jax.tree.leaves(head_like))
    assert jax.tree.leaves(a.value_network) == []
    assert len(jax.tree.leaves(_agent(with_v=True))) > len(jax.tree.leaves(a))


# --- 2. SARSA-V behaviour ----------------------------------------------------------------------------------------

def test_sarsa_actor_half_identical_to_default_and_independent_of_v():
    """OGPO + SARSA-V: PPO advantages read the Q target critic; the actor update never sees V."""
    batch, bs, bc, f = _batch(0), _batch(1), _bc(), jnp.asarray(False)
    s_agent = _agent(with_v=True)
    d, di = s_agent.replace(value_network=None).update(batch, bs, bc, f)
    s, si = s_agent.update(batch, bs, bc, f)
    for k in ('actor_params', 'target_actor_params', 'actor_opt_state', 'rng'):
        _assert_tree_equal(getattr(s, k), getattr(d, k))
    _assert_info_equal({k: v for k, v in si.items() if k.startswith('actor/')},
                       {k: v for k, v in di.items() if k.startswith('actor/')})
    # V params perturbed -> actor identical, critic different
    vp = jax.tree.map(lambda x: x + 0.5, s_agent.value_network.params)
    s2, si2 = s_agent.replace(value_network=s_agent.value_network.replace(params=vp)).update(batch, bs, bc, f)
    _assert_tree_equal(s2.actor_params, s.actor_params)
    _assert_info_equal({k: v for k, v in si2.items() if k.startswith('actor/')},
                       {k: v for k, v in si.items() if k.startswith('actor/')})
    assert _differs(s2.critic_network.params, s.critic_network.params)
    # critic target differs from the SDE target (teeth)
    assert _differs(s.critic_network.params, d.critic_network.params)
    # Q target critic used by the actor: perturbing it changes the actor step
    qp = jax.tree.map(lambda x: x * 1.5, s_agent.critic_network.params)
    s3, si3 = s_agent.replace(critic_network=s_agent.critic_network.replace(params=qp)).update(batch, bs, bc, f)
    # (Adam's first step is ~lr * sign(g), so compare the advantage inputs, not the params)
    assert float(si3['actor/q_values']) != float(si['actor/q_values'])
    assert _differs(si3['actor/adv_std'], si['actor/adv_std']) or _differs(si3['actor/pg_loss'], si['actor/pg_loss'])


@pytest.mark.parametrize('sb,flag', [(False, False), (True, False), (True, True)])
def test_sarsa_update_critic_half_equals_critic_update(sb, flag):
    batch, bs, bc, f = _batch(0), _batch(1), _bc(), jnp.asarray(flag)
    a = _agent(with_v=True, use_success_buffer_q=sb)
    full, fi = a.update(batch, bs, bc, f)
    crit, ci = a.critic_update(batch, bs, f)
    _assert_tree_equal(full.critic_network, crit.critic_network)
    _assert_tree_equal(full.value_network, crit.value_network)
    _assert_tree_equal(full.rng, crit.rng)
    assert set(ci) == {k for k in fi if k.startswith('critic/')}
    for k in ('critic/v_loss', 'critic/v_mean', 'critic/v_max', 'critic/v_min', 'critic/v_target_mean',
              'critic/v_head_std', 'critic/v_grad/norm', 'critic/v_grad/max', 'critic/v_grad/min',
              'critic/next_v_head_std', 'critic/td_loss', 'critic/grad/norm', 'critic/q_mc_corr'):
        assert k in ci and np.isfinite(float(ci[k])), k
    assert 'critic/next_actions_std' not in ci
    assert int(crit.value_network.step) == int(a.value_network.step) + (2 if (sb and flag) else 1)
    assert int(crit.critic_network.step) == int(a.critic_network.step) + (2 if (sb and flag) else 1)


def test_sarsa_critic_path_never_touches_policy_or_prefix():
    a = _agent(cls=_NoPolicy, with_v=True, use_success_buffer_q=True, grad_accum_steps=2)
    new, _ = a.critic_update(_batch(0), _batch(1), jnp.asarray(True))
    assert _differs(new.value_network.params, a.value_network.params)


def test_sarsa_donate_matches_plain():
    batch, bs, bc, f = _batch(0), _batch(1), _bc(), jnp.asarray(True)
    p, pi = _agent(with_v=True, use_success_buffer_q=True).update(batch, bs, bc, f)
    d, di = _agent(with_v=True, use_success_buffer_q=True, donate_agent=True).update(batch, bs, bc, f)
    _assert_tree_equal(p, d, leaves_only=True)
    _assert_info_equal(pi, di)
    pc, _ = _agent(with_v=True, use_success_buffer_q=True).critic_update(batch, bs, f)
    dc, _ = _agent(with_v=True, use_success_buffer_q=True, donate_agent=True).critic_update(batch, bs, f)
    _assert_tree_equal(pc, dc, leaves_only=True)


LR = 0.05


def _ref_sarsa_subsample(agent, batch, critic_rng, tau):
    """Hand reference for v_agg='subsample' with the agent's key plumbing: q_rng, v_rng = split(critic_rng);
    agg_rng = split(q_rng)[1] picks 2 of v_ensemble_size V heads; v_rng picks 2 of num_qs Q heads."""
    q_rng, v_rng = jax.random.split(critic_rng)
    _, agg_rng = jax.random.split(q_rng)
    iv = jax.random.randint(agg_rng, (2,), 0, NUM_VS)
    iq = jax.random.randint(v_rng, (2,), 0, NUM_QS)
    qs, vs = agent.critic_network, agent.value_network
    q01, q99 = agent.consts['action_q01'][:7], agent.consts['action_q99'][:7]
    a = ((batch['actions'] - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0).reshape(B, -1)
    w = batch['valid'][:, -1]
    nv = vs.select('target_value')(batch['next_observations'], is_encoded=True)[iv].min(axis=0)
    y = batch['rewards'][:, -1] + agent.config['discount'] ** HL * batch['masks'][:, -1] * nv
    gq = jax.grad(lambda p: (jnp.square(qs.select('critic')(batch['observations'], a, params=p, is_encoded=True) - y) * w).mean())(qs.params)
    qp = jax.tree.map(lambda p, g: p - LR * g, qs.params, gq)
    qp['modules_target_critic'] = jax.tree.map(lambda p, tp: p * tau + tp * (1 - tau), qp['modules_critic'], qp['modules_target_critic'])
    qs = qs.replace(params=qp)
    vt = qs.select('target_critic')(batch['observations'], a, is_encoded=True)[iq].min(axis=0)
    gv = jax.grad(lambda p: (jnp.square(vs.select('value')(batch['observations'], params=p, is_encoded=True) - vt) * w).mean())(vs.params)
    vp = jax.tree.map(lambda p, g: p - LR * g, vs.params, gv)
    vp['modules_target_value'] = jax.tree.map(lambda p, tp: p * tau + tp * (1 - tau), vp['modules_value'], vp['modules_target_value'])
    return qp, vp


def test_sarsa_subsample_rng_plumbing_matches_hand_reference():
    hits = 0
    for seed in range(4):
        a = _agent(cls=_NoPolicy, with_v=True, v_agg='subsample', tx_q=optax.sgd(LR), tx_v=optax.sgd(LR))
        a = a.replace(rng=jax.random.PRNGKey(seed))
        batch = _batch(seed)
        new, _ = a.critic_update(batch, batch, jnp.asarray(False))
        _, _, critic_rng, _ = jax.random.split(a.rng, 4)
        qp, vp = _ref_sarsa_subsample(a, batch, critic_rng, a.config['tau'])
        _close(new.critic_network.params, qp)
        _close(new.value_network.params, vp)
        # teeth: a mean-reduction run must differ
        m, _ = a.replace(config={**a.config, 'v_agg': 'mean'}).critic_update(batch, batch, jnp.asarray(False))
        hits += _differs(m.value_network.params, new.value_network.params)
    assert hits >= 3


def test_sarsa_value_step_grad_accum_matches_full_batch():
    a1 = _agent(cls=_NoPolicy, with_v=True, use_success_buffer_q=True)
    a2 = _agent(cls=_NoPolicy, with_v=True, use_success_buffer_q=True, grad_accum_steps=2)
    batch, bs, f = _batch(0), _batch(1), jnp.asarray(True)
    n1, i1 = a1.critic_update(batch, bs, f)
    n2, i2 = a2.critic_update(batch, bs, f)
    assert set(i1) == set(i2)
    _close(n1.critic_network.params, n2.critic_network.params, rtol=1e-4, atol=2e-5)
    _close(n1.value_network.params, n2.value_network.params, rtol=1e-4, atol=2e-5)
    for k in ('critic/v_loss', 'critic/v_mean', 'critic/v_target_mean', 'critic/v_max', 'critic/v_min',
              'critic/td_loss', 'critic/v_grad/max', 'critic/v_grad/min'):
        np.testing.assert_allclose(float(i1[k]), float(i2[k]), rtol=1e-4, atol=1e-6, err_msg=k)
    assert _differs(n2.value_network.params['modules_value'], a2.value_network.params['modules_value'])


def test_sarsa_value_loss_per_task_info_by_hand():
    a = _agent(cls=_NoPolicy, with_v=True, n_tasks=2)
    batch = _batch(0, n_tasks=2)
    loss, info = a.value_loss(batch, a.value_network.params, jax.random.PRNGKey(0))
    q01, q99 = a.consts['action_q01'][:7], a.consts['action_q99'][:7]
    act = ((batch['actions'] - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0).reshape(B, -1)
    tgt = a.critic_network.select('target_critic')(batch['observations'], act, is_encoded=True).mean(axis=0)
    v = a.value_network.select('value')(batch['observations'], is_encoded=True)
    np.testing.assert_allclose(float(loss), float((jnp.square(v - tgt) * batch['valid'][:, -1]).mean()), rtol=1e-6)
    tid = np.asarray(batch['task_ids'])
    for k in range(2):
        np.testing.assert_allclose(float(info[f'v_target/task{k}']), float(np.asarray(tgt)[tid == k].mean()), rtol=1e-6)
        np.testing.assert_allclose(float(info[f'v_mean/task{k}']), float(np.asarray(v.mean(0))[tid == k].mean()), rtol=1e-6)
    # grads reach only the online V
    g = jax.grad(lambda p: a.value_loss(batch, p, jax.random.PRNGKey(0))[0])(a.value_network.params)
    assert all(float(jnp.abs(x).max()) == 0.0 for x in jax.tree.leaves(g['modules_target_value']))
    assert sum(float(jnp.abs(x).sum()) for x in jax.tree.leaves(g['modules_value'])) > 0
    # update-level per-task keys
    _, ci = a.critic_update(batch, batch, jnp.asarray(False))
    assert {'critic/v_mean/task0', 'critic/v_target/task1', 'critic/q_mean/task0'} <= set(ci)


def test_sarsa_critic_loss_no_gradient_into_v_or_target():
    a = _agent(cls=_NoPolicy, with_v=True)
    batch = _batch(0)
    g = jax.grad(lambda p: a.critic_loss(batch, p, jax.random.PRNGKey(0))[0])(a.critic_network.params)
    assert all(float(jnp.abs(x).max()) == 0.0 for x in jax.tree.leaves(g['modules_target_critic']))


# --- 3. create() -------------------------------------------------------------------------------------------------

@pytest.fixture
def fake_ckpt(monkeypatch):
    calls = []
    monkeypatch.setattr(P, 'Pi05', lambda **kw: None)
    monkeypatch.setattr(P, 'load_pi05_params', lambda path: calls.append(path) or {})
    monkeypatch.setattr(P, 'split_params', lambda p: ({}, {'w': np.float32(0.3)}))
    ns = {'actions': {'q01': -np.ones(32) * 0.8, 'q99': np.ones(32) * 1.3},
          'state': {'q01': np.zeros(32), 'q99': np.ones(32)}}
    monkeypatch.setattr(P.T, 'load_norm_stats', lambda path, asset_id: calls.append(path) or ns)

    class _Tok:
        def __init__(self, *a):
            calls.append('tok')

        def tokenize(self, prompt):
            return np.zeros(4, np.int32), np.ones(4, bool)

    monkeypatch.setattr(P.T, 'PromptTokenizer', _Tok)
    return calls


HID = 32


def _create_cfg(**kw):
    c = dict(pi05_checkpoint='/nonexistent', pi05_robot='libero', pi05_tokenizer='x', pi05_action_horizon=H_PI, horizon_length=HL,
             pi05_clip_grad_norm=1.0, pi05_lr=1e-3, pi05_weight_decay=0.0,
             q_variance_reduction=False, q_vr_num_samples=3, q_vr_reduction='mean',
             pi05_critic_pooling='all', value_hidden_dims=(16, 16), layer_norm=True, num_qs=NUM_QS,
             q_min=-100.0, q_max=0.0, critic_scheduler='constant', ppo_lr=1e-3, critic_warmup_steps=0,
             critic_decay_steps=1, critic_end_value=0.0, clip_grad_norm=1.0, critic_weight_decay=0.0,
             pi05_critic_arch='mlp', pi05_bronet_hidden_dim=HID, pi05_bronet_depth=2,
             q_agg='mean', discount=0.99, tau=0.1, actor_tau=0.1,
             subsample_bon=True, pi05_bon_sampler='sde', pi05_bon_agg='subsample',
             use_value_fn=False, v_training_mode='td_offpolicy', v_ensemble_size=NUM_VS, v_agg='mean',
             v_pair_with_q=False)
    c.update(kw)
    return c


SARSA = dict(use_value_fn=True, v_training_mode='sarsa')


@pytest.mark.parametrize('kw,match', [
    (dict(use_value_fn=True), 'v_training_mode=sarsa only'),
    (dict(use_value_fn=True, v_training_mode='bc_q_mean'), 'v_training_mode=sarsa only'),
    (dict(SARSA, q_variance_reduction=True), 'q_variance_reduction'),
    (dict(SARSA, v_pair_with_q=True), 'v_pair_with_q'),
    (dict(SARSA, v_agg='max'), 'v_agg=.*mean \\| min \\| subsample'),
    (dict(SARSA, v_ensemble_size=1), 'v_ensemble_size=1.*>= 2 heads'),
])
def test_create_sarsa_rejects_before_loading(fake_ckpt, kw, match):
    with pytest.raises(ValueError, match=match):
        OGPOPi05Agent.create(0, ['task'], _create_cfg(**kw))
    assert fake_ckpt == []


def test_create_ignores_v_keys_when_off(fake_ckpt):
    # use_value_fn=false: MLP-agent V keys (td_offpolicy, v_pair_with_q, q-VR) must not raise on pi05
    a = OGPOPi05Agent.create(0, ['task'], _create_cfg(v_pair_with_q=True, q_variance_reduction=True, v_agg='weird'))
    assert a.value_network is None


def test_create_default_rng_and_critic_unchanged(fake_ckpt):
    d = OGPOPi05Agent.create(0, ['task'], _create_cfg())
    s = OGPOPi05Agent.create(0, ['task'], _create_cfg(**SARSA))
    rng_head, _ = jax.random.split(jax.random.PRNGKey(0))       # 8891663: rng, critic_rng = split(PRNGKey(seed))
    np.testing.assert_array_equal(np.asarray(d.rng), np.asarray(rng_head))
    assert d.value_network is None
    _assert_tree_equal(d.critic_network.params, s.critic_network.params)  # critic init key untouched by V
    assert _differs(d.rng, s.rng)                                          # SARSA consumes one extra split (expected)


@pytest.mark.parametrize('arch', ['mlp', 'bronet', 'bronet_vla'])
def test_create_sarsa_value_architecture_and_optimiser(fake_ckpt, arch, capsys):
    a = OGPOPi05Agent.create(0, ['task'], _create_cfg(pi05_critic_arch=arch, **SARSA))
    out = capsys.readouterr().out
    vn = a.value_network
    assert vn is not None
    pv, pq = vn.params['modules_value'], a.critic_network.params['modules_critic']
    assert set(vn.params) == {'modules_value', 'modules_target_value'}
    # same module tree as the critic; only the head's first-layer input width (no action) and ensemble size differ
    assert jax.tree.structure(pv) == jax.tree.structure(pq)
    act = HL * T.ENV_ACTION_DIM
    k_v, k_q = pv['value_net']['Dense_0']['kernel'], pq['value_net']['Dense_0']['kernel']
    assert k_v.shape[0] == NUM_VS and k_q.shape[0] == NUM_QS
    assert k_q.shape[1] - k_v.shape[1] == act and k_v.shape[2] == k_q.shape[2]
    if arch == 'bronet_vla':
        assert k_v.shape[1] == PREFIX_DIM + STATE_DIM and 'two_tier_encoder' not in pv
    else:
        assert 'two_tier_encoder' in pv
    # target: equal values, distinct buffers (donation)
    for x, y in zip(jax.tree.leaves(pv), jax.tree.leaves(vn.params['modules_target_value'])):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
        assert x.unsafe_buffer_pointer() != y.unsafe_buffer_pointer()
    # separate optimiser instance with the critic's structure; its own state
    assert vn.tx is not a.critic_network.tx
    assert jax.tree.structure(vn.opt_state) != jax.tree.structure(None)
    n_v = sum(x.size for x in jax.tree.leaves(pv))
    assert f"[pi05] SARSA-V value {arch} x {NUM_VS} heads: {n_v / 1e6:.1f}M params (v_agg=mean)" in out
    # value forward is (v_ens, B)
    v = vn.select('value')(jnp.zeros((5, PREFIX_DIM + STATE_DIM)), is_encoded=True)
    assert v.shape == (NUM_VS, 5)


class _CreatedFake(OGPOPi05Agent):
    def _prefix(self, images, task_ids=None, kv_only=False):  # kv_only: 2026-10-26 pi05_drop_empty_camera signature
        raise AssertionError("SARSA-V critic path must not run the VLM prefix")

    def _sample_sde(self, *a, **k):
        raise AssertionError("SARSA-V critic path must not sample the policy")

    def actor_loss(self, batch, bc_batch, grad_params, rng, ppo_batch=None):
        loss = jnp.sum((grad_params['w'] - 1.0) ** 2)
        return loss, {'pg_loss': loss}


@pytest.mark.parametrize('arch', ['bronet', 'bronet_vla'])
@pytest.mark.parametrize('donate', [False, True])
def test_created_sarsa_agent_updates(fake_ckpt, arch, donate):
    real = OGPOPi05Agent.create(0, ['task'], _create_cfg(pi05_critic_arch=arch, use_success_buffer_q=True,
                                                         donate_agent=donate, **SARSA))
    a = _CreatedFake(**_fields(real))
    step_v0, step_q0 = int(a.value_network.step), int(a.critic_network.step)
    v0 = jax.tree.map(np.asarray, a.value_network.params)
    q0 = jax.tree.map(np.asarray, a.critic_network.params)
    batch, bs, f = _batch(0), _batch(1), jnp.asarray(True)
    a, ci = a.critic_update(batch, bs, f)
    a, fi = a.update(batch, bs, {'actions': jnp.ones((B, H_PI, 7))}, f)
    assert all(np.isfinite(np.asarray(v)).all() for v in {**ci, **fi}.values())
    assert _differs(a.value_network.params['modules_value'], v0['modules_value'])
    assert _differs(a.value_network.params['modules_target_value'], v0['modules_target_value'])
    assert _differs(a.critic_network.params['modules_critic'], q0['modules_critic'])
    assert int(a.value_network.step) == step_v0 + 4 and int(a.critic_network.step) == step_q0 + 4  # 2 calls x (RB + SB)
    # several critic updates on a fixed batch reduce both losses
    v_losses, q_losses = [], []
    for _ in range(15):
        a, inf = a.critic_update(batch, batch, jnp.asarray(False))
        v_losses.append(float(inf['critic/v_loss'])), q_losses.append(float(inf['critic/td_loss']))
    assert v_losses[-1] < v_losses[0] and q_losses[-1] < q_losses[0], (v_losses, q_losses)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
