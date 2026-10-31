"""
Independent verifier tests for pi05 group-AWR (docs/changes/2026-10-25-pi05-group-awr/).

Stub agent: OGPOPi05Agent with a fake `_prefix` / `_actor_fn` (as tests/test_pi05_sarsa_v_verify.py) and the REAL
`_sample_sde`, `_log_prob`, `_actor_terms`, `_group_awr_loss`, `_cfm_err`, `_bc_loss_per_sample`, critic, `update`.

  1. PPO path: `_actor_terms` (objective 'ppo' and key absent) and the jitted `update` / `grad_diagnostics` are
     bit-identical to VERBATIM copies of `_actor_terms` / `_bc_loss_per_sample` from HEAD (git show HEAD:...).
  2. BC refactor: `_bc_loss_per_sample` (now via `_cfm_err`) bit-identical to the verbatim HEAD copy (libero, droid,
     multi-task), values and gradients.
  3. group_awr math: `_group_awr_loss` vs a hand-rolled mean(min(exp(A/β), ω_max) · CFM) with an independent CFM
     (explicit (t, x0) draws from split(rng, awr_n_mc), block-sliced executed dims); stop-gradients; dim_mask
     gradient; `_actor_terms` routing / ordering; update, grad accumulation, donation, grad_diagnostics.
  4. create() validation for group_awr (raises before the checkpoint load).
  5. The new launch script: flags exist, diff vs its twin, resolved config passes create() / main_pi05 validation.
  6. Sharded (8 fake CPU devices, subprocess): group_awr update / grad_diagnostics with a mesh == single device.

Run:  JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_pi05_group_awr.py -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import copy
import difflib
import re
import subprocess
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P  # used by the verbatim HEAD _actor_terms

from ogpo.agents.modules.flax_utils import ModuleDict, TrainState
from ogpo.agents.modules.pg_helper import compute_ogpo_advantages, compute_ppo_loss
from ogpo.agents.ogpo_pi05 import OGPOPi05Agent, PREFIX_DIM, STATE_DIM, T, _per_task, _tile
from ogpo.networks import Value
from ogpo.utils.config_loader import AttrDict, config_to_agent_dict, load_config

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
B, NUM_QS, H_PI, HL, G = 4, 4, 10, 5, 3
OBS = PREFIX_DIM + STATE_DIM
FLAT = H_PI * T.MODEL_ACTION_DIM
L, S, KV, HD = 2, 3, 1, 2
N_MC = 3


class _Fake(OGPOPi05Agent):
    """Fake VLM prefix / velocity (scalar gain `w`, per-flat-dim bias `v`); everything else real."""

    def _prefix(self, images, task_ids=None, kv_only=False):  # kv_only: 2026-10-26 pi05_drop_empty_camera signature
        b_idx = images[:, 0, 0, 0, 0].astype(jnp.float32)
        n = images.shape[0]
        k = jnp.broadcast_to(b_idx[None, :, None, None, None], (L, n, S, KV, HD))
        return None, (k, k + 100.0), jnp.ones((n, S), bool)

    def _actor_fn(self, params, kv_cache, prefix_mask):
        k, _ = kv_cache
        b = k[0, :, 0, 0, 0]

        def actor_fn(obs, x, t, params_=None, is_encoded=True):
            return params['w'] * x + params['v'][None, :] + 0.5 * b[:, None] + 2.0 * t

        return actor_fn


def _velocity(params, x, t, b_rows):
    """The stub's velocity, written out independently for the hand references."""
    return params['w'] * x + params['v'][None, :] + 0.5 * b_rows[:, None] + 2.0 * t


def _config(objective='ppo', **kw):
    c = config_to_agent_dict(load_config('ogpo', {}))
    c.update(pi05_action_horizon=H_PI, horizon_length=HL, num_qs=NUM_QS, batch_size=B, ppo_batch_size=B,
             group_num_samples=G, flow_steps=4, tau=0.3, actor_tau=0.2, donate_agent=False, grad_accum_steps=1,
             use_success_buffer_q=False, q_min=-100.0, q_max=0.0, adv_clip_min=None, use_bc_regularization=True,
             use_tapered_noise=True, use_constant_noise=False, constant_noise_std=0.1, q_variance_reduction=False,
             awr_n_mc=N_MC, awr_beta=0.05, awr_weight_max=20.0, entropy_coeff=0.0, bc_coeff=0.7)
    if objective is None:
        c.pop('pi05_actor_objective')
    else:
        c['pi05_actor_objective'] = objective
    c.update(kw)
    return c


def _critic(seed=0):
    vdef = Value(hidden_dims=(16, 16), layer_norm=True, num_ensembles=NUM_QS, critic_loss_type='mse',
                 q_min=-100.0, q_max=0.0, obs_two_tier=True, two_tier_img_dim=PREFIX_DIM,
                 two_tier_proprio_dim=STATE_DIM, two_tier_fused_dim=0)
    net = ModuleDict({'critic': vdef, 'target_critic': copy.deepcopy(vdef)})
    ex = dict(observations=jnp.zeros((1, OBS)), actions=jnp.zeros((1, HL * T.ENV_ACTION_DIM)))
    k = jax.random.split(jax.random.PRNGKey(seed), 2)
    params = net.init(k[0], critic=ex, target_critic=ex)['params']
    params['modules_target_critic'] = net.init(k[1], critic=ex, target_critic=ex)['params']['modules_critic']
    return TrainState.create(net, params, optax.adam(1e-2))


def _dim_mask():
    m = np.zeros((H_PI, T.MODEL_ACTION_DIM), np.float32)
    m[:HL, :T.ENV_ACTION_DIM] = 1.0
    return m


def _agent(cls=_Fake, objective='ppo', n_tasks=1, robot='libero', **kw):
    consts = {'tokens': jnp.zeros((n_tasks, 4), jnp.int32), 'dim_mask': jnp.asarray(_dim_mask().reshape(-1)),
              'action_q01': -jnp.ones(T.MODEL_ACTION_DIM) * 0.8, 'action_q99': jnp.ones(T.MODEL_ACTION_DIM) * 1.3,
              'state_q01': -jnp.ones(STATE_DIM) * 0.5, 'state_q99': jnp.ones(STATE_DIM) * 2.0}
    v = jnp.asarray(np.random.RandomState(7).randn(FLAT) * 0.05, jnp.float32)
    params = {'w': jnp.float32(0.3), 'v': v}
    tx = optax.adam(1e-2)
    return cls(rng=jax.random.PRNGKey(3), actor_params=params,
               target_actor_params={'w': jnp.float32(0.7), 'v': v * 0.5},
               actor_opt_state=tx.init(params), init_actor_params=jax.tree.map(jnp.copy, params), frozen_params={},
               critic_network=_critic(), consts=consts, config=_config(objective, **kw), model=None, actor_tx=tx,
               robot=robot)


def _batch(seed=0, n_tasks=1):
    r = np.random.RandomState(seed)
    img = np.zeros((B, 2, 2, 2, 3), np.float32)
    img[:, 0] = (np.arange(B) + 10 * seed)[:, None, None, None]
    valid = np.ones((B, HL), np.float32)
    valid[0, -1] = 0.0
    b = {
        'images': jnp.asarray(img), 'next_images': jnp.asarray(img + 1.0),
        'observations': jnp.asarray(r.randn(B, OBS), jnp.float32),
        'next_observations': jnp.asarray(r.randn(B, OBS), jnp.float32),
        'actions': jnp.asarray(r.uniform(-1, 1, (B, HL, T.ENV_ACTION_DIM)), jnp.float32),
        'rewards': jnp.asarray(-r.rand(B, HL), jnp.float32), 'masks': jnp.ones((B, HL), jnp.float32),
        'valid': jnp.asarray(valid), 'mc_returns': jnp.asarray(-r.rand(B) * 10, jnp.float32),
        'is_success': jnp.asarray((r.rand(B, HL) > 0.5).astype(np.float32)),
    }
    if n_tasks > 1:
        b['task_ids'] = jnp.asarray(np.arange(B) % n_tasks, jnp.int32)
    return b


def _bc(seed=0, n_tasks=1, act_dim=7):
    r = np.random.RandomState(50 + seed)
    img = np.zeros((B, 2, 2, 2, 3), np.float32)
    img[:, 0] = np.arange(B)[:, None, None, None]
    valid = np.ones((B, H_PI), np.float32)
    valid[1, 6:] = 0.0
    b = {'images': jnp.asarray(img), 'observations': jnp.asarray(r.randn(B, OBS) * 0.3, jnp.float32),
         'actions': jnp.asarray(r.uniform(-1, 1, (B, H_PI, act_dim)), jnp.float32), 'valid': jnp.asarray(valid)}
    if n_tasks > 1:
        b['task_ids'] = jnp.asarray((np.arange(B) + 1) % n_tasks, jnp.int32)
    return b


def _with(agent, **kw):
    """agent with config keys overridden (keeps the AttrDict: compute_ogpo_advantages reads config.adv_strategy)."""
    return agent.replace(config=AttrDict({**agent.config, **kw}))


def _fields(agent):
    return {f: getattr(agent, f) for f in agent.__dataclass_fields__}


def _tree_equal(a, b, leaves_only=False):
    if not leaves_only:
        assert jax.tree.structure(a) == jax.tree.structure(b)
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    assert len(la) == len(lb)
    for x, y in zip(la, lb):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def _info_equal(a, b):
    assert set(a) == set(b), sorted(set(a) ^ set(b))
    for k in a:
        np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]), err_msg=k)


def _close(a, b, rtol=1e-5, atol=1e-6):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=rtol, atol=atol)


# --- VERBATIM ogpo/agents/ogpo_pi05.py @ HEAD (pre-change) ---------------------------------------------------------

def _head_bc_loss_per_sample(self, bc_batch, grad_params, rng):
    """(B,) masked flow-matching error per window; `bc_loss` is its mean, per-task diagnostics use it directly."""
    _, kv, pmask = self._prefix(bc_batch['images'], bc_batch.get('task_ids'))
    # observations = the window-start critic features (droid delta reference); normalise, then zero-pad (openpi order)
    env_actions = self._normalize_env_actions(bc_batch['actions'], bc_batch.get('observations'))
    a = jnp.pad(env_actions, ((0, 0), (0, 0), (0, T.MODEL_ACTION_DIM - self.env_action_dim)))
    b = a.shape[0]
    a = a.reshape(b, -1)
    x0_rng, t_rng = jax.random.split(rng)
    x0 = jax.random.normal(x0_rng, a.shape)
    t = jax.random.uniform(t_rng, (b, 1))
    x_t = (1 - t) * x0 + t * a
    pred = self._actor_fn(grad_params, kv, pmask)(None, x_t, t)
    err = jnp.square(pred - (a - x0)).reshape(b, *self._chunk_shape)
    return (err * bc_batch['valid'][..., None]).mean(axis=(1, 2))


def _head_actor_terms(self, batch, bc_batch, grad_params, rng, ppo_batch=None):
    """(pg_loss, entropy, bc_loss, info); split out so `grad_diagnostics` can differentiate each.

    PPO states come from `ppo_batch` when given (the runner's task-stratified batch), else from a random
    ppo_batch_size subsample of the TD batch (original behaviour).

    AWR (`pi05_actor_objective: awr`) returns before any of the PPO machinery runs: see `_awr_terms`.
    """
    if self.config.get('pi05_actor_objective', 'ppo') == 'awr':
        return self._awr_terms(bc_batch, grad_params, rng)
    c = self.config
    G = c['group_num_samples']
    if ppo_batch is not None:
        images, c_obs, task_ids = ppo_batch['images'], ppo_batch['observations'], ppo_batch.get('task_ids')
    else:
        images = batch['images']
        c_obs = batch['observations']
        task_ids = batch.get('task_ids')
        if c['ppo_batch_size'] < c['batch_size']:
            rng, key = jax.random.split(rng)
            n_ppo = c['ppo_batch_size'] // self._accum_steps  # per micro-batch under grad accumulation
            idxs = jax.random.choice(key, images.shape[0], (n_ppo,), replace=False)
            images, c_obs = images[idxs], c_obs[idxs]
            task_ids = None if task_ids is None else task_ids[idxs]
            if self.mesh is not None:
                # The gather returns the subsample replicated: without this every device would run all
                # n_ppo states x G chains. n_ppo divides over the devices (main_pi05 checks ppo_batch_size).
                shard = NamedSharding(self.mesh, P('data'))
                images, c_obs = (jax.lax.with_sharding_constraint(x, shard) for x in (images, c_obs))
                task_ids = None if task_ids is None else jax.lax.with_sharding_constraint(task_ids, shard)
    B = images.shape[0]

    _, kv, pmask = self._prefix(images, task_ids)
    kv_g, pmask_g = _tile(kv, G, axis=1), _tile(pmask, G, axis=0)  # index g * B + b

    rng, sample_rng, adv_rng, bc_rng = jax.random.split(rng, 4)
    actions, chains, old_lp, sigmas = self._sample_sde(self.target_actor_params, kv_g, pmask_g, sample_rng)
    actions_g = self._critic_actions(actions).reshape(G, B, -1)

    q_agg, q_full = jax.vmap(lambda a_g, r_g: self._target_q(c_obs, a_g, r_g))(
        actions_g, jax.random.split(adv_rng, G))
    adv, adv_pre = compute_ogpo_advantages(q_agg, q_full, c)

    lp, entropy_rate, lp_info = self._log_prob(grad_params, kv_g, pmask_g, chains, get_entropy=True)
    pg_loss, pg_stats = compute_ppo_loss(lp, old_lp, adv.reshape(-1), c)
    ent = jnp.mean(entropy_rate)

    K = self.consts['tokens'].shape[0]
    adv_g = adv.reshape(G, B)  # per-state stats over the group
    per_task = _per_task({
        'adv_abs': jnp.abs(adv_g).mean(axis=0), 'adv_zero_frac': (adv_g == 0).mean(axis=0),
        'q_values': adv_pre.mean(axis=0), 'q_group_std': jnp.std(adv_pre, axis=0),
    }, task_ids, K)
    if c['use_bc_regularization']:
        bc_per_sample = self._bc_loss_per_sample(bc_batch, grad_params, bc_rng)
        bc_loss = bc_per_sample.mean()
        per_task.update(_per_task({'bc_loss': bc_per_sample}, bc_batch.get('task_ids'), K))
    else:
        bc_loss = 0.0
    info = {
        **per_task,
        'pg_loss': pg_loss, 'bc_loss': bc_loss, 'entropy': ent,
        'lp': lp.mean(), 'lp_min': lp.min(), 'lp_max': lp.max(), 'old_lp': old_lp.mean(),
        'adv': adv.mean(), 'adv_min': adv.min(), 'adv_max': adv.max(), 'adv_std': adv.std(),
        'adv_zero_frac': jnp.mean(adv == 0),  # conservative: heads disagreed on the sign
        'q_values': adv_pre.mean(), 'sigmas': sigmas.mean(),
        'actions': actions_g.mean(), 'actions_std': actions_g.std(),
        'actions_clipped_frac': jnp.mean(jnp.abs(actions_g) >= c['act_max']),
        'grpo_adv_group_std': jnp.std(adv_pre, axis=0).mean(),
        'grpo_adv_between_group_std': jnp.std(adv_pre.mean(axis=0)),
        **pg_stats, **{f'info_{k}': v for k, v in lp_info.items()},
    }
    return pg_loss, ent, bc_loss, info


class _Head(_Fake):
    _actor_terms = _head_actor_terms
    _bc_loss_per_sample = _head_bc_loss_per_sample


def test_verbatim_copies_match_git_head():
    """The two reference functions above are literally HEAD's method bodies (guards against a stale copy)."""
    import inspect
    head = subprocess.run(['git', '--no-optional-locks', 'show', 'HEAD:ogpo/agents/ogpo_pi05.py'], cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout
    for fn, name in ((_head_bc_loss_per_sample, '_bc_loss_per_sample'), (_head_actor_terms, '_actor_terms')):
        mine = textwrap.dedent(inspect.getsource(fn)).replace(f'def _head{name}(', f'def {name}(').rstrip()
        m = re.search(rf'\n(    def {name}\(.*?)(?=\n    def |\n    @)', head, re.S)
        assert m, name
        theirs = textwrap.dedent(m.group(1)).rstrip()
        assert mine == theirs, '\n'.join(difflib.unified_diff(theirs.splitlines(), mine.splitlines(), lineterm=''))


# =====================================================================================================================
# 1. PPO path bit-identical to HEAD
# =====================================================================================================================

PPO_GRID = [(obj, mode, nt, bcr) for obj in (None, 'ppo') for mode in ('subsample', 'full', 'ppo_batch')
            for nt in (1, 2) for bcr in (True, False)]


def _ppo_inputs(mode, n_tasks):
    kw = dict(ppo_batch_size=2) if mode == 'subsample' else {}
    batch, bc = _batch(0, n_tasks), _bc(0, n_tasks)
    ppo = None
    if mode == 'ppo_batch':
        ppo = {'images': batch['images'][1:3], 'observations': batch['observations'][1:3]}
        if n_tasks > 1:
            ppo['task_ids'] = batch['task_ids'][1:3]
    return kw, batch, bc, ppo


@pytest.mark.parametrize('obj,mode,n_tasks,bc_reg', PPO_GRID)
def test_ppo_actor_terms_bit_identical_to_head(obj, mode, n_tasks, bc_reg):
    kw, batch, bc, ppo = _ppo_inputs(mode, n_tasks)
    new = _agent(objective=obj, n_tasks=n_tasks, use_bc_regularization=bc_reg, **kw)
    if obj is None:
        assert 'pi05_actor_objective' not in new.config
    head = _Head(**_fields(new))
    rng = jax.random.PRNGKey(11)
    pg1, e1, b1, i1 = new._actor_terms(batch, bc, new.actor_params, rng, ppo)
    pg0, e0, b0, i0 = head._actor_terms(batch, bc, head.actor_params, rng, ppo)
    for x, y in ((pg1, pg0), (e1, e0), (b1, b0)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    _info_equal(i1, i0)
    assert float(e1) != 0.0  # PPO path still computes the entropy
    # jitted loss + grads
    f = lambda a: jax.value_and_grad(lambda p: a.actor_loss(batch, bc, p, rng, ppo), has_aux=True)(a.actor_params)
    (l1, j1), g1 = jax.jit(f)(new)
    (l0, j0), g0 = jax.jit(f)(head)
    np.testing.assert_array_equal(np.asarray(l1), np.asarray(l0))
    _tree_equal(g1, g0)
    _info_equal(j1, j0)


@pytest.mark.parametrize('obj,K,mode,n_tasks', [(None, 1, 'subsample', 1), ('ppo', 1, 'subsample', 1),
                                                ('ppo', 2, 'subsample', 1), (None, 2, 'full', 2),
                                                ('ppo', 2, 'ppo_batch', 2)])
def test_ppo_update_bit_identical_to_head(obj, K, mode, n_tasks):
    kw, batch, bc, ppo = _ppo_inputs(mode, n_tasks)
    if mode == 'subsample' and K == 2:
        kw = dict(ppo_batch_size=2)
    if mode == 'ppo_batch' and K == 2:
        ppo = {k: v[:2] for k, v in {'images': batch['images'], 'observations': batch['observations'],
                                      'task_ids': batch['task_ids']}.items()}
    new = _agent(objective=obj, n_tasks=n_tasks, grad_accum_steps=K, use_success_buffer_q=True, **kw)
    head = _Head(**_fields(new))
    bs, flag = _batch(1, n_tasks), jnp.asarray(True)
    a1, i1 = new.update(batch, bs, bc, flag, ppo)
    a0, i0 = head.update(batch, bs, bc, flag, ppo)
    _tree_equal(a1, _Fake(**_fields(a0)))
    _info_equal(i1, i0)
    a1b, i1b = a1.update(batch, bs, bc, flag, ppo)
    a0b, i0b = _Head(**_fields(a0)).update(batch, bs, bc, flag, ppo)
    _tree_equal(a1b, _Fake(**_fields(a0b)))
    _info_equal(i1b, i0b)


@pytest.mark.parametrize('K', [1, 2])
def test_ppo_grad_diagnostics_bit_identical_to_head(K):
    new = _agent(objective='ppo', grad_accum_steps=K, ppo_batch_size=2)
    head = _Head(**_fields(new))
    batch, bc = _batch(0), _bc(0)
    _info_equal(new.grad_diagnostics(batch, bc), head.grad_diagnostics(batch, bc))


# =====================================================================================================================
# 2. `_bc_loss_per_sample` refactor bit-identical to HEAD
# =====================================================================================================================

@pytest.mark.parametrize('robot,n_tasks', [('libero', 1), ('libero', 2), ('droid', 1)])
def test_bc_loss_per_sample_bit_identical_to_head(robot, n_tasks):
    a = _agent(n_tasks=n_tasks, robot=robot)
    bc = _bc(3, n_tasks, act_dim=a.env_action_dim)
    for seed in range(3):
        rng = jax.random.PRNGKey(seed)
        np.testing.assert_array_equal(np.asarray(a._bc_loss_per_sample(bc, a.actor_params, rng)),
                                      np.asarray(_head_bc_loss_per_sample(a, bc, a.actor_params, rng)))
        np.testing.assert_array_equal(np.asarray(a.bc_loss(bc, a.actor_params, rng)),
                                      np.asarray(_head_bc_loss_per_sample(a, bc, a.actor_params, rng).mean()))
        g1 = jax.jit(jax.grad(lambda p: a._bc_loss_per_sample(bc, p, rng).mean()))(a.actor_params)
        g0 = jax.jit(jax.grad(lambda p: _head_bc_loss_per_sample(a, bc, p, rng).mean()))(a.actor_params)
        _tree_equal(g1, g0)
        l1 = jax.jit(lambda p: a._bc_loss_per_sample(bc, p, rng))(a.actor_params)
        l0 = jax.jit(lambda p: _head_bc_loss_per_sample(a, bc, p, rng))(a.actor_params)
        np.testing.assert_array_equal(np.asarray(l1), np.asarray(l0))


def test_cfm_err_shape_and_definition():
    a = _agent()
    _, kv, pm = a._prefix(_batch(0)['images'])
    x = jax.random.uniform(jax.random.PRNGKey(1), (B, FLAT), minval=-1, maxval=1)
    rng = jax.random.PRNGKey(2)
    err = a._cfm_err(x, kv, pm, a.actor_params, rng)
    assert err.shape == (B, H_PI, T.MODEL_ACTION_DIM) and err.dtype == jnp.float32
    k0, k1 = jax.random.split(rng)
    x0 = jax.random.normal(k0, x.shape)
    t = jax.random.uniform(k1, (B, 1))
    pred = _velocity(a.actor_params, (1 - t) * x0 + t * x, t, jnp.arange(B, dtype=jnp.float32))
    np.testing.assert_allclose(np.asarray(err), np.asarray(jnp.square(pred - (x - x0)).reshape(B, H_PI, -1)),
                               rtol=1e-6, atol=1e-7)


# =====================================================================================================================
# 3. group_awr math
# =====================================================================================================================

BETA, WMAX = 0.05, 20.0


def _ref_group_awr(params, b_rows, actions, adv, rng, n_mc, beta=BETA, wmax=WMAX):
    """mean_n[min(exp(A_n/β), ω_max) · CFM_n]; CFM_n = mean over n_mc draws (keys = split(rng, n_mc)) of the squared
    velocity error over the executed block [:HL, :7] (= dim_mask). w is a constant (built from numpy adv)."""
    n = actions.shape[0]
    w = jnp.asarray(np.minimum(np.exp(np.asarray(adv, np.float64) / beta), wmax), jnp.float32)
    cfm = jnp.zeros(n)
    for key in jax.random.split(rng, n_mc):
        k0, k1 = jax.random.split(key)
        x0 = jax.random.normal(k0, actions.shape)
        t = jax.random.uniform(k1, (n, 1))
        pred = _velocity(params, (1 - t) * x0 + t * actions, t, b_rows)
        err = jnp.square(pred - (actions - x0)).reshape(n, H_PI, T.MODEL_ACTION_DIM)
        cfm = cfm + err[:, :HL, :T.ENV_ACTION_DIM].mean(axis=(1, 2))
    cfm = cfm / n_mc
    return jnp.mean(w * cfm), w, cfm


def _awr_inputs(seed=0):
    a_imgs = _batch(seed)['images']
    n = G * B
    acts = jax.random.uniform(jax.random.PRNGKey(100 + seed), (n, FLAT), minval=-1, maxval=1)
    adv = jnp.asarray(np.linspace(-0.4, 0.4, n)[np.random.RandomState(seed).permutation(n)], jnp.float32)
    b_rows = jnp.asarray(np.tile(np.asarray(a_imgs[:, 0, 0, 0, 0]), G), jnp.float32)
    return a_imgs, acts, adv, b_rows


@pytest.mark.parametrize('n_mc', [1, 3])
def test_group_awr_loss_matches_hand_rolled(n_mc):
    a = _agent(objective='group_awr', awr_n_mc=n_mc)
    imgs, acts, adv, b_rows = _awr_inputs()
    _, kv, pm = a._prefix(imgs)
    kv_g, pm_g = _tile(kv, G, axis=1), _tile(pm, G, axis=0)
    rng = jax.random.PRNGKey(21)
    loss, info = a._group_awr_loss(a.actor_params, kv_g, pm_g, acts, adv, rng)
    ref, w, cfm = _ref_group_awr(a.actor_params, b_rows, acts, adv, rng, n_mc)
    np.testing.assert_allclose(float(loss), float(ref), rtol=1e-5)
    np.testing.assert_allclose(float(info['awr_loss']), float(ref), rtol=1e-5)
    np.testing.assert_allclose(float(info['cfm_loss']), float(cfm.mean()), rtol=1e-5)
    np.testing.assert_allclose(float(info['awr_w_mean']), float(w.mean()), rtol=1e-5)
    np.testing.assert_allclose(float(info['awr_w_min']), float(w.min()), rtol=1e-5)
    assert float(info['awr_w_max']) == WMAX
    raw = np.exp(np.asarray(adv, np.float64) / BETA)
    assert 0 < (raw >= WMAX).mean() < 1  # the input exercises the clip
    np.testing.assert_allclose(float(info['awr_w_clip_frac']), (raw >= WMAX).mean(), rtol=1e-6)
    wn = np.asarray(w, np.float64)
    np.testing.assert_allclose(float(info['awr_ess']), wn.sum() ** 2 / (len(wn) * (wn ** 2).sum()), rtol=1e-5)
    # teeth: no batch normalisation of the weights; clip after exp, not on A/β at ω_max
    assert abs(float(jnp.mean(w / w.mean() * cfm)) - float(loss)) > 1e-3 * abs(float(loss))
    assert abs(float(jnp.mean(jnp.exp(jnp.minimum(adv / BETA, WMAX)) * cfm)) - float(loss)) > 1e-3 * abs(float(loss))
    # n_mc averaging: a different n_mc gives a different estimate (keys are split(rng, n_mc))
    other, _ = _with(a, awr_n_mc=n_mc + 1)._group_awr_loss(
        a.actor_params, kv_g, pm_g, acts, adv, rng)
    assert float(other) != float(loss)


def test_group_awr_loss_jit_and_dtype():
    a = _agent(objective='group_awr')
    imgs, acts, adv, b_rows = _awr_inputs(1)
    _, kv, pm = a._prefix(imgs)
    kv_g, pm_g = _tile(kv, G, axis=1), _tile(pm, G, axis=0)
    rng = jax.random.PRNGKey(5)
    l_eager, _ = a._group_awr_loss(a.actor_params, kv_g, pm_g, acts, adv, rng)
    l_jit, info = jax.jit(lambda p: a._group_awr_loss(p, kv_g, pm_g, acts, adv, rng))(a.actor_params)
    np.testing.assert_allclose(float(l_eager), float(l_jit), rtol=1e-6)
    assert l_jit.dtype == jnp.float32 and all(v.dtype == jnp.float32 for v in info.values())


def test_group_awr_gradients():
    a = _agent(objective='group_awr')
    imgs, acts, adv, b_rows = _awr_inputs(2)
    _, kv, pm = a._prefix(imgs)
    kv_g, pm_g = _tile(kv, G, axis=1), _tile(pm, G, axis=0)
    rng = jax.random.PRNGKey(8)
    f = lambda p, ac, ad: a._group_awr_loss(p, kv_g, pm_g, ac, ad, rng)[0]
    g = jax.grad(f)(a.actor_params, acts, adv)
    g_ref = jax.grad(lambda p: _ref_group_awr(p, b_rows, acts, adv, rng, N_MC)[0])(a.actor_params)
    _close(g, g_ref, rtol=1e-4, atol=1e-7)
    # weights carry no gradient; no gradient into the sampled actions
    g_adv = jax.grad(f, argnums=2)(a.actor_params, acts, adv)
    g_act = jax.grad(f, argnums=1)(a.actor_params, acts, adv)
    assert float(jnp.abs(g_adv).max()) == 0.0 and float(jnp.abs(g_act).max()) == 0.0
    # padded / unexecuted dims: exactly zero gradient through the per-dim bias; executed dims all nonzero
    gv = np.asarray(g['v']).reshape(H_PI, T.MODEL_ACTION_DIM)
    m = _dim_mask().astype(bool)
    assert np.all(gv[~m] == 0.0)
    assert np.all(gv[m] != 0.0)
    # teeth: without stop_gradient on w the adv gradient would be nonzero
    f_leak = lambda ad: jnp.mean(jnp.minimum(jnp.exp(ad / BETA), WMAX) * _ref_group_awr(
        a.actor_params, b_rows, acts, adv, rng, N_MC)[2])
    assert float(jnp.abs(jax.grad(f_leak)(adv)).max()) > 0.0


def _hand_actor_terms_inputs(a, batch, rng):
    """Replicates `_actor_terms`' sample / advantage plumbing (no subsample: ppo_batch_size == batch_size)."""
    c = a.config
    _, kv, pm = a._prefix(batch['images'])
    kv_g, pm_g = _tile(kv, G, axis=1), _tile(pm, G, axis=0)
    rest, s_rng, adv_rng, bc_rng = jax.random.split(rng, 4)
    acts, _, _, _ = a._sample_sde(a.target_actor_params, kv_g, pm_g, s_rng)
    acts_g = a._critic_actions(acts).reshape(G, B, -1)
    q_agg, q_full = jax.vmap(lambda ag, r: a._target_q(batch['observations'], ag, r))(acts_g, jax.random.split(adv_rng, G))
    adv, _ = compute_ogpo_advantages(q_agg, q_full, c)
    return acts, adv, rest, bc_rng, kv_g, pm_g


@pytest.mark.parametrize('adv_strategy,norm', [('vanilla', False), ('vanilla', True), ('conservative', False),
                                               ('penultimate', False)])
def test_actor_terms_group_awr_routes_samples_and_advantages(adv_strategy, norm):
    a = _agent(objective='group_awr', adv_strategy=adv_strategy, normalize_group=norm, entropy_coeff=0.0)
    batch, bc = _batch(0), _bc(0)
    rng = jax.random.PRNGKey(31)
    pg, ent, bcl, info = a._actor_terms(batch, bc, a.actor_params, rng)
    acts, adv, rest, bc_rng, kv_g, pm_g = _hand_actor_terms_inputs(a, batch, rng)
    b_rows = jnp.asarray(np.tile(np.asarray(batch['images'][:, 0, 0, 0, 0]), G), jnp.float32)
    ref, _, _ = _ref_group_awr(a.actor_params, b_rows, acts, adv.reshape(-1), rest, N_MC)
    np.testing.assert_allclose(float(pg), float(ref), rtol=1e-5)
    assert float(ent) == 0.0 and ent.dtype == jnp.float32
    # teeth: (b * G + g) ordering of the advantages, or samples from the online actor, give a different loss
    perm, _, _ = _ref_group_awr(a.actor_params, b_rows, acts, adv.T.reshape(-1), rest, N_MC)
    if float(jnp.abs(adv).max()) > 0:
        assert abs(float(perm) - float(ref)) > 1e-6 * abs(float(ref))
    on, _, _, _ = a._sample_sde(a.actor_params, kv_g, pm_g, jax.random.split(rng, 4)[1])
    assert not np.allclose(np.asarray(on), np.asarray(acts))
    # BC anchor uses bc_rng exactly as the PPO path; shared info keys bit-identical to the PPO agent's
    np.testing.assert_array_equal(np.asarray(bcl),
                                  np.asarray(a._bc_loss_per_sample(bc, a.actor_params, bc_rng).mean()))
    p = _with(a, pi05_actor_objective='ppo')
    _, ent_p, bcl_p, info_p = p._actor_terms(batch, bc, p.actor_params, rng)
    np.testing.assert_array_equal(np.asarray(bcl), np.asarray(bcl_p))
    awr_keys = {'awr_loss', 'cfm_loss', 'awr_w_mean', 'awr_w_max', 'awr_w_min', 'awr_w_clip_frac', 'awr_ess'}
    assert awr_keys <= set(info) and not (awr_keys & set(info_p))
    ppo_only = set(info_p) - set(info)
    assert {'lp', 'lp_min', 'lp_max', 'old_lp'} <= ppo_only
    assert all(k in {'lp', 'lp_min', 'lp_max', 'old_lp'} or k.startswith('info_') or k in
               compute_ppo_loss(jnp.zeros(2), jnp.zeros(2), jnp.zeros(2), a.config)[1] for k in ppo_only), ppo_only
    for k in set(info) & set(info_p) - {'pg_loss', 'entropy'}:
        np.testing.assert_array_equal(np.asarray(info[k]), np.asarray(info_p[k]), err_msg=k)
    assert float(info['entropy']) == 0.0 and float(info['pg_loss']) == float(pg)


def test_actor_terms_group_awr_subsample_and_multitask():
    """ppo_batch_size < batch_size and multi-task task_ids: runs, per-task keys present, rng plumbing as PPO's."""
    a = _agent(objective='group_awr', n_tasks=2, ppo_batch_size=2)
    batch, bc = _batch(0, 2), _bc(0, 2)
    rng = jax.random.PRNGKey(4)
    pg, ent, bcl, info = a._actor_terms(batch, bc, a.actor_params, rng)
    p = _with(a, pi05_actor_objective='ppo')
    _, _, bcl_p, info_p = p._actor_terms(batch, bc, p.actor_params, rng)
    np.testing.assert_array_equal(np.asarray(bcl), np.asarray(bcl_p))
    for k in ('adv', 'adv_std', 'q_values', 'actions', 'sigmas', 'adv_abs/task0', 'bc_loss/task1'):
        np.testing.assert_array_equal(np.asarray(info[k]), np.asarray(info_p[k]), err_msg=k)
    assert np.isfinite(float(pg)) and float(ent) == 0.0


@pytest.mark.parametrize('K', [1, 2])
def test_group_awr_update(K):
    a = _agent(objective='group_awr', grad_accum_steps=K, ppo_batch_size=2, use_success_buffer_q=True)
    batch, bs, bc, flag = _batch(0), _batch(1), _bc(0), jnp.asarray(True)
    new, info = a.update(batch, bs, bc, flag)
    assert all(np.isfinite(np.asarray(v)).all() for v in info.values()), info
    for k in ('actor/awr_loss', 'actor/cfm_loss', 'actor/awr_w_mean', 'actor/awr_ess', 'actor/awr_w_clip_frac',
              'actor/adv_std', 'actor/bc_loss', 'actor/grad_norm', 'critic/td_loss'):
        assert k in info, k
    assert 'actor/lp' not in info and 'actor/approx_kl' not in info and 'actor/ratio_clipped_total' not in info
    assert float(info['actor/entropy']) == 0.0
    assert not np.array_equal(np.asarray(new.actor_params['v']), np.asarray(a.actor_params['v']))
    # the critic half is the PPO agent's (objective-independent), bit-for-bit
    p_new, p_info = _with(a, pi05_actor_objective='ppo').update(batch, bs, bc, flag)
    _tree_equal(new.critic_network, p_new.critic_network)
    _tree_equal(new.rng, p_new.rng)
    _info_equal({k: v for k, v in info.items() if k.startswith('critic/')},
                {k: v for k, v in p_info.items() if k.startswith('critic/')})
    if K == 1:
        actor_rng = jax.random.split(a.rng, 4)[1]
        (loss, _), g = jax.value_and_grad(lambda p: a.actor_loss(batch, bc, p, actor_rng), has_aux=True)(a.actor_params)
        pg, ent, bcl, _ = a._actor_terms(batch, bc, a.actor_params, actor_rng)
        np.testing.assert_allclose(float(loss), float(pg + a.config['bc_coeff'] * bcl), rtol=1e-6)
        np.testing.assert_allclose(float(info['actor/grad_norm']), float(optax.global_norm(g)), rtol=1e-5)
        np.testing.assert_allclose(float(info['actor/pg_loss']), float(pg), rtol=1e-5)
    # donation path == plain path
    d_new, d_info = _agent(objective='group_awr', grad_accum_steps=K, ppo_batch_size=2, use_success_buffer_q=True,
                           donate_agent=True).update(batch, bs, bc, flag)
    _tree_equal(d_new, new, leaves_only=True)
    _info_equal(d_info, info)


def test_group_awr_grad_accum_equals_mean_of_micro_batch_grads():
    """K=2: the accumulated actor gradient = mean of the per-micro-batch grads of actor_loss (rows k, k+K, ...)."""
    a = _agent(objective='group_awr', grad_accum_steps=2, ppo_batch_size=2)
    batch, bc = _batch(0), _bc(0)
    rng = jax.random.PRNGKey(17)
    g_acc, info = a._accum_grad(lambda p, b, k: a.actor_loss(b[0], b[1], p, k, b[2]), a.actor_params,
                                (batch, bc, None), rng)
    micro = jax.tree.map(lambda x: x.reshape(x.shape[0] // 2, 2, *x.shape[1:]).swapaxes(0, 1), (batch, bc))
    keys = jax.random.split(rng, 2)
    gs = [jax.grad(lambda p: a.actor_loss(*jax.tree.map(lambda x: x[k], micro), p, keys[k])[0])(a.actor_params)
          for k in range(2)]
    _close(g_acc, jax.tree.map(lambda x, y: (x + y) / 2, *gs), rtol=1e-5, atol=1e-7)
    assert float(info['awr_w_max']) <= a.config['awr_weight_max']


@pytest.mark.parametrize('K', [1, 2])
def test_group_awr_grad_diagnostics(K):
    a = _agent(objective='group_awr', grad_accum_steps=K, ppo_batch_size=2)
    batch, bc = _batch(0), _bc(0)
    d = a.grad_diagnostics(batch, bc)
    assert all(np.isfinite(np.asarray(v)).all() for v in d.values()), d
    g_pg = a._term_grad(batch, bc, 'pg')
    g_bc = a._term_grad(batch, bc, 'bc')
    if K == 1:
        rng = jax.random.fold_in(a.rng, 7)
        ref = jax.grad(lambda p: a._actor_terms(batch, bc, p, rng)[0])(a.actor_params)
        _close(g_pg, ref, rtol=1e-5, atol=1e-7)
    assert float(optax.global_norm(g_pg)) > 0 and float(optax.global_norm(g_bc)) > 0


# =====================================================================================================================
# 4. create() validation
# =====================================================================================================================

class _Reached(Exception):
    pass


@pytest.fixture
def stop_at_load(monkeypatch):
    import ogpo.agents.ogpo_pi05 as Pm

    def _load(path):
        raise _Reached(path)
    monkeypatch.setattr(Pm, 'load_pi05_params', _load)


def _yaml_agent_config(**over):
    from ogpo.utils.config_loader import parse_cli_args
    ac = dict(config_to_agent_dict(load_config(*parse_cli_args(['--algo=ogpo']))))
    ac.update(pi05_action_horizon=10, pi05_checkpoint='/nonexistent', pi05_robot='libero', pi05_critic_pooling='all')
    ac.update(over)
    return ac


GAWR = dict(pi05_actor_objective='group_awr')


@pytest.mark.parametrize('over', [{}, {'adv_strategy': 'conservative'}, {'adv_strategy': 'penultimate'},
                                  {'awr_n_mc': 1}, {'normalize_group': True},
                                  {'use_value_fn': True, 'v_training_mode': 'sarsa', 'v_ensemble_size': 2,
                                   'q_variance_reduction': False}])
def test_create_group_awr_reaches_checkpoint_load(stop_at_load, over, capsys):
    with pytest.raises(_Reached):
        OGPOPi05Agent.create(0, 'p', _yaml_agent_config(**GAWR, **over))


@pytest.mark.parametrize('over,match', [
    ({'awr_beta': 0.0}, 'awr_beta'),
    ({'awr_beta': -0.1}, 'awr_beta'),
    ({'awr_weight_max': 0.0}, 'awr_weight_max'),
    ({'awr_weight_max': -1.0}, 'awr_weight_max'),
    ({'awr_mode': 'asymmetric'}, 'symmetric'),
    ({'awr_mode': 'positive_only'}, 'symmetric'),
    ({'awr_n_mc': 0}, 'awr_n_mc'),
    ({'awr_n_mc': -2}, 'awr_n_mc'),
    ({'adv_strategy': 'awr'}, 'adv_strategy'),
    ({'adv_strategy': 'bogus'}, 'adv_strategy'),
    ({'entropy_coeff': 0.01}, 'entropy_coeff'),
    ({'entropy_coeff': -0.01}, 'entropy_coeff'),
    ({'awr_normalize_adv': True}, 'awr_normalize_adv'),
    ({'use_value_fn': True, 'v_training_mode': 'lambda_return', 'v_ensemble_size': 2}, 'lambda_return'),
])
def test_create_rejects_bad_group_awr_configs(stop_at_load, over, match):
    with pytest.raises(ValueError, match=match):
        OGPOPi05Agent.create(0, 'p', _yaml_agent_config(**{**GAWR, **over}))


@pytest.mark.parametrize('obj', ['group-awr', 'GROUP_AWR', 'grpo_awr'])
def test_create_rejects_unknown_objective(stop_at_load, obj):
    with pytest.raises(ValueError, match='expected ppo \\| awr \\| group_awr'):
        OGPOPi05Agent.create(0, 'p', _yaml_agent_config(pi05_actor_objective=obj))


def test_create_group_awr_keys_ignored_by_ppo(stop_at_load):
    """The group_awr checks gate on the objective: a PPO config with awr_n_mc=0 / entropy_coeff etc. still loads."""
    with pytest.raises(_Reached):
        OGPOPi05Agent.create(0, 'p', _yaml_agent_config(pi05_actor_objective='ppo', awr_n_mc=0, entropy_coeff=0.01,
                                                        awr_normalize_adv=True, adv_strategy='awr', awr_beta=0.0))


# =====================================================================================================================
# 6. Sharded group_awr (8 fake CPU devices, own process)
# =====================================================================================================================

_SHARDED = r'''
import os, sys
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, os.path.join(sys.argv[1], 'tests'))
import test_pi05_multi_device as MD   # sets XLA_FLAGS (8 devices) before importing jax
import jax, numpy as np
from ogpo.utils.sharding import shard_batch_flat, shard_pi05_agent
assert len(jax.devices()) == 8
for accum in (1, 2):
    kw = dict(pi05_actor_objective='group_awr', awr_n_mc=2, grad_accum_steps=accum, entropy_coeff=0.0)
    batch, bc = MD._batches()
    mesh = MD._mesh()
    placed = shard_pi05_agent(MD._agent(MD._Spy, **kw), mesh)
    rows = (MD.PPO // accum) * placed.config['group_num_samples']
    MD.SEEN.clear()
    a_s, i_s = MD._run(placed, batch, bc, mesh, steps=2)
    seen = [s for n, s in MD.SEEN if n == rows]
    assert seen and all(not s.is_fully_replicated for s in seen), MD.SEEN
    a_1, i_1 = MD._run(MD._agent(MD._Fake, **kw), batch, bc, steps=2)
    MD._close(jax.device_get(a_s.actor_params), a_1.actor_params, atol=1e-4)
    for x, y in zip(i_s, i_1):
        assert set(x) == set(y) and 'actor/awr_loss' in x
        MD._close(x, y, atol=1e-4)
    d_s = jax.device_get(placed.grad_diagnostics(shard_batch_flat(batch, mesh), shard_batch_flat(bc, mesh)))
    d_1 = jax.device_get(MD._agent(MD._Fake, **kw).grad_diagnostics(batch, bc))
    MD._close(d_s, d_1, atol=1e-4)
print('SHARDED_OK')
'''


def test_group_awr_sharded_matches_single_device():
    env = dict(os.environ, JAX_PLATFORMS='cpu')
    r = subprocess.run([sys.executable, '-c', _SHARDED, ROOT], capture_output=True, text=True, env=env, timeout=1800)
    assert r.returncode == 0 and 'SHARDED_OK' in r.stdout, (r.stdout[-3000:], r.stderr[-6000:])


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
