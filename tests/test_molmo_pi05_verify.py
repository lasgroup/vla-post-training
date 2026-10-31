"""
Independent verifier tests for the MolmoSpaces pi05 change (docs/changes/2026-10-14-molmo-pi05/).

Agent: OGPOPi05Agent subclass with a fake `_prefix` / `_actor_fn` (real SDE / ODE samplers, critic, value net,
encode, jitted acting methods, `update`).
  1. libero: jitted sample_actions_ode / plain / BON (sde + ode samplers), called the way the runner now calls them
     (with the replan features), and a full `update` (PPO + BC + TD critic + SARSA-V, grad_accum 1 / 2, bc_batch now
     carrying 'observations') are bit-identical to an agent running VERBATIM e546157 copies of the changed methods.
  2. droid: jitted ode / plain / BON -> env action = unnormalise(model)[:, :H, :8] + chunk-start raw state on dims 0-6
     (raw state computed by hand, not via `_window_state`); BoN picks the argmax-mean-Q candidate in model space.
  3. droid end to end through the runner: real MolmoPi05Env (fake MolmoSpaces task whose arm follows the absolute
     command) x VectorSlotWrapper x SyncVectorEnv x `_run_vector_waves`: executed 8-D actions = the policy chunk,
     chunk = model + state at the replan (state read from the sim, not the features), no [-1, 1] clip in eval,
     buffer stores the 8-D absolute actions, replan-aligned windows normalise back to exactly the critic actions of
     the sampled model chunk (acting <-> critic training consistent), critic_loss / value_loss q equal a hand-computed
     delta + normalise on the window-start state, info contract (length / success) through gymnasium.
  4. `_update_once`: bc_batch carries 'observations' = the window-start features of its 'actions' on the success
     path, the replay path and the task-stratified path.
  5. droid `update` with donation + grad accumulation + SARSA-V runs and keeps `robot`; a bc_batch without
     'observations' fails loudly; shard_pi05_agent keeps `robot`.
  6. get_env_maker('molmo_pi05-*') in an env without robomimic (the .venv-molmo case).

Run:  JAX_PLATFORMS=cpu python -m pytest tests/test_molmo_pi05_verify.py -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import copy
import functools
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from gymnasium.vector.utils import create_empty_array

import envs.molmo_pi05_utils as MU
import ogpo.runners.pi05_online_runner as R
from envs.molmo_pi05_utils import MolmoPi05Env
from ogpo.agents.modules.flax_utils import ModuleDict, TrainState
from ogpo.agents.modules.pg_helper import sample_flow_actions_ode
from ogpo.agents.modules.q_helper import aggregate_q_values, compute_td_loss, compute_td_target, select_best_of_n
from ogpo.agents.ogpo_pi05 import OGPOPi05Agent, PREFIX_DIM, STATE_DIM, T, _per_task
from ogpo.networks import Value
from ogpo.utils.config_loader import config_to_agent_dict, load_config

H_PI, HL, NUM_QS, NUM_VS, N_BON = 15, 5, 4, 3, 6
OBS = PREFIX_DIM + STATE_DIM
FLAT = H_PI * T.MODEL_ACTION_DIM
L, S, KV, HD = 2, 3, 1, 2

# pi05_droid_jointpos assets/droid/norm_stats.json (first 8 dims, rounded; the other 24 are 0 / 0)
DROID_ACT_Q01 = [-0.287, -0.459, -0.286, -0.475, -0.466, -0.442, -0.532, 0.0]
DROID_ACT_Q99 = [0.286, 0.527, 0.283, 0.47, 0.453, 0.5, 0.532, 1.0]
DROID_STATE_Q01 = [-0.828, -0.84, -0.843, -2.773, -1.843, 1.172, -2.047, 0.0]
DROID_STATE_Q99 = [0.9, 1.385, 0.692, -0.454, 1.732, 3.467, 2.198, 0.991]
HOME = np.array([0.05, -0.4, 0.02, -2.3, 0.03, 1.9, 0.8], np.float32)  # droid-like arm pose (joint 3 ~ -2.3 rad)


def _pad32(v):
    return np.concatenate([np.asarray(v, np.float32), np.zeros(32 - len(v), np.float32)])


# ---------------------------------------------------------------------------------------------------------------------
# Fake pi05 + verbatim e546157 copies
# ---------------------------------------------------------------------------------------------------------------------

class _Fake(OGPOPi05Agent):
    def _prefix(self, images, task_ids=None, kv_only=False):  # kv_only: 2026-10-26 pi05_drop_empty_camera signature
        b_idx = images[:, 0, 0, 0, 0].astype(jnp.float32)
        n = images.shape[0]
        k = jnp.broadcast_to(b_idx[None, :, None, None, None], (L, n, S, KV, HD))
        prefix_out = jnp.broadcast_to(0.01 * b_idx[:, None, None], (n, 2, PREFIX_DIM)) + jnp.arange(2.0)[None, :, None]
        return prefix_out, (k, k + 100.0), jnp.ones((n, S), bool)

    def _actor_fn(self, params, kv_cache, prefix_mask):
        k, _ = kv_cache
        b = k[0, :, 0, 0, 0]

        def actor_fn(obs, x, t, params_=None, is_encoded=True):
            # pushes samples well outside [-1, 1]; row-dependent through the (tiled) KV cache
            return params['w'] * x + 0.05 * b[:, None] + 2.0 * t

        return actor_fn


class _Head(_Fake):
    """The methods this change edited, VERBATIM from ogpo/agents/ogpo_pi05.py @ e546157 (docstrings trimmed)."""

    def _critic_actions(self, flat_actions):
        h, d = self._chunk_shape
        x = flat_actions.reshape(-1, h, d)[:, :self.config['horizon_length'], :T.ENV_ACTION_DIM]
        return x.reshape(x.shape[0], -1)

    def _normalize_env_actions(self, env_actions):
        q01, q99 = self.consts['action_q01'][:T.ENV_ACTION_DIM], self.consts['action_q99'][:T.ENV_ACTION_DIM]
        return (env_actions - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

    def _to_env_actions(self, flat_actions):
        h, d = self._chunk_shape
        x = flat_actions.reshape(-1, h, d)[:, :self.config['horizon_length'], :T.ENV_ACTION_DIM]
        q01, q99 = self.consts['action_q01'][:T.ENV_ACTION_DIM], self.consts['action_q99'][:T.ENV_ACTION_DIM]
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01

    def _bc_loss_per_sample(self, bc_batch, grad_params, rng):
        _, kv, pmask = self._prefix(bc_batch['images'], bc_batch.get('task_ids'))
        env_actions = self._normalize_env_actions(bc_batch['actions'])
        a = jnp.pad(env_actions, ((0, 0), (0, 0), (0, T.MODEL_ACTION_DIM - T.ENV_ACTION_DIM)))
        b = a.shape[0]
        a = a.reshape(b, -1)
        x0_rng, t_rng = jax.random.split(rng)
        x0 = jax.random.normal(x0_rng, a.shape)
        t = jax.random.uniform(t_rng, (b, 1))
        x_t = (1 - t) * x0 + t * a
        pred = self._actor_fn(grad_params, kv, pmask)(None, x_t, t)
        err = jnp.square(pred - (a - x0)).reshape(b, *self._chunk_shape)
        return (err * bc_batch['valid'][..., None]).mean(axis=(1, 2))

    def critic_loss(self, batch, grad_params, rng):
        c = self.config
        sample_rng, agg_rng = jax.random.split(rng)
        batch_actions = self._normalize_env_actions(batch['actions']).reshape(batch['actions'].shape[0], -1)

        if self.value_network is not None:
            next_vs = self.value_network.select('target_value')(batch['next_observations'], is_encoded=True)
            next_q = aggregate_q_values(next_vs, method=c['v_agg'], rng=agg_rng, num_qs=c['v_ensemble_size'])
            next_info = {'next_v_head_std': next_vs.std(axis=0).mean()}
        else:
            next_q, next_info = self._sde_next_q(batch, sample_rng, agg_rng)

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
            'next_q_mean': next_q.mean(), **next_info,
            'target_q_mean': target_q.mean(), 'q_head_std': q.std(axis=0).mean(),
            **self._mc_stats(jax.lax.stop_gradient(q).mean(axis=0), batch),
        }

    def value_loss(self, batch, grad_params, rng):
        c = self.config
        batch_actions = self._normalize_env_actions(batch['actions']).reshape(batch['actions'].shape[0], -1)
        q_tgt = self.critic_network.select('target_critic')(batch['observations'], batch_actions, is_encoded=True)
        target_v = jax.lax.stop_gradient(aggregate_q_values(q_tgt, method=c['v_agg'], rng=rng, num_qs=c['num_qs']))
        v = self.value_network.select('value')(batch['observations'], params=grad_params, is_encoded=True)
        v_loss, _ = compute_td_loss(v, target_v, valid_mask=batch['valid'][..., -1], loss_type='mse')
        per_task = _per_task({'v_mean': v.mean(axis=0), 'v_target': target_v},
                             batch.get('task_ids'), self.consts['tokens'].shape[0])
        return v_loss, {
            **per_task,
            'v_loss': v_loss, 'v_mean': v.mean(), 'v_max': v.max(), 'v_min': v.min(),
            'v_target_mean': target_v.mean(), 'v_head_std': v.std(axis=0).mean(),
        }

    @jax.jit
    def sample_actions_BON(self, images, critic_obs, rng, task_ids=None):
        B, N = images.shape[0], self.config['best_of_n']
        action_rng, sub_rng = jax.random.split(rng)
        _, kv, pmask = self._prefix(images, task_ids)
        kv_n = jax.tree.map(lambda x: jnp.repeat(x, N, axis=1), kv)
        pmask_n = jnp.repeat(pmask, N, axis=0)
        if self.config['pi05_bon_sampler'] == 'ode':
            actions = self._sample_ode(self.target_actor_params, kv_n, pmask_n, action_rng, clip=False)
        else:
            actions, _, _, _ = self._sample_sde(self.target_actor_params, kv_n, pmask_n, action_rng)
        q_values = self.critic_network.select('target_critic')(
            jnp.repeat(critic_obs, N, axis=0), self._critic_actions(actions), is_encoded=True,
        ).reshape(self.config['num_qs'], B, N)
        best = select_best_of_n(q_values, self.config['pi05_bon_agg'], sub_rng)
        best_actions = jax.vmap(lambda a, i: a[i])(actions.reshape(B, N, -1), best)
        return self._to_env_actions(best_actions)

    @jax.jit
    def sample_actions_ode(self, images, rng, task_ids=None):
        _, kv, pmask = self._prefix(images, task_ids)
        return self._to_env_actions(self._sample_ode(self.actor_params, kv, pmask, rng, clip=True))

    @jax.jit
    def sample_actions_plain(self, images, rng, task_ids=None):
        _, kv, pmask = self._prefix(images, task_ids)
        return self._to_env_actions(self._sample_ode(self.target_actor_params, kv, pmask, rng, clip=False))


# ---------------------------------------------------------------------------------------------------------------------
# Agent / batch builders
# ---------------------------------------------------------------------------------------------------------------------

def _config(**kw):
    c = config_to_agent_dict(load_config('ogpo', {}))
    c.update(pi05_action_horizon=H_PI, horizon_length=HL, num_qs=NUM_QS, v_ensemble_size=NUM_VS, best_of_n=N_BON,
             batch_size=4, ppo_batch_size=4, group_num_samples=3, flow_steps=4, tau=0.3, actor_tau=0.2,
             q_vr_num_samples=3, donate_agent=False, grad_accum_steps=1, use_success_buffer_q=True,
             q_min=-100.0, q_max=0.0, adv_clip_min=None, use_bc_regularization=True, discount=0.995,
             use_tapered_noise=True, use_constant_noise=False, constant_noise_std=0.1)
    c.update(kw)
    return c


def _vdef(n):
    return Value(hidden_dims=(16, 16), layer_norm=True, num_ensembles=n, critic_loss_type='mse',
                 q_min=-100.0, q_max=0.0, obs_two_tier=True, two_tier_img_dim=PREFIX_DIM,
                 two_tier_proprio_dim=STATE_DIM, two_tier_fused_dim=0)


def _nets(env_dim, seed=0):
    k = jax.random.split(jax.random.PRNGKey(seed), 4)
    qdef, vdef = _vdef(NUM_QS), _vdef(NUM_VS)
    qnet = ModuleDict({'critic': qdef, 'target_critic': copy.deepcopy(qdef)})
    vnet = ModuleDict({'value': vdef, 'target_value': copy.deepcopy(vdef)})
    exq = dict(observations=jnp.zeros((1, OBS)), actions=jnp.zeros((1, HL * env_dim)))
    exv = dict(observations=jnp.zeros((1, OBS)))
    qp = qnet.init(k[0], critic=exq, target_critic=exq)['params']
    qp['modules_target_critic'] = qnet.init(k[1], critic=exq, target_critic=exq)['params']['modules_critic']
    vp = vnet.init(k[2], value=exv, target_value=exv)['params']
    vp['modules_target_value'] = vnet.init(k[3], value=exv, target_value=exv)['params']['modules_value']
    return TrainState.create(qnet, qp, optax.adam(1e-2)), TrainState.create(vnet, vp, optax.adam(1e-2))


def _consts(robot, n_tasks=1):
    env_dim = T.ROBOTS[robot]['env_action_dim']
    dim_mask = np.zeros((H_PI, T.MODEL_ACTION_DIM), np.float32)
    dim_mask[:HL, :env_dim] = 1.0
    if robot == 'droid':
        aq01, aq99 = _pad32(DROID_ACT_Q01), _pad32(DROID_ACT_Q99)
        sq01, sq99 = np.asarray(DROID_STATE_Q01, np.float32), np.asarray(DROID_STATE_Q99, np.float32)
    else:
        aq01, aq99 = -np.ones(32, np.float32) * 0.8, np.ones(32, np.float32) * 1.3
        sq01, sq99 = -np.ones(STATE_DIM, np.float32) * 0.5, np.ones(STATE_DIM, np.float32) * 2.0
    return {'tokens': jnp.zeros((n_tasks, 4), jnp.int32), 'token_mask': jnp.ones((n_tasks, 4), bool),
            'dim_mask': jnp.asarray(dim_mask.reshape(-1)),
            'action_q01': jnp.asarray(aq01), 'action_q99': jnp.asarray(aq99),
            'state_q01': jnp.asarray(sq01), 'state_q99': jnp.asarray(sq99)}


def _agent(robot='libero', cls=_Fake, with_v=False, n_tasks=1, **kw):
    q, v = _nets(T.ROBOTS[robot]['env_action_dim'])
    actor_tx = optax.adam(1e-2)
    w = {'w': jnp.float32(0.3)}
    return cls(rng=jax.random.PRNGKey(3), actor_params=w, target_actor_params={'w': jnp.float32(0.7)},
               actor_opt_state=actor_tx.init(w), init_actor_params={'w': jnp.float32(0.3)}, frozen_params={},
               critic_network=q, consts=_consts(robot, n_tasks), config=_config(**kw), model=None, actor_tx=actor_tx,
               value_network=v if with_v else None, robot=robot)


def _images(B, seed=0):
    img = np.zeros((B, 2, 2, 2, 3), np.float32)
    img[:, 0] = (np.arange(B) + 10 * seed)[:, None, None, None]
    return jnp.asarray(img)


def _raw_states(B, seed=0):
    r = np.random.RandomState(seed)
    return r.uniform(DROID_STATE_Q01, DROID_STATE_Q99, (B, STATE_DIM)).astype(np.float32)


def _feats(agent, raw_states, seed=0):
    """[prefix | normalised state], written independently of encode (same openpi quantile formula)."""
    q01, q99 = np.asarray(agent.consts['state_q01']), np.asarray(agent.consts['state_q99'])
    st = (raw_states - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    pre = np.random.RandomState(100 + seed).randn(raw_states.shape[0], PREFIX_DIM).astype(np.float32)
    return jnp.asarray(np.concatenate([pre, st], -1), jnp.float32)


def _batch(agent, B, seed=0):
    env_dim = agent.env_action_dim
    r = np.random.RandomState(seed)
    valid = np.ones((B, HL), np.float32)
    valid[0, -1] = 0.0
    masks = np.ones((B, HL), np.float32)
    masks[1, -1] = 0.0
    raw = _raw_states(B, seed)
    act = r.uniform(-1, 1, (B, HL, env_dim)).astype(np.float32)
    if agent.robot == 'droid':
        act[..., :7] += raw[:, None, :7]  # absolute joint targets near the window-start state
    return {
        'images': _images(B, seed), 'next_images': _images(B, seed) + 1.0,
        'observations': _feats(agent, raw, seed), 'next_observations': _feats(agent, _raw_states(B, seed + 7), seed + 7),
        'actions': jnp.asarray(act), 'rewards': jnp.asarray(-r.rand(B, HL), jnp.float32),
        'masks': jnp.asarray(masks), 'valid': jnp.asarray(valid),
        'mc_returns': jnp.asarray(-r.rand(B) * 10, jnp.float32),
        'is_success': jnp.asarray((r.rand(B, HL) > 0.5).astype(np.float32)),
    }


def _bc(agent, B, seed=0, with_obs=True):
    env_dim = agent.env_action_dim
    r = np.random.RandomState(50 + seed)
    raw = _raw_states(B, 50 + seed)
    act = r.uniform(-1, 1, (B, H_PI, env_dim)).astype(np.float32)
    if agent.robot == 'droid':
        act[..., :7] += raw[:, None, :7]
    valid = np.ones((B, H_PI), np.float32)
    valid[0, 9:] = 0.0
    out = {'images': _images(B, seed), 'actions': jnp.asarray(act), 'valid': jnp.asarray(valid)}
    if with_obs:
        out['observations'] = _feats(agent, raw, 50 + seed)
    return out


def _assert_trees_equal(a, b):
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    assert len(la) == len(lb)
    for x, y in zip(la, lb):
        assert np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True)


def _hand_to_env(agent, model_flat, raw_state):
    """Independent droid / libero model -> env conversion (openpi Unnormalize + AbsoluteActions(mask 7, -1))."""
    a = agent.env_action_dim
    x = np.asarray(model_flat, np.float64).reshape(-1, H_PI, 32)[:, :HL, :a]
    q01 = np.asarray(agent.consts['action_q01'], np.float64)[:a]
    q99 = np.asarray(agent.consts['action_q99'], np.float64)[:a]
    x = (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    k = T.ROBOTS[agent.robot]['delta_dims']
    if k:
        x[..., :k] += np.asarray(raw_state, np.float64)[:, None, :k]
    return x


def _hand_normalize(agent, env_actions, raw_state):
    """Independent droid / libero env -> model conversion (openpi DeltaActions + Normalize)."""
    a = agent.env_action_dim
    x = np.asarray(env_actions, np.float64).copy()
    k = T.ROBOTS[agent.robot]['delta_dims']
    if k:
        x[..., :k] -= np.asarray(raw_state, np.float64)[:, None, :k]
    q01 = np.asarray(agent.consts['action_q01'], np.float64)[:a]
    q99 = np.asarray(agent.consts['action_q99'], np.float64)[:a]
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


# ---------------------------------------------------------------------------------------------------------------------
# 1. libero bit-exact vs e546157
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('mode,sampler', [('ode', None), ('plain', None), ('bon', 'sde'), ('bon', 'ode')])
def test_libero_acting_bit_exact_vs_e546157(mode, sampler):
    kw = {} if sampler is None else dict(pi05_bon_sampler=sampler)
    new, head = _agent('libero', **kw), _agent('libero', cls=_Head, **kw)
    B = 5
    imgs, key = _images(B), jax.random.PRNGKey(11)
    feats = _feats(new, _raw_states(B))
    ids = jnp.zeros(B, jnp.int32)
    a = np.asarray(R._policy(new, mode)(imgs, feats, ids, key))  # as the runner calls it now (feats passed)
    if mode == 'ode':
        b = head.sample_actions_ode(imgs, key, ids)
    elif mode == 'plain':
        b = head.sample_actions_plain(imgs, key, ids)
    else:
        b = head.sample_actions_BON(imgs, feats, key, ids)
    assert a.shape == (B, HL, 7)
    assert np.array_equal(a, np.asarray(b))


@pytest.mark.parametrize('with_v', [False, True])
@pytest.mark.parametrize('accum', [1, 2])
def test_libero_update_bit_exact_vs_e546157(with_v, accum):
    new = _agent('libero', with_v=with_v, grad_accum_steps=accum)
    head = _agent('libero', cls=_Head, with_v=with_v, grad_accum_steps=accum)
    batch, bs = _batch(new, 4, 0), _batch(new, 4, 1)
    na, ni = new.update(batch, bs, _bc(new, 4, with_obs=True), True)    # runner now ships 'observations'
    ha, hi = head.update(batch, bs, _bc(head, 4, with_obs=False), True)  # e546157 bc_batch
    _assert_trees_equal(na, ha)
    assert ni.keys() == hi.keys()
    for k in ni:
        assert np.array_equal(np.asarray(ni[k]), np.asarray(hi[k]), equal_nan=True), k
    nc, nci = new.critic_update(batch, bs, False)
    hc, hci = head.critic_update(batch, bs, False)
    _assert_trees_equal(nc, hc)


# ---------------------------------------------------------------------------------------------------------------------
# 2. droid acting through the real jitted methods
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('mode', ['ode', 'plain'])
def test_droid_ode_plain_env_actions(mode):
    agent = _agent('droid')
    B = 4
    imgs, key = _images(B), jax.random.PRNGKey(5)
    raw = _raw_states(B, 3)
    feats = _feats(agent, raw, 3)
    out = np.asarray(R._policy(agent, mode)(imgs, feats, jnp.zeros(B, jnp.int32), key))
    _, kv, pmask = agent._prefix(imgs)
    params = agent.actor_params if mode == 'ode' else agent.target_actor_params
    model = agent._sample_ode(params, kv, pmask, key, clip=(mode == 'ode'))
    expect = _hand_to_env(agent, model, raw)
    assert out.shape == (B, HL, 8)
    np.testing.assert_allclose(out, expect, rtol=0, atol=2e-5)
    assert np.abs(out).max() > 1.0  # absolute joint targets: the agent itself never clips env actions


@pytest.mark.parametrize('sampler', ['sde', 'ode'])
def test_droid_bon_picks_argmax_candidate_and_adds_state(sampler):
    agent = _agent('droid', pi05_bon_sampler=sampler, pi05_bon_agg='mean')
    B, N = 3, N_BON
    imgs, key = _images(B), jax.random.PRNGKey(9)
    raw = _raw_states(B, 4)
    feats = _feats(agent, raw, 4)
    out = np.asarray(agent.sample_actions_BON(imgs, feats, key, jnp.zeros(B, jnp.int32)))
    # candidates, as the method draws them
    action_rng, _ = jax.random.split(key)
    _, kv, pmask = agent._prefix(imgs)
    kv_n = jax.tree.map(lambda x: jnp.repeat(x, N, axis=1), kv)
    pmask_n = jnp.repeat(pmask, N, axis=0)
    if sampler == 'ode':
        cands = agent._sample_ode(agent.target_actor_params, kv_n, pmask_n, action_rng, clip=False)
    else:
        cands = agent._sample_sde(agent.target_actor_params, kv_n, pmask_n, action_rng)[0]
    q = agent.critic_network.select('target_critic')(jnp.repeat(feats, N, axis=0), agent._critic_actions(cands),
                                                       is_encoded=True).reshape(NUM_QS, B, N)
    best = np.asarray(q.mean(axis=0).argmax(axis=1))
    cands = np.asarray(cands).reshape(B, N, -1)
    expect = _hand_to_env(agent, cands[np.arange(B), best], raw)
    np.testing.assert_allclose(out, expect, rtol=0, atol=2e-5)


# ---------------------------------------------------------------------------------------------------------------------
# 3. droid end to end: MolmoPi05Env (fake task) -> VectorSlotWrapper -> SyncVectorEnv -> runner -> buffer -> losses
# ---------------------------------------------------------------------------------------------------------------------

class _ArmTask:
    """Fake MolmoSpaces task: the arm qpos becomes the commanded absolute joint target; success at `succeed_at`."""

    def __init__(self, succeed_at, offset):
        self.env = SimpleNamespace(n_batch=1)
        self._sensor_suite = SimpleNamespace(sensors={})
        self.succeed_at, self.offset = succeed_at, offset
        self.arm, self.grip, self.k = None, None, 0
        self.commands, self.states_at_step = [], []

    def register_policy(self, policy):
        pass

    def _obs(self):
        img = np.full((352, 624, 3), 40 + self.offset, np.uint8)
        return {'qpos': {'arm': self.arm.astype(np.float64), 'gripper': np.array([self.grip, self.grip])},
                'exo_camera_1': img, 'wrist_camera': img}

    def reset(self):
        self.arm = HOME + 0.01 * self.offset
        self.grip, self.k = 0.0, 0
        return [self._obs()], [{'success': False}]

    def step(self, action):
        self.states_at_step.append(np.concatenate([self.arm, [min(self.grip / 0.824033, 1.0)]]).astype(np.float32))
        self.commands.append({'arm': np.asarray(action['arm']).copy(), 'gripper': np.asarray(action['gripper']).copy()})
        self.arm = np.asarray(action['arm'], np.float32).copy()
        self.grip = 0.824033 if action['gripper'][0] > 0 else 0.0
        self.k += 1
        succ = self.k == self.succeed_at
        return [self._obs()], np.array([0.5]), np.array([succ]), np.array([False]), [{'success': succ}]


class _SyncShim(SyncVectorEnv):
    """As tests/test_pi05_vector.py: DISABLED autoreset, finished slots may be stepped (AsyncVectorEnv semantics)."""

    def __init__(self, env_fns):
        super().__init__(env_fns, autoreset_mode=AutoresetMode.DISABLED)
        zero = create_empty_array(self.single_observation_space, n=1, fn=np.zeros)
        self._env_obs = [{k: v[0].copy() for k, v in zero.items()} for _ in range(self.num_envs)]

    def step(self, actions):
        self._autoreset_envs[:] = False
        return super().step(actions)


@pytest.fixture
def molmo_vec(monkeypatch):
    for k in MU.REQUIRED_ENV_VARS:
        monkeypatch.setenv(k, '/x')

    def fake_load(self):
        if self._sampler is None:
            task = _ArmTask(self._succeed_at, self._offset)
            self._sampler = SimpleNamespace(sample_task=lambda **kw: task, close=lambda: None)
            self._episode = SimpleNamespace(house_index=0)
            self._prompt_sampler = SimpleNamespace(next=lambda: None, get_prompt=lambda t: 'pick up the box.')

    monkeypatch.setattr(MolmoPi05Env, '_load', fake_load)

    def build(names, m, succeed):
        fns = []
        for ti, name in enumerate(names):
            for j in range(m):
                def fn(name=name, off=ti * m + j, sa=succeed[ti * m + j]):
                    env = MU.make_env(name, vector=True)
                    env.unwrapped._succeed_at, env.unwrapped._offset = sa, off
                    return env
                fns.append(fn)
        vec = _SyncShim(fns)
        return R.VectorTasks(vec, list(names), list(range(len(names))), m, 16), vec

    return build


def _recording_policy(agent, mode, log):
    base = R._policy(agent, mode)

    def policy(imgs, feats, ids, key):
        out = base(imgs, feats, ids, key)
        _, kv, pmask = agent._prefix(jnp.asarray(imgs))
        model = agent._sample_ode(agent.actor_params if mode == 'ode' else agent.target_actor_params, kv, pmask, key,
                                  clip=(mode == 'ode'))
        log.append((np.asarray(feats), np.asarray(model), np.asarray(out)))
        return out

    return policy


def test_droid_vector_rollout_buffer_and_losses(molmo_vec):
    agent = _agent('droid', with_v=True)
    names, m = ['molmo_pi05-164', 'molmo_pi05-86'], 2
    succeed = [12, 7, 9, 13]  # per slot: success step (one episode per slot, one wave)
    vt, vec = molmo_vec(names, m, succeed)
    log, eps = [], []
    R._run_vector_waves(vt, m, lambda k, w, j, k_w: 1000 + 10 * k + j, agent, _recording_policy(agent, 'plain', log),
                        jax.random.PRNGKey(0), clip_actions=False, record=True, on_episode=eps.append)
    tasks = [e.unwrapped._task for e in vec.envs]
    assert sorted((e['task'], e['length']) for e in eps) == sorted([(0, 12), (0, 7), (1, 9), (1, 13)])
    assert all(e['success'] == 1.0 and e['info']['length'] == e['length'] for e in eps)

    # executed actions: chunk h of replan r == what the sim received, chunk == model + state at the replan (from sim)
    for s, task in enumerate(tasks):
        L_s = len(task.commands)
        assert L_s == succeed[s]
        for t in range(L_s):
            r, h = divmod(t, HL)
            feats_r, model_r, out_r = log[r]
            np.testing.assert_array_equal(task.commands[t]['arm'], out_r[s, h, :7].astype(np.float32))
            assert task.commands[t]['gripper'][0] == (255.0 if out_r[s, h, 7] > 0.5 else 0.0)
            start = task.states_at_step[r * HL]  # raw state at the replan, as the sim reported it
            expect = _hand_to_env(agent, model_r[s:s + 1], start[None])[0, h]
            np.testing.assert_allclose(out_r[s, h], expect, rtol=0, atol=3e-5)

    # buffer: 8-D absolute actions as executed
    ep0 = eps[0]['trajectory'][0]
    rb = R._make_buffer(ep0['observations'], ep0['images'], 500, 0, agent.env_action_dim)
    for e in eps:
        R._flush_episode(rb, e['trajectory'], e['info'], 0.995)
    assert rb['actions'].shape[-1] == 8
    stored = np.concatenate([np.stack([t['actions'] for t in e['trajectory']]) for e in eps])
    np.testing.assert_array_equal(np.asarray(rb['actions'][:rb.size]), stored)
    assert np.abs(stored).max() > 1.0  # absolute joint targets, never clipped during collection

    # acting <-> critic training: a replan-aligned window normalises back to the critic actions of its model chunk
    by_slot = {}
    for e in eps:
        s = next(i for i, task in enumerate(tasks) if len(task.commands) == e['length'] and
                 np.array_equal(task.commands[0]['arm'], e['trajectory'][0]['actions'][:7]))
        by_slot[s] = e['trajectory']
    for s, traj in by_slot.items():
        for r in range(len(traj) // HL):
            win = np.stack([t['actions'] for t in traj[r * HL:(r + 1) * HL]])[None]
            obs = traj[r * HL]['observations'][None]
            got = np.asarray(agent._normalize_env_actions(jnp.asarray(win), jnp.asarray(obs))).reshape(-1)
            want = np.asarray(agent._critic_actions(jnp.asarray(log[r][1][s:s + 1])))[0]
            np.testing.assert_allclose(got, want, rtol=0, atol=5e-4)

    # critic_loss / value_loss on a real buffer batch use the delta to the window-start state
    batch = rb.sample_sequence(8, sequence_length=HL, discount=0.995)
    batch = {k: jnp.asarray(v) for k, v in batch.items()}
    q01, q99 = np.asarray(agent.consts['state_q01']), np.asarray(agent.consts['state_q99'])
    raw_start = (np.asarray(batch['observations'])[:, -STATE_DIM:] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    hand = _hand_normalize(agent, batch['actions'], raw_start).reshape(8, -1).astype(np.float32)
    q_hand = agent.critic_network.select('critic')(batch['observations'], jnp.asarray(hand), is_encoded=True)
    _, info = agent.critic_loss(batch, agent.critic_network.params, jax.random.PRNGKey(1))
    np.testing.assert_allclose(float(info['q_mean']), float(q_hand.mean()), rtol=1e-5, atol=1e-5)
    qt_hand = agent.critic_network.select('target_critic')(batch['observations'], jnp.asarray(hand), is_encoded=True)
    _, vinfo = agent.value_loss(batch, agent.value_network.params, jax.random.PRNGKey(2))
    np.testing.assert_allclose(float(vinfo['v_target_mean']), float(qt_hand.mean(axis=0).mean()), rtol=1e-5, atol=1e-5)


def test_droid_vector_eval_not_clipped(molmo_vec):
    agent = _agent('droid')
    vt, vec = molmo_vec(['molmo_pi05-164'], 2, [6, 8])
    metrics = R.evaluate_pi05_vector(agent, vt, 0, 2, 'ode')
    assert metrics['molmo_pi05-164']['success'] == 1.0
    assert metrics['molmo_pi05-164']['length'] == 7.0
    arms = np.stack([c['arm'] for e in vec.envs for c in e.unwrapped._task.commands])
    assert (arms[:, 3] < -1.0).all()  # the runner did not clip droid's absolute joint targets to [-1, 1]
    assert not agent.clip_env_actions and _agent('libero').clip_env_actions


# ---------------------------------------------------------------------------------------------------------------------
# 4. _update_once ships window-start observations in bc_batch
# ---------------------------------------------------------------------------------------------------------------------

class _CaptureAgent:
    env_action_dim = 8

    def __init__(self):
        self.seen = []
        self.actor_params = jnp.zeros(1)

    def update(self, batch, batch_success, bc_batch, success_flag, ppo_batch=None):
        self.seen.append((batch, batch_success, bc_batch, success_flag, ppo_batch))
        return self, {}


def _tagged_buffer(n_eps, ep_len, success_eps, n_tasks=1):
    """obs[0] = action[0] = global step index; episodes contiguous."""
    obs0 = np.zeros(10, np.float32)
    rb = R._make_buffer(obs0, np.zeros((2, 2, 2, 3), np.uint8), 5000, 0, 8)
    g = 0
    for e in range(n_eps):
        traj = []
        for t in range(ep_len):
            o = np.zeros(10, np.float32)
            o[0] = g
            a = np.zeros(8, np.float32)
            a[0] = g
            traj.append(dict(observations=o, images=np.zeros((2, 2, 2, 3), np.uint8), actions=a, rewards=-1.0,
                             terminals=float(t == ep_len - 1), masks=1.0, next_observations=o,
                             next_images=np.zeros((2, 2, 2, 3), np.uint8), task_ids=np.int32(e % n_tasks)))
            g += 1
        R._flush_episode(rb, traj, {'success': 1.0 if e in success_eps else 0.0}, 0.99)
    return rb


@pytest.mark.parametrize('path', ['success', 'replay', 'stratified'])
def test_update_once_bc_batch_window_start_observations(path):
    if path == 'success':
        rb = _tagged_buffer(20, 30, success_eps=set(range(0, 20, 2)))
        kw = {}
    elif path == 'replay':
        rb = _tagged_buffer(20, 30, success_eps=set())
        kw = {}
    else:
        rb = _tagged_buffer(20, 30, success_eps=set(range(0, 20, 3)), n_tasks=2)
        kw = dict(task_balance='bc', num_tasks=2)
    agent = _CaptureAgent()
    R._update_once(agent, rb, lambda b: b, batch_size=16, success_buffer_batch_size=16, horizon_length=HL,
                   bc_horizon=H_PI, discount=0.99, bc_batch_size=32, **kw)
    _, _, bc, flag, _ = agent.seen[-1]
    assert set(bc) >= {'images', 'observations', 'actions', 'valid'}
    assert bc['actions'].shape == (32, H_PI, 8) and bc['observations'].shape == (32, 10)
    np.testing.assert_array_equal(bc['observations'][:, 0], bc['actions'][:, 0, 0])  # window start
    if path == 'success':
        succ_idx = np.flatnonzero(rb._traj_success_mask[:rb.size])
        assert np.isin(bc['actions'][:, 0, 0].astype(int), succ_idx).all()


# ---------------------------------------------------------------------------------------------------------------------
# 5. droid update: donation + grad accumulation + SARSA-V; missing observations fail loudly; sharding keeps robot
# ---------------------------------------------------------------------------------------------------------------------

def test_droid_update_donate_accum_runs_and_keeps_robot():
    agent = _agent('droid', with_v=True, grad_accum_steps=2, donate_agent=True)
    batch, bs = _batch(agent, 4, 0), _batch(agent, 4, 1)
    new, info = agent.update(batch, bs, _bc(agent, 4), True)
    assert new.robot == 'droid' and new.env_action_dim == 8
    assert all(np.isfinite(np.asarray(v)).all() for k, v in info.items() if 'task' not in k)
    new2, info2 = new.critic_update(batch, bs, True)
    assert new2.robot == 'droid'


def test_droid_bc_without_observations_fails_loudly():
    agent = _agent('droid')
    with pytest.raises(AssertionError, match='critic features'):
        agent.update(_batch(agent, 4, 0), _batch(agent, 4, 1), _bc(agent, 4, with_obs=False), True)


def test_shard_pi05_agent_keeps_robot():
    from jax.sharding import Mesh
    from ogpo.utils.sharding import shard_pi05_agent, shard_batch_flat
    mesh = Mesh(np.array(jax.devices()[:1]), ('data',))
    agent = shard_pi05_agent(_agent('droid', batch_size=4, ppo_batch_size=2), mesh)
    assert agent.robot == 'droid' and agent.mesh is mesh
    batch, bs = _batch(agent, 4, 0), _batch(agent, 4, 1)
    new, _ = agent.update(shard_batch_flat(batch, mesh), shard_batch_flat(bs, mesh), shard_batch_flat(_bc(agent, 4), mesh),
                          True)
    assert new.robot == 'droid'


# ---------------------------------------------------------------------------------------------------------------------
# 6. dispatch in an env without robomimic (.venv-molmo has no robomimic extra)
# ---------------------------------------------------------------------------------------------------------------------

def test_get_env_maker_molmo_without_robomimic(monkeypatch):
    for mod in [m for m in list(sys.modules) if m == 'robomimic' or m.startswith('robomimic.')]:
        monkeypatch.delitem(sys.modules, mod)
    monkeypatch.setitem(sys.modules, 'robomimic', None)  # import robomimic -> ImportError, as in .venv-molmo
    monkeypatch.delitem(sys.modules, 'envs.robomimic_utils', raising=False)
    from ogpo.utils.main_helper import get_env_maker
    assert get_env_maker('molmo_pi05-164') is MU.make_env
