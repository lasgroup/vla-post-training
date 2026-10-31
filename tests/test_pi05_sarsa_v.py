"""
Tests for the pi05 SARSA-V critic (docs/changes/2026-10-09-sarsa-v-critic/): vla-post-training's BoN critic update.

Per batch: Q step  Q(s, a_buf) <- r + gamma^H mask mean_heads V_tgt(s')
           V step  V(s) <- mean_heads Q_tgt(s, a_buf), Q_tgt AFTER the Q step's Polyak update.
Checked against a from-scratch reference (hand losses, SGD, Polyak) on a stub agent whose `_prefix` / `_sample_sde`
raise, so the SARSA path provably never touches the policy or next images.

Run:  JAX_PLATFORMS=cpu python -m pytest tests/test_pi05_sarsa_v.py -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import copy
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import ogpo.runners.pi05_online_runner as R
from ogpo.agents.modules.flax_utils import ModuleDict, TrainState
from ogpo.agents.ogpo_pi05 import OGPOPi05Agent, PREFIX_DIM, STATE_DIM, T
from ogpo.networks import Value

B, NUM_QS, NUM_VS, HL = 6, 4, 3, 5
OBS = PREFIX_DIM + STATE_DIM
GAMMA, TAU, LR = 0.99, 0.3, 0.05


class _NoPolicy(OGPOPi05Agent):
    def _prefix(self, images, task_ids=None, kv_only=False):  # kv_only: 2026-10-26 pi05_drop_empty_camera signature
        raise AssertionError("SARSA-V must not run the VLM prefix")

    def _sample_sde(self, *a, **k):
        raise AssertionError("SARSA-V must not sample the policy")


def _value_def(n):
    return Value(hidden_dims=(16, 16), layer_norm=True, num_ensembles=n, critic_loss_type='mse',
                 q_min=-100.0, q_max=0.0, obs_two_tier=True, two_tier_img_dim=PREFIX_DIM,
                 two_tier_proprio_dim=STATE_DIM, two_tier_fused_dim=0)


def _states(seed=0):
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
    return TrainState.create(qnet, qp, optax.sgd(LR)), TrainState.create(vnet, vp, optax.sgd(LR))


def _agent(with_v=True, **kw):
    c = dict(num_qs=NUM_QS, v_ensemble_size=NUM_VS, v_agg='mean', q_agg='mean', discount=GAMMA,
             horizon_length=HL, tau=TAU, use_success_buffer_q=False, q_variance_reduction=False, donate_agent=False)
    c.update(kw)
    consts = {'tokens': jnp.zeros((1, 4), jnp.int32),
              'action_q01': -jnp.ones(T.MODEL_ACTION_DIM) * 0.8, 'action_q99': jnp.ones(T.MODEL_ACTION_DIM) * 1.3}
    q, v = _states()
    return _NoPolicy(rng=jax.random.PRNGKey(0), actor_params={}, target_actor_params={}, actor_opt_state=None,
                     init_actor_params={}, frozen_params={}, critic_network=q, consts=consts, config=c,
                     model=None, actor_tx=None, value_network=v if with_v else None)


def _batch(seed=0):
    k = jax.random.split(jax.random.PRNGKey(100 + seed), 5)
    valid = np.ones((B, HL), np.float32)
    valid[0, -1] = 0.0                           # one window crossing an episode end
    masks = np.ones((B, HL), np.float32)
    masks[1, -1] = 0.0                           # one terminal
    return dict(
        observations=jax.random.normal(k[0], (B, OBS)),
        next_observations=jax.random.normal(k[1], (B, OBS)),
        actions=jax.random.uniform(k[2], (B, HL, T.ENV_ACTION_DIM), minval=-1, maxval=1),
        rewards=jax.random.normal(k[3], (B, HL)), masks=jnp.asarray(masks), valid=jnp.asarray(valid),
        next_images=jnp.zeros((B, 1)),   # present in real batches; must be unused
    )


# --- from-scratch reference ------------------------------------------------------------------------------------

def _norm_actions(agent, a):
    q01, q99 = agent.consts['action_q01'][:T.ENV_ACTION_DIM], agent.consts['action_q99'][:T.ENV_ACTION_DIM]
    return ((a - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0).reshape(a.shape[0], -1)


def _polyak(params, online, target):
    params = dict(params)
    params[target] = jax.tree.map(lambda p, tp: p * TAU + tp * (1 - TAU), params[online], params[target])
    return params


def _ref_sarsa_step(agent, qs, vs, batch):
    a = _norm_actions(agent, batch['actions'])
    w = batch['valid'][:, -1]

    next_v = vs.select('target_value')(batch['next_observations'], is_encoded=True).mean(axis=0)
    y = batch['rewards'][:, -1] + GAMMA ** HL * batch['masks'][:, -1] * next_v

    def q_loss(p):
        q = qs.select('critic')(batch['observations'], a, params=p, is_encoded=True)
        return (jnp.square(q - y) * w).mean()

    qp = jax.tree.map(lambda p, g: p - LR * g, qs.params, jax.grad(q_loss)(qs.params))
    qs_new = qs.replace(params=_polyak(qp, 'modules_critic', 'modules_target_critic'))

    v_tgt = qs_new.select('target_critic')(batch['observations'], a, is_encoded=True).mean(axis=0)  # post-Q-step

    def v_loss(p):
        v = vs.select('value')(batch['observations'], params=p, is_encoded=True)
        return (jnp.square(v - v_tgt) * w).mean()

    vp = jax.tree.map(lambda p, g: p - LR * g, vs.params, jax.grad(v_loss)(vs.params))
    return qs_new, vs.replace(params=_polyak(vp, 'modules_value', 'modules_target_value'))


def _close(a, b, atol=1e-5):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), atol=atol, rtol=1e-5)


def _differs(a, b):
    return any(not np.allclose(np.asarray(x), np.asarray(y), atol=1e-6) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))


# --- tests -----------------------------------------------------------------------------------------------------

def test_q_target_is_v_bootstrap_by_hand():
    agent, batch = _agent(), _batch()
    loss, info = agent.critic_loss(batch, agent.critic_network.params, jax.random.PRNGKey(1))
    a = _norm_actions(agent, batch['actions'])
    next_v = agent.value_network.select('target_value')(batch['next_observations'], is_encoded=True).mean(axis=0)
    y = batch['rewards'][:, -1] + GAMMA ** HL * batch['masks'][:, -1] * next_v
    q = agent.critic_network.select('critic')(batch['observations'], a, is_encoded=True)
    np.testing.assert_allclose(float(loss), float((jnp.square(q - y) * batch['valid'][:, -1]).mean()), rtol=1e-6)
    np.testing.assert_allclose(float(info['next_q_mean']), float(next_v.mean()), rtol=1e-6)
    assert 'next_actions_std' not in info


def test_critic_update_matches_reference_q_then_v():
    agent, batch = _agent(), _batch()
    new, info = jax.jit(lambda ag, b: ag.critic_update(b, b, False))(agent, batch)
    ref_q, ref_v = _ref_sarsa_step(agent, agent.critic_network, agent.value_network, batch)
    _close(new.critic_network.params, ref_q.params)
    _close(new.value_network.params, ref_v.params)
    assert 'critic/v_loss' in info and 'critic/v_grad/norm' in info and 'critic/grad/norm' in info


def test_v_step_uses_post_q_target_critic():
    """Teeth: a V step against the PRE-Q-step target critic gives different V params."""
    agent, batch = _agent(), _batch()
    new, _ = agent.critic_update(batch, batch, False)
    a = _norm_actions(agent, batch['actions'])
    w = batch['valid'][:, -1]
    stale_tgt = agent.critic_network.select('target_critic')(batch['observations'], a, is_encoded=True).mean(axis=0)
    vs = agent.value_network

    def v_loss(p):
        v = vs.select('value')(batch['observations'], params=p, is_encoded=True)
        return (jnp.square(v - stale_tgt) * w).mean()

    stale = jax.tree.map(lambda p, g: p - LR * g, vs.params, jax.grad(v_loss)(vs.params))['modules_value']
    assert _differs(new.value_network.params['modules_value'], stale)


def test_success_buffer_step_runs_after_rb_step_with_updated_v():
    agent, rb, sb = _agent(use_success_buffer_q=True), _batch(0), _batch(1)
    new, _ = agent.critic_update(rb, sb, True)
    q1, v1 = _ref_sarsa_step(agent, agent.critic_network, agent.value_network, rb)
    q2, v2 = _ref_sarsa_step(agent, q1, v1, sb)
    _close(new.critic_network.params, q2.params)
    _close(new.value_network.params, v2.params)
    # success_flag False: only the RB step
    new_f, _ = agent.critic_update(rb, sb, False)
    _close(new_f.critic_network.params, q1.params)
    _close(new_f.value_network.params, v1.params)


def test_separate_optimiser_states():
    agent, batch = _agent(), _batch()
    q0, v0 = agent.critic_network, agent.value_network
    new, _ = agent.critic_update(batch, batch, False)
    assert int(new.critic_network.step) == int(q0.step) + 1
    assert int(new.value_network.step) == int(v0.step) + 1
    assert jax.tree.structure(new.value_network.opt_state) == jax.tree.structure(v0.opt_state)


def test_value_network_none_is_an_empty_pytree_node():
    """Default agents carry value_network=None: no extra leaves (jit signature / donation unchanged)."""
    with_none, without_field = _agent(with_v=False), _agent(with_v=False)
    assert with_none.value_network is None
    assert jax.tree.leaves(with_none.value_network) == []
    assert len(jax.tree.leaves(with_none)) == len(jax.tree.leaves(without_field.replace(value_network=None)))


def test_save_checkpoint_value_keys(tmp_path):
    for with_v in (False, True):
        d = tmp_path / f"v{int(with_v)}"
        R.save_checkpoint(_agent(with_v=with_v), str(d), 3)
        with open(d / "checkpoint_latest.pkl", "rb") as f:
            state = pickle.load(f)
        assert ('value_params' in state) == with_v and ('value_opt_state' in state) == with_v
        assert {'actor_params', 'critic_params', 'critic_opt_state'} <= set(state)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
