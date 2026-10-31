"""
Multi-device tests for the pi05 data-parallel memory fixes (docs/changes/2026-10-11-pi05-zero-actor-state/), on 8
fake CPU devices. MUST run in its own process (XLA_FLAGS is read at jax import):

    JAX_PLATFORMS=cpu python -m pytest tests/test_pi05_multi_device.py -q

  Fix A: with a mesh, the PPO states subsampled in `_actor_terms` reach the SDE sampler sharded over 'data' (they
         were replicated: every device ran every state's chains); `update` equals the single-device result.
  Fix B: `shard_pi05_agent(..., zero_actor_state=True)` shards the actor opt state / init copy leaf-wise
         (`zero_spec`), keeps params replicated, and the donated update keeps that layout (no recompile) with the
         same numbers as the replicated run.
"""
import os
import sys

os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_force_host_platform_device_count=8'
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import copy

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import PartitionSpec as P

from ogpo.agents.modules.flax_utils import ModuleDict, TrainState
from ogpo.agents.ogpo_pi05 import OGPOPi05Agent, PREFIX_DIM, STATE_DIM, T
from ogpo.networks import Value
from ogpo.utils.config_loader import config_to_agent_dict, load_config
from ogpo.utils.sharding import create_mesh, shard_batch_flat, shard_pi05_agent, zero_spec

if len(jax.devices()) < 8:
    pytest.skip("needs 8 (fake) devices: run this file in its own process", allow_module_level=True)

N_DEV = 8
B, PPO, BC_B, NUM_QS, H_PI, HL = 32, 16, 16, 4, 10, 5
OBS = PREFIX_DIM + STATE_DIM
L, S, KV, HD = 2, 3, 1, 2
SEEN = []


class _Fake(OGPOPi05Agent):
    """Fake VLM prefix / velocity; real SDE sampler, log-prob, PPO + BC actor terms, critic."""

    def _prefix(self, images, task_ids=None, kv_only=False):  # kv_only: 2026-10-26 pi05_drop_empty_camera signature
        b_idx = images[:, 0, 0, 0, 0].astype(jnp.float32)
        n = images.shape[0]
        k = jnp.broadcast_to(b_idx[None, :, None, None, None], (L, n, S, KV, HD))
        return None, (k, k + 100.0), jnp.ones((n, S), bool)

    def _actor_fn(self, params, kv_cache, prefix_mask):
        k, _ = kv_cache
        b = k[0, :, 0, 0, 0]

        def actor_fn(obs, x, t, params_=None, is_encoded=True):
            return params['w'] * x + 0.5 * b[:, None] + 2.0 * t + params['m'].mean()

        return actor_fn


class _Spy(_Fake):
    """Records (rows, sharding) of the K cache entering the SDE sampler: the K/V caches are the memory hog. (The
    fake prefix mask is a constant, so it never reflects the input layout.)"""

    def _sample_sde(self, params, kv_cache, prefix_mask, rng):
        k = kv_cache[0]
        jax.debug.inspect_array_sharding(k, callback=lambda s, n=k.shape[1]: SEEN.append((n, s)))
        return super()._sample_sde(params, kv_cache, prefix_mask, rng)


def _config(**kw):
    c = config_to_agent_dict(load_config('ogpo', {}))
    c.update(pi05_action_horizon=H_PI, horizon_length=HL, num_qs=NUM_QS, batch_size=B, ppo_batch_size=PPO,
             group_num_samples=4, flow_steps=3, tau=0.3, actor_tau=0.2, donate_agent=False, grad_accum_steps=1,
             use_success_buffer_q=False, q_min=-100.0, q_max=0.0, adv_clip_min=None, use_bc_regularization=True,
             use_tapered_noise=True, use_constant_noise=False, constant_noise_std=0.1, q_variance_reduction=False)
    c.update(kw)
    return c


def _critic():
    vdef = Value(hidden_dims=(16, 16), layer_norm=True, num_ensembles=NUM_QS, critic_loss_type='mse',
                 q_min=-100.0, q_max=0.0, obs_two_tier=True, two_tier_img_dim=PREFIX_DIM,
                 two_tier_proprio_dim=STATE_DIM, two_tier_fused_dim=0)
    net = ModuleDict({'critic': vdef, 'target_critic': copy.deepcopy(vdef)})
    ex = dict(observations=jnp.zeros((1, OBS)), actions=jnp.zeros((1, HL * T.ENV_ACTION_DIM)))
    params = net.init(jax.random.PRNGKey(0), critic=ex, target_critic=ex)['params']
    return TrainState.create(net, params, optax.adam(1e-2))


def _agent(cls=_Fake, **kw):
    dim_mask = np.zeros((H_PI, T.MODEL_ACTION_DIM), np.float32)
    dim_mask[:HL, :T.ENV_ACTION_DIM] = 1.0
    consts = {'tokens': jnp.zeros((1, 4), jnp.int32), 'dim_mask': jnp.asarray(dim_mask.reshape(-1)),
              'action_q01': -jnp.ones(T.MODEL_ACTION_DIM) * 0.8, 'action_q99': jnp.ones(T.MODEL_ACTION_DIM) * 1.3}
    params = {'w': jnp.float32(0.3), 'm': jnp.asarray(np.random.RandomState(1).randn(64, 8) * 0.01, jnp.float32)}
    actor_tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-2, b1=0.9, b2=0.95, weight_decay=1e-4))
    return cls(rng=jax.random.PRNGKey(3), actor_params=params, target_actor_params=jax.tree.map(lambda x: x + 0.1, params),
               actor_opt_state=actor_tx.init(params), init_actor_params=jax.tree.map(jnp.copy, params),
               frozen_params={}, critic_network=_critic(), consts=consts, config=_config(**kw), model=None,
               actor_tx=actor_tx)


def _batches(seed=0):
    r = np.random.RandomState(seed)
    img = np.zeros((B, 2, 2, 2, 3), np.float32)
    img[:, 0] = np.arange(B)[:, None, None, None]
    batch = {
        'images': jnp.asarray(img), 'next_images': jnp.asarray(img + 1.0),
        'observations': jnp.asarray(r.randn(B, OBS), jnp.float32),
        'next_observations': jnp.asarray(r.randn(B, OBS), jnp.float32),
        'actions': jnp.asarray(r.uniform(-1, 1, (B, HL, T.ENV_ACTION_DIM)), jnp.float32),
        'rewards': jnp.asarray(-r.rand(B, HL), jnp.float32), 'masks': jnp.ones((B, HL), jnp.float32),
        'valid': jnp.ones((B, HL), jnp.float32), 'mc_returns': jnp.asarray(-r.rand(B) * 10, jnp.float32),
        'is_success': jnp.asarray((r.rand(B, HL) > 0.5).astype(np.float32)),
    }
    bimg = np.zeros((BC_B, 2, 2, 2, 3), np.float32)
    bimg[:, 0] = np.arange(BC_B)[:, None, None, None]
    bc = {'images': jnp.asarray(bimg), 'actions': jnp.asarray(r.uniform(-1, 1, (BC_B, H_PI, T.ENV_ACTION_DIM)), jnp.float32),
          'valid': jnp.ones((BC_B, H_PI), jnp.float32)}
    return batch, bc


def _mesh():
    mesh = create_mesh('multi')
    assert mesh is not None and mesh.devices.size == N_DEV
    return mesh


def _run(agent, batch, bc, mesh=None, steps=1):
    if mesh is not None:
        batch, bc = shard_batch_flat(batch, mesh), shard_batch_flat(bc, mesh)
    infos = []
    for _ in range(steps):
        agent, info = agent.update(batch, batch, bc, False)
        infos.append(jax.device_get(info))
    return agent, infos


def _close(a, b, rtol=1e-4, atol=1e-5):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=rtol, atol=atol, equal_nan=True)


def _spec(x):
    return x.sharding.spec


# --- zero_spec ---------------------------------------------------------------------------------------------------

def test_zero_spec_rule():
    assert zero_spec((), 8) == P()
    assert zero_spec((7,), 8) == P()
    assert zero_spec((4,), 8) == P()            # smaller than the device count: replicate
    assert zero_spec((64, 8), 8) == P('data')
    assert zero_spec((3, 16), 8) == P(None, 'data')
    assert zero_spec((18, 2, 1024, 4096), 8) == P(None, None, 'data')


# --- Fix A -------------------------------------------------------------------------------------------------------

FIELDS = ('actor_params', 'target_actor_params', 'actor_opt_state', 'critic_network', 'rng')


@pytest.mark.parametrize('accum', [1, 2])
def test_fix_a_update_matches_unconstrained_and_single_device(accum):
    """Fix A changes only the layout: same numbers as the pre-fix data-parallel run (mesh=None on the same placed
    agent) and as the single-device run. Tolerance: cross-device reduction order, which Adam (lr 1e-2) amplifies on
    near-zero-gradient elements (seen: 1 of 16384 critic weights off by 4e-5 single vs multi at accum 2, also
    without the fix)."""
    batch, bc = _batches()
    mesh = _mesh()
    single, info_s = _run(_agent(grad_accum_steps=accum), batch, bc)
    placed = shard_pi05_agent(_agent(grad_accum_steps=accum), mesh)
    fixed, info_f = _run(placed, batch, bc, mesh=mesh)
    unfixed, info_u = _run(placed.replace(mesh=None), batch, bc, mesh=mesh)
    for f in FIELDS:
        _close(getattr(fixed, f), getattr(unfixed, f), atol=1e-4)
        _close(getattr(fixed, f), getattr(single, f), atol=1e-4)
    _close(info_f, info_u, atol=1e-4)
    _close(info_f, info_s, atol=1e-4)


@pytest.mark.parametrize('accum', [1, 2])
def test_fix_a_ppo_chains_are_sharded_only_with_mesh(accum):
    """Teeth: the same placed agent with mesh=None gets replicated PPO chains."""
    batch, bc = _batches()
    mesh = _mesh()
    placed = shard_pi05_agent(_agent(_Spy, grad_accum_steps=accum), mesh)
    ppo_rows = (PPO // accum) * placed.config['group_num_samples']   # n_ppo states x G chains per micro-batch

    def ppo(seen):
        out = [s for n, s in seen if n == ppo_rows]
        assert out, seen
        return out

    SEEN.clear()
    _run(placed, batch, bc, mesh=mesh)
    assert all(not s.is_fully_replicated for s in ppo(SEEN)), SEEN
    SEEN.clear()
    _run(placed.replace(mesh=None), batch, bc, mesh=mesh)
    assert all(s.is_fully_replicated for s in ppo(SEEN)), SEEN


# --- Fix B -------------------------------------------------------------------------------------------------------

def test_fix_b_layout():
    agent = shard_pi05_agent(_agent(), _mesh(), zero_actor_state=True)
    assert agent.zero_actor_state and agent.mesh is not None
    mu = agent.actor_opt_state[1][0].mu
    assert _spec(mu['m']) == P('data') and _spec(mu['w']) == P()
    assert _spec(agent.actor_opt_state[1][0].count) == P()
    assert _spec(agent.init_actor_params['m']) == P('data')
    for x in jax.tree.leaves((agent.actor_params, agent.target_actor_params, agent.critic_network)):
        assert x.sharding.is_fully_replicated
    plain = shard_pi05_agent(_agent(), _mesh())
    assert not plain.zero_actor_state
    assert all(x.sharding.is_fully_replicated for x in jax.tree.leaves(plain))


@pytest.mark.parametrize('accum', [1, 2])
def test_fix_b_donated_steps_keep_layout_and_match_replicated(accum):
    batch, bc = _batches()
    mesh = _mesh()
    zero = shard_pi05_agent(_agent(grad_accum_steps=accum, donate_agent=True), mesh, zero_actor_state=True)
    ref, info_r = _run(shard_pi05_agent(_agent(grad_accum_steps=accum), mesh), batch, bc, mesh=mesh, steps=3)
    specs0 = jax.tree.map(_spec, (zero.actor_opt_state, zero.init_actor_params, zero.actor_params))
    zero, info_z = _run(zero, batch, bc, mesh=mesh, steps=1)
    n_compiled = OGPOPi05Agent._update_donate._cache_size()
    zero, more = _run(zero, batch, bc, mesh=mesh, steps=2)
    info_z += more
    assert OGPOPi05Agent._update_donate._cache_size() == n_compiled   # no recompile on later steps
    assert jax.tree.map(_spec, (zero.actor_opt_state, zero.init_actor_params, zero.actor_params)) == specs0
    for f in ('actor_params', 'target_actor_params', 'actor_opt_state', 'init_actor_params', 'critic_network', 'rng'):
        _close(getattr(ref, f), getattr(zero, f))
    _close(info_r, info_z)


def test_fix_b_critic_only_update_keeps_layout():
    batch, _ = _batches()
    mesh = _mesh()
    zero = shard_pi05_agent(_agent(), mesh, zero_actor_state=True)
    specs0 = jax.tree.map(_spec, zero.actor_opt_state)
    zero, _ = zero.critic_update(shard_batch_flat(batch, mesh), shard_batch_flat(batch, mesh), False)
    assert jax.tree.map(_spec, zero.actor_opt_state) == specs0


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
