"""
Independent verifier, 8 fake CPU devices (docs/changes/2026-10-11-pi05-zero-actor-state/). MUST run in its own
process (XLA_FLAGS is read at jax import); on hosts with a small `ulimit -v`, run one test function per process.

  Fix A: grad_diagnostics' `_term_grad` PPO chains are sharded (accum 1 / 2) and match mesh=None; the runner's
         ppo_batch path and the ppo_batch_size == batch_size path reach the sampler sharded; multi-task task_ids;
         a non-divisible subsample (outside main_pi05's check) is documented.
  Fix B: zero_spec edge cases; donated update / critic_update alternating (actor_update_interval > 1): layout stable,
         no recompile, donation consumed (old buffers deleted, no "donated buffers were not usable" warning), same
         numbers as the replicated run, for the Q-only and SARSA-V critics; save_checkpoint on a sharded agent
         gathers full arrays equal to the replicated run's; grad_diagnostics with sharded init_actor_params.
"""
import os
import sys

os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_force_host_platform_device_count=8'
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import copy
import pickle
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

if len(jax.devices()) < 8:
    pytest.skip("needs 8 (fake) devices: run this file in its own process", allow_module_level=True)

import test_pi05_multi_device as MD  # helpers only (module attribute: its tests are not re-collected)
from ogpo.agents.modules.flax_utils import ModuleDict, TrainState
from ogpo.agents.ogpo_pi05 import OGPOPi05Agent, PREFIX_DIM, STATE_DIM, T
from ogpo.networks import Value
from ogpo.runners import pi05_online_runner as R
from ogpo.utils.sharding import shard_batch_flat, shard_pi05_agent, zero_shardings, zero_spec

B, PPO = MD.B, MD.PPO
SEEN = MD.SEEN


def _spec(x):
    return x.sharding.spec


def _ppo_specs(ppo_rows):
    out = [s for n, s in SEEN if n == ppo_rows]
    assert out, SEEN
    return out


def _value_net():
    vdef = Value(hidden_dims=(16, 16), layer_norm=True, num_ensembles=3, critic_loss_type='mse', q_min=-100.0,
                 q_max=0.0, obs_two_tier=True, two_tier_img_dim=PREFIX_DIM, two_tier_proprio_dim=STATE_DIM,
                 two_tier_fused_dim=0)
    net = ModuleDict({'value': vdef, 'target_value': copy.deepcopy(vdef)})
    ex = dict(observations=jnp.zeros((1, MD.OBS)))
    params = net.init(jax.random.PRNGKey(5), value=ex, target_value=ex)['params']
    params['modules_target_value'] = jax.tree.map(jnp.copy, params['modules_value'])
    return TrainState.create(net, params, optax.adam(1e-2))


def _agent(cls=MD._Fake, with_v=False, **kw):
    a = MD._agent(cls, v_ensemble_size=3, **kw)
    return a.replace(value_network=_value_net()) if with_v else a


# --- zero_spec edge cases ----------------------------------------------------------------------------------------

def test_zero_spec_edge_cases():
    assert zero_spec((1, 8), 8) == P(None, 'data')
    assert zero_spec((8, 1), 8) == P('data')
    assert zero_spec((1, 1, 1), 8) == P()
    assert zero_spec((0,), 8) == P()
    assert zero_spec((0, 16), 8) == P(None, 'data')
    assert zero_spec((12, 24), 8) == P(None, 'data')
    assert zero_spec((7, 9), 8) == P()
    assert zero_spec((16,), 1) == P('data')          # 1-device mesh: trivially 'data'
    mesh = MD._mesh()
    tree = {'bf': jnp.arange(64, dtype=jnp.bfloat16).reshape(8, 8), 'cnt': jnp.zeros((), jnp.int32),
            'odd': jnp.ones((3, 5), jnp.float32), 'one': jnp.ones((1, 16), jnp.float32)}
    placed = jax.device_put(tree, zero_shardings(tree, mesh))
    assert _spec(placed['bf']) == P('data') and _spec(placed['cnt']) == P()
    assert _spec(placed['odd']) == P() and _spec(placed['one']) == P(None, 'data')
    assert placed['bf'].dtype == jnp.bfloat16 and placed['bf'].addressable_shards[0].data.shape == (1, 8)
    for k in tree:
        np.testing.assert_array_equal(np.asarray(jax.device_get(placed[k])), np.asarray(tree[k]))


# --- Fix A -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('accum', [1, 2])
def test_fix_a_grad_diagnostics_sharded_and_matches(accum):
    batch, bc = MD._batches()
    mesh = MD._mesh()
    sb, sbc = shard_batch_flat(batch, mesh), shard_batch_flat(bc, mesh)
    placed = shard_pi05_agent(_agent(MD._Spy, grad_accum_steps=accum), mesh)
    rows = (PPO // accum) * placed.config['group_num_samples']
    SEEN.clear()
    d_fix = jax.device_get(placed.grad_diagnostics(sb, sbc))
    assert all(not s.is_fully_replicated for s in _ppo_specs(rows)), SEEN
    SEEN.clear()
    d_unfix = jax.device_get(placed.replace(mesh=None).grad_diagnostics(sb, sbc))
    assert all(s.is_fully_replicated for s in _ppo_specs(rows)), SEEN
    d_single = jax.device_get(_agent(MD._Spy, grad_accum_steps=accum).grad_diagnostics(batch, bc))
    for k in d_fix:
        np.testing.assert_allclose(d_fix[k], d_unfix[k], rtol=1e-4, atol=1e-6, err_msg=k)
        np.testing.assert_allclose(d_fix[k], d_single[k], rtol=1e-4, atol=1e-6, err_msg=k)


@pytest.mark.parametrize('accum', [1, 2])
def test_fix_a_ppo_batch_path_sharded(accum):
    """task_balance=bc_ppo/all: the runner passes a sharded ppo_batch (no subsample, no constraint needed)."""
    batch, bc = MD._batches()
    mesh = MD._mesh()
    ppo = {'images': batch['images'][:PPO], 'observations': batch['observations'][:PPO],
           'task_ids': jnp.arange(PPO, dtype=jnp.int32) % 1}
    placed = shard_pi05_agent(_agent(MD._Spy, grad_accum_steps=accum), mesh)
    rows = (PPO // accum) * placed.config['group_num_samples']
    SEEN.clear()
    a, info = placed.update(shard_batch_flat(batch, mesh), shard_batch_flat(batch, mesh), shard_batch_flat(bc, mesh),
                            False, shard_batch_flat(ppo, mesh))
    assert all(not s.is_fully_replicated for s in _ppo_specs(rows)), SEEN
    s_agent, s_info = _agent(MD._Spy, grad_accum_steps=accum).update(batch, batch, bc, False, ppo)
    MD._close(a.actor_params, s_agent.actor_params, atol=1e-4)
    MD._close(jax.device_get(info), jax.device_get(s_info), atol=1e-4)


def test_fix_a_full_batch_path_sharded():
    """ppo_batch_size >= batch_size: no subsample; the sharded TD batch reaches the sampler sharded."""
    batch, bc = MD._batches()
    mesh = MD._mesh()
    placed = shard_pi05_agent(_agent(MD._Spy, ppo_batch_size=B), mesh)
    SEEN.clear()
    placed.update(shard_batch_flat(batch, mesh), shard_batch_flat(batch, mesh), shard_batch_flat(bc, mesh), False)
    assert all(not s.is_fully_replicated for s in _ppo_specs(B * placed.config['group_num_samples'])), SEEN


def test_fix_a_multitask_task_ids():
    batch, bc = MD._batches()
    batch = dict(batch, task_ids=jnp.arange(B, dtype=jnp.int32) % 2)
    mesh = MD._mesh()
    a0 = _agent(grad_accum_steps=2)
    a0 = a0.replace(consts=dict(a0.consts, tokens=jnp.zeros((2, 4), jnp.int32)))
    placed = shard_pi05_agent(a0, mesh)
    sb = shard_batch_flat(batch, mesh)
    fixed, info_f = placed.update(sb, sb, shard_batch_flat(bc, mesh), False)
    single, info_s = a0.update(batch, batch, bc, False)
    MD._close(fixed.actor_params, single.actor_params, atol=1e-4)
    info_f, info_s = jax.device_get(info_f), jax.device_get(info_s)
    assert set(info_f) == set(info_s)
    MD._close(info_f, info_s, atol=1e-4)


def test_fix_a_non_divisible_subsample_documented():
    """Outside main_pi05's divisibility check (ppo_batch_size % (n_dev*accum) != 0): n_ppo=12 over 8 devices.
    Records whether with_sharding_constraint pads (uneven) and still matches mesh=None, or raises."""
    batch, bc = MD._batches()
    mesh = MD._mesh()
    placed = shard_pi05_agent(_agent(ppo_batch_size=12), mesh)
    sb = shard_batch_flat(batch, mesh)
    try:
        fixed, info_f = placed.update(sb, sb, shard_batch_flat(bc, mesh), False)
    except Exception as e:  # recorded, not asserted: main_pi05 rejects this config before training
        print(f"[verify] n_ppo=12 over 8 devices raised {type(e).__name__}: {str(e)[:300]}")
        return
    unfixed, info_u = placed.replace(mesh=None).update(sb, sb, shard_batch_flat(bc, mesh), False)
    MD._close(fixed.actor_params, unfixed.actor_params, atol=1e-4)
    MD._close(jax.device_get(info_f), jax.device_get(info_u), atol=1e-4)
    print("[verify] n_ppo=12 over 8 devices: runs (uneven sharding), matches mesh=None")


# --- Fix B -------------------------------------------------------------------------------------------------------

SCHEDULE = ('U', 'C', 'C', 'U', 'C', 'U', 'U')  # actor_update_interval > 1 interleaving


def _run_schedule(agent, batch, bc, mesh):
    sb, sbc = shard_batch_flat(batch, mesh), shard_batch_flat(bc, mesh)
    infos = []
    for s in SCHEDULE:
        if s == 'U':
            agent, info = agent.update(sb, sb, sbc, False)
        else:
            agent, info = agent.critic_update(sb, sb, False)
        infos.append(jax.device_get(info))
    return agent, infos


@pytest.mark.parametrize('with_v', [False, True])
def test_fix_b_alternating_donated_layout_cache_donation_numbers(with_v):
    batch, bc = MD._batches()
    mesh = MD._mesh()
    ref, info_r = _run_schedule(shard_pi05_agent(_agent(with_v=with_v), mesh), batch, bc, mesh)

    zero = shard_pi05_agent(_agent(with_v=with_v, donate_agent=True), mesh, zero_actor_state=True)
    specs0 = jax.tree.map(_spec, (zero.actor_opt_state, zero.init_actor_params, zero.actor_params,
                                  zero.target_actor_params, zero.critic_network, zero.value_network))
    sb, sbc = shard_batch_flat(batch, mesh), shard_batch_flat(bc, mesh)
    infos = []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        sizes = None
        for i, s in enumerate(SCHEDULE):
            old_mu = zero.actor_opt_state[1][0].mu['m']
            old_crit = jax.tree.leaves(zero.critic_network.params)[0]
            if s == 'U':
                zero, info = zero.update(sb, sb, sbc, False)
            else:
                zero, info = zero.critic_update(sb, sb, False)
            jax.block_until_ready(zero)
            infos.append(jax.device_get(info))
            assert old_crit.is_deleted(), f"step {i} ({s}): donated critic buffer still alive"
            if s == 'U':
                assert old_mu.is_deleted(), f"step {i}: donated sharded opt-state buffer still alive"
            assert jax.tree.map(_spec, (zero.actor_opt_state, zero.init_actor_params, zero.actor_params,
                                        zero.target_actor_params, zero.critic_network,
                                        zero.value_network)) == specs0, f"layout changed at step {i} ({s})"
            if i >= 2:  # both jits have compiled once (steps 0 'U' and 1 'C')
                now = (OGPOPi05Agent._update_donate._cache_size(), OGPOPi05Agent._critic_update_donate._cache_size())
                if sizes is None:
                    sizes = now
                assert now == sizes, f"recompile at step {i} ({s}): {sizes} -> {now}"
    bad = [str(x.message) for x in w if 'donat' in str(x.message).lower()]
    assert not bad, bad
    for f in ('actor_params', 'target_actor_params', 'actor_opt_state', 'init_actor_params', 'critic_network',
              'value_network', 'rng'):
        MD._close(getattr(ref, f), getattr(zero, f))
    for a, b in zip(info_r, infos):
        MD._close(a, b)


@pytest.mark.parametrize('with_v', [False, True])
def test_fix_b_save_checkpoint_gathers_full_arrays(tmp_path, with_v):
    batch, bc = MD._batches()
    mesh = MD._mesh()
    ref, _ = _run_schedule(shard_pi05_agent(_agent(with_v=with_v), mesh), batch, bc, mesh)
    zero, _ = _run_schedule(shard_pi05_agent(_agent(with_v=with_v, donate_agent=True), mesh, zero_actor_state=True),
                            batch, bc, mesh)
    assert _spec(zero.actor_opt_state[1][0].mu['m']) == P('data')
    R.save_checkpoint(ref, str(tmp_path / 'ref'), 7)
    R.save_checkpoint(zero, str(tmp_path / 'zero'), 7)
    with open(tmp_path / 'ref' / 'checkpoint_latest.pkl', 'rb') as f:
        s_ref = pickle.load(f)
    with open(tmp_path / 'zero' / 'checkpoint_latest.pkl', 'rb') as f:
        s_zero = pickle.load(f)
    assert set(s_ref) == set(s_zero)
    assert jax.tree.structure(s_ref) == jax.tree.structure(s_zero)
    for x, y in zip(jax.tree.leaves(s_ref), jax.tree.leaves(s_zero)):
        assert isinstance(y, (np.ndarray, np.generic, int)), type(y)
        assert np.shape(x) == np.shape(y)
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-4, atol=1e-5)
    assert s_zero['actor_opt_state'][1][0].mu['m'].shape == (64, 8)


def test_fix_b_grad_diagnostics_with_sharded_init():
    batch, bc = MD._batches()
    mesh = MD._mesh()
    sb, sbc = shard_batch_flat(batch, mesh), shard_batch_flat(bc, mesh)
    ref = shard_pi05_agent(_agent(), mesh)
    zero = shard_pi05_agent(_agent(), mesh, zero_actor_state=True)
    assert _spec(zero.init_actor_params['m']) == P('data')
    ref, _ = ref.update(sb, sb, sbc, False)
    zero, _ = zero.update(sb, sb, sbc, False)
    d_ref, d_zero = jax.device_get(ref.grad_diagnostics(sb, sbc)), jax.device_get(zero.grad_diagnostics(sb, sbc))
    for k in d_ref:
        np.testing.assert_allclose(d_ref[k], d_zero[k], rtol=1e-4, atol=1e-6, err_msg=k)
    assert d_zero['drift_online_init_rel'] > 0


def test_fix_b_sampling_unaffected():
    batch, _ = MD._batches()
    mesh = MD._mesh()
    ref = shard_pi05_agent(_agent(best_of_n=4), mesh)
    zero = shard_pi05_agent(_agent(best_of_n=4), mesh, zero_actor_state=True)
    rng = jax.random.PRNGKey(1)
    imgs, obs = batch['images'][:8], batch['observations'][:8]
    np.testing.assert_allclose(np.asarray(ref.sample_actions_BON(imgs, obs, rng)),
                               np.asarray(zero.sample_actions_BON(imgs, obs, rng)), atol=1e-5)
    np.testing.assert_allclose(np.asarray(ref.sample_actions_ode(imgs, rng)),
                               np.asarray(zero.sample_actions_ode(imgs, rng)), atol=1e-5)


def test_main_pi05_per_device_print_on_placed_agent():
    """main_pi05's size print: every leaf of a shard_pi05_agent output is a jax.Array (TrainState.step was int)."""
    mesh = MD._mesh()
    raw = _agent(with_v=True)
    assert any(not isinstance(x, jax.Array) for x in jax.tree.leaves(raw))  # python int step before placement
    for zero in (False, True):
        agent = shard_pi05_agent(_agent(with_v=True), mesh, zero_actor_state=zero)
        assert all(isinstance(x, jax.Array) for x in jax.tree.leaves(agent))
        per_dev = sum(x.addressable_shards[0].data.nbytes for x in jax.tree.leaves(agent))
        total = sum(x.nbytes for x in jax.tree.leaves(agent))
        assert (per_dev < total) == zero


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
