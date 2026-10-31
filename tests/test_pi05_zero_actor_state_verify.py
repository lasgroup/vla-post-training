"""
Independent verifier, single device (docs/changes/2026-10-11-pi05-zero-actor-state/): with mesh=None (default) every
OGPOPi05Agent path is bit-identical to the pre-change file. The HEAD (565feff) ogpo/agents/ogpo_pi05.py is loaded
verbatim from git as a separate module; both classes get the same fake prefix / velocity (real SDE sampler, log-prob,
PPO + BC actor terms, critic, SARSA-V value step), are built from the same leaves, and are compared bit-exactly on
update / critic_update (plain + donated, accum 1 / 2, SARSA on / off, success buffer on / off), grad_diagnostics,
sample_actions_BON / _ode / _plain.

Run:  JAX_PLATFORMS=cpu python -m pytest tests/test_pi05_zero_actor_state_verify.py -q
"""
import copy
import importlib.util
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from ogpo.agents.modules.flax_utils import ModuleDict, TrainState
from ogpo.agents.ogpo_pi05 import OGPOPi05Agent, PREFIX_DIM, STATE_DIM, T
from ogpo.networks import Value
from ogpo.utils.config_loader import config_to_agent_dict, load_config

REPO = os.path.join(os.path.dirname(__file__), '..')
HEAD_REV = '565feff'


def _load_head():
    src = subprocess.run(['git', 'show', f'{HEAD_REV}:ogpo/agents/ogpo_pi05.py'], cwd=REPO, check=True,
                         capture_output=True, text=True).stdout
    assert 'zero_actor_state' not in src and 'mesh' not in src
    d = tempfile.mkdtemp()
    path = os.path.join(d, 'ogpo_pi05_head.py')
    with open(path, 'w') as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location('ogpo_pi05_head', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HEAD = _load_head()

B, PPO, BC_B, NUM_QS, NUM_VS, H_PI, HL = 8, 4, 8, 4, 3, 10, 5
OBS = PREFIX_DIM + STATE_DIM
L, S, KV, HD = 2, 3, 1, 2


class _FakeMixin:
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


class NewFake(_FakeMixin, OGPOPi05Agent):
    pass


class HeadFake(_FakeMixin, HEAD.OGPOPi05Agent):
    pass


def _config(**kw):
    c = config_to_agent_dict(load_config('ogpo', {}))
    c.update(pi05_action_horizon=H_PI, horizon_length=HL, num_qs=NUM_QS, v_ensemble_size=NUM_VS, batch_size=B,
             ppo_batch_size=PPO, group_num_samples=3, flow_steps=3, tau=0.3, actor_tau=0.2, donate_agent=False,
             grad_accum_steps=1, use_success_buffer_q=False, q_min=-100.0, q_max=0.0, adv_clip_min=None,
             use_bc_regularization=True, use_tapered_noise=True, use_constant_noise=False, constant_noise_std=0.1,
             q_variance_reduction=False, best_of_n=4)
    c.update(kw)
    return c


def _vdef(n):
    return Value(hidden_dims=(16, 16), layer_norm=True, num_ensembles=n, critic_loss_type='mse',
                 q_min=-100.0, q_max=0.0, obs_two_tier=True, two_tier_img_dim=PREFIX_DIM,
                 two_tier_proprio_dim=STATE_DIM, two_tier_fused_dim=0)


def _fields(with_v, **kw):
    k = jax.random.split(jax.random.PRNGKey(0), 4)
    qdef, vdef = _vdef(NUM_QS), _vdef(NUM_VS)
    qnet = ModuleDict({'critic': qdef, 'target_critic': copy.deepcopy(qdef)})
    vnet = ModuleDict({'value': vdef, 'target_value': copy.deepcopy(vdef)})
    exq = dict(observations=jnp.zeros((1, OBS)), actions=jnp.zeros((1, HL * T.ENV_ACTION_DIM)))
    exv = dict(observations=jnp.zeros((1, OBS)))
    qp = qnet.init(k[0], critic=exq, target_critic=exq)['params']
    qp['modules_target_critic'] = qnet.init(k[1], critic=exq, target_critic=exq)['params']['modules_critic']
    vp = vnet.init(k[2], value=exv, target_value=exv)['params']
    vp['modules_target_value'] = vnet.init(k[3], value=exv, target_value=exv)['params']['modules_value']
    dim_mask = np.zeros((H_PI, T.MODEL_ACTION_DIM), np.float32)
    dim_mask[:HL, :T.ENV_ACTION_DIM] = 1.0
    consts = {'tokens': jnp.zeros((1, 4), jnp.int32), 'dim_mask': jnp.asarray(dim_mask.reshape(-1)),
              'action_q01': -jnp.ones(T.MODEL_ACTION_DIM) * 0.8, 'action_q99': jnp.ones(T.MODEL_ACTION_DIM) * 1.3}
    params = {'w': jnp.float32(0.3), 'm': jnp.asarray(np.random.RandomState(1).randn(64, 8) * 0.01, jnp.float32)}
    actor_tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-2, b1=0.9, b2=0.95, weight_decay=1e-4))
    return dict(rng=jax.random.PRNGKey(3), actor_params=params,
                target_actor_params=jax.tree.map(lambda x: x + 0.1, params),
                actor_opt_state=actor_tx.init(params), init_actor_params=jax.tree.map(lambda x: x - 0.05, params),
                frozen_params={}, critic_network=TrainState.create(qnet, qp, optax.adam(1e-2)), consts=consts,
                config=_config(**kw), model=None, actor_tx=actor_tx,
                value_network=TrainState.create(vnet, vp, optax.adam(1e-2)) if with_v else None)


def _pair(with_v=False, **kw):
    # Two independent copies of identical leaves (donation must not alias between them).
    return NewFake(**_fields(with_v, **kw)), HeadFake(**_fields(with_v, **kw))


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
    bc = {'images': jnp.asarray(img), 'actions': jnp.asarray(r.uniform(-1, 1, (BC_B, H_PI, T.ENV_ACTION_DIM)),
                                                            jnp.float32),
          'valid': jnp.ones((BC_B, H_PI), jnp.float32)}
    return batch, bc


PYTREE_FIELDS = ('rng', 'actor_params', 'target_actor_params', 'actor_opt_state', 'init_actor_params',
                 'frozen_params', 'critic_network', 'consts', 'value_network')


def _eq(a, b):
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    assert len(la) == len(lb)
    for x, y in zip(la, lb):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def _eq_info(a, b):
    a, b = jax.device_get(a), jax.device_get(b)
    assert set(a) == set(b), set(a) ^ set(b)
    for k in a:
        np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]), err_msg=k)


def _eq_agent(n, h):
    for f in PYTREE_FIELDS:
        _eq(getattr(n, f), getattr(h, f))


def test_new_fields_are_static_and_leaves_unchanged():
    n, h = _pair(with_v=True)
    assert n.mesh is None and n.zero_actor_state is False
    assert len(jax.tree.leaves(n)) == len(jax.tree.leaves(h))
    for x, y in zip(jax.tree.leaves(n), jax.tree.leaves(h)):
        assert np.shape(x) == np.shape(y) and np.result_type(x) == np.result_type(y)
    # aux data differs only by the two static fields; they are hashable (jit static aux)
    hash(jax.tree.structure(n))
    leaves, tdef = jax.tree.flatten(n)
    back = jax.tree.unflatten(tdef, leaves)
    assert back.mesh is None and back.zero_actor_state is False


@pytest.mark.parametrize('with_v', [False, True])
@pytest.mark.parametrize('accum', [1, 2])
@pytest.mark.parametrize('sb', [False, True])
def test_update_and_critic_update_bit_identical_to_head(with_v, accum, sb):
    batch, bc = _batches()
    batch_s, _ = _batches(1)
    n, h = _pair(with_v=with_v, grad_accum_steps=accum, use_success_buffer_q=sb)
    for step in range(2):
        n, info_n = n.update(batch, batch_s, bc, sb)
        h, info_h = h.update(batch, batch_s, bc, sb)
        _eq_agent(n, h)
        _eq_info(info_n, info_h)
        n, info_n = n.critic_update(batch, batch_s, sb)
        h, info_h = h.critic_update(batch, batch_s, sb)
        _eq_agent(n, h)
        _eq_info(info_n, info_h)


@pytest.mark.parametrize('with_v', [False, True])
def test_donated_update_bit_identical_to_head(with_v):
    batch, bc = _batches()
    n, h = _pair(with_v=with_v, donate_agent=True, grad_accum_steps=2)
    for _ in range(2):
        n, info_n = n.update(batch, batch, bc, False)
        h, info_h = h.update(batch, batch, bc, False)
        _eq_agent(n, h)
        _eq_info(info_n, info_h)
        n, _ = n.critic_update(batch, batch, False)
        h, _ = h.critic_update(batch, batch, False)
        _eq_agent(n, h)


@pytest.mark.parametrize('accum', [1, 2])
def test_grad_diagnostics_bit_identical_to_head(accum):
    batch, bc = _batches()
    n, h = _pair(grad_accum_steps=accum)
    _eq_info(n.grad_diagnostics(batch, bc), h.grad_diagnostics(batch, bc))


@pytest.mark.parametrize('sampler', ['sde', 'ode'])
def test_sampling_bit_identical_to_head(sampler):
    batch, _ = _batches()
    n, h = _pair(pi05_bon_sampler=sampler)
    rng = jax.random.PRNGKey(11)
    np.testing.assert_array_equal(np.asarray(n.sample_actions_BON(batch['images'], batch['observations'], rng)),
                                  np.asarray(h.sample_actions_BON(batch['images'], batch['observations'], rng)))
    np.testing.assert_array_equal(np.asarray(n.sample_actions_ode(batch['images'], rng)),
                                  np.asarray(h.sample_actions_ode(batch['images'], rng)))
    np.testing.assert_array_equal(np.asarray(n.sample_actions_plain(batch['images'], rng)),
                                  np.asarray(h.sample_actions_plain(batch['images'], rng)))


def test_head_vs_new_has_teeth():
    """A different rng in the new agent must break equality (the comparison is not vacuous)."""
    batch, bc = _batches()
    n, h = _pair()
    n = n.replace(rng=jax.random.PRNGKey(4))
    n, _ = n.update(batch, batch, bc, False)
    h, _ = h.update(batch, batch, bc, False)
    with pytest.raises(AssertionError):
        _eq(n.actor_params, h.actor_params)


# --- main_pi05 wiring --------------------------------------------------------------------------------------------

def _main_pi05(monkeypatch, devices, *extra):
    from types import SimpleNamespace
    import ogpo.main_pi05 as M
    monkeypatch.setattr(sys, 'argv', ['main_pi05.py', '--algo=ogpo', '--experiment.log=false',
                                      '--agent.pi05_action_horizon=10', '--env.name=libero_pi05-libero_90-1', *extra])
    built = {'order': []}

    class _Stop(Exception):
        pass

    def make_env(name, seed, post_success_steps):
        built['order'].append('env')
        return SimpleNamespace(name=name, seed=seed, prompt=f'prompt:{name}')

    def create(cls, seed, prompt, config):
        built['order'].append('agent')
        return SimpleNamespace(tag='agent')

    def shard(agent, mesh, zero_actor_state=False):
        built['order'].append('shard')
        built['zero'] = zero_actor_state
        return {'x': jnp.zeros((8, 4))}  # leaves must be jax Arrays for main's per-device size print

    def run(agent, env, eval_env, config, logger, **kw):
        built['order'].append('run')
        built['mesh'] = kw['mesh']
        raise _Stop()
    monkeypatch.setattr(M, 'make_env', make_env)
    monkeypatch.setattr(M.OGPOPi05Agent, 'create', classmethod(create))
    monkeypatch.setattr(M, 'create_mesh', lambda mode: None if devices is None else
                        SimpleNamespace(devices=np.zeros(devices)))
    monkeypatch.setattr(M, 'shard_pi05_agent', shard)
    monkeypatch.setattr(M, 'setup_experiment_logging', lambda **kw: (None, SimpleNamespace(close=lambda: None), None))
    monkeypatch.setattr(M, 'run_pi05_round_rl', run)
    monkeypatch.setattr(M, 'run_pi05_online_rl', run)
    return M, built, _Stop


@pytest.mark.parametrize('flag', [None, 'false', 'true'])
def test_main_pi05_forwards_zero_actor_state(monkeypatch, flag):
    extra = [] if flag is None else [f'--training.zero_actor_state={flag}']
    M, built, stop = _main_pi05(monkeypatch, 4, '--training.gpu_mode=multi', *extra)
    with pytest.raises(stop):
        M.main()
    assert built['zero'] is (flag == 'true')   # parsed to a real bool, default false
    assert built['order'] == ['env', 'env', 'agent', 'shard', 'run']


def test_main_pi05_zero_without_mesh_raises(monkeypatch):
    """gpu_mode=single: raised before any env / agent work (the early check)."""
    M, built, stop = _main_pi05(monkeypatch, None, '--training.zero_actor_state=true')
    with pytest.raises(ValueError, match='zero_actor_state'):
        M.main()
    assert built['order'] == []


def test_main_pi05_zero_multi_but_single_device_raises_late(monkeypatch):
    """gpu_mode=multi with one visible device: create_mesh -> None, caught by the mesh check (after env / agent)."""
    M, built, stop = _main_pi05(monkeypatch, None, '--training.zero_actor_state=true', '--training.gpu_mode=multi')
    with pytest.raises(ValueError, match='zero_actor_state'):
        M.main()
    assert 'shard' not in built['order'] and 'run' not in built['order']


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
