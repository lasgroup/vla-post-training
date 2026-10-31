"""CPU tests for data-parallel vector-env inference (`pi05_online_runner._Sharder`) and the workers' EGL device
assignment (`make_vectorized_env(egl_devices=...)`, `env.worker_egl_devices`), 2026-10-10.

  a. _Sharder: identity without a mesh; with a mesh, chunk rounding to the device count and P('data') input sharding.
  b. sharded vs unsharded inputs give the same encode / policy outputs through _VecEncoder and _chunked_policy
     (a jitted per-row toy agent, as the real agent's calls are per-row).
  c. _run_vector_waves with a sharder: same episodes / transitions / rng as without (StubAgent sees jax inputs);
     with a chunk that rounds up, the same up to the stub's key-dependent action column.
  d. run_pi05_round_rl builds a _Sharder over its mesh and passes it to collection and eval.
  e. make_vectorized_env(egl_devices=...) under a real spawned AsyncVectorEnv: slot i's worker sees
     MUJOCO_EGL_DEVICE_ID / EGL_DEVICE_ID = egl_devices[i % len]; None inherits the parent's.
  f. main_pi05: env.worker_egl_devices parsing ("" / auto / list / errors) and wiring into make_vectorized_env.

The sharded tests (a-d) need two host devices, so run this file on its own with the flag set before JAX starts:
  XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu uv run pytest tests/test_pi05_vector_sharded.py -q
In a wider pytest run (JAX already initialised with one device) they skip.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=2')  # before jax is imported

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

import ogpo.runners.pi05_online_runner as R
import ogpo.utils.main_helper as MH
from ogpo.utils.sharding import create_mesh
from tests._pi05_stubs import Logger, StubAgent, assert_call_logs_equal
from tests._pi05_vector_env import _reg, make_egl_probe_env
from tests.test_pi05_vector import GOOD, GOOD2, KW, _Built, _main, _traj_equal, _vt

two_devices = pytest.mark.skipif(jax.device_count() < 2,
                                 reason='needs XLA_FLAGS=--xla_force_host_platform_device_count=2 before JAX starts')


@pytest.fixture
def tiny_images(monkeypatch):
    monkeypatch.setattr(R.T, 'resize_images', lambda x: x)


# ===========================================================================
# a. _Sharder
# ===========================================================================

def test_sharder_identity_without_mesh():
    s = R._Sharder()
    assert s.mesh is None and s.n_dev == 1
    assert [s.chunk(k) for k in (1, 5, 16)] == [1, 5, 16]
    a, b = np.zeros((3, 2)), np.ones(3)
    out = s(a, b)
    assert out[0] is a and out[1] is b


@two_devices
def test_sharder_mesh_chunk_and_sharding():
    mesh = create_mesh('multi')
    s = R._Sharder(mesh)
    assert s.n_dev == 2
    assert [s.chunk(k) for k in (1, 2, 3, 7, 8)] == [2, 2, 4, 8, 8]
    imgs = np.arange(4 * 2 * 3 * 3 * 3, dtype=np.uint8).reshape(4, 2, 3, 3, 3)
    feats = np.arange(8, dtype=np.float32).reshape(4, 2)
    rows = np.array([0, 1, 1, 0], np.int32)
    for x, y in zip(s(imgs, feats, rows), (imgs, feats, rows)):
        assert isinstance(x, jax.Array) and np.array_equal(np.asarray(x), y) and x.dtype == y.dtype
        assert x.sharding == NamedSharding(mesh, P('data'))
        assert len(x.addressable_shards) == 2 and all(sh.data.shape[0] == 2 for sh in x.addressable_shards)


# ===========================================================================
# b. sharded == unsharded through the encoder / policy helpers
# ===========================================================================

class JitAgent:
    """Per-row jitted encode / policies, shaped like the real agent's calls; outputs depend on every input."""

    @staticmethod
    @jax.jit
    def encode(imgs, states, task_ids):
        x = jnp.mean(imgs.astype(jnp.float32), axis=(1, 2, 3, 4))[:, None]
        return jnp.concatenate([x, states * (task_ids[:, None].astype(jnp.float32) + 1.0)], axis=-1)

    @staticmethod
    @jax.jit
    def sample_actions_BON(imgs, feats, key, task_ids):
        u = jax.random.uniform(key)  # one scalar per call, as the key is not sharded
        base = feats[:, None, :1] * jnp.ones((feats.shape[0], 3, 7)) + task_ids[:, None, None].astype(jnp.float32)
        return base + u + jnp.mean(imgs.astype(jnp.float32), axis=(1, 2, 3, 4))[:, None, None]

    @staticmethod
    @jax.jit
    def sample_actions_ode(imgs, key, task_ids, critic_obs=None):  # critic_obs: the chunk-start features (droid delta reference)
        u = jax.random.uniform(key)
        x = jnp.mean(imgs.astype(jnp.float32), axis=(1, 2, 3, 4))[:, None, None]
        return x * jnp.ones((imgs.shape[0], 3, 7)) + u + task_ids[:, None, None].astype(jnp.float32)


@two_devices
@pytest.mark.parametrize('ib', [1, 2, 3, 7, 16])
def test_sharded_encoder_and_policy_match_unsharded(tiny_images, ib):
    mesh = create_mesh('multi')
    agent = JitAgent()
    rng = np.random.default_rng(0)
    n = 7
    imgs = rng.integers(0, 255, (n, 2, 4, 4, 3)).astype(np.uint8)
    states = rng.normal(size=(n, 8)).astype(np.float32)
    rows = (np.arange(n) % 3).astype(np.int32)
    obs = {'image': imgs[:, 0], 'wrist_image': imgs[:, 1], 'state': states}
    plain, sharded = R._Sharder(), R._Sharder(mesh)
    c = sharded.chunk(min(ib, n))  # same call width for both, so the rng cadence matches too
    assert c % 2 == 0 and c >= min(ib, n)
    i_a, f_a = R._VecEncoder(agent, rows, c, plain).batch(obs)
    i_b, f_b = R._VecEncoder(agent, rows, c, sharded).batch(obs)
    assert np.array_equal(i_a, imgs) and np.array_equal(i_b, imgs)
    assert f_a.shape == (n, 9) and np.allclose(f_a, f_b, atol=1e-6)
    for mode in ('bon', 'ode'):
        a, rng_a = R._chunked_policy(R._policy(agent, mode), i_a, f_a, rows, jax.random.PRNGKey(3), c, plain)
        b, rng_b = R._chunked_policy(R._policy(agent, mode), i_b, f_b, rows, jax.random.PRNGKey(3), c, sharded)
        assert a.shape == (n, 3, 7) and np.allclose(a, b, atol=1e-6) and np.array_equal(rng_a, rng_b)
    # the sharded path really is sharded: a direct call with its inputs returns P('data') outputs
    out = agent.encode(*sharded(imgs[:4], states[:4], rows[:4]))
    assert out.sharding == NamedSharding(mesh, P('data'))


# ===========================================================================
# c. _run_vector_waves with a sharder
# ===========================================================================

def _waves(sharder, ib, tiny=True):
    reg, log, eps = _reg(), [], []
    agent = StubAgent(log)
    vt = _vt([0, 1, 2], 2, reg, ib=ib)  # 6 slots
    rng = R._run_vector_waves(vt, 3, lambda k, w, j, k_w: 100 + w * 6 + k * k_w + j, agent, R._policy(agent, 'bon'),
                              jax.random.PRNGKey(5), clip_actions=False, record=True, on_episode=eps.append,
                              sharder=sharder)
    return eps, log, rng, reg


@two_devices
def test_run_vector_waves_sharded_matches_unsharded(tiny_images):
    mesh = create_mesh('multi')
    (ea, la, ra, rga), (eb, lb, rb_, rgb) = _waves(R._Sharder(), 6), _waves(R._Sharder(mesh), 6)  # chunk 6 both
    assert np.array_equal(ra, rb_) and rga['resets'] == rgb['resets']
    assert [(e['task'], e['length'], e['success'], e['ret']) for e in ea] == \
           [(e['task'], e['length'], e['success'], e['ret']) for e in eb]
    for x, y in zip(ea, eb):
        _traj_equal(x['trajectory'], y['trajectory'])
    assert_call_logs_equal(la, lb)  # the stub saw the same (now sharded) inputs, call for call
    assert {len(c[1]) for c in lb if c[0] in ('encode', 'bon')} == {6}


@two_devices
def test_run_vector_waves_sharded_chunk_rounds_up(tiny_images):
    """ib=5 over 6 slots: 5-row calls unsharded, one 6-row call sharded (rounded up to the 2 devices). Same
    episodes and transitions up to the stub's key-dependent action column (one rng split per call)."""
    mesh = create_mesh('multi')
    (ea, la, _, _), (eb, lb, _, _) = _waves(R._Sharder(), 5), _waves(R._Sharder(mesh), 5)
    assert {len(c[1]) for c in la if c[0] in ('encode', 'bon')} == {5}
    assert {len(c[1]) for c in lb if c[0] in ('encode', 'bon')} == {6}
    assert [(e['task'], e['length'], e['success'], e['ret']) for e in ea] == \
           [(e['task'], e['length'], e['success'], e['ret']) for e in eb]
    for x, y in zip(ea, eb):
        assert len(x['trajectory']) == len(y['trajectory'])
        for tx, ty in zip(x['trajectory'], y['trajectory']):
            for k in tx:
                ax, ay = np.array(tx[k]), np.array(ty[k])
                if k == 'actions':
                    ax[3] = ay[3] = 0.0
                assert np.array_equal(ax, ay), k


# ===========================================================================
# d. run_pi05_round_rl wires the mesh into a _Sharder
# ===========================================================================

@pytest.mark.parametrize('with_mesh', [False, True])
def test_round_rl_passes_sharder(tiny_images, monkeypatch, with_mesh):
    if with_mesh and jax.device_count() < 2:
        pytest.skip('needs two host devices')
    mesh = create_mesh('multi') if with_mesh else None
    seen = []
    orig = R._run_vector_waves
    monkeypatch.setattr(R, '_run_vector_waves', lambda *a, **kw: seen.append(kw.get('sharder')) or orig(*a, **kw))
    reg, log = _reg(), []
    env = _vt([0, 1], 1, reg, ib=16)
    R.run_pi05_round_rl(StubAgent(log), env, env, {'grad_diagnostics': True, 'ppo_batch_size': 2}, Logger(),
                        num_rounds=1, rollouts_per_round=2, updates_per_round=0, eval_interval=1, eval_episodes=1,
                        log_interval=2, actor_update_interval=2, mesh=mesh, **KW)
    assert len(seen) == 3  # initial eval (ode + bon) + round-0 collection
    assert all(isinstance(s, R._Sharder) and s.mesh is mesh for s in seen)
    assert {len(c[1]) for c in log if c[0] in ('encode', 'bon', 'ode')} == {2}


# ===========================================================================
# e. make_vectorized_env(egl_devices=...) in spawned workers
# ===========================================================================

def test_make_vectorized_env_egl_round_robin_spawn(monkeypatch):
    monkeypatch.setattr(MH, 'get_env_maker', lambda name: make_egl_probe_env)
    for egl, want in ((None, [7, 7, 7, 7]), ([3, 5], [3, 5, 3, 5]), ((1, 2, 4), [1, 2, 4, 1]), ([], [7, 7, 7, 7])):
        monkeypatch.setenv('MUJOCO_EGL_DEVICE_ID', '7')  # the parent's; gymnasium's slot-0 dummy env runs env_fn(0) here
        monkeypatch.setenv('EGL_DEVICE_ID', '7')
        vec = MH.make_vectorized_env(['a'] * 4, 4, seed=0, context='spawn', egl_devices=egl)
        try:
            obs = R._vec_reset(vec, np.ones(4, bool), [1, 2, 3, 4])
            assert obs['state'][:, 3].tolist() == want, (egl, obs['state'][:, 3])
            assert obs['state'][:, 4].tolist() == want, (egl, obs['state'][:, 4])
            assert obs['state'][:, 1].tolist() == [1, 2, 3, 4]
        finally:
            vec.close(terminate=True)


# ===========================================================================
# f. main_pi05: env.worker_egl_devices
# ===========================================================================

def test_worker_egl_devices_parsing(monkeypatch):
    import ogpo.main_pi05 as M
    assert M.worker_egl_devices('') is None and M.worker_egl_devices('  ') is None
    assert M.worker_egl_devices('0,2, 5') == [0, 2, 5] and M.worker_egl_devices(3) == [3]
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '4,6')
    assert M.worker_egl_devices('auto') == [4, 6]
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES')
    with pytest.raises(ValueError, match='CUDA_VISIBLE_DEVICES'):
        M.worker_egl_devices('auto')
    for bad in ('x', '0,y', ','):
        with pytest.raises(ValueError, match='worker_egl_devices'):
            M.worker_egl_devices(bad)


def test_main_vector_wiring_egl_devices(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,3')
    M, built = _main(monkeypatch, f'--env.name={GOOD},{GOOD2}', '--training.collection_mode=rounds',
                     '--env.vector_envs=true', '--env.parallel_envs=2', '--env.worker_egl_devices=auto')
    with pytest.raises(_Built):
        M.main()
    assert [v.kw['egl_devices'] for v in built['vecs']] == [[2, 3]]
    M, built = _main(monkeypatch, f'--env.name={GOOD}', '--training.collection_mode=rounds', '--env.vector_envs=true',
                     '--env.worker_egl_devices=0,1')
    with pytest.raises(_Built):
        M.main()
    assert [v.kw['egl_devices'] for v in built['vecs']] == [[0, 1]]
    M, built = _main(monkeypatch, f'--env.name={GOOD}', '--training.collection_mode=rounds', '--env.vector_envs=true')
    with pytest.raises(_Built):
        M.main()
    assert [v.kw['egl_devices'] for v in built['vecs']] == [None]
    M, _ = _main(monkeypatch, f'--env.name={GOOD}', '--training.collection_mode=rounds', '--env.worker_egl_devices=0')
    with pytest.raises(ValueError, match='worker_egl_devices'):
        M.main()
