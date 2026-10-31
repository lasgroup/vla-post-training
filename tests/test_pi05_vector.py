"""
CPU tests for the pi05 vector-env path (docs/changes/2026-10-08-pi05-async-vector-env).

  a. _run_vector_waves vs the in-process _run_waves at T=1 (m=1, m=3; full and partial last wave; record / eval):
     identical transitions, reset seeds, executed actions, lengths / successes / on_episode order, rng, agent calls.
  b. T=3, m=2, partial last wave: (task, seed) assignment equals the in-process round loop's; per-slot prompt rows
     reach encode / policy / stored transitions; episodes contiguous in the buffer; eval metrics per task.
  c. run_pi05_round_rl with a VectorTasks (SyncVectorEnv shim) vs the in-process list of envs: same buffer up to
     the action-noise column, same reset seeds, same env/*, eval/*, eval_sde/* logs (incl. success/<task>).
  d. inference_batch chunking: constant call shape, same result as one big call.
  e. make_vectorized_env / get_env_maker.
  f. main_pi05 validation and vector wiring (incl. try/finally close).
  g. real AsyncVectorEnv(context="spawn"): masked reset / seed lists, differential vs the Sync shim, step timeout,
     VectorSlotWrapper inside a worker.
  h. VectorSlotWrapper freeze / resize semantics.

The Sync shim clears `_autoreset_envs` before each step (gymnasium 1.2.3 SyncVectorEnv(DISABLED) asserts on
stepping a finished slot; AsyncVectorEnv does not) and zero-initialises unset slot obs (as AsyncVectorEnv's shared
memory is), so a masked reset before every slot was reset works as it does under AsyncVectorEnv.

Run:  JAX_PLATFORMS=cpu uv run pytest tests/test_pi05_vector.py -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import functools
import multiprocessing
from collections import Counter
from types import SimpleNamespace

import gymnasium as gym
import jax
import numpy as np
import pytest
from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv
from gymnasium.vector.utils import create_empty_array

import ogpo.runners.pi05_online_runner as R
from ogpo.networks.pi05 import transforms as PT
from tests._pi05_stubs import Logger, StubAgent, assert_call_logs_equal, assert_logs_equal, episodes_in_buffer
from tests._pi05_vector_env import FakeLiberoEnv, SlotEnv, _reg, ep_len, make_slot_env, make_wrapped_fake


@pytest.fixture
def tiny_images(monkeypatch):
    monkeypatch.setattr(R.T, 'resize_images', lambda x: x)


class SyncShim(SyncVectorEnv):
    def __init__(self, env_fns):
        super().__init__(env_fns, autoreset_mode=AutoresetMode.DISABLED)
        zero = create_empty_array(self.single_observation_space, n=1, fn=np.zeros)
        self._env_obs = [{k: v[0].copy() for k, v in zero.items()} for _ in range(self.num_envs)]

    def step(self, actions):
        self._autoreset_envs[:] = False
        return super().step(actions)


def _vt(rows, m, reg, *, ib=16, names=None):
    """VectorTasks over a Sync shim; slot env task = prompt row (so episodes_in_buffer checks task_ids == row)."""
    vec = SyncShim([functools.partial(SlotEnv, row, reg) for row in rows for _ in range(m)])
    return R.VectorTasks(vec, names or [f't{r}' for r in rows], list(rows), m, ib)


def _traj_equal(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.keys() == y.keys()
        for k in x:
            assert np.array_equal(np.asarray(x[k]), np.asarray(y[k])), k
            assert np.asarray(x[k]).dtype == np.asarray(y[k]).dtype or k == 'rewards', (k, np.asarray(x[k]).dtype,
                                                                                        np.asarray(y[k]).dtype)


# ===========================================================================
# a. differential vs _run_waves at T=1
# ===========================================================================

@pytest.mark.parametrize('m,n', [(1, 4), (3, 6), (3, 7), (3, 2)])
@pytest.mark.parametrize('record,mode', [(True, 'bon'), (False, 'ode'), (False, 'bon')])
def test_vector_waves_match_run_waves_T1(tiny_images, m, n, record, mode):
    B, row = 500, 2
    # in-process
    reg_a, log_a, eps_a = _reg(), [], []
    agent_a = StubAgent(log_a)
    envs = [SlotEnv(row, reg_a) for _ in range(m)]
    rng_a = R._run_waves(envs, n, lambda i: B + i, R._Encoder(agent_a, row, m), R._policy(agent_a, mode),
                         jax.random.PRNGKey(7), clip_actions=not record, record=record, on_episode=eps_a.append)
    # vector
    reg_b, log_b, eps_b = _reg(), [], []
    agent_b = StubAgent(log_b)
    vt = _vt([row], m, reg_b)
    rng_b = R._run_vector_waves(vt, n, lambda k, w, j, k_w: B + w * m + j, agent_b, R._policy(agent_b, mode),
                                jax.random.PRNGKey(7), clip_actions=not record, record=record,
                                on_episode=eps_b.append)

    assert np.array_equal(rng_a, rng_b)
    assert reg_a['resets'] == reg_b['resets'] == [(row, B + i) for i in range(n)]
    assert len(reg_a['actions']) == len(reg_b['actions'])
    for p, q in zip(reg_a['actions'], reg_b['actions']):
        assert p[:3] == q[:3] and np.array_equal(p[3], q[3])
    assert [(e['length'], e['success'], e['ret']) for e in eps_a] == [(e['length'], e['success'], e['ret'])
                                                                      for e in eps_b]
    assert [e['length'] for e in eps_b] == [e['info']['length'] for e in eps_b]
    for ea, eb in zip(eps_a, eps_b):
        _traj_equal(ea['trajectory'], eb['trajectory'])
        assert eb['task'] == 0 and not any(k.startswith('_') for k in eb['info'])
    # completion order equals the in-process one (checked above); lengths match the env
    assert sorted(e['length'] for e in eps_b) == sorted(ep_len(B + i) for i in range(n))
    if n % m == 0:  # no padded rows: the agent sees exactly the same calls (args, task rows, RNG keys)
        assert_call_logs_equal(log_a, log_b)
    else:  # padded rows differ (stale slot obs vs repeated last row); kinds and RNG keys still match
        assert [c[0] for c in log_a] == [c[0] for c in log_b]
        key_idx = {'bon': 3, 'ode': 2}
        for ca, cb in zip(log_a, log_b):
            if ca[0] in key_idx:
                assert np.array_equal(ca[key_idx[ca[0]]], cb[key_idx[cb[0]]])
    # idle slots never stepped the sim with a real action (SlotEnv asserts zero actions on frozen slots)
    if n < m:
        # never-reset slots are stepped (zero action) through the "sim" while the wave runs
        assert reg_b['unreset_steps'] > 0


# ===========================================================================
# b. T=3, m=2, partial last wave
# ===========================================================================

def _old_round_seeds(rows, m, n, base):
    """(task row, seed) per episode, as the in-process round loop assigns them."""
    out, episodes = [], base
    for w in range(0, n, m):
        for row in rows:
            k = min(m, n - w)
            out += [(row, episodes + i) for i in range(k)]
            episodes += k
    return out


@pytest.mark.parametrize('n', [5, 4, 1])
def test_vector_waves_T3_seeds_rows_contiguity(tiny_images, n):
    rows, m, S, start = [2, 0, 1], 2, 3, 6  # rows != task index; `start` = episodes before this round
    reg_b, log, eps = _reg(), [], []
    agent = StubAgent(log)
    vt = _vt(rows, m, reg_b)
    T_ = len(rows)
    R._run_vector_waves(vt, n, lambda k, w, j, k_w: S + 10000 + start + w * T_ * m + k * k_w + j, agent,
                        R._policy(agent, 'bon'), jax.random.PRNGKey(0), clip_actions=False, record=True,
                        on_episode=eps.append)
    expected = _old_round_seeds(rows, m, n, S + 10000 + start)
    assert Counter(reg_b['resets']) == Counter(expected)      # same (task, seed) pairs, not only the same seeds
    assert len(set(s for _, s in expected)) == len(expected)
    # every call carries the per-slot rows, full width (6 rows: one call covers all tasks)
    slot_rows = np.repeat(rows, m)
    for c in log:
        if c[0] == 'encode':
            assert np.array_equal(c[3], slot_rows)
        if c[0] == 'bon':
            assert np.array_equal(c[4], slot_rows)
    # episode -> task index / row consistent; transitions stamped with the slot's row
    for e in eps:
        assert all(int(t['task_ids']) == rows[e['task']] for t in e['trajectory'])
        assert int(e['trajectory'][0]['observations'][0]) == rows[e['task']]
        assert e['length'] == ep_len(int(e['trajectory'][0]['observations'][1])) == len(e['trajectory'])
    # flushed through the real buffer: contiguous, self-consistent episodes
    rb = R._make_buffer(eps[0]['trajectory'][0]['observations'], eps[0]['trajectory'][0]['images'], 1000, 0, 7)
    for e in eps:
        R._flush_episode(rb, e['trajectory'], e['info'], 0.99)
    assert Counter(episodes_in_buffer(rb)) == Counter(expected)


def test_evaluate_vector_matches_per_task(tiny_images):
    rows, m, n, step = [0, 1, 2], 2, 5, 9
    vt = _vt(rows, m, _reg(), names=['a', 'b', 'c'])
    agent = StubAgent([])
    for mode in ('ode', 'bon'):
        got = R.evaluate_pi05_vector(agent, vt, step, n, mode)
        for name, row in zip(['a', 'b', 'c'], rows):
            ref = R.evaluate_pi05(agent, [SlotEnv(row, _reg()) for _ in range(m)], step, n, mode, task_id=row)
            assert got[name] == ref


# ===========================================================================
# c. full run_pi05_round_rl: VectorTasks vs in-process
# ===========================================================================

KW = dict(batch_size=2, success_buffer_batch_size=2, horizon_length=3, bc_horizon=4, discount=0.99,
          buffer_size=2000, seed=3)


def _round_run(vector, train_rows, eval_rows, *, m, rollouts, ib=16, reuse_eval=False, NR=2, EV=3):
    tr, ev, log = _reg(), _reg(), []
    agent, logger = StubAgent(log), Logger()
    if vector:
        env = _vt(train_rows, m, tr, ib=ib)
        eval_env = env if reuse_eval else _vt(eval_rows, m, ev, ib=ib)
    else:
        env = [(f't{r}', [SlotEnv(r, tr) for _ in range(m)], r) for r in train_rows]
        eval_env = [(f't{r}', [SlotEnv(r, ev) for _ in range(m)], r) for r in eval_rows]
    _, rb = R.run_pi05_round_rl(agent, env, eval_env, {'grad_diagnostics': True, 'ppo_batch_size': 2}, logger,
                                num_rounds=NR, rollouts_per_round=rollouts, updates_per_round=3, eval_interval=3,
                                eval_episodes=EV, log_interval=2, actor_update_interval=2, **KW)
    return SimpleNamespace(rb=rb, log=log, logs=logger.logs, tr=tr, ev=ev)


def _transitions(rb):
    out = {}
    for i in range(rb.size):
        o = rb['observations'][i]
        key = (int(rb['task_ids'][i]), int(o[1]), int(o[2]))
        assert key not in out
        rec = {k: np.array(rb[k][i]) for k in rb._dict}
        rec['actions'][3] = 0.0  # stub action-noise column (rng cadence differs across tasks by design)
        rec['success_mask'] = rb._traj_success_mask[i]
        out[key] = rec
    return out


@pytest.mark.parametrize('m,rollouts,ib,reuse', [(2, 5, 16, False), (2, 4, 4, False), (1, 3, 2, True),
                                                 (3, 2, 16, True)])
def test_round_vector_matches_in_process(tiny_images, m, rollouts, ib, reuse):
    rows = [0, 1, 2]
    a = _round_run(False, rows, rows, m=m, rollouts=rollouts)
    b = _round_run(True, rows, rows, m=m, rollouts=rollouts, ib=ib, reuse_eval=reuse)
    ta, tb = _transitions(a.rb), _transitions(b.rb)
    assert ta.keys() == tb.keys()
    for k in ta:
        for f in ta[k]:
            assert np.array_equal(ta[k][f], tb[k][f]), (k, f)
    episodes_in_buffer(b.rb)  # contiguous episodes in the vector run's buffer too
    tr_b = [x for x in b.tr['resets'] if x[1] >= 10000] if reuse else b.tr['resets']
    assert Counter(a.tr['resets']) == Counter(tr_b)
    ev_b = b.tr if reuse else b.ev
    ev_resets_b = ev_b['resets'] if not reuse else [x for x in ev_b['resets'] if x[1] < 10000]
    assert Counter(a.ev['resets']) == Counter(ev_resets_b)
    for prefix in ('env', 'eval', 'eval_sde'):
        la = [x for x in a.logs if x[0] == prefix]
        lb = [x for x in b.logs if x[0] == prefix]
        assert la and len(la) == len(lb)
        assert_logs_equal(la, lb)
    for p, _, d in b.logs:
        if p in ('eval', 'eval_sde'):
            assert set(d) == {'success', 'return', 'length', 'success/t0', 'success/t1', 'success/t2'}
        if p == 'env':
            assert {'success/t0', 'success/t1', 'success/t2'} <= set(d)
    # inference: one call width per run (ib-capped), all rows' prompt ids
    n = len(rows) * m
    widths = {len(c[1]) for c in b.log if c[0] in ('encode', 'bon', 'ode')}
    assert widths == {min(ib, n)}


def test_round_vector_single_task_eval_only_task(tiny_images):
    # train on one task, eval on it plus an eval-only task on a later prompt row
    a = _round_run(False, [0], [0, 1], m=2, rollouts=3)
    b = _round_run(True, [0], [0, 1], m=2, rollouts=3)
    for prefix in ('env', 'eval', 'eval_sde'):
        assert_logs_equal([x for x in a.logs if x[0] == prefix], [x for x in b.logs if x[0] == prefix])
    assert all('success/t1' in d for p, _, d in b.logs if p == 'eval')
    assert all(not any(k.startswith('success/') for k in d) for p, _, d in b.logs if p == 'env')


# ===========================================================================
# d. inference_batch chunking
# ===========================================================================

def test_chunked_policy_and_encoder_equal_one_call(tiny_images):
    rng = np.random.default_rng(0)
    n = 7
    imgs = rng.integers(0, 255, (n, 2, 4, 4, 3)).astype(np.uint8)
    feats = rng.normal(size=(n, 8)).astype(np.float32)
    rows = np.arange(n, dtype=np.int32) % 3
    shapes = []

    def policy(i, f, ids, key):  # deterministic in the rows, independent of the key
        shapes.append((i.shape[0], f.shape[0], ids.shape[0]))
        out = np.zeros((i.shape[0], 3, 7), np.float32)
        out[..., 0] = f[:, None, 0]
        out[..., 1] = ids[:, None]
        out[..., 2] = i[:, None, 0, 0, 0, 0]
        return out
    ref, rng_ref = R._chunked_policy(policy, imgs, feats, rows, jax.random.PRNGKey(1), n)
    for c in (1, 2, 3, 4, 6):
        shapes.clear()
        got, rng_c = R._chunked_policy(policy, imgs, feats, rows, jax.random.PRNGKey(1), c)
        assert np.array_equal(got, ref)
        assert set(shapes) == {(c, c, c)} and len(shapes) == -(-n // c)
        # one split per call
        k = jax.random.PRNGKey(1)
        for _ in range(-(-n // c)):
            k, _ = jax.random.split(k)
        assert np.array_equal(rng_c, k)

    log = []
    agent = StubAgent(log)
    obs = {'image': imgs[:, 0], 'wrist_image': imgs[:, 1], 'state': feats}
    i_ref, f_ref = R._VecEncoder(agent, rows, n).batch(obs)
    for c in (1, 3, 4):
        log.clear()
        i_c, f_c = R._VecEncoder(agent, rows, c).batch(obs)
        assert np.array_equal(i_c, i_ref) and np.array_equal(f_c, f_ref)
        assert all(len(e[1]) == c and len(e[3]) == c for e in log)
        assert np.array_equal(np.concatenate([e[3] for e in log])[:n], rows)


# ===========================================================================
# e. make_vectorized_env / get_env_maker
# ===========================================================================

def test_get_env_maker_and_make_vectorized_env():
    # subprocess: get_env_maker imports torch (envs.env_utils); torch + a later XLA compile in this process aborts
    import subprocess
    root = os.path.join(os.path.dirname(__file__), '..')
    out = subprocess.run([sys.executable, '-c', 'import sys; sys.path.insert(0, "."); '
                          'from tests._pi05_vector_env import check_env_maker; check_env_maker()'],
                         cwd=root, capture_output=True, text=True, env={**os.environ, 'JAX_PLATFORMS': 'cpu'})
    assert out.returncode == 0 and 'CHECK_ENV_MAKER_OK' in out.stdout, out.stderr[-3000:]


# ===========================================================================
# f. main_pi05
# ===========================================================================

GOOD, GOOD2, GOOD7 = 'libero_pi05-libero_90-1', 'libero_pi05-libero_90-2', 'libero_pi05-libero_90-7'


class _Built(Exception):
    pass


def _main(monkeypatch, *extra, finish=False, devices=None):
    import ogpo.main_pi05 as M
    monkeypatch.setattr(sys, 'argv', ['main_pi05.py', '--algo=ogpo', '--experiment.log=false',
                                      '--agent.pi05_action_horizon=10', '--training.num_rounds=1',
                                      '--training.rollouts_per_round=2', '--training.updates_per_round=1', *extra])
    built = {'vecs': [], 'envs': []}

    class FakeVec:
        def __init__(self, names, n, **kw):
            self.names, self.n, self.kw, self.closed = list(names), n, kw, []

        def close(self, terminate=False):
            self.closed.append(terminate)

    def make_vec(names, n, **kw):
        built['vecs'].append(FakeVec(names, n, **kw))
        return built['vecs'][-1]

    def create(cls, seed, prompt, config):
        built['prompt'] = prompt
        return SimpleNamespace(tag='agent')

    def run(agent, env, eval_env, config, logger, **kw):
        built.update(env=env, eval_env=eval_env, kw=kw)
        if not finish:
            raise _Built()
    monkeypatch.setattr(M, 'make_vectorized_env', make_vec)
    monkeypatch.setattr(M, 'get_prompt', lambda n: f'prompt:{n}')
    monkeypatch.setattr(M, 'make_env', lambda *a, **k: built['envs'].append(a) or _Built())
    monkeypatch.setattr(M.OGPOPi05Agent, 'create', classmethod(create))
    monkeypatch.setattr(M, 'create_mesh', lambda mode: None if devices is None else
                        SimpleNamespace(devices=np.zeros(devices)))
    monkeypatch.setattr(M, 'shard_pi05_agent', lambda agent, mesh, zero_actor_state=False: agent)
    monkeypatch.setattr(M, 'setup_experiment_logging', lambda **kw: (None, SimpleNamespace(close=lambda: None), None))
    monkeypatch.setattr(M, 'run_pi05_round_rl', run)
    monkeypatch.setattr(M, 'run_pi05_online_rl', run)
    return M, built


@pytest.mark.parametrize('extra,match', [
    ([f'--env.name={GOOD}', '--env.vector_envs=true'], 'vector_envs'),
    ([f'--env.name={GOOD}', '--env.vector_envs=true', '--training.collection_mode=online'], 'vector_envs'),
    ([f'--env.name={GOOD}', '--env.inference_batch=0', '--training.collection_mode=rounds'], 'inference_batch'),
])
def test_main_vector_validation(monkeypatch, extra, match):
    M, built = _main(monkeypatch, *extra)
    with pytest.raises(ValueError, match=match):
        M.main()
    assert not built['vecs'] and not built['envs']


def test_main_vector_wiring_separate_eval(monkeypatch):
    M, built = _main(monkeypatch, f'--env.name={GOOD},{GOOD2}', f'--env.eval_name={GOOD2},{GOOD7}',
                     '--env.vector_envs=true', '--env.parallel_envs=2', '--env.inference_batch=5',
                     '--env.post_success_steps=4', '--training.collection_mode=rounds')
    with pytest.raises(_Built):
        M.main()
    assert not built['envs']  # no in-process sim built
    assert built['prompt'] == [f'prompt:{GOOD}', f'prompt:{GOOD2}', f'prompt:{GOOD7}']
    env, ev = built['env'], built['eval_env']
    assert isinstance(env, M.VectorTasks) and isinstance(ev, M.VectorTasks) and env is not ev
    assert (env.names, env.rows, env.m, env.inference_batch) == ([GOOD, GOOD2], [0, 1], 2, 5)
    assert (ev.names, ev.rows, ev.m, ev.inference_batch) == ([GOOD2, GOOD7], [1, 2], 2, 5)
    assert env.vec.names == [GOOD, GOOD, GOOD2, GOOD2] and env.vec.n == 4
    assert ev.vec.names == [GOOD2, GOOD2, GOOD7, GOOD7]
    for v in (env.vec, ev.vec):
        assert v.kw['context'] == 'spawn' and v.kw['env_kwargs'] == {'vector': True}
        assert v.kw['post_success_steps'] == 4
        assert v.closed == [True]  # runner raised -> terminate, each vec closed once


@pytest.mark.parametrize('names', [[GOOD], [GOOD, GOOD2]])
def test_main_vector_reuses_eval_and_closes_cleanly(monkeypatch, names):
    M, built = _main(monkeypatch, f'--env.name={",".join(names)}', '--env.vector_envs=true',
                     '--training.collection_mode=rounds', finish=True)
    M.main()
    assert built['env'] is built['eval_env'] and len(built['vecs']) == 1
    assert built['vecs'][0].closed == [False]  # finished -> graceful close, once
    assert built['prompt'] == [f'prompt:{n}' for n in names]
    assert built['env'].m == 1 and built['env'].inference_batch == 16


# ===========================================================================
# g. real AsyncVectorEnv(context="spawn")
# ===========================================================================

def _async(fns):
    return AsyncVectorEnv(fns, autoreset_mode=AutoresetMode.DISABLED, context='spawn')


def test_async_spawn_reset_mask_and_waves(tiny_images):
    rows, m, n = [0, 1], 2, 3
    vec = _async([functools.partial(make_slot_env, r) for r in rows for _ in range(m)])
    try:
        obs = R._vec_reset(vec, np.array([False, True, False, False]), [11, 12, 13, 14])
        assert obs['state'][1, 1] == 12 and obs['state'][1, 0] == 0
        assert not obs['state'][[0, 2, 3]].any()  # untouched slots: zero shared memory
        obs = R._vec_reset(vec, np.array([True, False, False, True]), [21, 22, 23, 24])
        assert (obs['state'][0, 1], obs['state'][1, 1], obs['state'][3, 1]) == (21, 12, 24)  # slot 1 kept
        assert obs['state'][3, 0] == 1
        o, r, te, tr, info = R._vec_step(vec, np.zeros((4, 7), np.float32))
        assert o['state'][0, 2] == 1 and r.shape == (4,)

        runs = []
        for v in (vec, SyncShim([functools.partial(SlotEnv, r, _reg()) for r in rows for _ in range(m)])):
            eps, log = [], []
            agent = StubAgent(log)
            vt = R.VectorTasks(v, ['a', 'b'], rows, m, 3)
            rng = R._run_vector_waves(vt, n, lambda k, w, j, k_w: 100 + w * 4 + k * k_w + j, agent,
                                      R._policy(agent, 'bon'), jax.random.PRNGKey(5), clip_actions=False,
                                      record=True, on_episode=eps.append)
            runs.append((eps, log, rng))
        (ea, la, ra), (eb, lb, rb_) = runs
        assert np.array_equal(ra, rb_)
        assert [(e['task'], e['length'], e['success'], e['ret']) for e in ea] == \
               [(e['task'], e['length'], e['success'], e['ret']) for e in eb]
        for x, y in zip(ea, eb):
            _traj_equal(x['trajectory'], y['trajectory'])
            assert x['info'].keys() == y['info'].keys()
        seeds = sorted(int(e['trajectory'][0]['observations'][1]) for e in ea)
        assert seeds == [100, 101, 102, 103, 104, 105]
        assert_call_logs_equal(la, lb)
    finally:
        vec.close(terminate=True)


def test_async_spawn_step_timeout(monkeypatch):
    vec = _async([functools.partial(make_slot_env, 0, slow_on=-7.0) for _ in range(2)])
    try:
        R._vec_reset(vec, np.array([True, True]), [1, 2])
        monkeypatch.setattr(R, '_WORKER_TIMEOUT_S', 1.0)
        a = np.zeros((2, 7), np.float32)
        a[1, 0] = -7.0
        with pytest.raises(multiprocessing.TimeoutError):
            R._vec_step(vec, a)
    finally:
        vec.close(terminate=True)


def test_async_spawn_vector_slot_wrapper():
    vec = _async([functools.partial(make_wrapped_fake, 2) for _ in range(2)])
    try:
        assert vec.single_observation_space['image'].shape == (224, 224, 3)
        assert vec.single_observation_space['state'].shape == (8,)
        obs = R._vec_reset(vec, np.array([True, True]), [3, 4])
        ref = FakeLiberoEnv(2)
        raw, _ = ref.reset(seed=4)
        assert obs['image'].shape == (2, 224, 224, 3) and obs['image'].dtype == np.uint8
        assert np.array_equal(obs['image'][1], PT.resize_images(raw['image']))
        assert np.array_equal(obs['wrist_image'][1], PT.resize_images(raw['wrist_image']))
        steps = []
        for _ in range(4):
            o, r, te, tr, info = R._vec_step(vec, np.zeros((2, 7), np.float32))
            steps.append(info['inner_steps'].copy())
        assert [s.tolist() for s in steps] == [[1, 1], [2, 2], [2, 2], [2, 2]]  # frozen after done
        assert te.all() and np.array_equal(info['length'], [2, 2])
    finally:
        vec.close(terminate=True)


# ===========================================================================
# h. VectorSlotWrapper
# ===========================================================================

def test_vector_slot_wrapper_freeze_and_resize():
    from envs.libero_pi05_utils import VectorSlotWrapper
    inner = FakeLiberoEnv(horizon=3)
    env = VectorSlotWrapper(inner)
    assert env.observation_space['image'] == gym.spaces.Box(0, 255, (224, 224, 3), np.uint8)
    assert env.observation_space['wrist_image'].shape == (224, 224, 3)
    assert env.observation_space['state'] is inner.observation_space['state']
    assert env.observation_space['full_state'] is inner.observation_space['full_state']
    assert inner.observation_space['image'].shape == (256, 256, 3)  # inner space untouched

    obs, info = env.reset(seed=5, options={'x': 1})
    assert info == {'seed': 5} and inner.last_reset == 5
    raw = FakeLiberoEnv(3)
    raw_obs, _ = raw.reset(seed=5)
    assert np.array_equal(obs['image'], PT.resize_images(raw_obs['image']))
    assert np.array_equal(obs['state'], raw_obs['state'])
    outs = [env.step(np.zeros(7)) for _ in range(3)]
    assert outs[-1][2] is True and inner.inner_steps == 3
    for _ in range(4):
        again = env.step(np.ones(7))
        assert again is outs[-1]
    assert inner.inner_steps == 3
    obs2, _ = env.reset(seed=6)
    assert env.step(np.zeros(7))[2] is False and inner.inner_steps == 4
    assert obs2['image'].shape == (224, 224, 3)


def test_vector_slot_wrapper_truncation_freezes():
    from envs.libero_pi05_utils import VectorSlotWrapper

    class Trunc(FakeLiberoEnv):
        def step(self, action):
            o, r, te, tr, info = super().step(action)
            return o, r, False, te, info
    env = VectorSlotWrapper(Trunc(horizon=2))
    env.reset(seed=0)
    env.step(np.zeros(7))
    last = env.step(np.zeros(7))
    assert last[3] is True and env.step(np.zeros(7)) is last and env.env.inner_steps == 2
