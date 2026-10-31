"""
Verifier tests for the pranav/vla-ogpo merge (docs/changes/2026-10-30-merge-vla-branches/): the places where the
molmo-vla side (resume_rebase_rounds / step_offset, live-snapshot resume, buffer.storage_dir memmaps, kv_only prefix
passes) meets the pi05-awr side (AWR round schedule, lambda_returns buffer column, sample_actions_explore).

  a. run_pi05_round_rl under awr with a REBASED ResumeStore (step_offset != 0): the V fit runs at every round start
     (`fresh`), is skipped only on a mid-round resume, and every round runs exactly updates_per_round policy steps.
     Pre-merge AWR code (`fresh = step == r * updates_per_round`) fails the first and third case.
  b. run_pi05_round_rl under awr from a live snapshot (real `_resume_snapshot`, `save_buffer_dump` /
     `load_buffer_dump`) with buffer.storage_dir memmaps reused: lambda_returns survives the dump, the V fit runs.
  c. `_make_buffer` with storage_dir + lambda_returns: schema, memmaps, seed + 30000, reuse, printed byte counts.
  d. Structure / signatures: every round-boundary comparison in run_pi05_round_rl carries `off`; main_pi05's kwargs
     are accepted by both runners; every test `_prefix` stub takes kv_only; every KV-only `_prefix` call in the agent
     passes kv_only=True (sample_actions_explore, from the AWR side, was the one miss).

Run:  JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_merge_vla_ogpo.py -q -p no:cacheprovider
"""
import ast
import inspect
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

import ogpo.runners.pi05_online_runner as R
from ogpo.utils.resume import ResumeStore, save_buffer_dump, truncate_csvs
from tests._pi05_stubs import Logger, TaskEnv, episodes_in_buffer, reg
from tests.test_pi05_awr import AKW, AWR_CFG, _AWRStubAgent

ROOT = os.path.join(os.path.dirname(__file__), '..')
VS = 3  # awr_value_steps


class _Crash(Exception):
    pass


def _run_awr(monkeypatch, store, *, num_rounds, upr, crash_at=None, save_dir=None, agent_gen=0, on_log=None, **kw):
    """AWR round run on the stub agent / envs. -> (rb or None on crash, logger rows, agent call log, captured rb)."""
    monkeypatch.setattr(R.T, 'resize_images', lambda x: x)
    monkeypatch.setattr(R, '_agent_state', lambda agent, step: {'step': step, 'gen': agent.gen})

    def fake_resume(resume, man, agent, rb, save_dir, upr_, seed):  # offset-aware, as the real _resume_state
        state, csv_dir, dirs = resume.fetch(man)
        meta = resume.load_buffer(rb, dirs)
        rnd = (state['step'] - resume.step_offset) // upr_
        truncate_csvs(csv_dir, save_dir, state['step'], keep_at_step=rnd == man['round'])
        return _AWRStubAgent(agent.log, gen=state['gen']), state['step'], meta['episodes'], meta['env_steps']

    monkeypatch.setattr(R, '_resume_state', fake_resume)
    captured = []
    real_make = R._make_buffer
    monkeypatch.setattr(R, '_make_buffer', lambda *a, **k: captured.append(real_make(*a, **k)) or captured[-1])
    log, logger = [], Logger()
    base = logger.log

    def lg(data, prefix, step):
        if on_log is not None:
            on_log(prefix, step, captured[-1])
        if crash_at == (prefix, step):
            raise _Crash
        base(data, prefix, step)

    logger.log = lg
    try:
        _, rb = R.run_pi05_round_rl(_AWRStubAgent(log, gen=agent_gen), TaskEnv(0, reg()), TaskEnv(0, reg()), AWR_CFG,
                                    logger, num_rounds=num_rounds, rollouts_per_round=2, updates_per_round=upr,
                                    eval_interval=0, eval_episodes=1, log_interval=2, awr_value_steps=VS,
                                    save_dir=save_dir, resume=store, **{**AKW, **kw})
    except _Crash:
        rb = None
    return rb, logger.logs, log, captured


# =====================================================================================================================
# a. AWR x resume_rebase_rounds (step_offset != 0)
# =====================================================================================================================

def _store(tmp_path, name, upr, rebase=False):
    return ResumeStore(tmp_path / f'{name}_local', tmp_path / 'remote', upr, ckpt_interval=2, rebase_rounds=rebase)


# crash_at: where the first rebased run is killed (None: it is not). Expected for the FINAL run:
#   awr rows (step, value_fit_steps), actor_update / value_update counts, collected rounds.
# Rounds after the rebase: round 3 = updates 13..20, round 4 = updates 21..28 (step_offset = 12 - 3 * 8 = -12).
CASES = {
    'no_crash':          (None,                  [(12, VS), (20, VS)], 16, 2 * VS, [3, 4]),
    # killed at round 4's awr row: chunk 4 + agent@20 are a consistent pair -> resume at a round START, skip_collect
    'round_start':       (('awr', 20),           [(20, VS)],           8,  VS,     []),
    # killed mid round 4 (agent@22): V already fitted, skip the fit, run the 6 remaining policy steps
    'mid_round':         (('online_agent', 24),  [(22, 0)],            6,  0,      []),
}


@pytest.mark.parametrize('case', list(CASES))
def test_awr_rebase_rounds_fresh_and_update_counts(tmp_path, monkeypatch, case):
    crash_at, rows, n_actor, n_value, collected = CASES[case]
    # 3 rounds x 4 updates, finished; drop step_offset from the manifest as the pre-rebase runs wrote it
    store = _store(tmp_path, 'a', 4)
    assert _run_awr(monkeypatch, store, num_rounds=3, upr=4, save_dir=str(tmp_path / 'sa'))[0] is not None
    assert store.wait() is None
    man = _store(tmp_path, 'peek', 4).manifest()
    assert (man['round'], man['step']) == (2, 12)
    man.pop('step_offset')
    (tmp_path / 'remote' / 'MANIFEST.json').write_text(json.dumps(man))

    if crash_at is not None:
        crashed = _store(tmp_path, 'b', 8, rebase=True)
        assert _run_awr(monkeypatch, crashed, num_rounds=5, upr=8, crash_at=crash_at, save_dir=str(tmp_path / 'sb'))[0] is None
        assert crashed.wait() is None
        man2 = _store(tmp_path, 'peek2', 8).manifest()
        assert man2['step_offset'] == -12
        assert (man2['round'], man2['step']) == {('awr', 20): (4, 20), ('online_agent', 24): (4, 22)}[crash_at]

    final = _store(tmp_path, 'c', 8, rebase=True)
    rb, logs, log, _ = _run_awr(monkeypatch, final, num_rounds=5, upr=8, save_dir=str(tmp_path / 'sc'))
    assert final.wait() is None
    awr_rows = [(s, d['value_fit_steps']) for p, s, d in logs if p == 'awr']
    assert awr_rows == rows, awr_rows  # pre-merge `fresh = step == r * upr` gives value_fit_steps 0 at 12 / 20
    kinds = [c[0] for c in log]
    assert kinds.count('actor_update') == n_actor
    assert kinds.count('value_update') == n_value
    assert [d['round'] for p, _, d in logs if p == 'env'] == collected
    assert max(s for p, s, _ in logs if p == 'online_agent') == 28  # = num_rounds * upr + step_offset
    end = _store(tmp_path, 'peek3', 8).manifest()
    assert (end['round'], end['step'], end['step_offset']) == (4, 28, -12)
    assert 'lambda_returns' in rb._dict and np.any(rb['lambda_returns'][:rb.size] != 0.0)


def test_round_boundaries_all_carry_the_offset():
    """Structural: in run_pi05_round_rl every line comparing / counting against r * updates_per_round also uses `off`,
    and the AWR `fresh` test and the update loop see the same `off` (one assignment, inside the round loop)."""
    src = inspect.getsource(R.run_pi05_round_rl)
    lines = [ln.strip() for ln in src.splitlines() if 'r * updates_per_round' in ln or '(r + 1) * updates_per_round' in ln]
    assert len(lines) >= 3, lines  # fresh, the update loop, the end-of-round checkpoint
    assert all('off' in ln for ln in lines), [ln for ln in lines if 'off' not in ln]
    tree = ast.parse(inspect.cleandoc('\n' + src) if src.startswith(' ') else src)
    fn = tree.body[0]
    loop = next(n for n in fn.body if isinstance(n, ast.For) and getattr(n.target, 'id', None) == 'r')
    in_loop = [n for n in ast.walk(loop) if isinstance(n, ast.Assign) and any(getattr(t, 'id', None) == 'off' for t in n.targets)]
    assert len(in_loop) == 1 and ast.unparse(in_loop[0].value) == 'resume.step_offset if resume is not None else 0'
    fresh = [n for n in ast.walk(loop) if isinstance(n, ast.Assign) and any(getattr(t, 'id', None) == 'fresh' for t in n.targets)]
    assert len(fresh) == 1 and ast.unparse(fresh[0].value) == 'step - off == r * updates_per_round'
    # `off` is assigned before the `if awr:` block, not inside it, so the update loop never reads a stale value
    idx = {id(n): i for i, n in enumerate(loop.body)}
    awr_if = next(n for n in loop.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'awr')
    assert idx[id(in_loop[0])] < idx[id(awr_if)]


# =====================================================================================================================
# b. AWR x live-snapshot resume x buffer.storage_dir (reused memmaps) x lambda_returns
# =====================================================================================================================

def test_awr_snapshot_resume_with_storage_dir(tmp_path, monkeypatch):
    UPR, R_SNAP, NR = 4, 1, 3
    store_dir = tmp_path / 'scratch'
    store_dir.mkdir()
    snap = tmp_path / 'snap'

    def take_snapshot(prefix, step, rb):  # between round 1's collection and its updates, as a live-run snapshot is taken
        if (prefix, step) == ('env', R_SNAP * UPR):
            snap.mkdir()
            save_buffer_dump(rb, snap / 'buffer', {'round': R_SNAP, 'episodes': 4, 'env_steps': 0})
            with open(snap / 'agent.pkl', 'wb') as f:
                pickle.dump({'step': R_SNAP * UPR, 'gen': R_SNAP * (VS + UPR)}, f)
            (snap / 'csv').mkdir()
            (snap / 'csv' / 'env.csv').write_text(f'success,step\n0.5,0\n0.5,{R_SNAP * UPR}\n')

    rb0, _, _, cap = _run_awr(monkeypatch, None, num_rounds=NR, upr=UPR, crash_at=('env', R_SNAP * UPR),
                              on_log=take_snapshot, buffer_storage_dir=str(store_dir))
    assert rb0 is None and isinstance(cap[-1]._dict['images'], np.memmap)
    dumped = json.loads((snap / 'buffer' / 'meta.json').read_text())
    assert 'lambda_returns' in dumped['fields'] and 'images' not in dumped['fields']  # RAM fields only
    imgs_before = np.array(cap[-1]._dict['images'][:cap[-1].size])
    lam_before = np.array(cap[-1]._dict['lambda_returns'][:cap[-1].size])
    del cap

    monkeypatch.setattr(R, '_restore_agent', lambda agent, state: _AWRStubAgent(agent.log, gen=state['gen']))
    rb, logs, log, cap = _run_awr(monkeypatch, None, num_rounds=NR, upr=UPR, save_dir=str(tmp_path / 'save'),
                                  resume_snapshot_dir=str(snap), buffer_storage_dir=str(store_dir),
                                  buffer_reuse_storage=True)
    # the memmaps were reopened with round 0-1's images, not zeroed
    np.testing.assert_array_equal(np.asarray(rb._dict['images'][:len(imgs_before)]), imgs_before)
    assert isinstance(rb._dict['images'], np.memmap) and isinstance(rb._dict['next_images'], np.memmap)
    # resumed at round 1's start: V fit runs (fresh), round 1 is not re-collected, round 2 is
    assert [(s, d['value_fit_steps']) for p, s, d in logs if p == 'awr'] == [(4, VS), (8, VS)]
    assert [d['round'] for p, _, d in logs if p == 'env'] == [2]
    kinds = [c[0] for c in log]
    assert kinds.count('actor_update') == (NR - R_SNAP) * UPR and kinds.count('value_update') == 2 * VS
    assert max(s for p, s, _ in logs if p == 'online_agent') == NR * UPR
    # the first state_values pass bootstraps with the snapshot agent (gen 7)
    assert next(c[1] for c in log if c[0] == 'state_values') == R_SNAP * (VS + UPR)
    # the dumped (pre-fit) λ field was overwritten by the round-1 passes before the first policy step
    first_actor = next(c for c in log if c[0] == 'actor_update')
    assert np.any(first_actor[1]['lambda_returns'] != 0.0)
    assert not np.allclose(rb['lambda_returns'][:len(lam_before)], lam_before)
    episodes_in_buffer(rb)


# =====================================================================================================================
# c. _make_buffer: storage_dir + lambda_returns
# =====================================================================================================================

FEATS = np.zeros(5, np.float32)
IMGS = np.zeros((2, 4, 4, 3), np.uint8)
BASE_KEYS = {'observations', 'images', 'actions', 'rewards', 'terminals', 'masks', 'next_observations', 'next_images',
             'mc_returns', 'task_ids', 'is_success', 'score'}


@pytest.mark.parametrize('storage,lam', [(False, False), (False, True), (True, False), (True, True)])
def test_make_buffer_schema_memmaps_and_seed(tmp_path, capsys, storage, lam):
    sd = str(tmp_path) if storage else None
    rb = R._make_buffer(FEATS, IMGS, 50, 7, 3, storage_dir=sd, lambda_returns=lam)
    out = capsys.readouterr().out
    assert BASE_KEYS <= set(rb._dict) and ('lambda_returns' in rb._dict) == lam
    assert set(rb._dict) - BASE_KEYS == ({'lambda_returns'} if lam else set())
    if lam:
        assert rb._dict['lambda_returns'].shape == (50,) and rb._dict['lambda_returns'].dtype == np.float32
        assert not isinstance(rb._dict['lambda_returns'], np.memmap)
    disk = ('images', 'next_images') if storage else ()
    for k, v in rb._dict.items():
        assert isinstance(v, np.memmap) == (k in disk), k
    for k in disk:
        assert os.path.isfile(tmp_path / f'{k}.npy') and rb._dict[k].shape == (50, *IMGS.shape)
    # seed offset seed + 30000 unchanged
    ref = np.random.default_rng(7 + 30000)
    ref.integers(2**31)  # Dataset.__init__ draws the augmentation PRNGKey seed from numpy_rng first
    assert np.array_equal(rb.numpy_rng.integers(0, 1 << 30, 8), ref.integers(0, 1 << 30, 8))
    # printed byte counts: RAM excludes the memmapped fields
    on_disk = sum(rb._dict[k].nbytes for k in disk)
    ram = sum(v.nbytes for v in rb._dict.values()) - on_disk
    assert f'{ram / 1e9:.1f} GB in RAM' in out
    assert ('memmapped in' in out) == storage
    if storage:
        assert f"{on_disk / 1e9:.1f} GB of ['images', 'next_images'] memmapped in {tmp_path}" in out


def test_make_buffer_reuse_keeps_images_and_lambda_field_samples(tmp_path):
    rb = R._make_buffer(FEATS, IMGS, 20, 0, 3, storage_dir=str(tmp_path), lambda_returns=True)
    for i in range(10):
        rb.add_transition(dict(observations=np.full(5, i, np.float32), images=np.full(IMGS.shape, i, np.uint8),
                               actions=np.zeros(3, np.float32), rewards=np.float32(-1), terminals=np.float32(i == 9),
                               masks=np.float32(1), next_observations=np.full(5, i + 1, np.float32),
                               next_images=np.full(IMGS.shape, i + 1, np.uint8), mc_returns=np.float32(0),
                               task_ids=np.int32(0), is_success=0.0, score=1e-9, lambda_returns=np.float32(i / 10)))
    rb._dict['images'].flush()
    rb.write_field('lambda_returns', np.arange(10), np.arange(10, dtype=np.float32) * 2)
    b = rb.sample_sequence(4, sequence_length=2, discount=0.99)
    np.testing.assert_allclose(b['lambda_returns'], b['observations'][:, 0] * 2)  # λ-return at the window start
    rb2 = R._make_buffer(FEATS, IMGS, 20, 0, 3, storage_dir=str(tmp_path), reuse_storage=True, lambda_returns=True)
    assert np.array_equal(np.asarray(rb2._dict['images'][:10, 0, 0, 0, 0]), np.arange(10))
    assert not np.any(rb2._dict['lambda_returns'])  # RAM fields start empty: load_buffer_dump restores them
    with pytest.raises(ValueError, match='the buffer needs'):
        R._make_buffer(FEATS, IMGS, 21, 0, 3, storage_dir=str(tmp_path), reuse_storage=True)  # wrong size


# =====================================================================================================================
# d. Signatures
# =====================================================================================================================

def _main_calls():
    tree = ast.parse(open(os.path.join(ROOT, 'ogpo', 'main_pi05.py')).read())
    common = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                  and any(getattr(t, 'id', None) == 'common' for t in n.targets))
    common_kw = {k.arg for k in common.keywords}
    calls = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, 'id', None) in ('run_pi05_round_rl', 'run_pi05_online_rl'):
            kws = {k.arg for k in n.keywords if k.arg is not None}
            if any(k.arg is None and getattr(k.value, 'id', None) == 'common' for k in n.keywords):
                kws |= common_kw
            calls[n.func.id] = (len(n.args), kws)
    return calls


def test_main_pi05_kwargs_match_both_runner_signatures():
    calls = _main_calls()
    assert set(calls) == {'run_pi05_round_rl', 'run_pi05_online_rl'}
    for name, (n_pos, kws) in calls.items():
        sig = inspect.signature(getattr(R, name))
        params = sig.parameters
        pos = [p for p in params.values() if p.kind == p.POSITIONAL_OR_KEYWORD]
        kwonly = {p.name for p in params.values() if p.kind == p.KEYWORD_ONLY}
        assert n_pos == len(pos), (name, n_pos, [p.name for p in pos])
        assert kws <= kwonly, (name, sorted(kws - kwonly))
        required = {p.name for p in params.values() if p.kind == p.KEYWORD_ONLY and p.default is p.empty}
        assert required <= kws, (name, sorted(required - kws))
    # both kwarg sets of the merge reach the round runner
    assert {'resume_snapshot_dir', 'buffer_reuse_storage', 'buffer_storage_dir', 'awr_value_steps',
            'awr_value_refreshes', 'resume'} <= calls['run_pi05_round_rl'][1]


def test_every_test_prefix_stub_accepts_kv_only():
    bad = []
    for name in sorted(os.listdir(os.path.join(ROOT, 'tests'))):
        if not name.endswith('.py'):
            continue
        tree = ast.parse(open(os.path.join(ROOT, 'tests', name)).read())
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef) and n.name == '_prefix':
                args = {a.arg for a in n.args.args + n.args.kwonlyargs}
                if 'kv_only' not in args and n.args.kwarg is None:
                    bad.append(f'{name}:{n.lineno}')
    assert not bad, bad


def test_every_kv_only_prefix_call_in_the_agent_passes_kv_only():
    """`_, kv, pmask = self._prefix(...)` discards prefix_out, so it should pass kv_only=True (926eb41 did so for every
    such call that existed on molmo-vla)."""
    tree = ast.parse(open(os.path.join(ROOT, 'ogpo', 'agents', 'ogpo_pi05.py')).read())
    missing = []
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        for n in ast.walk(fn):
            if (isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                    and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == '_prefix'
                    and isinstance(n.targets[0], ast.Tuple) and getattr(n.targets[0].elts[0], 'id', None) == '_'):
                kv = [k for k in n.value.keywords if k.arg == 'kv_only']
                if not (kv and isinstance(kv[0].value, ast.Constant) and kv[0].value.value is True):
                    missing.append(f'{fn.name}:{n.lineno}')
    assert not missing, missing


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
