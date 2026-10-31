"""Verifier (2026-10-27-pi05-snapshot-resume): main_pi05 guards / forwarding for training.resume_snapshot_dir and
buffer.reuse_storage, with stub env / agent / logging (harness as tests/test_pi05_multitask.py's _patch_main).

Run:  JAX_PLATFORMS=cpu pytest tests/verify_main_guards.py -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from types import SimpleNamespace

import numpy as np
import pytest

GOOD = 'libero_pi05-libero_90-1'


class _Built(Exception):
    pass


def _main(monkeypatch, *extra):
    import ogpo.main_pi05 as M
    monkeypatch.delenv("OGPO_RESUME_S3", raising=False)
    monkeypatch.setattr(sys, 'argv', ['main_pi05.py', '--algo=ogpo', '--experiment.log=false',
                                      '--agent.pi05_action_horizon=10', f'--env.name={GOOD}', *extra])
    built = {}

    def make_env(name, seed, post_success_steps):
        return SimpleNamespace(name=name, seed=seed, prompt=f"prompt:{name}")

    def create(cls, seed, prompt, config):
        return SimpleNamespace(tag='agent')

    def logging(**kw):
        built['log_kw'] = kw
        return None, SimpleNamespace(close=lambda: None), None

    def capture(name):
        def run(agent, env, eval_env, config, logger, **kw):
            built['runner'], built['kw'] = name, kw
            raise _Built()
        return run
    monkeypatch.setattr(M, 'make_env', make_env)
    monkeypatch.setattr(M.OGPOPi05Agent, 'create', classmethod(create))
    monkeypatch.setattr(M, 'create_mesh', lambda mode: None)
    monkeypatch.setattr(M, 'shard_pi05_agent', lambda agent, mesh, zero_actor_state=False: agent)
    monkeypatch.setattr(M, 'setup_experiment_logging', logging)
    monkeypatch.setattr(M, 'load_checkpoint', lambda agent, path, keep_base_actor=False: (agent, 0))
    monkeypatch.setattr(M, 'ResumeStore', lambda *a, **k: SimpleNamespace(manifest=lambda: None))
    monkeypatch.setattr(M, 'run_pi05_round_rl', capture('rounds'))
    monkeypatch.setattr(M, 'run_pi05_online_rl', capture('online'))
    return M, built


ROUNDS = '--training.collection_mode=rounds'


@pytest.mark.parametrize('extra', [[ROUNDS], [ROUNDS, '--buffer.reuse_storage=false'], [ROUNDS, '--training.resume_snapshot_dir=null']])
def test_defaults_forward_off(monkeypatch, extra):
    M, built = _main(monkeypatch, *extra)
    with pytest.raises(_Built):
        M.main()
    kw = built['kw']
    assert kw['resume_snapshot_dir'] is None and kw['buffer_reuse_storage'] is False and kw['buffer_storage_dir'] is None
    assert built['log_kw']['append_csv'] is False


def test_snapshot_forwarded(monkeypatch):
    M, built = _main(monkeypatch, ROUNDS, '--training.resume_snapshot_dir=/x/snap', '--buffer.storage_dir=/x/rb',
                     '--buffer.reuse_storage=true')
    with pytest.raises(_Built):
        M.main()
    kw = built['kw']
    assert (kw['resume_snapshot_dir'], kw['buffer_reuse_storage'], kw['buffer_storage_dir']) == ('/x/snap', True, '/x/rb')
    assert built['log_kw']['append_csv'] is True


@pytest.mark.parametrize('extra,env,match', [
    (['--training.resume_snapshot_dir=/x/snap'], None, 'collection_mode=rounds'),
    ([ROUNDS, '--training.resume_snapshot_dir=/x/snap', '--training.restore_checkpoint=/x/c.pkl'], None, 'exclusive'),
    ([ROUNDS, '--training.resume_snapshot_dir=/x/snap', '--training.resume_s3=s3://b/x'], None, 'exclusive'),
    ([ROUNDS, '--training.resume_snapshot_dir=/x/snap'], 's3://b/x', 'exclusive'),
    ([ROUNDS, '--training.resume_snapshot_dir=/x/snap', '--training.restore_checkpoint=/x/c.pkl', '--training.eval_only=true'],
     None, 'exclusive'),
    ([ROUNDS, '--buffer.reuse_storage=true'], None, 'reuse_storage needs'),
    ([ROUNDS, '--buffer.reuse_storage=true', '--buffer.storage_dir=/x/rb'], None, 'reuse_storage needs'),
])
def test_guards(monkeypatch, extra, env, match):
    M, built = _main(monkeypatch, *extra)
    if env:
        monkeypatch.setenv("OGPO_RESUME_S3", env)
    with pytest.raises(ValueError, match=match):
        M.main()
    assert 'kw' not in built
