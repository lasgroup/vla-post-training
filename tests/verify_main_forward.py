"""Verifier 2 (2026-10-27-pi05-snapshot-resume): main_pi05 forwards the defaults (off) and a snapshot dir. Same harness as
tests/verify_main_guards.py, plus the rounds keys that ogpo.yaml does not define (num_rounds / rollouts_per_round /
updates_per_round are script-only; pre-existing).

Run:  JAX_PLATFORMS=cpu pytest tests/verify_main_forward.py -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pytest

from verify_main_guards import ROUNDS, _Built, _main

RKEYS = ['--training.num_rounds=2', '--training.rollouts_per_round=1', '--training.updates_per_round=4']


@pytest.mark.parametrize('extra', [[], ['--buffer.reuse_storage=false'], ['--training.resume_snapshot_dir=null']])
def test_defaults_forward_off(monkeypatch, extra):
    M, built = _main(monkeypatch, ROUNDS, *RKEYS, *extra)
    with pytest.raises(_Built):
        M.main()
    kw = built['kw']
    assert kw['resume_snapshot_dir'] is None and kw['buffer_reuse_storage'] is False and kw['resume'] is None
    assert kw['buffer_storage_dir'] is None
    assert built['log_kw']['append_csv'] is False


def test_snapshot_forwarded(monkeypatch):
    M, built = _main(monkeypatch, ROUNDS, *RKEYS, '--training.resume_snapshot_dir=/x/snap', '--buffer.reuse_storage=true',
                     '--buffer.storage_dir=/x/rb')
    with pytest.raises(_Built):
        M.main()
    kw = built['kw']
    assert (kw['resume_snapshot_dir'], kw['buffer_reuse_storage'], kw['buffer_storage_dir']) == ('/x/snap', True, '/x/rb')
    assert built['log_kw']['append_csv'] is True
