"""Verifier (2026-10-27-pi05-snapshot-resume): save_buffer_dump / load_buffer_dump / ReplayBuffer.create(reuse_disk) edge cases.

Run:  JAX_PLATFORMS=cpu pytest tests/verify_buffer_dump.py -q
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

import ogpo.utils.resume as RS
from ogpo.utils.datasets import ReplayBuffer
from ogpo.utils.resume import load_buffer_dump, save_buffer_dump

SIZE = 13
ROW = (5, 3, 3)  # images row = 45 bytes


def ex(obs_dim=3, img_dtype=np.uint8):
    return dict(observations=np.zeros(obs_dim, np.float32), images=np.zeros(ROW, img_dtype), actions=np.zeros(2, np.float32),
                rewards=np.array(0.0, np.float32), masks=np.array(1.0, np.float32), terminals=np.array(0.0, np.float32),
                next_images=np.zeros(ROW, img_dtype), task_ids=np.array(0, np.int32))


def make(disk=None, reuse=False, size=SIZE, **kw):
    keys = ("images", "next_images") if disk else ()
    return ReplayBuffer.create(ex(**kw), size, numpy_rng=np.random.default_rng(0), disk_dir=disk, disk_keys=keys,
                               reuse_disk=reuse)


def fill(rb, n, start=0):
    rng = np.random.default_rng(start)
    for i in range(start, start + n):
        rb.add_transition(dict(observations=rng.normal(size=3).astype(np.float32),
                               images=rng.integers(0, 255, ROW, dtype=np.uint8), actions=rng.normal(size=2).astype(np.float32),
                               rewards=np.array(-1.0, np.float32), masks=np.array(float(i % 5 != 4), np.float32),
                               terminals=np.array(float(i % 5 == 4), np.float32),
                               next_images=rng.integers(0, 255, ROW, dtype=np.uint8), task_ids=np.array(i % 3, np.int32),
                               is_success=float(i % 2)))


def same(a, b):
    assert (a.pointer, a.size, a.total_added, a.max_size) == (b.pointer, b.size, b.total_added, b.max_size)
    assert a._traj_success_mask.dtype == b._traj_success_mask.dtype == np.bool_
    assert np.array_equal(a._traj_success_mask, b._traj_success_mask)
    assert sorted(a._dict) == sorted(b._dict)
    for k in a._dict:
        assert a._dict[k].dtype == b._dict[k].dtype, k
        assert np.array_equal(np.asarray(a._dict[k]), np.asarray(b._dict[k])), k


@pytest.mark.parametrize("chunk", [1, 4 * 3, 45, 46, 2 * 45 + 7, 1 << 30])
@pytest.mark.parametrize("n", [0, 1, SIZE - 1, SIZE, SIZE + 5, 3 * SIZE + 2])
def test_ram_roundtrip_chunks_and_wraparound(tmp_path, chunk, n):
    rb = make()
    fill(rb, n)
    meta = save_buffer_dump(rb, str(tmp_path / "d"), dict(round=3, episodes=9, env_steps=99), chunk_bytes=chunk)
    assert not os.path.exists(str(tmp_path / "d") + ".tmp")
    assert json.load(open(tmp_path / "d" / "meta.json")) == meta
    other = make()
    fill(other, 4, start=1000)  # pre-existing junk must be overwritten
    load_buffer_dump(other, str(tmp_path / "d"))
    same(rb, other)


@pytest.mark.parametrize("n", [0, SIZE + 4])
def test_disk_roundtrip_with_reuse(tmp_path, n):
    live = tmp_path / "live"
    live.mkdir()
    rb = make(str(live))
    fill(rb, n)
    save_buffer_dump(rb, str(tmp_path / "snap" / "buffer"), dict(round=1, episodes=1, env_steps=1), chunk_bytes=50)
    assert sorted(json.load(open(tmp_path / "snap" / "buffer" / "meta.json"))["fields"]) == \
        sorted(k for k in rb._dict if k not in ("images", "next_images"))
    assert not (tmp_path / "snap" / "buffer" / "images.npy").exists()
    new = tmp_path / "new"
    new.mkdir()
    for k in ("images", "next_images"):  # what snapshot.py's reflink + the script's clone produce (no explicit flush first)
        with open(live / f"{k}.npy", "rb") as s, open(new / f"{k}.npy", "wb") as d:
            d.write(s.read())
    rb2 = make(str(new), reuse=True)
    load_buffer_dump(rb2, str(tmp_path / "snap" / "buffer"))
    same(rb, rb2)
    assert isinstance(rb2._dict["images"], np.memmap)
    # continue writing into the reused memmaps: the next transition lands at the restored pointer
    fill(rb, 2, start=500)
    fill(rb2, 2, start=500)
    same(rb, rb2)


def test_changes_during_dump_raise_and_leave_no_out_dir(tmp_path, monkeypatch):
    rb = make()
    fill(rb, 5)
    real = RS._write_npy
    calls = []

    def write(path, arr, chunk):
        real(path, arr, chunk)
        calls.append(path)
        if len(calls) == 2:
            fill(rb, 1, start=77)  # a collection step lands mid-dump

    monkeypatch.setattr(RS, "_write_npy", write)
    with pytest.raises(RuntimeError, match="changed during the dump"):
        save_buffer_dump(rb, str(tmp_path / "d"), {})
    assert not (tmp_path / "d").exists() and (tmp_path / "d.tmp").exists()
    with pytest.raises(ValueError, match="already exists"):  # a leftover .tmp blocks the next attempt (snapshot.py removes it)
        save_buffer_dump(rb, str(tmp_path / "d"), {})


def test_trailing_slash_out_dir(tmp_path):
    rb = make()
    fill(rb, 3)
    save_buffer_dump(rb, str(tmp_path / "d") + "/", {})
    load_buffer_dump(make(), str(tmp_path / "d") + "/")


def test_mismatches_raise_value_error(tmp_path):
    rb = make()
    fill(rb, 4)
    save_buffer_dump(rb, str(tmp_path / "d"), {})
    with pytest.raises(ValueError, match="buffer size"):
        load_buffer_dump(make(size=SIZE + 1), str(tmp_path / "d"))
    with pytest.raises(ValueError, match=r"observations\.npy is"):
        load_buffer_dump(make(obs_dim=4), str(tmp_path / "d"))
    with pytest.raises(ValueError, match=r"images\.npy is"):
        load_buffer_dump(make(img_dtype=np.float32), str(tmp_path / "d"))
    d2 = tmp_path / "disk"
    d2.mkdir()
    with pytest.raises(ValueError, match="RAM fields"):
        load_buffer_dump(make(str(d2)), str(tmp_path / "d"))


def test_reuse_disk_errors(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    rb = make(str(d))
    fill(rb, 3)
    rb._dict["images"].flush()
    with pytest.raises(ValueError, match="the buffer needs"):  # dtype mismatch
        make(str(d), reuse=True, img_dtype=np.float32)
    with pytest.raises(ValueError, match="the buffer needs"):  # size mismatch
        make(str(d), reuse=True, size=SIZE - 1)
    os.remove(d / "next_images.npy")
    with pytest.raises(ValueError, match="does not exist"):
        make(str(d), reuse=True)
    with pytest.raises(ValueError, match="needs a disk_dir"):
        ReplayBuffer.create(ex(), SIZE, disk_keys=("images",), reuse_disk=True)


def test_reuse_disk_skips_free_space_check(tmp_path, monkeypatch):
    d = tmp_path / "d"
    d.mkdir()
    make(str(d))
    real = os.statvfs

    class Full:
        def __init__(self, st):
            self.f_frsize, self.f_bavail = st.f_frsize, 0

    monkeypatch.setattr(os, "statvfs", lambda p: Full(real(p)))
    with pytest.raises(ValueError, match="GB free"):
        make(str(tmp_path))
    make(str(d), reuse=True)  # no check: the files already exist


def test_write_npy_empty(tmp_path):
    RS._write_npy(str(tmp_path / "e.npy"), np.zeros((0, 3), np.float32), 1 << 20)
    assert np.load(tmp_path / "e.npy").shape == (0, 3)
