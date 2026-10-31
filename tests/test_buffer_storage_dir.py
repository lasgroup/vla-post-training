"""
CPU tests for `buffer.storage_dir` (docs/changes/2026-10-26-pi05-buffer-scratch): ReplayBuffer.create(disk_dir, disk_keys)
memmaps the pi05 buffer's images / next_images; the default path must be unchanged.

  1. differential: pi05 _make_buffer RAM vs memmap, same episodes through _flush_episode (FIFO wrap-around):
     bit-identical sample_sequence (T, success_only, per_task), _sample_success, _task_counts, counters, masks.
  2. resume: ResumeStore.save_chunk from a memmap buffer, load_buffer into a RAM and a fresh memmap buffer.
  3. default path vs a verbatim copy of the pre-change ReplayBuffer.create (pi05 schema, nested dicts).
  4. ValueError paths (no dir, missing dir, file as dir, unknown key, not enough free space).
  5. <dir>/<key>.npy files: shape / dtype, readable with np.load(mmap_mode='r') after writes.
  6. config: buffer.storage_dir defaults to null, CLI override arrives as a string.

Run:  JAX_PLATFORMS=cpu pytest tests/test_buffer_storage_dir.py -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import numpy as np
import pytest

import ogpo.runners.pi05_online_runner as R
import ogpo.utils.datasets as D
from ogpo.utils.datasets import ReplayBuffer
from ogpo.utils.resume import ResumeStore

FEAT_SHAPE = (3, 4)
IMG_SHAPE = (2, 6, 6, 3)
ACT_DIM = 7
NUM_TASKS = 3
DISCOUNT = 0.99


def _example():
    return np.zeros(FEAT_SHAPE, np.float32), np.zeros(IMG_SHAPE, np.uint8)


def _make(size, seed=0, storage_dir=None):
    feats, imgs = _example()
    return R._make_buffer(feats, imgs, size, seed, ACT_DIM, storage_dir=None if storage_dir is None else str(storage_dir))


def _episodes(seed, n_eps):
    """Episodes as the runner builds them (dicts per step) + the final info."""
    rng = np.random.default_rng(seed)
    eps = []
    for _ in range(n_eps):
        L = int(rng.integers(2, 12))
        task = int(rng.integers(NUM_TASKS))
        succ = float(rng.random() < 0.5)
        traj = []
        for t in range(L):
            done = t == L - 1
            terminated = done and succ > 0
            traj.append(dict(
                observations=rng.standard_normal(FEAT_SHAPE).astype(np.float32),
                images=rng.integers(0, 256, IMG_SHAPE, dtype=np.uint8),
                actions=rng.standard_normal(ACT_DIM).astype(np.float32),
                rewards=0.0 if terminated else -1.0,
                terminals=float(done), masks=1.0 - float(terminated),
                next_observations=rng.standard_normal(FEAT_SHAPE).astype(np.float32),
                next_images=rng.integers(0, 256, IMG_SHAPE, dtype=np.uint8),
                task_ids=np.int32(task),
            ))
        eps.append((traj, {'success': succ}))
    return eps


def _feed(rb, eps):
    for traj, info in eps:
        R._flush_episode(rb, [dict(t) for t in traj], info, DISCOUNT)


def _assert_tree_equal(a, b, what):
    assert sorted(a) == sorted(b), what
    for k in a:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        assert x.dtype == y.dtype and x.shape == y.shape, (what, k, x.dtype, y.dtype, x.shape, y.shape)
        assert np.array_equal(x, y), (what, k)


def _assert_buffers_equal(a, b, what):
    assert (a.size, a.pointer, a.total_added, a.max_size) == (b.size, b.pointer, b.total_added, b.max_size), what
    _assert_tree_equal(a._dict, b._dict, what)
    assert np.array_equal(a._traj_success_mask, b._traj_success_mask), what


def _both(fn, a, b):
    """fn on both buffers: same result or the same exception type."""
    out = []
    for rb in (a, b):
        try:
            out.append(("ok", fn(rb)))
        except ValueError as e:
            out.append(("err", type(e)))
    assert out[0][0] == out[1][0], out
    return out


def _compare_samples(ram, mm, what):
    for T in (1, 2, 5):
        for success_only in (None, True, False):
            for per_task in (None, NUM_TASKS):
                fn = lambda rb: rb.sample_sequence(16, sequence_length=T, discount=DISCOUNT, success_only=success_only,
                                                   per_task=per_task)
                (s1, r1), (_, r2) = _both(fn, ram, mm)
                if s1 == "ok":
                    _assert_tree_equal(r1, r2, (what, T, success_only, per_task))
                    for k in ('images', 'next_images'):  # fancy indexing a memmap returns a plain in-RAM array
                        assert type(r2[k]) is np.ndarray, (k, type(r2[k]))
                else:
                    assert r1 == r2
        (s1, r1), (_, r2) = _both(lambda rb: R._sample_success(rb, 4, T, DISCOUNT, per_task=NUM_TASKS), ram, mm)
        if s1 == "ok":
            assert (r1 is None) == (r2 is None)
            if r1 is not None:
                _assert_tree_equal(r1, r2, (what, "success", T))
    assert R._task_counts(ram, NUM_TASKS) == R._task_counts(mm, NUM_TASKS)
    assert ram.numpy_rng.bit_generator.state == mm.numpy_rng.bit_generator.state


# ===========================================================================
# 1. differential RAM vs memmap
# ===========================================================================

def test_memmap_buffer_matches_ram_buffer(tmp_path):
    ram, mm = _make(60, seed=5), _make(60, seed=5, storage_dir=tmp_path)
    assert type(ram._dict['images']) is np.ndarray and isinstance(mm._dict['images'], np.memmap)
    assert all(type(mm._dict[k]) is np.ndarray for k in mm._dict if k not in ('images', 'next_images'))
    _assert_buffers_equal(ram, mm, "empty")
    eps = _episodes(1, 30)
    _feed(ram, eps[:4])
    _feed(mm, eps[:4])
    assert ram.size < ram.max_size  # partial fill
    _assert_buffers_equal(ram, mm, "partial")
    _compare_samples(ram, mm, "partial")
    _feed(ram, eps[4:])
    _feed(mm, eps[4:])
    assert ram.total_added > 2 * ram.max_size  # wrapped more than once
    _assert_buffers_equal(ram, mm, "wrapped")
    _compare_samples(ram, mm, "wrapped")
    assert ram._traj_success_mask.any() and not ram._traj_success_mask.all()


def test_online_runner_add_transition_path_matches(tmp_path):
    """run_pi05_online_rl adds via _flush_episode too; also add_transition directly, single step at a time."""
    ram, mm = _make(25, seed=2), _make(25, seed=2, storage_dir=tmp_path)
    for traj, info in _episodes(7, 12):
        for t in traj:
            t = dict(t, mc_returns=0.5, is_success=info['success'])
            ram.add_transition(dict(t))
            mm.add_transition(dict(t))
    _assert_buffers_equal(ram, mm, "add_transition")
    _compare_samples(ram, mm, "add_transition")


# ===========================================================================
# 2. resume round trip
# ===========================================================================

UPR = 10


def test_resume_roundtrip_from_memmap(tmp_path):
    (tmp_path / "src_rb").mkdir()
    src = _make(40, seed=3, storage_dir=tmp_path / "src_rb")
    st = ResumeStore(tmp_path / "s_local", tmp_path / "remote", UPR, ckpt_interval=5)
    st.save_agent({"step": 0}, 0, [])
    eps = _episodes(11, 32)
    for r in range(4):
        p0, t0 = src.pointer, src.total_added
        _feed(src, eps[8 * r:8 * (r + 1)])
        st.save_chunk(src, r, p0, t0, {"episodes": 8 * (r + 1), "env_steps": int(src.total_added)})
        st.save_agent({"step": (r + 1) * UPR}, (r + 1) * UPR, [])
    assert st.wait() is None
    st.close()
    assert src.total_added > src.max_size  # the ring wrapped

    fresh = ResumeStore(tmp_path / "f_local", tmp_path / "remote", UPR, ckpt_interval=5)
    man = fresh.manifest()
    assert man["round"] == 3 and man["step"] == 4 * UPR
    _, _, dirs = fresh.fetch(man)
    (tmp_path / "dst_rb").mkdir()
    for dst in (_make(40, seed=3), _make(40, seed=3, storage_dir=tmp_path / "dst_rb")):
        meta = fresh.load_buffer(dst, dirs)
        assert meta["episodes"] == 32
        _assert_buffers_equal(src, dst, type(dst._dict['images']).__name__)
    for k in ('images', 'next_images'):  # what the fresh memmap wrote is on disk
        assert np.array_equal(np.load(tmp_path / "dst_rb" / f"{k}.npy", mmap_mode="r"), src._dict[k])
    fresh.close()


# ===========================================================================
# 3. default path vs the pre-change create
# ===========================================================================

def _old_create(cls, transition, size, numpy_rng=None):
    """Verbatim ReplayBuffer.create at git HEAD c07ee40 (cls passed explicitly)."""

    def create_buffer(example):
        example = np.array(example)
        return np.zeros((size, *example.shape), dtype=example.dtype)

    buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
    if 'is_success' not in buffer_dict:
        buffer_dict['is_success'] = np.zeros(size, dtype=np.float32)
    if 'score' not in buffer_dict:
        buffer_dict['score'] = np.zeros(size, dtype=np.float32)
    return cls(buffer_dict, numpy_rng=numpy_rng)


def _pi05_transition():
    feats, imgs = _example()
    return dict(
        observations=feats, images=imgs, actions=np.zeros(ACT_DIM, np.float32),
        rewards=np.array(0.0, np.float32), terminals=np.array(0.0, np.float32), masks=np.array(1.0, np.float32),
        next_observations=feats, next_images=imgs, mc_returns=np.array(0.0, np.float32),
        task_ids=np.array(0, np.int32),
    )


def _nested_transition():
    """As online_rl_runner.py:367 builds it, with dict observations (multi-modal obs)."""
    obs = {'state': np.zeros(5, np.float32), 'pixels': np.zeros((8, 8, 3), np.uint8),
           'deep': {'x': np.zeros(2, np.float64)}}
    return dict(observations=obs, actions=np.zeros(4, np.float32), rewards=np.array(0.0), terminals=np.array(0.0),
                masks=np.array(1.0), next_observations=obs, images=np.zeros((8, 8, 3), np.uint8),
                next_images=np.zeros((8, 8, 3), np.uint8), full_states=np.zeros(9), next_full_states=np.zeros(9),
                mc_returns=np.array(0.0))


@pytest.mark.parametrize("make_tr", [_pi05_transition, _nested_transition])
@pytest.mark.parametrize("extra", [{}, {"disk_keys": ()}, {"disk_dir": "TMP", "disk_keys": ()}])
def test_default_path_equals_pre_change_create(tmp_path, make_tr, extra):
    extra = {k: (str(tmp_path) if v == "TMP" else v) for k, v in extra.items()}
    old = _old_create(ReplayBuffer, make_tr(), 17, numpy_rng=np.random.default_rng(0))
    new = ReplayBuffer.create(make_tr(), 17, numpy_rng=np.random.default_rng(0), **extra)
    assert type(new) is type(old)
    assert list(new._dict.keys()) == list(old._dict.keys())
    assert jax.tree_util.tree_structure(new._dict) == jax.tree_util.tree_structure(old._dict)
    for lo, ln in zip(jax.tree_util.tree_leaves(old._dict), jax.tree_util.tree_leaves(new._dict)):
        assert type(ln) is type(lo) is np.ndarray
        assert (ln.dtype, ln.shape) == (lo.dtype, lo.shape)
        assert np.array_equal(ln, lo)
    assert (new.size, new.pointer, new.max_size, new.total_added) == (old.size, old.pointer, old.max_size, old.total_added)
    assert new.numpy_rng.bit_generator.state == old.numpy_rng.bit_generator.state
    assert np.array_equal(new.rng, old.rng)
    assert os.listdir(tmp_path) == []  # nothing written to disk


def test_make_buffer_default_equals_pre_change_create():
    feats, imgs = _example()
    new = R._make_buffer(feats, imgs, 13, 4, ACT_DIM)
    old = _old_create(ReplayBuffer, _pi05_transition(), 13, numpy_rng=np.random.default_rng(4 + 30000))
    _assert_buffers_equal(old, new, "make_buffer")
    assert all(type(v) is np.ndarray for v in new._dict.values())
    assert new.numpy_rng.bit_generator.state == old.numpy_rng.bit_generator.state


def test_memmap_with_nested_transition(tmp_path):
    """disk_keys on top-level leaves of a nested transition: nested RAM fields keep their structure."""
    old = _old_create(ReplayBuffer, _nested_transition(), 9, numpy_rng=np.random.default_rng(0))
    new = ReplayBuffer.create(_nested_transition(), 9, numpy_rng=np.random.default_rng(0), disk_dir=str(tmp_path),
                              disk_keys=('images', 'next_images'))
    assert jax.tree_util.tree_structure(new._dict) == jax.tree_util.tree_structure(old._dict)
    for lo, ln in zip(jax.tree_util.tree_leaves(old._dict), jax.tree_util.tree_leaves(new._dict)):
        assert (ln.dtype, ln.shape) == (lo.dtype, lo.shape) and np.array_equal(ln, lo)
    assert sorted(os.listdir(tmp_path)) == ['images.npy', 'next_images.npy']


# ===========================================================================
# 4. error paths
# ===========================================================================

def test_disk_keys_without_dir_raises():
    with pytest.raises(ValueError, match="needs a disk_dir"):
        ReplayBuffer.create(_pi05_transition(), 5, disk_keys=('images',))


def test_missing_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="not an existing directory"):
        ReplayBuffer.create(_pi05_transition(), 5, disk_dir=str(tmp_path / "nope"), disk_keys=('images',))
    f = tmp_path / "file"
    f.write_text("x")
    with pytest.raises(ValueError, match="not an existing directory"):
        ReplayBuffer.create(_pi05_transition(), 5, disk_dir=str(f), disk_keys=('images',))
    with pytest.raises(ValueError, match="not an existing directory"):
        ReplayBuffer.create(_pi05_transition(), 5, disk_dir="", disk_keys=('images',))


def test_unknown_key_raises(tmp_path):
    with pytest.raises(ValueError, match=r"\['imgs'\] not in the transition"):
        ReplayBuffer.create(_pi05_transition(), 5, disk_dir=str(tmp_path), disk_keys=('images', 'imgs'))
    assert os.listdir(tmp_path) == []


class _Statvfs:
    def __init__(self, free):
        self.f_bavail, self.f_frsize = free, 1


def test_insufficient_space_raises(tmp_path, monkeypatch):
    size = 11
    need = 2 * size * int(np.prod(IMG_SHAPE))  # uint8
    monkeypatch.setattr(D.os, "statvfs", lambda p: _Statvfs(need - 1))
    with pytest.raises(ValueError, match="GB free"):
        ReplayBuffer.create(_pi05_transition(), size, disk_dir=str(tmp_path), disk_keys=('images', 'next_images'))
    assert os.listdir(tmp_path) == []  # refused before any file is created
    monkeypatch.setattr(D.os, "statvfs", lambda p: _Statvfs(need))  # exactly enough
    rb = ReplayBuffer.create(_pi05_transition(), size, disk_dir=str(tmp_path), disk_keys=('images', 'next_images'))
    assert rb._dict['images'].shape == (size, *IMG_SHAPE)


def test_make_buffer_storage_dir_falsy_is_ram():
    feats, imgs = _example()
    rb = R._make_buffer(feats, imgs, 5, 0, ACT_DIM, storage_dir="")
    assert all(type(v) is np.ndarray for v in rb._dict.values())


# ===========================================================================
# 5. the memmap files
# ===========================================================================

def test_memmap_files_on_disk(tmp_path):
    rb = _make(20, storage_dir=tmp_path)
    assert sorted(os.listdir(tmp_path)) == ['images.npy', 'next_images.npy']
    for k in ('images', 'next_images'):
        a = np.load(tmp_path / f"{k}.npy", mmap_mode="r")
        assert a.shape == (20, *IMG_SHAPE) and a.dtype == np.uint8 and not a.any()
    _feed(rb, _episodes(3, 6))
    for k in ('images', 'next_images'):  # MAP_SHARED: visible to a second mapping without a flush
        assert np.array_equal(np.load(tmp_path / f"{k}.npy", mmap_mode="r"), rb._dict[k])
        assert rb._dict[k].any()
    for k in ('images', 'next_images'):
        rb._dict[k].flush()
        assert np.array_equal(np.load(tmp_path / f"{k}.npy"), rb._dict[k])


def test_second_buffer_in_same_dir_zeroes_the_first(tmp_path):
    """Documents a gotcha (not reachable from the runner, which makes one buffer per process): open_memmap(w+)
    truncates and re-extends the same inode, so an earlier buffer in the same dir silently reads zeros."""
    a = _make(10, storage_dir=tmp_path)
    _feed(a, _episodes(4, 3))
    assert a._dict['images'].any()
    _make(10, storage_dir=tmp_path)
    assert not a._dict['images'].any()


# ===========================================================================
# 6. config
# ===========================================================================

def test_config_storage_dir_default_and_override():
    from ogpo.utils.config_loader import load_config, parse_cli_args
    assert load_config('ogpo')['buffer']['storage_dir'] is None
    path = "/scratch/someone/ogpo_rb_ogpo_molmo_mt64_pi05_bon_s0_10934680"
    algo, ov = parse_cli_args(['--algo=ogpo', f'--buffer.storage_dir={path}'])
    assert load_config(algo, ov)['buffer']['storage_dir'] == path
