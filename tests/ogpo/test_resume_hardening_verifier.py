"""Independent verification of docs/changes/2026-08-27-resume-hardening/.

Written by the step-3 verifier from the change spec + the diff, without the
implementing session's reasoning. Adversarial: every claim in BLAST-RADIUS §4.6
(the crash table), §4.2 (the rebase), §4.3 (the cutoff) and §9 is re-derived here
rather than restated from the implementation.

Two things distinguish these tests from ``test_resume_hardening.py``:

1. ``restore_shards`` is a refactor of a data path, so the no-cutoff behavior is
   checked **differentially against a verbatim copy of the pre-change
   implementation** (``_restore_shards_pre_change`` below, transcribed from
   ``git show 5b94510:src/rl/replay_buffer.py``). Comparing the new code to
   itself would prove nothing. The copy is frozen in this file on purpose: a
   ``git show HEAD`` differential silently degenerates into a self-comparison the
   moment the change is committed — exactly the rot that has already disabled
   three ``test_verifier_alignment.py`` legs.
2. The crash table is exercised through the **real** ``save_epoch_state`` against
   a stand-in agent that emulates orbax's ``max_to_keep=1`` GC, killing the
   process at each write point, and then resolving the surviving tree with the
   real ``resolve_resume_step``. The implementer's tests assert the resolver
   given hand-written step sets; these assert that the implemented save order
   actually produces those step sets.

CPU-only, model-free. ``OGPOAgentLearner.__init__`` device_puts the EMA to
``pinned_host`` and is GPU-only, so learner methods are called unbound against
stand-ins (or on ``object.__new__`` instances that never ran ``__init__``).
"""
import dataclasses
import json
import logging
import types
from pathlib import Path

import numpy as np
import pytest

from src.rl.advantage_weighted_sft.advantage_weighted_sft_learner import (
    AdvantageWeightedSFTLearner,
)
from src.rl.best_of_n.best_of_n_learner import BestofNLearner
from src.rl.filtered_sft_agent.filtered_sft_learner import FilteredSFTLearner
from src.rl.flow_grpo.flow_grpo_learner import FlowGRPOLearner
from src.rl.mpo_weighted_sft.mpo_weighted_sft_learner import MPOWeightedSFTLearner
from src.rl.ogpo.ogpo_learner import OGPOAgentLearner, _rebase_task_ranges
from src.rl.replay_buffer import ShardedReplayBuffer
from src.rl.dataset import read_nested
from src.training.runtime_state import (
    ResumeState,
    load_resume_state,
    resolve_resume_step,
    resume_state_path,
    save_epoch_state,
    step_manifest_path,
    step_manifest_steps,
    success_shard_dir,
    success_shard_path,
    replay_shard_dir,
    replay_shard_path,
)

import h5py

_S, _AH, _AD = 3, 2, 4


# --------------------------------------------------------------------------- #
# VERBATIM pre-change implementation (git 5b94510:src/rl/replay_buffer.py:257-)
# Transcribed as a free function taking the buffer as `self`; the body below is
# character-for-character the pre-change method body.
# --------------------------------------------------------------------------- #

def _restore_shards_pre_change(self, shard_dir, *, rng_state_json=None):
    shard_dir = Path(shard_dir)
    if not shard_dir.exists():
        raise FileNotFoundError(f"Replay shard directory does not exist: {shard_dir}")

    self.ptr = 0
    self.size = 0
    self.total_inserted = 0
    self.obs_ptr = 0
    self.obs_total = 0
    self.valid_start = 0
    shard_paths = sorted(shard_dir.glob("step_*.h5"))

    for shard_path in shard_paths:
        with h5py.File(shard_path, "r") as f:
            restored = {
                "observations": read_nested(f["observations"]),
                "obs_index": f["links/obs_index"][()],
                "next_obs_index": f["links/next_obs_index"][()],
            }
            restored.update(read_nested(f["transitions"]))
        self.insert(restored)

    self.persisted_total_inserted = self.total_inserted
    self.set_rng_state_json(rng_state_json)

    logging.info(
        "Restored replay buffer from shards in %s (transitions=%d)",
        shard_dir, self.size,
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _buffer_dummy() -> dict:
    return {
        "observations": {"state": np.zeros((1, _S), np.float32)},
        "actions": np.zeros((1, _AH, _AD), np.float32),
        "reward": np.zeros((1,), np.float32),
        "mc_return": np.zeros((1,), np.float32),
        "discount": np.zeros((1,), np.float32),
        "is_success": np.zeros((1,), np.float32),
    }


def _new_buffer(max_capacity: int, seed: int = 0) -> ShardedReplayBuffer:
    return ShardedReplayBuffer(
        dummy_data=_buffer_dummy(),
        max_capacity=max_capacity,
        seed=seed,
        freeze_dict=False,
    )


def _insert(buf: ShardedReplayBuffer, n: int, task: float = 0.0) -> tuple[int, int]:
    """Insert `n` transitions stamped with their global ordinal. Returns (lo, hi)."""
    lo = buf.total_inserted
    buf.insert(
        {
            "observations": {
                "state": np.repeat(
                    (lo + np.arange(n + 1, dtype=np.float32))[:, None], _S, axis=1
                )
            },
            "obs_index": np.arange(n, dtype=np.int64),
            "next_obs_index": np.arange(n, dtype=np.int64) + 1,
            "actions": np.full((n, _AH, _AD), float(lo), np.float32),
            "reward": np.full((n,), task, np.float32),
            "mc_return": lo + np.arange(n, dtype=np.float32),
            "discount": np.full((n,), 0.99, np.float32),
            "is_success": np.ones((n,), np.float32),
        }
    )
    return lo, buf.total_inserted


def _full_state(buf: ShardedReplayBuffer) -> dict:
    """Every piece of restorable buffer state, for a byte-level differential."""
    return {
        "ptr": buf.ptr,
        "size": buf.size,
        "total_inserted": buf.total_inserted,
        "obs_ptr": buf.obs_ptr,
        "obs_total": buf.obs_total,
        "valid_start": buf.valid_start,
        "persisted_total_inserted": buf.persisted_total_inserted,
        "obs_pos": buf.obs_pos.copy(),
        "next_obs_pos": buf.next_obs_pos.copy(),
        "obs_state": buf.obs_storage["state"].copy(),
        "reward": buf.storage["reward"].copy(),
        "mc_return": buf.storage["mc_return"].copy(),
        "discount": buf.storage["discount"].copy(),
        "is_success": buf.storage["is_success"].copy(),
        "actions": buf.storage["actions"].copy(),
        "rng": buf.rng_state_json(),
    }


def _assert_same_state(a: dict, b: dict, what: str) -> None:
    assert set(a) == set(b)
    for k in sorted(a):
        if isinstance(a[k], np.ndarray):
            np.testing.assert_array_equal(a[k], b[k], err_msg=f"{what}: {k} differs")
        else:
            assert a[k] == b[k], f"{what}: {k} differs ({a[k]!r} != {b[k]!r})"


@pytest.fixture(scope="module")
def shard_tree(tmp_path_factory):
    """Three shards, unclipped, plus the source buffer's state at each save.

    Module-scoped: written once, read by several differential legs.
    """
    root = tmp_path_factory.mktemp("shards")
    d = root / "replay_shards"
    src = _new_buffer(64)
    at_step = {}
    for step, n in ((0, 5), (1000, 4), (2000, 6)):
        _insert(src, n)
        src.save_shard(d / f"step_{step:08d}.h5")
        at_step[step] = src.total_inserted
    return types.SimpleNamespace(dir=d, at_step=at_step, total=src.total_inserted)


# --------------------------------------------------------------------------- #
# A. restore_shards(max_step=None) is byte-identical to the pre-change code
# --------------------------------------------------------------------------- #

def test_no_cutoff_is_bit_identical_to_the_pre_change_implementation(shard_tree):
    """Differential against the verbatim pre-change body, not against itself."""
    new = _new_buffer(64)
    new.restore_shards(shard_tree.dir)
    old = _new_buffer(64)
    _restore_shards_pre_change(old, shard_tree.dir)
    _assert_same_state(_full_state(new), _full_state(old), "unclipped no-cutoff")
    assert new.total_inserted == 15


def test_no_cutoff_differential_with_an_rng_state(shard_tree):
    donor = _new_buffer(64, seed=1234)
    donor._rng.integers(0, 10, size=7)  # advance it so the state is non-initial
    rng_json = donor.rng_state_json()

    new = _new_buffer(64)
    new.restore_shards(shard_tree.dir, rng_state_json=rng_json)
    old = _new_buffer(64)
    _restore_shards_pre_change(old, shard_tree.dir, rng_state_json=rng_json)
    _assert_same_state(_full_state(new), _full_state(old), "no-cutoff + rng")
    assert json.loads(new.rng_state_json()) == json.loads(rng_json)


def test_no_cutoff_differential_in_the_clipped_case(tmp_path):
    """The save_shard delta clip (replay_buffer.py:213) is the only path that
    moves ordinals; the refactor must not perturb it."""
    d = tmp_path / "clipped"
    cap = 16
    src = _new_buffer(cap)
    _insert(src, 6)
    src.save_shard(d / "step_00000000.h5")
    for _ in range(4):
        _insert(src, 6)
    src.save_shard(d / "step_00001000.h5")

    new = _new_buffer(cap)
    new.restore_shards(d)
    old = _new_buffer(cap)
    _restore_shards_pre_change(old, d)
    _assert_same_state(_full_state(new), _full_state(old), "clipped no-cutoff")
    assert new.total_inserted < src.total_inserted, "test did not reach the clip"


def test_no_cutoff_differential_on_an_empty_shard_directory(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    new = _new_buffer(32)
    _insert(new, 4)  # dirty it first, so the reset path is exercised
    new.restore_shards(d)
    old = _new_buffer(32)
    _insert(old, 4)
    _restore_shards_pre_change(old, d)
    _assert_same_state(_full_state(new), _full_state(old), "empty dir")
    assert (new.total_inserted, new.size, new.valid_start) == (0, 0, 0)


def test_no_cutoff_differential_on_a_missing_shard_directory(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError) as new_exc:
        _new_buffer(32).restore_shards(missing)
    with pytest.raises(FileNotFoundError) as old_exc:
        _restore_shards_pre_change(_new_buffer(32), missing)
    assert str(new_exc.value) == str(old_exc.value)


def test_no_cutoff_still_ignores_shard_name_parsing(tmp_path):
    """DIFF divergence D3: `_shard_step` runs only under a cutoff, so a foreign
    `step_*.h5` reaches h5py exactly as before rather than raising the new
    ValueError. Asserted so the divergence is a decision, not an accident."""
    d = tmp_path / "foreign"
    src = _new_buffer(32)
    _insert(src, 4)
    src.save_shard(d / "step_00000000.h5")
    (d / "step_latest.h5").write_bytes(b"not an hdf5 file")

    with pytest.raises(Exception) as new_exc:
        _new_buffer(32).restore_shards(d)
    with pytest.raises(Exception) as old_exc:
        _restore_shards_pre_change(_new_buffer(32), d)
    assert type(new_exc.value) is type(old_exc.value)
    assert not isinstance(new_exc.value, ValueError) or "foreign file" not in str(
        new_exc.value
    )


# --------------------------------------------------------------------------- #
# A2. the cutoff itself, differentially against the pre-change code on a
#     directory that only ever held the retained shards
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("cut", [0, 1000, 2000])
def test_cutoff_equals_pre_change_restore_of_a_truncated_directory(shard_tree, tmp_path, cut):
    truncated = tmp_path / f"trunc_{cut}"
    truncated.mkdir()
    for p in sorted(shard_tree.dir.glob("step_*.h5")):
        if int(p.stem[len("step_"):]) <= cut:
            (truncated / p.name).write_bytes(p.read_bytes())

    new = _new_buffer(64)
    new.restore_shards(shard_tree.dir, max_step=cut)
    old = _new_buffer(64)
    _restore_shards_pre_change(old, truncated)
    _assert_same_state(_full_state(new), _full_state(old), f"cutoff {cut}")
    assert new.total_inserted == shard_tree.at_step[cut]


def test_cutoff_below_every_shard_yields_an_empty_buffer(shard_tree):
    buf = _new_buffer(64)
    buf.restore_shards(shard_tree.dir, max_step=-1)
    assert (buf.total_inserted, buf.size, buf.valid_start) == (0, 0, 0)
    assert buf.persisted_total_inserted == 0


def test_cutoff_is_inclusive_and_rejects_a_foreign_name(shard_tree, tmp_path):
    d = tmp_path / "foreign_cut"
    d.mkdir()
    for p in shard_tree.dir.glob("step_*.h5"):
        (d / p.name).write_bytes(p.read_bytes())
    (d / "step_final.h5").write_bytes(b"")
    with pytest.raises(ValueError, match="foreign file"):
        _new_buffer(64).restore_shards(d, max_step=1000)


# --------------------------------------------------------------------------- #
# B. the §4.6 crash table, driven through the REAL save_epoch_state
# --------------------------------------------------------------------------- #

class _Boom(RuntimeError):
    """Stands in for SIGKILL at a chosen point inside save_epoch_state."""


class _CrashingBuffer(ShardedReplayBuffer):
    fail = False

    def save_shard(self, path):
        if self.fail:
            # Emulate a torn write: the atomic writer's temp file is discarded,
            # so nothing lands at `path`.
            raise _Boom("crash during the online shard write")
        return super().save_shard(path)


class _FakeCkptMgr:
    """Emulates openpi's CheckpointManager with max_to_keep=1."""

    def __init__(self, root: Path):
        self.root = root

    def all_steps(self):
        if not self.root.exists():
            return []
        return sorted(int(p.name) for p in self.root.iterdir() if p.name.isdigit())

    def commit(self, step: int):
        for p in list(self.root.iterdir()) if self.root.exists() else []:
            if p.name.isdigit():
                for q in sorted(p.rglob("*"), reverse=True):
                    q.unlink()
                p.rmdir()
        (self.root / str(step)).mkdir(parents=True)
        (self.root / str(step) / "params").write_text(str(step))

    def wait_until_finished(self):
        pass


class _FakeAgent:
    """Minimal stand-in with OGPO's + AWR's write shape, and a kill switch."""

    def __init__(self, ckpt_dir: Path, fail_at: str | None = None):
        self.config = types.SimpleNamespace(checkpoint_dir=str(ckpt_dir))
        self.training_steps = 0
        self.total_collected_episodes = 0
        self.fail_at = fail_at
        self._ckpt_root = ckpt_dir / "orbax"
        self._ckpt_root.mkdir(parents=True, exist_ok=True)
        self._checkpoint_manager = _FakeCkptMgr(self._ckpt_root)
        self._online_data_buffer = _CrashingBuffer(
            dummy_data=_buffer_dummy(), max_capacity=256, seed=0, freeze_dict=False
        )
        self._success_data_buffer = _new_buffer(256, seed=1)
        self._success_task_ranges: dict[str, list[tuple[int, int]]] = {}
        self._adv_scale = 1.0
        self._rl_state_dir = ckpt_dir / "rl_state"

    # --- the two hooks save_epoch_state calls -----------------------------
    def rng_state_json(self) -> str:
        return json.dumps({"step": self.training_steps})

    def save_extra_resume_state(self, step: int) -> dict:
        if self.fail_at == "extra":
            raise _Boom("crash inside save_extra_resume_state")
        self._success_data_buffer.save_shard(success_shard_path(self.config, step))
        return {
            "adv_scale": float(self._adv_scale),
            "success_shard_dir": str(success_shard_dir(self.config)),
            "success_total_inserted": int(self._success_data_buffer.total_inserted),
            "success_task_ranges": {
                t: [[int(lo), int(hi)] for lo, hi in rs]
                for t, rs in self._success_task_ranges.items()
            },
            "success_rng_state_json": self._success_data_buffer.rng_state_json(),
        }

    def save_checkpoint(self, step: int | None = None):
        # AWR's implemented order: rl_state, then the orbax commit (which GCs).
        if self.fail_at == "rl_state":
            raise _Boom("crash before the rl_state write")
        self._rl_state_dir.mkdir(parents=True, exist_ok=True)
        (self._rl_state_dir / str(int(step))).mkdir(exist_ok=True)
        if self.fail_at == "orbax":
            raise _Boom("crash after rl_state, before the orbax commit")
        self._checkpoint_manager.commit(int(step))
        if self.fail_at == "post_orbax":
            raise _Boom("crash after the orbax commit, before the pointer refresh")

    # --- what a resume would do -------------------------------------------
    def resume_required_paths(self, step: int):
        return [self._rl_state_dir / str(int(step))]

    def resolve(self) -> int:
        pointer = None
        if resume_state_path(self.config).exists():
            pointer = int(load_resume_state(self.config).step)
        return resolve_resume_step(
            orbax_steps=set(self._checkpoint_manager.all_steps()),
            manifest_steps=step_manifest_steps(self.config),
            required_ok=lambda s: all(p.exists() for p in self.resume_required_paths(s)),
            pointer_step=pointer,
        )


def _advance_and_save(agent: _FakeAgent, step: int, n_online: int, n_success: int):
    agent.training_steps = step
    _insert(agent._online_data_buffer, n_online)
    lo, hi = _insert(agent._success_data_buffer, n_success)
    agent._success_task_ranges.setdefault("t", []).append((lo, hi))
    agent.total_collected_episodes += 1
    save_epoch_state(agent, agent.config, prepare_for_resume=True)


def _kill_at_nth_atomic_write(monkeypatch, n: int) -> None:
    """SIGKILL stand-in for the k-th `_atomic_write_text` inside save_epoch_state.

    n=1 kills the per-step manifest write (between (2) and (3)); n=2 kills the
    pointer refresh (between (4) and (5)).
    """
    from src.training import runtime_state as _rs

    real = _rs._atomic_write_text
    calls = {"n": 0}

    def fake(path, text):
        calls["n"] += 1
        if calls["n"] == n:
            raise _Boom(f"crash on atomic write #{n} ({path})")
        return real(path, text)

    monkeypatch.setattr(_rs, "_atomic_write_text", fake)


@pytest.mark.parametrize(
    "fail_at,expected_step,expect_shard_n,expect_success_shard_n,expect_manifest_n",
    [
        # BLAST-RADIUS §4.6 crash table, row by row.
        ("shard", 10000, False, False, False),      # during (1)
        ("extra", 10000, True, False, False),       # between (1) and (2)
        ("manifest", 10000, True, True, False),     # between (2) and (3)
        ("rl_state", 10000, True, True, True),      # inside (4), before rl_state
        ("orbax", 10000, True, True, True),         # inside (4), before the commit
        ("post_orbax", 20000, True, True, True),    # inside (4), after the commit
        ("pointer", 20000, True, True, True),       # between (4) and (5)
        (None, 20000, True, True, True),            # no crash
    ],
)
def test_crash_table_rows_resolve_to_the_claimed_step(
    tmp_path, monkeypatch, fail_at, expected_step, expect_shard_n,
    expect_success_shard_n, expect_manifest_n,
):
    ckpt = tmp_path / "ckpt"
    agent = _FakeAgent(ckpt)
    _advance_and_save(agent, 10000, 7, 5)          # step M, clean
    assert agent.resolve() == 10000

    agent.fail_at = fail_at if fail_at not in ("manifest", "pointer") else None
    agent._online_data_buffer.fail = fail_at == "shard"
    if fail_at == "manifest":
        _kill_at_nth_atomic_write(monkeypatch, 1)
    elif fail_at == "pointer":
        _kill_at_nth_atomic_write(monkeypatch, 2)
    if fail_at is None:
        _advance_and_save(agent, 20000, 6, 4)
    else:
        with pytest.raises(_Boom):
            _advance_and_save(agent, 20000, 6, 4)

    # 1. what is on disk
    assert replay_shard_path(agent.config, 20000).exists() is expect_shard_n
    assert success_shard_path(agent.config, 20000).exists() is expect_success_shard_n
    assert step_manifest_path(agent.config, 20000).is_file() is expect_manifest_n

    # 2. the resolver picks the claimed step
    assert agent.resolve() == expected_step

    # 3. and the tree it picks is internally consistent
    step = agent.resolve()
    assert step in agent._checkpoint_manager.all_steps()
    assert (agent._rl_state_dir / str(step)).exists()
    state = load_resume_state(
        agent.config, step=step if step in step_manifest_steps(agent.config) else None
    )
    assert state.step == step
    assert state.extra["adv_scale"] == 1.0

    # 4. restoring at that step never ingests a newer shard
    restored = _new_buffer(256)
    restored.restore_shards(replay_shard_dir(agent.config), max_step=step)
    reference_total = 7 if step == 10000 else 13
    assert restored.total_inserted == reference_total


def test_pointer_stays_behind_after_a_post_commit_crash_and_the_resolver_wins(tmp_path):
    """§4.6 last row: the pointer still names M while N is fully durable."""
    ckpt = tmp_path / "ckpt"
    agent = _FakeAgent(ckpt)
    _advance_and_save(agent, 10000, 7, 5)
    agent.fail_at = "post_orbax"
    with pytest.raises(_Boom):
        _advance_and_save(agent, 20000, 6, 4)

    assert int(load_resume_state(agent.config).step) == 10000  # stale pointer
    assert agent.resolve() == 20000                            # resolver wins
    # ...and nothing is lost: the step-20000 manifest carries the newer counts.
    assert load_resume_state(agent.config, step=20000).total_collected_episodes == 2


def test_save_epoch_state_writes_the_pointer_last_and_identical_to_the_step_manifest(tmp_path):
    agent = _FakeAgent(tmp_path / "ckpt")
    _advance_and_save(agent, 10000, 7, 5)
    step_text = step_manifest_path(agent.config, 10000).read_text()
    assert resume_state_path(agent.config).read_text() == step_text
    payload = json.loads(step_text)
    assert set(payload) == {
        "step",
        "agent_rng_state_json",
        "total_collected_episodes",
        "replay_shard_dir",
        "replay_rng_state_json",
        "extra",
    }
    assert ResumeState(**payload).extra["success_total_inserted"] == 5


def test_prepare_for_resume_false_writes_only_the_checkpoint(tmp_path):
    agent = _FakeAgent(tmp_path / "ckpt")
    agent.training_steps = 500
    save_epoch_state(agent, agent.config, prepare_for_resume=False)
    assert agent._checkpoint_manager.all_steps() == [500]
    assert not resume_state_path(agent.config).exists()
    assert step_manifest_steps(agent.config) == set()
    assert not replay_shard_dir(agent.config).exists()


def test_pointer_file_is_not_mistaken_for_a_per_step_manifest(tmp_path):
    agent = _FakeAgent(tmp_path / "ckpt")
    _advance_and_save(agent, 10000, 7, 5)
    assert resume_state_path(agent.config).exists()
    assert step_manifest_steps(agent.config) == {10000}


def test_legacy_directory_without_per_step_manifests_falls_back_to_the_pointer(tmp_path):
    """Every in-flight run and both --resume probes take this path (D1 case 1)."""
    agent = _FakeAgent(tmp_path / "ckpt")
    _advance_and_save(agent, 10000, 7, 5)
    step_manifest_path(agent.config, 10000).unlink()  # pre-change tree shape
    assert step_manifest_steps(agent.config) == set()
    assert agent.resolve() == 10000
    assert load_resume_state(agent.config).step == 10000


def test_mixed_legacy_tree_should_still_resume_the_complete_older_step(tmp_path):
    agent = _FakeAgent(tmp_path / "ckpt")
    _advance_and_save(agent, 10000, 7, 5)
    step_manifest_path(agent.config, 10000).unlink()   # pre-change tree shape at M
    # First post-change boundary: manifest N lands, the orbax commit does not.
    agent.fail_at = "orbax"
    with pytest.raises(_Boom):
        _advance_and_save(agent, 20000, 6, 4)

    # Step 10000 is complete: orbax has it, the pointer names it, rl_state exists.
    assert agent._checkpoint_manager.all_steps() == [10000]
    assert int(load_resume_state(agent.config).step) == 10000
    assert (agent._rl_state_dir / "10000").exists()
    assert step_manifest_steps(agent.config) == {20000}
    assert agent.resolve() == 10000  # raises FileNotFoundError instead


def test_legacy_pointer_naming_a_gc_d_checkpoint_raises(tmp_path):
    """D1 case 3."""
    agent = _FakeAgent(tmp_path / "ckpt")
    _advance_and_save(agent, 10000, 7, 5)
    step_manifest_path(agent.config, 10000).unlink()
    agent._checkpoint_manager.commit(30000)  # GC'd 10000, pointer still says 10000
    with pytest.raises(FileNotFoundError, match="no per-step"):
        agent.resolve()


def test_missing_rl_state_for_the_newest_step_falls_back_one_step(tmp_path):
    """`keep_period` pins an older orbax step; the newest one lost its sidecar."""
    agent = _FakeAgent(tmp_path / "ckpt")
    _advance_and_save(agent, 10000, 7, 5)
    pinned = agent._ckpt_root / "10000"
    _advance_and_save(agent, 20000, 6, 4)
    pinned.mkdir(exist_ok=True)  # emulate keep_period pinning 10000
    (agent._rl_state_dir / "20000").rmdir()
    assert agent.resolve() == 10000


# --------------------------------------------------------------------------- #
# C. _rebase_task_ranges vs an independent reference (BLAST-RADIUS §4.2)
# --------------------------------------------------------------------------- #

def _reference_rebase(saved, saved_total, restored_total, valid_start):
    """Spec transcription, written from BLAST-RADIUS §4.2 rather than the code:

        shift = restored - saved
        shift every (lo, hi)
        drop ranges with hi + shift <= valid_start
        clamp hi = min(hi + shift, total_inserted)
    """
    shift = restored_total - saved_total
    out = {}
    for task, ranges in saved.items():
        kept = []
        for lo, hi in ranges:
            if hi + shift <= valid_start:
                continue
            lo2, hi2 = lo + shift, min(hi + shift, restored_total)
            if hi2 <= lo2:
                continue
            kept.append((lo2, hi2))
        if kept:
            out[task] = kept
    return out


def test_rebase_matches_the_spec_transcription_on_randomized_inputs():
    rng = np.random.default_rng(20260827)
    for _ in range(400):
        saved_total = int(rng.integers(1, 200))
        restored_total = int(rng.integers(0, saved_total + 1))
        valid_start = int(rng.integers(0, restored_total + 1))
        saved = {}
        for t in range(int(rng.integers(1, 5))):
            ranges = []
            for _ in range(int(rng.integers(1, 4))):
                lo = int(rng.integers(0, saved_total))
                hi = int(rng.integers(lo + 1, saved_total + 1))
                ranges.append((lo, hi))
            saved[f"t{t}"] = ranges
        got = _rebase_task_ranges(saved, saved_total, restored_total, valid_start)
        want = _reference_rebase(saved, saved_total, restored_total, valid_start)
        assert got == want, (saved, saved_total, restored_total, valid_start)


def test_rebase_output_is_always_samplable(tmp_path):
    """The invariant that matters: every rebased range must survive
    `_balanced_success_ordinals` -> `sample(ordinals=...)`'s guard."""
    rng = np.random.default_rng(7)
    for _ in range(200):
        saved_total = int(rng.integers(4, 80))
        restored_total = int(rng.integers(1, saved_total + 1))
        valid_start = int(rng.integers(0, restored_total))
        saved = {
            f"t{t}": [
                (lo, min(lo + int(rng.integers(1, 12)), saved_total))
                for lo in [int(rng.integers(0, saved_total - 1))]
            ]
            for t in range(3)
        }
        out = _rebase_task_ranges(saved, saved_total, restored_total, valid_start)
        for task, ranges in out.items():
            for lo, hi in ranges:
                assert hi <= restored_total, (task, lo, hi, restored_total)
                assert hi > valid_start
                assert hi > lo


def test_rebase_is_identity_when_nothing_was_dropped():
    saved = {"a": [(0, 4), (9, 12)], "b": [(4, 9)]}
    assert _rebase_task_ranges(saved, 12, 12, 0) == saved


def test_rebase_drops_a_task_whose_every_range_was_evicted():
    out = _rebase_task_ranges({"a": [(0, 5)], "b": [(5, 20)]}, 20, 20, 10)
    assert set(out) == {"b"} and out["b"] == [(5, 20)]


# --------------------------------------------------------------------------- #
# D. OGPO restore: the raise paths and the happy path, unbound
# --------------------------------------------------------------------------- #

def _rng_json(seed: int = 5) -> str:
    """A real PCG64 state — `restore_shards` feeds it to `set_rng_state_json`,
    which rejects anything else."""
    return _new_buffer(4, seed=seed).rng_state_json()


def _ogpo_stand_in(extra, step=20000, success_buf=None, adv_scale=0.125):
    return types.SimpleNamespace(
        _resume_state=ResumeState(
            step=step,
            total_collected_episodes=3,
            replay_shard_dir="/unused",
            agent_rng_state_json="{}",
            replay_rng_state_json="{}",
            extra=extra,
        ),
        _success_data_buffer=success_buf,
        _success_task_ranges={},
        _adv_scale=adv_scale,
    )


def test_restore_extra_degrades_on_a_pre_change_manifest(caplog):
    me = _ogpo_stand_in(extra={}, success_buf=_new_buffer(32))
    with caplog.at_level(logging.WARNING):
        OGPOAgentLearner._restore_extra_resume_state(me)
    assert "no `extra` block" in caplog.text
    assert me._adv_scale == 0.125          # untouched: today's default stands
    assert me._success_task_ranges == {}
    assert me._success_data_buffer.total_inserted == 0


def test_restore_extra_raises_when_the_manifest_has_state_but_the_run_has_no_buffer():
    me = _ogpo_stand_in(
        extra={"adv_scale": 3.0, "success_shard_dir": "/nope",
               "success_total_inserted": 4, "success_task_ranges": {},
               "success_rng_state_json": _rng_json()},
        success_buf=None,
    )
    with pytest.raises(ValueError, match="use_success_buffer off"):
        OGPOAgentLearner._restore_extra_resume_state(me)


def test_restore_extra_raises_when_the_run_has_a_buffer_but_the_manifest_does_not():
    me = _ogpo_stand_in(extra={"adv_scale": 3.0}, success_buf=_new_buffer(32))
    with pytest.raises(ValueError, match="no success-buffer state"):
        OGPOAgentLearner._restore_extra_resume_state(me)


def test_restore_extra_restores_adv_scale_even_with_the_success_buffer_off():
    me = _ogpo_stand_in(extra={"adv_scale": 7.5}, success_buf=None)
    OGPOAgentLearner._restore_extra_resume_state(me)
    assert me._adv_scale == 7.5


def test_restore_extra_raises_on_a_missing_shard_directory(tmp_path):
    me = _ogpo_stand_in(
        extra={"adv_scale": 1.0, "success_shard_dir": str(tmp_path / "gone"),
               "success_total_inserted": 8, "success_task_ranges": {},
               "success_rng_state_json": _rng_json()},
        success_buf=_new_buffer(32),
    )
    with pytest.raises(FileNotFoundError, match="does not exist"):
        OGPOAgentLearner._restore_extra_resume_state(me)


def test_restore_extra_raises_when_the_declared_shard_is_newer_than_the_step(tmp_path):
    d = tmp_path / "success_shards"
    src = _new_buffer(32)
    _insert(src, 8)
    src.save_shard(d / "step_00030000.h5")  # written for a step ahead of the resume
    me = _ogpo_stand_in(
        extra={"adv_scale": 1.0, "success_shard_dir": str(d),
               "success_total_inserted": 8, "success_task_ranges": {"a": [[0, 8]]},
               "success_rng_state_json": _rng_json()},
        success_buf=_new_buffer(32),
        step=20000,
    )
    with pytest.raises(FileNotFoundError, match="holds no shard at or below"):
        OGPOAgentLearner._restore_extra_resume_state(me)


def test_restore_extra_happy_path_restores_buffer_ranges_and_scale(tmp_path):
    d = tmp_path / "success_shards"
    src = _new_buffer(64)
    ranges = {"a": [], "b": []}
    for i in range(4):
        task = "a" if i % 2 == 0 else "b"
        ranges[task].append(_insert(src, 6, task=float(i % 2)))
    src.save_shard(d / "step_00020000.h5")
    saved_total = src.total_inserted

    donor = _new_buffer(64, seed=99)
    donor._rng.integers(0, 5, size=3)
    me = _ogpo_stand_in(
        extra={
            "adv_scale": 2.75,
            "success_shard_dir": str(d),
            "success_total_inserted": saved_total,
            "success_task_ranges": {t: [[lo, hi] for lo, hi in rs] for t, rs in ranges.items()},
            "success_rng_state_json": donor.rng_state_json(),
        },
        success_buf=_new_buffer(64),
        step=20000,
    )
    OGPOAgentLearner._restore_extra_resume_state(me)
    buf = me._success_data_buffer
    assert buf.total_inserted == saved_total
    assert me._adv_scale == 2.75
    assert me._success_task_ranges == {t: [(lo, hi) for lo, hi in rs] for t, rs in ranges.items()}
    assert json.loads(buf.rng_state_json()) == json.loads(donor.rng_state_json())

    # ...and the package property: the restored ranges drive a balanced sample.
    ords = OGPOAgentLearner._balanced_success_ordinals(me, 8)
    batch = buf.sample(batch_size=8, ordinals=ords)
    counts = np.bincount(np.asarray(batch["reward"]).astype(np.int64), minlength=2)
    np.testing.assert_array_equal(counts, [4, 4])


def test_restore_extra_ignores_a_shard_newer_than_the_resolved_step(tmp_path):
    """A crash between the success-shard write and the orbax commit leaves a
    shard the restored weights never saw; the cutoff must drop it."""
    d = tmp_path / "success_shards"
    src = _new_buffer(64)
    for i in range(2):
        _insert(src, 6, task=float(i))
    src.save_shard(d / "step_00020000.h5")
    saved_total = src.total_inserted
    saved_ranges = {"a": [[0, 6]], "b": [[6, 12]]}
    _insert(src, 6, task=0.0)
    src.save_shard(d / "step_00030000.h5")   # the orphan

    me = _ogpo_stand_in(
        extra={"adv_scale": 1.0, "success_shard_dir": str(d),
               "success_total_inserted": saved_total,
               "success_task_ranges": saved_ranges,
               "success_rng_state_json": _rng_json()},
        success_buf=_new_buffer(64),
        step=20000,
    )
    OGPOAgentLearner._restore_extra_resume_state(me)
    assert me._success_data_buffer.total_inserted == saved_total == 12


def test_success_and_online_shards_live_in_separate_directories(tmp_path):
    cfg = types.SimpleNamespace(checkpoint_dir=str(tmp_path))
    assert success_shard_dir(cfg) != replay_shard_dir(cfg)
    assert success_shard_dir(cfg).name == "success_shards"
    assert replay_shard_dir(cfg).name == "replay_shards"


# --------------------------------------------------------------------------- #
# E. inheritance / clone family
# --------------------------------------------------------------------------- #

def test_base_hooks_are_no_ops_and_only_ogpo_overrides_the_save_hook():
    me = object.__new__(FilteredSFTLearner)
    assert FilteredSFTLearner.save_extra_resume_state(me, 123) == {}
    assert FilteredSFTLearner._resume_required_paths(me, 123) == []
    overriders = [
        c for c in (AdvantageWeightedSFTLearner, BestofNLearner, MPOWeightedSFTLearner,
                    FlowGRPOLearner, OGPOAgentLearner)
        if "save_extra_resume_state" in c.__dict__
    ]
    assert overriders == [OGPOAgentLearner]


def test_only_awr_declares_resume_required_paths_bofn_keeps_oq6():
    assert "_resume_required_paths" in AdvantageWeightedSFTLearner.__dict__
    assert "_resume_required_paths" not in BestofNLearner.__dict__
    # MPO / FlowGRPO inherit AWR's, so they require rl_state too.
    assert MPOWeightedSFTLearner._resume_required_paths is (
        AdvantageWeightedSFTLearner._resume_required_paths
    )
    assert FlowGRPOLearner._resume_required_paths is (
        AdvantageWeightedSFTLearner._resume_required_paths
    )
    # BofN still warns rather than raising on a missing rl_state (OQ-6): the
    # warn lives in its own _restore_rl_checkpoint, which AWR does not share.
    assert "_restore_rl_checkpoint" in BestofNLearner.__dict__
    import inspect
    assert "starting critics from scratch" in inspect.getsource(
        BestofNLearner._restore_rl_checkpoint
    )
    assert "starting critics from scratch" not in inspect.getsource(
        AdvantageWeightedSFTLearner._restore_rl_checkpoint
    )


def test_awr_resume_required_paths_without_a_registry(tmp_path):
    me = object.__new__(AdvantageWeightedSFTLearner)
    me._config = types.SimpleNamespace(checkpoint_dir=str(tmp_path))
    me._task_registry = None
    paths = AdvantageWeightedSFTLearner._resume_required_paths(me, 20000)
    assert [str(p) for p in paths] == [str(tmp_path / "rl_state" / "20000")]


def test_awr_resume_required_paths_with_a_registry(tmp_path):
    me = object.__new__(AdvantageWeightedSFTLearner)
    me._config = types.SimpleNamespace(checkpoint_dir=str(tmp_path))
    me._task_registry = object()
    paths = [str(p) for p in AdvantageWeightedSFTLearner._resume_required_paths(me, 20000)]
    assert paths == [
        str(tmp_path / "rl_state" / "20000"),
        str(tmp_path / "rl_state" / "task_registry_20000.json"),
    ]


class _RecordingCheckpointer:
    def __init__(self, log):
        self.log = log

    def save(self, path, state):
        self.log.append(("rl_state", str(path)))
        Path(path).mkdir(parents=True, exist_ok=True)


def _awr_stand_in(tmp_path, log, registry=None):
    me = object.__new__(AdvantageWeightedSFTLearner)
    me._config = types.SimpleNamespace(checkpoint_dir=str(tmp_path))
    me._task_registry = registry
    me._rl_state_checkpointer = _RecordingCheckpointer(log)
    me._state_action_critic_state = "q"
    me._value_state = "v"
    me._normalizer_state = "n"
    return me


def test_awr_save_checkpoint_writes_rl_state_before_the_orbax_commit(tmp_path, monkeypatch):
    log = []
    monkeypatch.setattr(
        FilteredSFTLearner, "save_checkpoint",
        lambda self, step=None: log.append(("orbax", step)),
    )
    me = _awr_stand_in(tmp_path, log)
    AdvantageWeightedSFTLearner.save_checkpoint(me, step=20000)
    assert [k for k, _ in log] == ["rl_state", "orbax"]


def test_awr_save_checkpoint_writes_the_registry_before_the_orbax_commit(tmp_path, monkeypatch):
    log = []
    monkeypatch.setattr(
        FilteredSFTLearner, "save_checkpoint",
        lambda self, step=None: log.append(("orbax", step)),
    )
    registry = types.SimpleNamespace(to_json=lambda p: log.append(("registry", str(p))))
    me = _awr_stand_in(tmp_path, log, registry=registry)
    AdvantageWeightedSFTLearner.save_checkpoint(me, step=20000)
    assert [k for k, _ in log] == ["rl_state", "registry", "orbax"]


def test_awr_still_crashes_on_step_none_the_documented_divergence(tmp_path, monkeypatch):
    """BLAST-RADIUS §2.2: unreachable in production, and NOT harmonized here."""
    log = []
    monkeypatch.setattr(FilteredSFTLearner, "save_checkpoint", lambda self, step=None: None)
    me = _awr_stand_in(tmp_path, log)
    with pytest.raises(TypeError):
        AdvantageWeightedSFTLearner.save_checkpoint(me, step=None)


def _bofn_stand_in(tmp_path, log):
    me = object.__new__(BestofNLearner)
    me._config = types.SimpleNamespace(checkpoint_dir=str(tmp_path))
    me._rl_state_checkpointer = _RecordingCheckpointer(log)
    me._state_action_critic_state = "q"
    me._value_state = "v"
    me.training_steps = 30000
    return me


def test_bofn_save_checkpoint_writes_rl_state_before_the_orbax_commit(tmp_path, monkeypatch):
    log = []
    monkeypatch.setattr(
        FilteredSFTLearner, "save_checkpoint",
        lambda self, step=None: log.append(("orbax", step)),
    )
    me = _bofn_stand_in(tmp_path, log)
    BestofNLearner.save_checkpoint(me, step=20000)
    assert [k for k, _ in log] == ["rl_state", "orbax"]


def test_bofn_normalizes_step_none(tmp_path, monkeypatch):
    log = []
    monkeypatch.setattr(
        FilteredSFTLearner, "save_checkpoint",
        lambda self, step=None: log.append(("orbax", step)),
    )
    me = _bofn_stand_in(tmp_path, log)
    BestofNLearner.save_checkpoint(me, step=None)
    assert log == [("rl_state", str(tmp_path / "rl_state" / "30000")), ("orbax", 30000)]


def test_bofn_existing_rl_state_skips_only_the_rl_state_write(tmp_path, monkeypatch):
    """DIFF divergence D2: the pre-change early `return` would now also skip the
    orbax commit. The reshaped guard must still commit."""
    log = []
    monkeypatch.setattr(
        FilteredSFTLearner, "save_checkpoint",
        lambda self, step=None: log.append(("orbax", step)),
    )
    me = _bofn_stand_in(tmp_path, log)
    (tmp_path / "rl_state" / "20000").mkdir(parents=True)
    BestofNLearner.save_checkpoint(me, step=20000)
    assert log == [("orbax", 20000)]


# --------------------------------------------------------------------------- #
# F. manifest schema strictness
# --------------------------------------------------------------------------- #

def test_manifest_schema_is_strict_and_extra_is_per_instance():
    old = {
        "step": 1,
        "total_collected_episodes": 2,
        "replay_shard_dir": "/d",
        "agent_rng_state_json": "{}",
        "replay_rng_state_json": "{}",
    }
    a, b = ResumeState(**old), ResumeState(**old)
    assert a.extra == {} and a.extra is not b.extra
    with pytest.raises(TypeError):
        ResumeState(**(old | {"success_shard_dir": "/x"}))
    with pytest.raises(TypeError):
        ResumeState(**{k: v for k, v in old.items() if k != "step"})
    assert dataclasses.fields(ResumeState)[-1].name == "extra"
    # frozen: a resume manifest must not be mutated in place
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.step = 5


def test_replay_shard_dir_annotation_now_matches_what_json_produces():
    ann = {f.name: f.type for f in dataclasses.fields(ResumeState)}
    assert ann["replay_shard_dir"] in ("str", str)


# --------------------------------------------------------------------------- #
# G. FilteredSFTLearner._resolve_resume_state, called unbound
# --------------------------------------------------------------------------- #

class _Mgr:
    def __init__(self, steps):
        self._steps = steps

    def all_steps(self):
        return list(self._steps)


def _resolver_stand_in(cfg, orbax_steps, required=lambda s: True):
    return types.SimpleNamespace(
        _config=cfg,
        _checkpoint_manager=_Mgr(orbax_steps),
        _resume_required_paths=lambda s: [] if required(s) else [Path("/definitely/missing")],
    )


def test_resolve_resume_state_returns_the_per_step_manifest(tmp_path, caplog):
    agent = _FakeAgent(tmp_path / "ckpt")
    _advance_and_save(agent, 10000, 7, 5)
    _advance_and_save(agent, 20000, 6, 4)
    me = _resolver_stand_in(agent.config, [20000])
    with caplog.at_level(logging.INFO):
        state = FilteredSFTLearner._resolve_resume_state(me)
    assert state.step == 20000
    assert state.extra["success_total_inserted"] == 9
    assert "Resume resolved to step 20000" in caplog.text


def test_resolve_resume_state_falls_back_to_the_pointer_manifest(tmp_path):
    agent = _FakeAgent(tmp_path / "ckpt")
    _advance_and_save(agent, 10000, 7, 5)
    step_manifest_path(agent.config, 10000).unlink()
    me = _resolver_stand_in(agent.config, [10000])
    state = FilteredSFTLearner._resolve_resume_state(me)
    assert state.step == 10000


def test_resolve_resume_state_warning_text_matches_the_direction(tmp_path, caplog):
    agent = _FakeAgent(tmp_path / "ckpt")
    _advance_and_save(agent, 10000, 7, 5)
    agent.fail_at = "post_orbax"          # pointer stays at 10000, step 20000 durable
    with pytest.raises(_Boom):
        _advance_and_save(agent, 20000, 6, 4)
    me = _resolver_stand_in(agent.config, [20000])
    with caplog.at_level(logging.WARNING):
        state = FilteredSFTLearner._resolve_resume_state(me)
    assert state.step == 20000
    assert "is discarded" not in caplog.text


# --------------------------------------------------------------------------- #
# H. BLAST-RADIUS §3.3: the success shard inherits the per-task-critic schema
#    fail-fast (a per-task shard must not load into a shared-critic buffer)
# --------------------------------------------------------------------------- #

def _buffer_dummy_with_task_index() -> dict:
    d = _buffer_dummy()
    d["task_index"] = np.zeros((1,), np.int32)
    return d


def test_a_per_task_success_shard_refuses_to_load_into_a_shared_critic_buffer(tmp_path):
    d = tmp_path / "success_shards"
    per_task = ShardedReplayBuffer(
        dummy_data=_buffer_dummy_with_task_index(), max_capacity=32, seed=0, freeze_dict=False
    )
    n = 4
    per_task.insert(
        {
            "observations": {"state": np.zeros((n + 1, _S), np.float32)},
            "obs_index": np.arange(n, dtype=np.int64),
            "next_obs_index": np.arange(n, dtype=np.int64) + 1,
            "actions": np.zeros((n, _AH, _AD), np.float32),
            "reward": np.zeros((n,), np.float32),
            "mc_return": np.zeros((n,), np.float32),
            "discount": np.zeros((n,), np.float32),
            "is_success": np.ones((n,), np.float32),
            "task_index": np.zeros((n,), np.int32),
        }
    )
    per_task.save_shard(d / "step_00020000.h5")

    shared = _new_buffer(32)
    with pytest.raises(ValueError, match="Insert transition structure"):
        shared.restore_shards(d, max_step=20000)


# --------------------------------------------------------------------------- #
# I. source-level invariants the change asserts about itself
# --------------------------------------------------------------------------- #

def test_exactly_one_sanctioned_best_effort_comment_in_the_changed_files():
    import inspect
    from src.rl.ogpo import ogpo_learner
    from src.training import runtime_state
    from src.rl import replay_buffer
    from src.rl.filtered_sft_agent import filtered_sft_learner
    from src.rl.advantage_weighted_sft import advantage_weighted_sft_learner
    from src.rl.best_of_n import best_of_n_learner

    counts = {
        m.__name__: inspect.getsource(m).count("# best-effort:")
        for m in (ogpo_learner, runtime_state, replay_buffer, filtered_sft_learner,
                  advantage_weighted_sft_learner, best_of_n_learner)
    }
    assert sum(counts.values()) == 1, counts
    assert counts["src.rl.ogpo.ogpo_learner"] == 1


def test_the_ogpo_restore_never_takes_the_buffer_without_the_ranges():
    """§4.2's package property, as a source invariant: the only
    `_success_data_buffer.restore_shards` call in the tree sits inside
    `_restore_extra_resume_state`, and that method assigns
    `_success_task_ranges` on every path that reaches the restore."""
    import inspect
    from src.rl.ogpo import ogpo_learner

    module_src = inspect.getsource(ogpo_learner)
    assert module_src.count("_success_data_buffer.restore_shards") == 1
    method_src = inspect.getsource(OGPOAgentLearner._restore_extra_resume_state)
    assert method_src.count("_success_data_buffer.restore_shards") == 1
    restore_at = method_src.index("_success_data_buffer.restore_shards")
    ranges_at = method_src.index("self._success_task_ranges =")
    assert ranges_at > restore_at
    tail = method_src[restore_at:ranges_at]
    # nothing between them may return early
    assert "return" not in tail
