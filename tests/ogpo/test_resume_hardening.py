"""Resume hardening (docs/changes/2026-08-27-resume-hardening/).

Pure-data, CPU-only, model-free. Everything under test was factored so it can be
exercised without a learner: ``OGPOAgentLearner.__init__`` device_puts the EMA to
``pinned_host`` and is GPU-only by construction, so the range rebase and the
resume-step resolver are module-level functions and ``_balanced_success_ordinals``
is called unbound against a stand-in ``self``.

Covers:
  - shard round-trip ordinal/payload exactness (the evidence for the "uniform
    shift" claim the range rebase rests on);
  - the clipped case (``save_shard``'s delta clamp, replay_buffer.py:200) and the
    shift it induces;
  - ``restore_shards(max_step=...)``, differentially against a buffer that only
    ever received the early shards;
  - ``_rebase_task_ranges`` + its ``_balanced_success_ordinals`` consumer;
  - the ``ResumeState`` manifest schema (old payloads, ``extra``, strictness);
  - ``resolve_resume_step`` over the crash table.
"""
import json
import types

import numpy as np
import pytest

from src.rl.ogpo.ogpo_learner import OGPOAgentLearner, _rebase_task_ranges
from src.rl.replay_buffer import ShardedReplayBuffer
from src.training.runtime_state import (
    ResumeState,
    load_resume_state,
    resolve_resume_step,
    resume_state_path,
    step_manifest_path,
    step_manifest_steps,
)

_S, _AH, _AD = 3, 2, 4


def _buffer_dummy() -> dict:
    return {
        "observations": {"state": np.zeros((1, _S), np.float32)},
        "actions": np.zeros((1, _AH, _AD), np.float32),
        "reward": np.zeros((1,), np.float32),
        "mc_return": np.zeros((1,), np.float32),
        "discount": np.zeros((1,), np.float32),
        "is_success": np.zeros((1,), np.float32),
    }


def _new_buffer(max_capacity: int) -> ShardedReplayBuffer:
    return ShardedReplayBuffer(
        dummy_data=_buffer_dummy(),
        max_capacity=max_capacity,
        seed=0,
        freeze_dict=False,
    )


def _insert_episode(buf: ShardedReplayBuffer, n: int, task: float = 0.0) -> tuple[int, int]:
    """Insert `n` transitions stamped with their GLOBAL ordinal and a task tag.

    `mc_return` carries the ordinal the transition had in this buffer, so after a
    restore the shift between an ordinal and its payload is directly readable.
    Returns the (lo, hi) ordinal range, as `save_episode` records it.
    """
    lo = buf.total_inserted
    ordinals = lo + np.arange(n, dtype=np.float32)
    buf.insert(
        {
            "observations": {
                "state": np.repeat(
                    (lo + np.arange(n + 1, dtype=np.float32))[:, None], _S, axis=1
                )
            },
            "obs_index": np.arange(n, dtype=np.int64),
            "next_obs_index": np.arange(n, dtype=np.int64) + 1,
            "actions": np.zeros((n, _AH, _AD), np.float32),
            "reward": np.full((n,), task, np.float32),
            "mc_return": ordinals,
            "discount": np.full((n,), 0.99, np.float32),
            "is_success": np.ones((n,), np.float32),
        }
    )
    return lo, buf.total_inserted


def _live(buf: ShardedReplayBuffer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ordinals, stamped original ordinal, stamped obs state) over the live window."""
    ordinals = np.arange(buf.valid_start, buf.total_inserted)
    batch = buf.sample(batch_size=len(ordinals), ordinals=ordinals)
    return (
        ordinals,
        np.asarray(batch["mc_return"]),
        np.asarray(batch["observation"]["state"])[:, 0],
    )


# --------------------------------------------------------------------------- #
# Shard round trip / the shift formula
# --------------------------------------------------------------------------- #

def test_shard_round_trip_preserves_ordinals_and_payloads(tmp_path):
    """Unclipped restore is exact: same ordinals, same window, same payloads.

    This is the evidence for the rebase design — with nothing dropped the shift is
    zero and persisted ranges are already correct.
    """
    shard_dir = tmp_path / "shards"
    src = _new_buffer(64)
    for n in (5, 4, 6):
        _insert_episode(src, n)
    src.save_shard(shard_dir / "step_000000.h5")
    _insert_episode(src, 5)
    src.save_shard(shard_dir / "step_001000.h5")

    restored = _new_buffer(64)
    restored.restore_shards(shard_dir)

    assert restored.total_inserted == src.total_inserted == 20
    assert restored.valid_start == src.valid_start == 0
    assert restored.size == src.size
    src_ords, src_stamp, src_state = _live(src)
    res_ords, res_stamp, res_state = _live(restored)
    np.testing.assert_array_equal(res_ords, src_ords)
    np.testing.assert_array_equal(res_stamp, src_stamp)
    np.testing.assert_array_equal(res_state, src_state)
    # The stamp IS the ordinal, so nothing moved.
    np.testing.assert_array_equal(res_stamp, res_ords.astype(np.float32))


def test_clipped_save_shifts_every_live_transition_by_the_dropped_count(tmp_path):
    """`save_shard` clamps its delta to `size` (replay_buffer.py:200). The
    transitions it drops are a contiguous block, so every transition that is still
    live after the restore moves by the SAME shift — the assumption
    `_rebase_task_ranges` encodes.
    """
    shard_dir = tmp_path / "shards"
    cap = 16
    src = _new_buffer(cap)
    _insert_episode(src, 6)
    src.save_shard(shard_dir / "step_000000.h5")
    persisted = src.total_inserted
    for _ in range(4):  # 24 > cap inserted between saves => the clamp fires
        _insert_episode(src, 6)
    saved_total = src.total_inserted
    size_at_save = src.size
    expected_dropped = (saved_total - persisted) - size_at_save
    assert expected_dropped > 0, "test did not reach the clipped branch"
    src.save_shard(shard_dir / "step_001000.h5")

    restored = _new_buffer(cap)
    restored.restore_shards(shard_dir)

    shift = restored.total_inserted - saved_total
    assert shift == -expected_dropped
    ordinals, stamp, state = _live(restored)
    # Every live transition carries the payload of `ordinal - shift`.
    np.testing.assert_array_equal(stamp, (ordinals - shift).astype(np.float32))
    np.testing.assert_array_equal(state, (ordinals - shift).astype(np.float32))


# --------------------------------------------------------------------------- #
# restore_shards(max_step=...)
# --------------------------------------------------------------------------- #

def test_max_step_cutoff_equals_a_buffer_that_only_saw_the_early_shards(tmp_path):
    """Differential: restoring three shards with `max_step=1000` must reproduce a
    buffer built from a shard directory that only ever received the first two.
    """
    full_dir, early_dir = tmp_path / "full", tmp_path / "early"
    full_src, early_src = _new_buffer(64), _new_buffer(64)
    for step, n in ((0, 5), (1000, 4)):
        _insert_episode(full_src, n)
        _insert_episode(early_src, n)
        full_src.save_shard(full_dir / f"step_{step:08d}.h5")
        early_src.save_shard(early_dir / f"step_{step:08d}.h5")
    _insert_episode(full_src, 6)  # step 2000 exists only in `full`
    full_src.save_shard(full_dir / "step_00002000.h5")

    cut = _new_buffer(64)
    cut.restore_shards(full_dir, max_step=1000)
    early = _new_buffer(64)
    early.restore_shards(early_dir)

    assert (cut.total_inserted, cut.valid_start, cut.size) == (
        early.total_inserted,
        early.valid_start,
        early.size,
    )
    assert cut.total_inserted == 9  # the cutoff is inclusive of step 1000
    for cut_leg, early_leg in zip(_live(cut), _live(early)):
        np.testing.assert_array_equal(cut_leg, early_leg)

    # No cutoff keeps today's behavior: every shard is replayed.
    uncut = _new_buffer(64)
    uncut.restore_shards(full_dir)
    assert uncut.total_inserted == 15


def test_unparseable_shard_name_raises_under_a_cutoff(tmp_path):
    shard_dir = tmp_path / "shards"
    src = _new_buffer(32)
    _insert_episode(src, 4)
    src.save_shard(shard_dir / "step_000000.h5")
    (shard_dir / "step_latest.h5").write_bytes(b"")

    with pytest.raises(ValueError, match="foreign file"):
        _new_buffer(32).restore_shards(shard_dir, max_step=1000)


# --------------------------------------------------------------------------- #
# Task-range rebase
# --------------------------------------------------------------------------- #

def test_rebase_task_ranges_shifts_drops_and_clamps():
    # Nothing dropped => identity.
    assert _rebase_task_ranges(
        {"a": [(0, 4)], "b": [(4, 10)]},
        saved_total_inserted=10,
        restored_total_inserted=10,
        valid_start=0,
    ) == {"a": [(0, 4)], "b": [(4, 10)]}

    # Uniform negative shift, and a range that ends at or below valid_start goes.
    assert _rebase_task_ranges(
        {"a": [(0, 4), (10, 14)], "b": [(4, 10)]},
        saved_total_inserted=14,
        restored_total_inserted=12,
        valid_start=4,
    ) == {"a": [(8, 12)], "b": [(2, 8)]}

    # A task with no surviving range disappears entirely.
    assert _rebase_task_ranges(
        {"a": [(0, 4)], "b": [(4, 12)]},
        saved_total_inserted=12,
        restored_total_inserted=12,
        valid_start=6,
    ) == {"b": [(4, 12)]}

    # `hi` is clamped to the restored buffer, and a range emptied by the clamp is
    # dropped rather than kept inverted.
    assert _rebase_task_ranges(
        {"a": [(0, 12)], "b": [(12, 16)]},
        saved_total_inserted=10,
        restored_total_inserted=10,
        valid_start=0,
    ) == {"a": [(0, 10)]}


def test_rebased_ranges_feed_balanced_success_sampling_without_raising(tmp_path):
    """End-to-end on the real buffer: save a two-task success buffer through a
    clipped shard, restore it, rebase the ranges, and let
    `_balanced_success_ordinals` sample. The failure mode this guards is a raise
    from replay_buffer.py:155-156, so assert both no-raise and the balance.
    """
    shard_dir = tmp_path / "success_shards"
    cap = 32
    src = _new_buffer(cap)
    saved_ranges: dict[str, list[tuple[int, int]]] = {"a": [], "b": []}
    for i in range(2):
        task = "a" if i % 2 == 0 else "b"
        saved_ranges[task].append(_insert_episode(src, 8, task=float(i % 2)))
    src.save_shard(shard_dir / "step_000000.h5")
    for i in range(2, 6):
        task = "a" if i % 2 == 0 else "b"
        saved_ranges[task].append(_insert_episode(src, 8, task=float(i % 2)))
    saved_total = src.total_inserted
    src.save_shard(shard_dir / "step_001000.h5")

    buf = _new_buffer(cap)
    buf.restore_shards(shard_dir, max_step=1000)
    assert buf.total_inserted < saved_total, "test did not reach the clipped branch"

    rebased = _rebase_task_ranges(
        saved_ranges,
        saved_total_inserted=saved_total,
        restored_total_inserted=buf.total_inserted,
        valid_start=buf.valid_start,
    )
    assert set(rebased) == {"a", "b"}

    stand_in = types.SimpleNamespace(
        _success_data_buffer=buf, _success_task_ranges=rebased
    )
    batch_size = 8
    ordinals = OGPOAgentLearner._balanced_success_ordinals(stand_in, batch_size)
    assert ordinals is not None and ordinals.shape == (batch_size,)
    assert ordinals.min() >= buf.valid_start and ordinals.max() < buf.total_inserted
    batch = buf.sample(batch_size=batch_size, ordinals=ordinals)  # must not raise
    counts = np.bincount(np.asarray(batch["reward"]).astype(np.int64), minlength=2)
    np.testing.assert_array_equal(counts, [batch_size // 2, batch_size // 2])


def test_unclamped_hi_is_what_the_clamp_prevents(tmp_path):
    """Why the `hi` clamp is load-bearing: `_balanced_success_ordinals` clamps only
    the low end, so an `hi` past `total_inserted` reaches `sample`."""
    buf = _new_buffer(32)
    _insert_episode(buf, 8, task=0.0)
    _insert_episode(buf, 8, task=1.0)
    over_high = {"a": [(0, 8)], "b": [(8, buf.total_inserted + 1)]}
    stand_in = types.SimpleNamespace(
        _success_data_buffer=buf, _success_task_ranges=over_high
    )
    ordinals = OGPOAgentLearner._balanced_success_ordinals(stand_in, 64)
    with pytest.raises(ValueError, match="evicted or unwritten"):
        buf.sample(batch_size=64, ordinals=ordinals)


# --------------------------------------------------------------------------- #
# Manifest schema
# --------------------------------------------------------------------------- #

_OLD_PAYLOAD = {
    "step": 20000,
    "total_collected_episodes": 137,
    "replay_shard_dir": "/ckpt/runtime_state/replay_shards",
    "agent_rng_state_json": "{}",
    "replay_rng_state_json": "{}",
}


def _cfg(tmp_path):
    return types.SimpleNamespace(checkpoint_dir=str(tmp_path))


def test_pre_change_manifest_still_constructs_and_new_keys_still_raise():
    old = ResumeState(**_OLD_PAYLOAD)
    assert old.extra == {}
    # Two ResumeStates must not share one dict.
    assert ResumeState(**_OLD_PAYLOAD).extra is not old.extra

    with pytest.raises(TypeError):
        ResumeState(**(_OLD_PAYLOAD | {"adv_scale": 1.0}))


def test_manifest_round_trip_keeps_extra_and_step_lookup_finds_it(tmp_path):
    cfg = _cfg(tmp_path)
    payload = _OLD_PAYLOAD | {
        "extra": {
            "adv_scale": 2.5,
            "success_shard_dir": "/ckpt/runtime_state/success_shards",
            "success_total_inserted": 40,
            "success_task_ranges": {"a": [[0, 8]], "b": [[8, 40]]},
            "success_rng_state_json": "{}",
        }
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    step_path = step_manifest_path(cfg, payload["step"])
    step_path.parent.mkdir(parents=True, exist_ok=True)
    step_path.write_text(text)
    resume_state_path(cfg).write_text(text)

    by_step = load_resume_state(cfg, step=payload["step"])
    by_pointer = load_resume_state(cfg)
    assert by_step == by_pointer
    assert by_step.extra["adv_scale"] == 2.5
    assert by_step.extra["success_task_ranges"] == {"a": [[0, 8]], "b": [[8, 40]]}
    assert by_step.replay_shard_dir == _OLD_PAYLOAD["replay_shard_dir"]

    assert step_manifest_steps(cfg) == {payload["step"]}  # ignores resume_state.json
    with pytest.raises(FileNotFoundError):
        load_resume_state(cfg, step=payload["step"] + 1)


def test_step_manifest_steps_rejects_a_foreign_file(tmp_path):
    cfg = _cfg(tmp_path)
    path = step_manifest_path(cfg, 10)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    (path.parent / "resume_state_backup.json").write_text("{}")
    with pytest.raises(ValueError, match="foreign file"):
        step_manifest_steps(cfg)


def test_step_manifest_steps_is_empty_before_any_save(tmp_path):
    assert step_manifest_steps(_cfg(tmp_path / "nothing_here")) == set()


# --------------------------------------------------------------------------- #
# resolve_resume_step — one case per crash-table row
# --------------------------------------------------------------------------- #

_ALL_OK = lambda step: True  # noqa: E731 - inline predicate, one per call site


def test_resolver_picks_the_completed_step_for_each_crash_window():
    # Crash during/after the shard write, before the per-step manifest: only M is
    # on disk in full.
    assert resolve_resume_step({20000}, {20000}, _ALL_OK, 20000) == 20000
    # Crash between the manifest write and the checkpoint save: manifest N exists,
    # orbax is still at M.
    assert resolve_resume_step({20000}, {20000, 30000}, _ALL_OK, 30000) == 20000
    # Crash inside save_checkpoint before the orbax commit: rl_state N exists,
    # orbax is still at M.
    assert (
        resolve_resume_step(
            {20000}, {20000, 30000}, lambda s: s in {20000, 30000}, 30000
        )
        == 20000
    )
    # Crash after the orbax commit (M was GC'd by max_to_keep=1), and the crash
    # between the commit and the pointer refresh: N is complete either way.
    assert resolve_resume_step({30000}, {20000, 30000}, _ALL_OK, 20000) == 30000
    assert resolve_resume_step({30000}, {20000, 30000}, _ALL_OK, 30000) == 30000


def test_resolver_skips_a_step_missing_its_learner_sidecar():
    assert (
        resolve_resume_step({20000, 30000}, {20000, 30000}, lambda s: s == 20000, 30000)
        == 20000
    )


def test_resolver_falls_back_to_the_pointer_for_pre_change_directories():
    assert resolve_resume_step({20000}, set(), _ALL_OK, 20000) == 20000


def test_resolver_raises_when_the_pointer_names_a_missing_checkpoint():
    with pytest.raises(FileNotFoundError, match="no per-step"):
        resolve_resume_step({20000}, set(), _ALL_OK, 30000)
    with pytest.raises(FileNotFoundError):
        resolve_resume_step(set(), set(), _ALL_OK, None)


def test_resolver_raises_when_no_step_is_consistent():
    with pytest.raises(FileNotFoundError, match="learner rl_state"):
        resolve_resume_step({30000}, {30000}, lambda s: False, 30000)
    with pytest.raises(FileNotFoundError, match="learner rl_state"):
        resolve_resume_step({10000}, {30000}, _ALL_OK, 30000)
