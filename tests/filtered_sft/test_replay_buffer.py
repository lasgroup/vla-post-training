import os
import tempfile
import types
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from flax.core import frozen_dict

from scripts.launcher_util import validate_unique_exp_names
from src.rl.replay_buffer import ReplayBuffer, ShardedReplayBuffer
from src.training.runtime_state import (
    current_training_step,
    load_resume_state,
    save_epoch_state,
    restore_train_state,
    write_resume_state,
)


def _assert_nested_equal(lhs, rhs):
    if isinstance(lhs, dict):
        assert lhs.keys() == rhs.keys()
        for key in lhs:
            _assert_nested_equal(lhs[key], rhs[key])
        return
    assert np.array_equal(np.asarray(lhs), np.asarray(rhs))


def test_replay_buffer_insert_sample_and_resize():
    obs_space = gym.spaces.Dict(
        {
            "obs": gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            "img": gym.spaces.Box(low=0, high=255, shape=(2, 2, 3), dtype=np.uint8),
        }
    )
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    rb = ReplayBuffer(observation_space=obs_space, action_space=act_space, capacity=2)

    def step(i: int):
        return {
            "observations": {
                "obs": np.full((3,), i, np.float32),
                "img": np.full((2, 2, 3), i, np.uint8),
            },
            "next_observations": {
                "obs": np.full((3,), i + 1, np.float32),
                "img": np.full((2, 2, 3), i + 1, np.uint8),
            },
            "actions": np.array([i, i + 0.5], np.float32),
            "next_actions": np.array([i + 1, i + 1.5], np.float32),
            "rewards": np.float32(i),
            "masks": np.float32(1.0),
            "discount": np.float32(0.99),
        }

    rb.insert(step(0))
    rb.insert(step(1))
    # next insert triggers resize from 2 -> 4
    rb.insert(step(2))

    assert rb.size == 3
    assert rb.capacity == 4

    batch = rb.sample(batch_size=2)
    assert isinstance(batch, frozen_dict.FrozenDict)
    assert batch["observations"]["obs"].shape == (2, 3)
    assert batch["observations"]["img"].shape == (2, 2, 2, 3)
    assert batch["actions"].shape == (2, 2)


def test_replay_buffer_action_stats_only_use_populated_prefix():
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    rb = ReplayBuffer(observation_space=obs_space, action_space=act_space, capacity=10)

    # Insert two known actions. The rest of the storage is uninitialized, so
    # compute_action_stats must slice to rb.size.
    for a in [0.0, 2.0]:
        rb.insert(
            {
                "observations": np.array([0.0], np.float32),
                "next_observations": np.array([0.0], np.float32),
                "actions": np.array([a], np.float32),
                "next_actions": np.array([a], np.float32),
                "rewards": np.float32(0.0),
                "masks": np.float32(1.0),
                "discount": np.float32(1.0),
            }
        )

    stats = rb.compute_action_stats()
    assert np.allclose(stats["mean"], np.array([1.0], np.float32))
    assert np.allclose(stats["std"], np.array([1.0], np.float32))


def test_replay_buffer_save_restore_roundtrip():
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    rb = ReplayBuffer(observation_space=obs_space, action_space=act_space, capacity=2)

    rb.insert(
        {
            "observations": np.array([1.0, 2.0], np.float32),
            "next_observations": np.array([3.0, 4.0], np.float32),
            "actions": np.array([0.1, 0.2, 0.3], np.float32),
            "next_actions": np.array([0.4, 0.5, 0.6], np.float32),
            "rewards": np.float32(1.0),
            "masks": np.float32(1.0),
            "discount": np.float32(0.99),
        }
    )

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "rb.pkl")
        rb.save(path)

        rb2 = ReplayBuffer(
            observation_space=obs_space, action_space=act_space, capacity=1
        )
        rb2.restore(path)

        assert rb2.size == rb.size
        assert rb2.capacity >= rb2.size
        assert np.allclose(
            rb2.data["actions"][: rb2.size], rb.data["actions"][: rb.size]
        )
        assert np.allclose(
            rb2.data["next_actions"][: rb2.size], rb.data["next_actions"][: rb.size]
        )
        assert np.allclose(
            rb2.data["rewards"][: rb2.size], rb.data["rewards"][: rb.size]
        )


def test_sharded_replay_buffer_insert_and_sample_structure():
    dummy = {
        "observations": {
            "state": np.zeros((1, 4), dtype=np.float32),
            "pixels": np.zeros((1, 8, 8, 3), dtype=np.uint8),
        },
        "actions": np.zeros((1, 2), dtype=np.float32),
    }

    buf = ShardedReplayBuffer(
        dummy_data=dummy, max_capacity=100, batch_size=7, data_sharding=None, seed=0
    )

    data = {
        "observations": {
            "state": np.ones((10, 4), dtype=np.float32),
            "pixels": np.zeros((10, 8, 8, 3), dtype=np.uint8),
        },
        "actions": np.full((10, 2), 3.0, dtype=np.float32),
    }

    buf.insert(data)
    assert len(buf) == 10

    batch = buf.sample()
    assert isinstance(batch, frozen_dict.FrozenDict)
    assert batch["observations"]["state"].shape == (7, 4)
    assert batch["observations"]["pixels"].shape == (7, 8, 8, 3)
    assert batch["actions"].shape == (7, 2)


def test_sharded_replay_buffer_sample_empty_raises():
    dummy = {"x": np.zeros((1, 1), dtype=np.float32)}
    buf = ShardedReplayBuffer(
        dummy_data=dummy, max_capacity=10, batch_size=2, data_sharding=None, seed=0
    )
    with pytest.raises(ValueError, match="empty buffer"):
        buf.sample()


def test_sharded_replay_buffer_snapshot_roundtrip_preserves_state(tmp_path):
    dummy = {
        "observations": {
            "state": np.zeros((1, 4), dtype=np.float32),
            "pixels": np.zeros((1, 4, 4, 3), dtype=np.uint8),
        },
        "actions": np.zeros((1, 2), dtype=np.float32),
    }
    data = {
        "observations": {
            "state": np.arange(24, dtype=np.float32).reshape(6, 4),
            "pixels": np.arange(6 * 4 * 4 * 3, dtype=np.uint8).reshape(6, 4, 4, 3),
        },
        "actions": np.arange(12, dtype=np.float32).reshape(6, 2),
    }

    buf = ShardedReplayBuffer(
        dummy_data=dummy,
        max_capacity=8,
        batch_size=3,
        data_sharding=None,
        seed=123,
        freeze_dict=False,
    )
    buf.insert(data)

    shard_dir = tmp_path / "replay_shards"
    shard_path = shard_dir / "step_00000042.h5"
    buf.save_shard(shard_path)
    with h5py.File(shard_path, "r") as f:
        assert f["transitions"]["observations"]["state"].shape[0] == buf.size
        assert f["transitions"]["actions"].shape[0] == buf.size

    restored = ShardedReplayBuffer(
        dummy_data=dummy,
        max_capacity=3,
        batch_size=3,
        data_sharding=None,
        seed=999,
        freeze_dict=False,
    )
    restored_info = restored.restore_shards(
        shard_dir,
        step=42,
        total_inserted=buf.total_inserted,
        latest_shard_path=shard_path,
        rng_state_json=buf.rng_state_json(),
    )

    assert restored.size == buf.size
    assert restored.ptr == buf.ptr
    assert restored.max_capacity == buf.max_capacity
    _assert_nested_equal(restored.storage, buf.storage)

    original_sample = buf.sample(batch_size=3)
    restored_sample = restored.sample(batch_size=3)
    _assert_nested_equal(original_sample, restored_sample)


def test_restore_train_state_uses_resume_manifest_step(tmp_path):
    replay_dir = tmp_path / "runtime_state" / "replay_shards"
    replay_dir.mkdir(parents=True, exist_ok=True)
    replay_path = replay_dir / "step_00000017.h5"
    replay_path.write_bytes(b"snapshot")
    (tmp_path / "999").mkdir()

    config = types.SimpleNamespace(checkpoint_dir=tmp_path)
    write_resume_state(
        config,
        step=17,
        replay_size=23,
        replay_total_inserted=29,
        replay_shards=replay_dir,
        latest_replay_shard=replay_path,
        replay_rng_state_json="{}",
    )
    resume_state = load_resume_state(config)

    assert resume_state is not None
    assert resume_state.step == 17
    assert resume_state.replay_size == 23
    assert resume_state.replay_total_inserted == 29
    assert resume_state.replay_shard_dir == replay_dir
    assert resume_state.latest_replay_shard_path == replay_path

    calls = {}

    def fake_restore_fn(manager, train_state, data_loader, step=None):
        calls["step"] = step
        calls["manager"] = manager
        calls["train_state"] = train_state
        calls["data_loader"] = data_loader
        return "restored"

    manager = object()
    train_state = object()
    data_loader = object()
    restored = restore_train_state(
        fake_restore_fn,
        manager,
        train_state,
        data_loader,
        resume_state=resume_state,
    )

    assert restored == "restored"
    assert calls["step"] == 17
    assert calls["manager"] is manager
    assert calls["train_state"] is train_state
    assert calls["data_loader"] is data_loader


def test_current_training_step_prefers_agent_training_steps():
    agent = types.SimpleNamespace(training_steps=12, _train_state=types.SimpleNamespace(step=3))
    assert current_training_step(agent) == 12


def test_save_epoch_state_uses_agent_training_steps_for_manifest(tmp_path):
    class _CheckpointManager:
        def __init__(self):
            self.wait_calls = 0

        def wait_until_finished(self):
            self.wait_calls += 1

    class _ReplayBuffer:
        def __init__(self):
            self.calls = []
            self._rng_state_json = '{"state": "ok"}'

        def save_shard(self, path):
            snapshot_path = Path(path)
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_path.write_bytes(b"snapshot")
            self.calls.append(snapshot_path)
            return {
                "size": 7,
                "total_inserted": 11,
                "path": str(snapshot_path),
            }

        def rng_state_json(self):
            return self._rng_state_json

    class _Agent:
        def __init__(self):
            self._train_state = types.SimpleNamespace(step=42)
            self._checkpoint_manager = _CheckpointManager()
            self._online_data_buffer = _ReplayBuffer()
            self.saved_steps = []

        def save_checkpoint(self, step):
            self.saved_steps.append(step)

    agent = _Agent()
    config = types.SimpleNamespace(
        checkpoint_dir=tmp_path,
        requeue=True,
    )

    agent.training_steps = 42
    resume_state = save_epoch_state(agent, config)
    loaded_resume_state = load_resume_state(config)

    assert resume_state is not None
    assert agent.saved_steps == [42]
    assert agent._checkpoint_manager.wait_calls == 1
    assert agent._online_data_buffer.calls[0].name == "step_00000042.h5"
    assert loaded_resume_state is not None
    assert loaded_resume_state.step == 42
    assert loaded_resume_state.replay_size == 7
    assert loaded_resume_state.replay_total_inserted == 11


def test_validate_unique_exp_names_raises_for_collisions():
    with pytest.raises(ValueError, match="unique exp_name"):
        validate_unique_exp_names(
            [
                {"exp_name": "same"},
                {"exp_name": "same"},
            ]
        )
