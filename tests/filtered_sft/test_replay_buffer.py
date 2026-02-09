import os
import tempfile

import gymnasium as gym
import numpy as np
import pytest

from flax.core import frozen_dict

from src.rl_training.replay_buffer import ReplayBuffer, ShardedReplayBuffer


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
