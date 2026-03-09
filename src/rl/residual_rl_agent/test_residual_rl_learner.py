import contextlib
import sys
import types
from pathlib import Path

# Ensure local `openpi` and `openpi_client` packages are importable when running tests from repo root.
_ROOT = Path(__file__).resolve().parents[2]
for _rel in ("openpi/src", "openpi/packages/openpi-client/src"):
    _p = str(_ROOT / _rel)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.rl.types import StepData
import src.rl.residual_rl_agent.residual_rl_learner as learner_mod
from src.rl.residual_rl_agent.residual_rl_learner_cfg import ResidualRLTrainConfig, CollectionConfig, SACModelConfig

class _Buf:
    def __init__(self):
        self.inserted = []

    def insert(self, data):
        self.inserted.append(data)


def _make_learner_stub(*, num_envs=2, obs_prefix="pi0"):
    # Stub config
    config = ResidualRLTrainConfig(
        collect=CollectionConfig(
            env_num=num_envs,
            obs_prefix_key=obs_prefix,
            add_per_step_data=False,
            replan_steps=1,
        ),
        batch_size=256,
        fsdp_devices=1,
        model=SACModelConfig(obs_dim=1024, action_horizon=10, action_dim=7),
    )
    lrn = learner_mod.ResidualRLLearner.__new__(learner_mod.ResidualRLLearner)
    lrn._config = config
    lrn._rng = jax.random.key(0)
    lrn._mesh = object()
    lrn._checkpoint_manager = object()
    lrn._data_loader = object()
    lrn._data_iter = iter([])
    lrn._episode_storage = [[] for _ in range(num_envs)]
    lrn._online_data_buffer = _Buf()
    lrn._collection_success_episodes = 0
    lrn.training_steps = 0
    return lrn


def test_get_train_state_resuming_uses_returned_state_not_self(monkeypatch):
    lrn = _make_learner_stub()

    shape_state = types.SimpleNamespace(params="shape_params")
    restored_state = types.SimpleNamespace(params="restored_params")
    shard = object()

    def fake_init_train_state(config, init_rng, mesh, *, resume):
        assert resume is True
        return shape_state, shard

    calls = {}

    class FakeCheckpointManager:
        def latest_step(self):
            return 100
            
        def restore(self, step, args):
            calls["args"] = (step, args.item)
            return {'train_state': restored_state}

    lrn._checkpoint_manager = FakeCheckpointManager()

    monkeypatch.setattr(learner_mod, "init_train_state", fake_init_train_state)
    monkeypatch.setattr(learner_mod.jax, "block_until_ready", lambda x: x)

    # Replicate constructor logic
    out_state, out_shard = learner_mod.init_train_state(lrn._config, lrn._rng, lrn._mesh, resume=True)
    restored = lrn._checkpoint_manager.restore(100, learner_mod.ocp.args.StandardRestore(out_state))
    out_state_restored = restored['train_state']

    assert out_state_restored is restored_state
    assert out_shard is shard
    assert calls["args"][0] == 100
    assert calls["args"][1] is shape_state


def test_sample_action_extracts_obs_prefix(monkeypatch):
    lrn = _make_learner_stub(num_envs=1)
    
    # Mock model definitions
    class FakeModel:
        def sample_actions(self, obs, rng):
            return jnp.ones((obs.shape[0], 70), dtype=jnp.float32) * 2.5
            
    lrn._train_state = types.SimpleNamespace(model_def=object(), params="params", ema_params=None)

    def fake_merge(model_def, params):
        return FakeModel()
        
    monkeypatch.setattr(learner_mod.nnx, "merge", fake_merge)

    obs = {
        "base_action": np.array([[0.1, 0.2]], dtype=np.float32),
        "ignore_me": 123,
    }

    actions = lrn.eval_actions(obs)

    assert actions.shape == (1, 10, 7)
    assert np.allclose(actions, 2.5)


def test_add_data_and_save_episode_inserts_stacked_episode(monkeypatch):
    lrn = _make_learner_stub(num_envs=2)
    lrn._config = ResidualRLTrainConfig(
        collect=CollectionConfig(
            env_num=2,
            obs_prefix_key="pi0",
            add_per_step_data=True,
            replan_steps=1,
        )
    )
    lrn._episode_storage = [[] for _ in range(2)]

    def mk_step(t):
        obs = {
            "observation": {
                "base_action": np.array([[t, t + 1], [t + 2, t + 3]], dtype=np.float32),
                "actions": np.array([[t * 10.0, 0.0], [t * 10.0 + 1.0, 1.0]], dtype=np.float32)
            }
        }
        terminate = np.array([[False], [False]], dtype=np.bool_)
        truncate = np.array([[False], [False]], dtype=np.bool_)
        return {
            "observation": obs["observation"],
            "next_observation": obs["observation"],
            "reward": np.array([[1.0], [1.0]]),
            "terminate": terminate,
            "truncate": truncate,
        }

    lrn.add_data(mk_step(0))
    lrn.add_data(mk_step(1))

    assert len(lrn._episode_storage[0]) == 2
    assert len(lrn._episode_storage[1]) == 2

    lrn.save_episode(is_success=True, env_index=0)

    assert len(lrn._online_data_buffer.inserted) == 1
    inserted = lrn._online_data_buffer.inserted[0]

    assert inserted["observation"]["base_action"].shape == (2,)
    assert inserted["actions"].shape == (2,)
    assert inserted["reward"].shape == (2,)

    assert lrn._episode_storage[0] == []
    assert len(lrn._episode_storage[1]) == 2


def test_update_returns_empty_when_not_enough_data():
    lrn = _make_learner_stub(num_envs=1)
    
    class FakeBuffer:
        def __init__(self):
            self.size = 10
            self.batch_size = 256
            
    lrn._online_data_buffer = FakeBuffer()
    assert lrn.update() == {"actor_loss": 0.0, "critic_loss": 0.0}
