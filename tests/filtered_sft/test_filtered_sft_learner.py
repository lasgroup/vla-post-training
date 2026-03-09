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

from src.rl_training.agent import StepData
from src.rl_training.types import EnvKwargs, OnlineLearningConfig


# `openpi.training.data_loader` imports torch at module import time and uses a few symbols
# in type annotations. These unit tests don't need torch, so provide a tiny stub.
if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.Generator = object  # only used at runtime in data loader init
    torch_stub.utils = types.SimpleNamespace(
        data=types.SimpleNamespace(DataLoader=object, Dataset=object)
    )
    sys.modules["torch"] = torch_stub

import src.rl_training.filtered_sft_learner as learner_mod


class _Buf:
    def __init__(self):
        self.inserted = []

    def insert(self, data):
        self.inserted.append(data)


class _Policy:
    def __init__(self):
        self.calls = []

    def infer_with_model(self, *, model, obs, rng, noise_level):
        self.calls.append(
            {"model": model, "obs": obs, "rng": rng, "noise_level": noise_level}
        )
        return {"actions": jnp.array([1.0, 2.0], dtype=jnp.float32)}


def _make_learner_stub(*, num_envs=2, obs_prefix="Pi0"):
    lrn = learner_mod.PiFilteredSFTLearner.__new__(learner_mod.PiFilteredSFTLearner)
    lrn._obs_prefix_key = obs_prefix
    lrn._rng = jax.random.PRNGKey(0)
    lrn._config = types.SimpleNamespace(
        base_policy_config=object(),
        online_learning_config=OnlineLearningConfig(
            variant=EnvKwargs(),
            num_envs=num_envs,
            episode_update_frequency=9999,
        ),
    )
    lrn._mesh = object()
    lrn._checkpoint_manager = object()
    lrn._offline_data_loader = object()
    lrn._episode_storage = [[] for _ in range(num_envs)]
    lrn._online_data_buffer = _Buf()
    lrn.env_steps = 0
    lrn.episodes = 0
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

    def fake_restore_state(mngr, state, data_loader, step=None):
        calls["args"] = (mngr, state, data_loader, step)
        return restored_state

    monkeypatch.setattr(learner_mod, "init_train_state", fake_init_train_state)
    monkeypatch.setattr(learner_mod._checkpoints, "restore_state", fake_restore_state)
    monkeypatch.setattr(learner_mod.jax, "block_until_ready", lambda x: x)
    monkeypatch.setattr(
        learner_mod.training_utils, "array_tree_to_info", lambda _: "info"
    )

    out_state, out_shard = lrn._get_train_state(resuming=True)

    assert out_state is restored_state
    assert out_shard is shard
    assert calls["args"][0] is lrn._checkpoint_manager
    assert (
        calls["args"][1] is shape_state
    )  # critical: use the local var, not self._train_state
    assert calls["args"][2] is lrn._offline_data_loader


def test_sample_action_filters_keys_and_resizes_images(monkeypatch):
    lrn = _make_learner_stub(num_envs=1)

    # Minimal train_state; we won't execute the jitted _sample_action in this unit test.
    lrn._train_state = types.SimpleNamespace(model_def="graph", params="params")

    # Track image processing.
    called = {"resize": [], "to_uint8": 0}

    def fake_resize_with_pad(x, h, w):
        called["resize"].append((x, h, w))
        return x

    def fake_convert_to_uint8(x):
        called["to_uint8"] += 1
        return x

    monkeypatch.setattr(
        learner_mod.image_tools, "resize_with_pad", fake_resize_with_pad
    )
    monkeypatch.setattr(
        learner_mod.image_tools, "convert_to_uint8", fake_convert_to_uint8
    )

    captured = {}

    def fake_sample_action(
        *,
        observations,
        rng,
        train_state,
        noise_level=0.0,
        num_steps=None,
        batch_actions=True,
    ):
        captured["observations"] = observations
        captured["rng"] = rng
        captured["train_state"] = train_state
        captured["noise_level"] = noise_level
        captured["num_steps"] = num_steps
        captured["batch_actions"] = batch_actions
        return jnp.array([1.0, 2.0], dtype=jnp.float32)

    lrn._sample_action = fake_sample_action

    obs = {
        "Pi0image_front": np.zeros((32, 32, 3), dtype=np.uint8),
        "Pi0state": np.array([0.1, 0.2], dtype=np.float32),
        "Pi0prompt": "do the thing",
        "ignore_me": 123,
    }

    actions = lrn.eval_actions(obs)

    assert np.allclose(np.asarray(actions), np.array([1.0, 2.0], dtype=np.float32))

    sent = captured["observations"]
    assert set(sent.keys()) == {"observation/image_front", "observation/state"}

    assert called["resize"]
    assert called["resize"][0][1:] == (224, 224)
    assert called["to_uint8"] == 1


def test_add_data_and_save_episode_inserts_stacked_episode(monkeypatch):
    lrn = _make_learner_stub(num_envs=2)

    def mk_step(t):
        obs = {
            "Pi0state": np.array([[t, t + 1], [t + 2, t + 3]], dtype=np.float32),
            "Pi0image": np.zeros((2, 4, 4, 3), dtype=np.uint8) + t,
            "other": np.array([999, 999], dtype=np.int32),
        }
        action = np.array([[t * 10.0, 0.0], [t * 10.0 + 1.0, 1.0]], dtype=np.float32)
        terminate = np.array([False, False])
        return StepData(
            obs=obs,
            action=action,
            next_obs=None,
            reward=None,
            terminate=terminate,
            truncate=None,
        )

    lrn.add_data(mk_step(0))
    lrn.add_data(mk_step(1))

    assert lrn.env_steps == 4
    assert len(lrn._episode_storage[0]) == 2
    assert len(lrn._episode_storage[1]) == 2

    lrn.save_episode(is_success=True, env_index=0)

    assert len(lrn._online_data_buffer.inserted) == 1
    inserted = lrn._online_data_buffer.inserted[0]

    assert inserted["observations"]["state"].shape == (2, 2)
    assert inserted["observations"]["image"].shape == (2, 4, 4, 3)
    assert inserted["actions"].shape == (2, 2)

    assert lrn._episode_storage[0] == []
    assert len(lrn._episode_storage[1]) == 2


def test_update_returns_empty_when_not_scheduled():
    lrn = _make_learner_stub(num_envs=1)
    lrn._config = types.SimpleNamespace(
        base_policy_config=object(),
        online_learning_config=OnlineLearningConfig(
            variant=EnvKwargs(), num_envs=1, episode_update_frequency=2
        ),
    )
    lrn.episodes = 1

    assert lrn.update() == {}


def test_update_handles_none_frequencies():
    lrn = _make_learner_stub(num_envs=1)
    lrn._config = types.SimpleNamespace(
        base_policy_config=object(),
        online_learning_config=OnlineLearningConfig(
            variant=EnvKwargs(),
            num_envs=1,
            episode_update_frequency=None,
            env_steps_update_frequency=10,
        ),
    )
    lrn.env_steps = 5
    lrn.episodes = 123

    assert lrn.update() == {}


def test_update_uses_offline_and_online_batches(monkeypatch):
    lrn = _make_learner_stub(num_envs=1)

    batch_size = 4
    state_dim = 3

    def make_obs(val: float):
        return learner_mod._model.Observation(
            images={
                "base_0_rgb": jnp.full((batch_size, 2, 2, 3), val, dtype=jnp.float32)
            },
            image_masks={"base_0_rgb": jnp.ones((batch_size,), dtype=bool)},
            state=jnp.full((batch_size, state_dim), val, dtype=jnp.float32),
            tokenized_prompt=jnp.zeros((batch_size, 5), dtype=jnp.int32),
            tokenized_prompt_mask=jnp.ones((batch_size, 5), dtype=bool),
        )

    offline_batch = (make_obs(0.0), jnp.zeros((batch_size, 1, 2), dtype=jnp.float32))

    class _OfflineLoader:
        def __iter__(self):
            yield offline_batch

    def _to_jax(x):
        return jnp.asarray(x)

    dummy_obs_dict = make_obs(0.0).to_dict()
    dummy_actions = jnp.zeros((batch_size, 1, 2), dtype=jnp.float32)

    def _postprocess_online(batch):
        obs_dict, actions = batch
        obs_dict = learner_mod.jax.tree.map(_to_jax, obs_dict)
        actions = _to_jax(actions)
        return learner_mod._model.Observation.from_dict(obs_dict), actions

    online_buf = learner_mod.ShardedReplayBuffer(
        dummy_data=(dummy_obs_dict, dummy_actions),
        max_capacity=100,
        batch_size=batch_size,
        data_sharding=None,
        seed=0,
        postprocess_fn=_postprocess_online,
    )

    online_obs_dict = make_obs(1.0).to_dict()
    online_actions = np.full((batch_size, 1, 2), 2.0, dtype=np.float32)
    online_buf.insert((online_obs_dict, online_actions))

    lrn._offline_data_loader = _OfflineLoader()
    lrn._online_data_buffer = online_buf
    lrn._train_state = object()
    lrn._mesh = object()
    lrn.episodes = 0  # makes episode_update_frequency=1 trigger update

    lrn._config = types.SimpleNamespace(
        base_policy_config=object(),
        online_learning_config=OnlineLearningConfig(
            variant=EnvKwargs(task_description="do the thing"),
            num_envs=1,
            episode_update_frequency=1,
            start_step=0,
            num_train_steps_per_update=1,
        ),
    )

    @contextlib.contextmanager
    def _no_mesh(_mesh):
        yield

    monkeypatch.setattr(learner_mod.sharding, "set_mesh", _no_mesh)
    monkeypatch.setattr(learner_mod.tqdm, "tqdm", lambda it, **_: it)
    monkeypatch.setattr(
        learner_mod.common_utils, "stack_forest", lambda infos: infos[0]
    )
    monkeypatch.setattr(learner_mod.jax, "device_get", lambda x: x)

    captured = {}

    def fake_update(rng, train_state, batch):
        captured["rng"] = rng
        captured["train_state"] = train_state
        captured["batch"] = batch
        return train_state, {"loss": jnp.array(1.0)}

    lrn._update = fake_update

    info = lrn.update()

    assert "batch" in captured
    stacked_obs, stacked_actions = captured["batch"]

    # Offline batch is concatenated first, then online batch.
    assert np.allclose(np.asarray(stacked_obs.state[0]), 0.0)
    assert np.allclose(np.asarray(stacked_obs.state[batch_size]), 1.0)
    assert np.allclose(np.asarray(stacked_actions[0]), 0.0)
    assert np.allclose(np.asarray(stacked_actions[batch_size]), 2.0)

    assert "loss" in info
