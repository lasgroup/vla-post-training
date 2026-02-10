import contextlib
import dataclasses
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


class _CheckpointManager:
    def __init__(self, directory: str | Path):
        self._directory = Path(directory)
        self.wait_calls = 0

    def wait_until_finished(self):
        self.wait_calls += 1


@dataclasses.dataclass
class _DummyTrainState:
    model_def: object = "graphdef"
    params: object = "params"
    opt_state: object = dataclasses.field(
        default_factory=lambda: jnp.array([1.0], dtype=jnp.float32)
    )
    ema_params: object = dataclasses.field(
        default_factory=lambda: jnp.array([2.0], dtype=jnp.float32)
    )


class _Actor:
    def __init__(self):
        self.infer_calls = []
        self.infer_with_model_calls = []

    def infer(self, element, *, sharding_spec):
        self.infer_calls.append({"element": element, "sharding_spec": sharding_spec})
        return {"actions": np.zeros((1, 1, 2), dtype=np.float32)}

    def infer_with_model(self, **kwargs):
        self.infer_with_model_calls.append(kwargs)
        return {"actions": jnp.array([[1.0, 2.0]], dtype=jnp.float32)}


def _make_initialized_learner(monkeypatch, *, num_envs=2, resuming=False):
    actor = _Actor()
    checkpoint_manager = _CheckpointManager("/tmp/openpi-test-checkpoint")
    train_state = _DummyTrainState()
    train_state_sharding = types.SimpleNamespace(opt_state=None, ema_params=None)

    base_policy_config = types.SimpleNamespace(
        seed=123,
        fsdp_devices=(0,),
        checkpoint_dir=Path("/tmp/checkpoints"),
        keep_period=None,
        overwrite=False,
        resume=False,
        default_prompt="test prompt",
        weight_loader=object(),
        trainable_filter=object(),
    )
    online_learning_config = OnlineLearningConfig(
        variant=EnvKwargs(task_description="test task"),
        num_envs=num_envs,
        episode_update_frequency=1,
        num_train_steps_per_update=1,
        start_step=0,
        pi0_num_steps=7,
        fm_noise_level=0.4,
        online_ratio=0.0,
    )
    config = types.SimpleNamespace(
        base_policy_config=base_policy_config,
        online_learning_config=online_learning_config,
        obs_prefix_key="pi0/",
    )

    monkeypatch.setenv("OPENPI_POLICY_CHECKPOINT_DIR", "/tmp/openpi-policy")
    monkeypatch.setattr(learner_mod.sharding, "make_mesh", lambda _devices: object())
    monkeypatch.setattr(
        learner_mod.jax.sharding,
        "NamedSharding",
        lambda mesh, spec: types.SimpleNamespace(mesh=mesh, spec=spec),
    )
    monkeypatch.setattr(
        learner_mod._checkpoints,
        "initialize_checkpoint_dir",
        lambda *args, **kwargs: (checkpoint_manager, resuming),
    )
    monkeypatch.setattr(
        learner_mod.PiFilteredSFTLearner,
        "_get_offline_data_loader",
        lambda self: types.SimpleNamespace(data_config=lambda: object()),
    )
    monkeypatch.setattr(
        learner_mod.PiFilteredSFTLearner,
        "_get_online_replay_buffer",
        lambda self, data_sharding: _Buf(),
    )
    monkeypatch.setattr(
        learner_mod.PiFilteredSFTLearner,
        "_get_train_state",
        lambda self, resuming=False: (train_state, train_state_sharding),
    )
    monkeypatch.setattr(
        learner_mod.policy_config, "create_trained_policy", lambda *args, **kwargs: actor
    )
    monkeypatch.setattr(learner_mod.jax, "jit", lambda fn, **kwargs: fn)

    learner = learner_mod.PiFilteredSFTLearner(config)
    return learner, actor, train_state, train_state_sharding, checkpoint_manager


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


def test_initialized_instance_init_and_properties(monkeypatch):
    lrn, actor, train_state, _sharding, _ckpt = _make_initialized_learner(
        monkeypatch, num_envs=2, resuming=True
    )
    assert isinstance(lrn, learner_mod.PiFilteredSFTLearner)
    assert lrn.actor is actor
    assert lrn._train_state is train_state
    assert lrn.resuming is True
    assert lrn.base_policy_config is lrn._config.base_policy_config
    assert lrn.online_learning_config is lrn._config.online_learning_config
    assert len(lrn._episode_storage) == 2


def test_sample_action_eval_actions_and_sample_actions_on_initialized_instance(monkeypatch):
    lrn, actor, train_state, _sharding, _ckpt = _make_initialized_learner(
        monkeypatch, num_envs=1
    )

    monkeypatch.setattr(learner_mod.nnx, "merge", lambda graphdef, params: "model")
    obs = {"observation/state": np.array([0.1, 0.2], dtype=np.float32)}
    act = lrn._sample_action(
        observations=obs,
        rng=jax.random.PRNGKey(0),
        train_state=train_state,
        num_steps=3,
        batch_actions=True,
    )
    assert act.shape == (1, 1, 2)
    assert actor.infer_with_model_calls
    infer_call = actor.infer_with_model_calls[0]
    assert infer_call["model"] == "model"
    assert infer_call["num_steps"] == 3

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
        captured["noise_level"] = noise_level
        captured["num_steps"] = num_steps
        return jnp.array([3.0, 4.0], dtype=jnp.float32)

    lrn._sample_action = fake_sample_action
    monkeypatch.setattr(
        lrn, "_process_obs_for_pi0", lambda _obs: {"observation/state": np.array([1.0])}
    )

    out_eval = lrn.eval_actions({"pi0/state": np.array([1.0])})
    assert np.allclose(np.asarray(out_eval), np.array([3.0, 4.0], dtype=np.float32))
    assert captured["noise_level"] == 0.0
    assert captured["num_steps"] == lrn.online_learning_config.pi0_num_steps

    out_sample = lrn.sample_actions({"pi0/state": np.array([1.0])})
    assert np.allclose(np.asarray(out_sample), np.array([3.0, 4.0], dtype=np.float32))
    assert captured["noise_level"] == lrn.online_learning_config.fm_noise_level
    assert captured["num_steps"] == lrn.online_learning_config.pi0_num_steps


def test_save_checkpoint_start_data_collection_and_start_training(monkeypatch):
    lrn, _actor, _state, _sharding, ckpt = _make_initialized_learner(
        monkeypatch, num_envs=2
    )
    saved = {}

    def fake_save_state(manager, train_state, offline_loader, step):
        saved["args"] = (manager, train_state, offline_loader, step)

    monkeypatch.setattr(learner_mod._checkpoints, "save_state", fake_save_state)

    lrn.save_checkpoint(step=7)
    assert saved["args"][0] is lrn._checkpoint_manager
    assert saved["args"][1] is lrn._train_state
    assert saved["args"][2] is lrn._offline_data_loader
    assert saved["args"][3] == 7
    assert ckpt.wait_calls == 1

    lrn._episode_storage[0].append({"x": 1})
    lrn._episode_storage[1].append({"x": 2})
    assert lrn.start_data_collection() is True
    assert lrn._phase == "collect"
    assert lrn._episode_storage == [[], []]
    assert lrn._offloaded_training_state is not None

    assert lrn.start_training() is True
    assert lrn._phase == "train"
    assert lrn._offloaded_training_state is None


def test_get_offline_data_loader_uses_data_loader_factory(monkeypatch):
    lrn = learner_mod.PiFilteredSFTLearner.__new__(learner_mod.PiFilteredSFTLearner)
    lrn._config = types.SimpleNamespace(base_policy_config="base-config")
    lrn._data_sharding = "data-shard"

    batch = ("obs", "actions")

    class _Loader:
        def __iter__(self):
            yield batch

    loader = _Loader()
    calls = {}

    def fake_create_data_loader(config, sharding, shuffle):
        calls["args"] = (config, sharding, shuffle)
        return loader

    monkeypatch.setattr(learner_mod._data_loader, "create_data_loader", fake_create_data_loader)
    monkeypatch.setattr(learner_mod.training_utils, "array_tree_to_info", lambda x: "info")

    out = lrn._get_offline_data_loader()
    assert out is loader
    assert calls["args"] == ("base-config", "data-shard", True)


def test_get_online_replay_buffer_smoke(monkeypatch):
    lrn = learner_mod.PiFilteredSFTLearner.__new__(learner_mod.PiFilteredSFTLearner)

    class _Spec:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype

    class _ObsSpec:
        def to_dict(self):
            return {
                "image": {"base_0_rgb": _Spec((2, 2, 2, 3), np.uint8)},
                "image_mask": {"base_0_rgb": _Spec((2,), np.bool_)},
                "state": _Spec((2, 3), np.float32),
            }

    class _ActSpec:
        shape = (2, 1, 2)
        dtype = np.float32

    class _ModelConfig:
        action_horizon = 1

        def inputs_spec(self, batch_size=1):
            return _ObsSpec(), _ActSpec()

    class _FakeTokenizePrompt:
        def __call__(self, data):
            return {
                "tokenized_prompt": np.array([1, 2, 3], dtype=np.int32),
                "tokenized_prompt_mask": np.array([True, True, True], dtype=np.bool_),
            }

    fake_group = types.SimpleNamespace(inputs=[], outputs=[])
    data_config = types.SimpleNamespace(
        repack_transforms=fake_group,
        data_transforms=fake_group,
        model_transforms=types.SimpleNamespace(inputs=[_FakeTokenizePrompt()], outputs=[]),
        norm_stats={},
        use_quantile_norm=False,
    )

    lrn._offline_data_loader = types.SimpleNamespace(data_config=lambda: data_config)
    lrn._config = types.SimpleNamespace(
        base_policy_config=types.SimpleNamespace(
            model=_ModelConfig(),
            batch_size=8,
            seed=7,
            default_prompt="default task",
        ),
        online_learning_config=OnlineLearningConfig(
            variant=EnvKwargs(task_description="task"),
            num_envs=1,
            online_batch_size=4,
            online_max_samples=20,
        ),
    )

    def fake_compose(_fns):
        def _apply(raw):
            obs = raw["observation"]
            if isinstance(obs, np.ndarray) and obs.dtype == object:
                obs = obs.item()
            state = np.asarray(obs["state"], dtype=np.float32)
            image = np.asarray(obs["image"], dtype=np.uint8)
            actions = np.asarray(raw["actions"], dtype=np.float32)
            return {
                "state": state,
                "image": {"base_0_rgb": image},
                "image_mask": {"base_0_rgb": np.ones((state.shape[0],), dtype=np.bool_)},
                "actions": actions,
            }

        return _apply

    class _FakeReplayBuffer:
        def __init__(
            self,
            *,
            dummy_data,
            max_capacity,
            batch_size,
            data_sharding,
            seed,
            preprocess_fn,
            postprocess_fn,
            freeze_dict,
        ):
            self.dummy_data = dummy_data
            self.max_capacity = max_capacity
            self.batch_size = batch_size
            self.data_sharding = data_sharding
            self.seed = seed
            self.preprocess_fn = preprocess_fn
            self.postprocess_fn = postprocess_fn
            self.freeze_dict = freeze_dict

    monkeypatch.setattr(learner_mod._transforms, "compose", fake_compose)
    monkeypatch.setattr(learner_mod._transforms, "Normalize", lambda *a, **k: object())
    monkeypatch.setattr(learner_mod._transforms, "TokenizePrompt", _FakeTokenizePrompt)
    monkeypatch.setattr(
        learner_mod._transforms, "TokenizeFASTInputs", type("DummyFAST", (), {})
    )
    monkeypatch.setattr(learner_mod, "ShardedReplayBuffer", _FakeReplayBuffer)

    buffer = lrn._get_online_replay_buffer(data_sharding="shard")
    assert isinstance(buffer, _FakeReplayBuffer)
    assert buffer.batch_size == 4
    assert buffer.max_capacity == 256

    obs_dict, actions = buffer.preprocess_fn(
        {
            "observation": {
                "state": np.zeros((2, 3), dtype=np.float32),
                "image": np.zeros((2, 2, 2, 3), dtype=np.uint8),
            },
            "actions": np.zeros((2, 1, 2), dtype=np.float32),
            "prompt": "hello",
        }
    )
    assert "tokenized_prompt" in obs_dict
    assert "tokenized_prompt_mask" in obs_dict
    assert actions.shape == (2, 1, 2)


def test_update_helper_update_and_wrap_env(monkeypatch):
    lrn, _actor, _state, _sharding, _ckpt = _make_initialized_learner(
        monkeypatch, num_envs=1
    )

    lrn._update_actor = lambda rng, train_state, batch: ("new-state", {"loss": jnp.array(1.0)})
    out_state, out_info = lrn._update(jax.random.PRNGKey(0), "old-state", ("obs", "act"))
    assert out_state == "new-state"
    assert "loss" in out_info

    class _OfflineLoader:
        def __iter__(self):
            yield ("offline-obs", "offline-actions")

    lrn._offline_data_loader = _OfflineLoader()
    lrn._online_data_buffer = types.SimpleNamespace(size=0, batch_size=4)
    lrn._train_state = "train-state"
    lrn.episodes = 0
    lrn.env_steps = 0

    start_training_calls = {"n": 0}

    def fake_start_training():
        start_training_calls["n"] += 1
        return True

    lrn.start_training = fake_start_training
    lrn._update = lambda rng, train_state, batch: (train_state, {"loss": jnp.array(2.0)})

    @contextlib.contextmanager
    def _no_mesh(_mesh):
        yield

    monkeypatch.setattr(learner_mod.sharding, "set_mesh", _no_mesh)
    monkeypatch.setattr(learner_mod.tqdm, "tqdm", lambda it, **_: it)
    monkeypatch.setattr(learner_mod.common_utils, "stack_forest", lambda infos: infos[0])
    monkeypatch.setattr(learner_mod.jax, "device_get", lambda x: x)

    info = lrn.update()
    assert start_training_calls["n"] == 1
    assert "loss" in info

    class _FakeVecEnv:
        def __init__(self, factories):
            self.factories = factories
            self.instances = [fn() for fn in factories]
            self.seed_values = None

        def seed(self, values):
            self.seed_values = values

    monkeypatch.setattr(learner_mod, "DummyVectorEnv", _FakeVecEnv)
    monkeypatch.setattr(learner_mod, "SubprocVectorEnv", _FakeVecEnv)
    monkeypatch.setattr(learner_mod, "ensure_gymnasium_env", lambda env: ("ensure", env))
    monkeypatch.setattr(
        learner_mod,
        "TimeLimit",
        lambda env, max_episode_steps: ("timelimit", max_episode_steps, env),
    )
    monkeypatch.setattr(
        learner_mod,
        "QueryFrequencyWrapper",
        lambda env, query_frequency, discount: ("query", query_frequency, discount, env),
    )
    monkeypatch.setattr(
        learner_mod,
        "Pi0ObservationWrapper",
        lambda env, variant: ("pi0", variant.task_description, env),
    )
    monkeypatch.setattr(
        learner_mod,
        "WarmUpOnResetWrapper",
        lambda env, num_steps_wait, warm_up_action: ("warmup", num_steps_wait, warm_up_action, env),
    )

    def env_factory(render_gpu_device_id=None):
        return {"render_gpu_device_id": render_gpu_device_id}

    vec_env = lrn.wrap_env(env_factory)
    assert isinstance(vec_env, _FakeVecEnv)
    assert vec_env.seed_values == [lrn.base_policy_config.seed]
    assert vec_env.instances[0][0] == "pi0"
