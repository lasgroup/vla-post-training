import contextlib
import types

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import openpi.models.model as _model
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import src.rl.advantage_weighted_regression.advantage_weigthed_filtered_sft_learner as learner_mod
from src.rl.advantage_weighted_regression.update_critic import (
    StateActionValueCritic,
    StateValueCritic,
    create_critic,
    critic_hidden_dims,
    init_critic_train_state,
    train_q_step,
    train_value_step,
)
from src.training.config import get_config


def _build_train_state(
    model: nnx.Module,
    *,
    ema_params: nnx.State | None = None,
) -> training_utils.TrainState:
    params = nnx.state(model)
    tx = optax.adam(1e-3)
    return training_utils.TrainState(
        step=0,
        params=params,
        model_def=nnx.graphdef(model),
        tx=tx,
        opt_state=tx.init(nnx.filter_state(params, nnx.Param)),
        ema_decay=0.99,
        ema_params=params if ema_params is None else ema_params,
    )


def test_critic_hidden_dims_supports_tuple_and_int():
    assert critic_hidden_dims(types.SimpleNamespace(rl=None)) == (256, 256)
    assert (
        critic_hidden_dims(types.SimpleNamespace(rl=types.SimpleNamespace(critic_hidden_dims=128)))
        == (128,)
    )
    assert (
        critic_hidden_dims(types.SimpleNamespace(rl=types.SimpleNamespace(critic_hidden_dims=(64, 32))))
        == (64, 32)
    )


def test_create_critic_respects_use_ema_flag():
    model = StateValueCritic(state_dim=4, hidden_dims=(8,), rngs=nnx.Rngs(0))
    params = nnx.state(model)
    ema_params = jax.tree.map(lambda p: p + 10.0, params)
    state = _build_train_state(model, ema_params=ema_params)
    observation = _model.Observation(
        images={"base_0_rgb": jnp.zeros((2, 2, 2, 3), dtype=jnp.float32)},
        image_masks={"base_0_rgb": jnp.ones((2,), dtype=bool)},
        state=jnp.ones((2, 4), dtype=jnp.float32),
    )

    critic_params = create_critic(
        state, types.SimpleNamespace(rl=types.SimpleNamespace(use_ema_critic=False))
    )
    critic_ema = create_critic(
        state, types.SimpleNamespace(rl=types.SimpleNamespace(use_ema_critic=True))
    )

    out_params = np.asarray(critic_params(observation))
    out_ema = np.asarray(critic_ema(observation))
    assert not np.allclose(out_params, out_ema)


def test_train_q_and_value_step_smoke():
    config = get_config("pi05_libero_online")
    mesh = sharding.make_mesh(config.fsdp_devices)
    obs_spec, action_spec = config.model.inputs_spec(batch_size=1)
    state_dim = int(obs_spec.state.shape[-1])
    action_horizon = int(action_spec.shape[-2])
    action_dim = int(action_spec.shape[-1])

    q_state, _ = init_critic_train_state(
        config,
        jax.random.key(0),
        mesh,
        critic_factory=lambda rng: StateActionValueCritic(
            state_dim=state_dim,
            action_horizon=action_horizon,
            action_dim=action_dim,
            hidden_dims=(64, 64),
            rngs=nnx.Rngs(rng),
        ),
    )
    value_state, _ = init_critic_train_state(
        config,
        jax.random.key(1),
        mesh,
        critic_factory=lambda rng: StateValueCritic(
            state_dim=state_dim,
            hidden_dims=(64, 64),
            rngs=nnx.Rngs(rng),
        ),
    )

    observation = config.model.fake_obs(batch_size=2)
    actions = config.model.fake_act(batch_size=2)
    next_state = jax.random.normal(jax.random.key(2), observation.state.shape)
    reward = jnp.ones((2,), dtype=jnp.float32)
    discount = jnp.full((2,), 0.99, dtype=jnp.float32)
    batch = (observation, actions, next_state, reward, discount)

    q_state_1, q_info = train_q_step(
        config, jax.random.key(3), q_state, value_state, batch
    )
    value_state_1, value_info = train_value_step(
        config, jax.random.key(4), value_state, q_state_1, batch
    )

    assert int(q_state_1.step) == int(q_state.step) + 1
    assert int(value_state_1.step) == int(value_state.step) + 1
    assert {"loss", "grad_norm", "param_norm", "q_value_mean"} <= set(q_info)
    assert {"loss", "grad_norm", "param_norm", "value_mean"} <= set(value_info)
    assert np.isfinite(np.asarray(q_info["loss"]))
    assert np.isfinite(np.asarray(value_info["loss"]))


def test_online_batch_to_critic_batch_conversion():
    learner = learner_mod.AdvantageWeightedFilteredSFTLearner.__new__(
        learner_mod.AdvantageWeightedFilteredSFTLearner
    )
    batch_size = 3
    online_batch = {
        "observation": {
            "image": {
                "base_0_rgb": jnp.zeros((batch_size, 4, 4, 3), dtype=jnp.uint8),
            },
            "image_mask": {
                "base_0_rgb": jnp.ones((batch_size,), dtype=jnp.bool_),
            },
            "state": jnp.ones((batch_size, 7), dtype=jnp.float32),
            "tokenized_prompt": jnp.zeros((batch_size, 5), dtype=jnp.int32),
            "tokenized_prompt_mask": jnp.ones((batch_size, 5), dtype=jnp.bool_),
        },
        "actions": jnp.zeros((batch_size, 2, 7), dtype=jnp.float32),
        "next_observation": {
            "state": jnp.full((batch_size, 7), 2.0, dtype=jnp.float32)
        },
        "reward": jnp.ones((batch_size,), dtype=jnp.float32),
        "discount": jnp.full((batch_size,), 0.95, dtype=jnp.float32),
    }

    observation, actions, next_state, reward, discount = learner._online_batch_to_critic_batch(
        online_batch
    )

    assert isinstance(observation, _model.Observation)
    assert observation.state.shape == (batch_size, 7)
    assert observation.tokenized_prompt is not None
    assert actions.shape == (batch_size, 2, 7)
    assert next_state.shape == (batch_size, 7)
    assert reward.shape == (batch_size,)
    assert discount.shape == (batch_size,)


def test_update_critics_averages_info(monkeypatch):
    learner = learner_mod.AdvantageWeightedFilteredSFTLearner.__new__(
        learner_mod.AdvantageWeightedFilteredSFTLearner
    )
    learner._critic_updates_per_step = 2
    learner._rng = jax.random.key(0)
    learner._mesh = object()
    learner._state_action_critic_state = 0
    learner._value_state = 0
    learner._online_data_buffer = types.SimpleNamespace(size=11, batch_size=4)
    learner.sample_online_transitions = lambda: {"dummy": 1}
    learner._online_batch_to_critic_batch = lambda _: "critic_batch"

    def q_step(rng, q_state, value_state, critic_batch):
        del rng, value_state, critic_batch
        return q_state + 1, {
            "loss": jnp.asarray(float(q_state + 1), dtype=jnp.float32),
        }

    def value_step(rng, value_state, q_state, critic_batch):
        del rng, q_state, critic_batch
        return value_state + 1, {
            "loss": jnp.asarray(float(value_state + 1), dtype=jnp.float32),
        }

    learner._q_train_step = q_step
    learner._value_train_step = value_step

    @contextlib.contextmanager
    def _no_mesh(_mesh):
        yield

    monkeypatch.setattr(learner_mod.sharding, "set_mesh", _no_mesh)
    info = learner._update_critics()

    assert float(info["critic/q_loss"]) == 1.5
    assert float(info["critic/value_loss"]) == 1.5
    assert float(info["critic/online_buffer_size"]) == 11.0
    assert learner._state_action_critic_state == 2
    assert learner._value_state == 2


def test_update_merges_actor_and_critic_info(monkeypatch):
    learner = learner_mod.AdvantageWeightedFilteredSFTLearner.__new__(
        learner_mod.AdvantageWeightedFilteredSFTLearner
    )
    learner._online_data_buffer = types.SimpleNamespace(size=8, batch_size=4)
    learner._update_critics = lambda: {"critic/q_loss": jnp.array(0.25)}

    monkeypatch.setattr(
        learner_mod.LegacyFilteredSFTLearner,
        "update",
        lambda self: {"actor/loss": jnp.array(1.0)},
    )

    info = learner.update()
    assert float(info["actor/loss"]) == 1.0
    assert float(info["critic/q_loss"]) == 0.25
