# ruff: noqa: F722
import dataclasses
from collections.abc import Callable
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
from src.training.config import OnlineTrainConfig, BestofNLearnerConfig
from src.rl.networks.rl_networks import (
    ObsType,
    ActionType,
    StateActionCritic,
    StateValue,
)
from src.rl.value_distribution import get_value_bounds, make_value_distribution


CriticBatch = tuple[
    ObsType,
    _model.Actions,
    ObsType,
    at.Float[at.Array, " b"],
    at.Float[at.Array, " b"],
    at.Float[at.Array, " b"],  # MC returns
]

StateActionCriticDef = Callable[[ObsType, ActionType, nnx.Rngs], StateActionCritic]
StateValueDef = Callable[[ObsType, nnx.Rngs], StateValue]


def _use_ema_critic(config: OnlineTrainConfig) -> bool:
    assert isinstance(config.rl, BestofNLearnerConfig)
    return config.rl.use_ema_critic


def _critic_ema_decay(config: OnlineTrainConfig) -> float | None:
    assert isinstance(config.rl, BestofNLearnerConfig)
    return config.rl.critic_ema_decay


def create_critic(
    critic_state: training_utils.TrainState,
    config: OnlineTrainConfig,
) -> StateActionCritic | StateValue:
    critic_params = critic_state.params
    if critic_state.ema_params is not None and _use_ema_critic(config):
        critic_params = critic_state.ema_params
    return nnx.merge(critic_state.model_def, critic_params)


@at.typecheck
def _as_scalar_batch(values: at.ArrayLike) -> at.Float[at.Array, " b"]:
    values = jnp.asarray(values, dtype=jnp.float32)
    if values.ndim == 0:
        return values[jnp.newaxis]
    if values.ndim > 1:
        return values.reshape((values.shape[0], -1))[:, 0]
    return values


@at.typecheck
def summarize_critic_values(
    critic_logits: at.ArrayLike,
    config: OnlineTrainConfig,
    critic_reduction: str = "min",
) -> at.Float[at.Array, " b"]:
    lower, upper = get_value_bounds(config)
    dist = make_value_distribution(critic_logits, config.rl.num_value_bins, lower, upper)
    expected_values = dist.mean()
    if expected_values.ndim > 1:
        # Take min across the ensemble members
        if critic_reduction == "min":
            expected_values = jnp.min(expected_values, axis=0)
        elif critic_reduction == "mean":
            expected_values = jnp.mean(expected_values, axis=0)
        else:
            raise NotImplementedError(
                f"Critic reduction {critic_reduction} is not implemented."
            )
    return _as_scalar_batch(expected_values)


@at.typecheck
def flatten_action_horizon(values: ActionType) -> at.Float[at.Array, "b a"]:
    return values.reshape((values.shape[0], -1))


def _ensure_rngs(rng: at.KeyArrayLike | nnx.Rngs) -> nnx.Rngs:
    if isinstance(rng, nnx.Rngs):
        return rng
    return nnx.Rngs(rng)


def init_state_action_critic_train_state(
    config: OnlineTrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    critic_def: StateActionCriticDef,
    dummy_obs: ObsType,
    dummy_act: ActionType,
) -> tuple[training_utils.TrainState, Any]:
    assert isinstance(config.rl, BestofNLearnerConfig)
    tx = _optimizer.create_optimizer(
        config.rl.critic_optimizer, config.rl.critic_lr_schedule, weight_decay_mask=None
    )
    ema_decay = _critic_ema_decay(config)
    # flatten the array across the array dim
    dummy_act = flatten_action_horizon(dummy_act)
    dummy_act = jax.tree.map(lambda x: x.reshape(*x.shape[:-1], -1), dummy_act)

    def init(obs, act, rng) -> training_utils.TrainState:
        critic = critic_def(obs, act, _ensure_rngs(rng))
        params = nnx.state(critic)
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(critic),
            tx=tx,
            opt_state=tx.init(nnx.filter_state(params, nnx.Param)),
            ema_decay=ema_decay,
            ema_params=None if ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, dummy_obs, dummy_act, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=False)
    # Initialize on the CPU backend, then move params to the GPU sharding.
    cpu = jax.devices("cpu")[0]
    args_cpu = jax.device_put((dummy_obs, dummy_act, init_rng), cpu)
    cpu_state = jax.jit(init, backend="cpu")(*args_cpu)
    train_state = jax.device_put(cpu_state, state_sharding)
    return train_state, state_sharding


def init_state_value_train_state(
    config: OnlineTrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    critic_def: StateValueDef,
    dummy_obs: ObsType,
) -> tuple[training_utils.TrainState, Any]:
    assert isinstance(config.rl, BestofNLearnerConfig)
    tx = _optimizer.create_optimizer(
        config.rl.critic_optimizer, config.rl.critic_lr_schedule, weight_decay_mask=None
    )
    ema_decay = _critic_ema_decay(config)

    def init(obs, rng) -> training_utils.TrainState:
        critic = critic_def(obs, _ensure_rngs(rng))
        params = nnx.state(critic)
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(critic),
            tx=tx,
            opt_state=tx.init(nnx.filter_state(params, nnx.Param)),
            ema_decay=ema_decay,
            ema_params=None if ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, dummy_obs, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=False)
    # See init_state_action_critic_train_state: init on CPU to avoid GPU cuSolver
    # in orthogonal init, then move to the GPU sharding.
    cpu = jax.devices("cpu")[0]
    args_cpu = jax.device_put((dummy_obs, init_rng), cpu)
    cpu_state = jax.jit(init, backend="cpu")(*args_cpu)
    train_state = jax.device_put(cpu_state, state_sharding)
    return train_state, state_sharding


def _update_train_state(
    state: training_utils.TrainState,
    model: nnx.Module,
    grads: nnx.State,
) -> training_utils.TrainState:
    params = nnx.filter_state(state.params, nnx.Param)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(
        state,
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
    )
    if state.ema_decay is not None and state.ema_params is not None:
        # Only EMA nnx.Param leaves. BatchStat (BatchNorm running stats) and RngState
        # (PRNG keys) are non-arithmetic — pass them through as-is from new_params.
        param_keys = set(nnx.filter_state(new_params, nnx.Param).flat_state())
        old_flat = dict(state.ema_params.flat_state())
        def _ema_or_copy(path, new_val):
            if path in param_keys:
                return state.ema_decay * old_flat[path] + (1 - state.ema_decay) * new_val
            return new_val
        new_state = dataclasses.replace(
            new_state,
            ema_params=new_params.map(_ema_or_copy),
        )
    return new_state


def _kernel_param_norm(model: nnx.Module) -> at.Float[at.Array, ""]:
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(
                nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")
            ),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    return optax.global_norm(kernel_params)


def _q_loss_and_aux(
    config: OnlineTrainConfig,
    critic_model: StateActionCritic,
    target_value_model: StateValue,
    batch: CriticBatch,
    step: at.ArrayLike,
) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:
    assert isinstance(config.rl, BestofNLearnerConfig)
    observation, actions, next_observation, reward, discount, mc_return = batch
    reward = _as_scalar_batch(reward)
    discount = _as_scalar_batch(discount)
    mc_return = _as_scalar_batch(mc_return)
    actions = flatten_action_horizon(actions)

    q_logits = critic_model(observation, actions)
    bootstrapped_values = summarize_critic_values(
        target_value_model(next_observation),
        config,
        critic_reduction=config.rl.critic_reduction,
    )
    td_targets = reward + discount * jax.lax.stop_gradient(bootstrapped_values)
    lower, upper = get_value_bounds(config)
    q_dist = make_value_distribution(
        q_logits,
        config.rl.num_value_bins,
        lower,
        upper,
        config.rl.value_target_type,
    )
    td_weight = config.rl.td_weight_schedule.create()(step)
    td_weight = jnp.clip(td_weight, 0.0, 1.0)
    td_loss = -jnp.mean(q_dist.log_prob(td_targets))
    mc_loss = -jnp.mean(q_dist.log_prob(mc_return))
    loss = td_weight * td_loss + (1 - td_weight) * mc_loss
    return loss, {
        "value_mean": jnp.mean(q_dist.mean()),
        "mc_loss": mc_loss,
        "td_loss": td_loss,
        "td_weight": td_weight,
    }


def _value_loss_and_aux(
    config: OnlineTrainConfig,
    critic_model: StateValue,
    target_q_model: StateActionCritic,
    batch: CriticBatch,
    step: at.ArrayLike,
) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:
    assert isinstance(config.rl, BestofNLearnerConfig)
    observation, actions, _, _, _, mc_return = batch

    actions = flatten_action_horizon(actions)
    mc_return = _as_scalar_batch(mc_return)

    td_weight = config.rl.td_weight_schedule.create()(step)
    td_weight = jnp.clip(td_weight, 0.0, 1.0)
    value_logits = critic_model(observation)
    q_values = summarize_critic_values(
        target_q_model(observation, actions),
        config,
        critic_reduction=config.rl.critic_reduction,
    )
    lower, upper = get_value_bounds(config)
    v_dist = make_value_distribution(
        value_logits,
        config.rl.num_value_bins,
        lower,
        upper,
        config.rl.value_target_type,
    )
    mc_loss = -jnp.mean(v_dist.log_prob(mc_return))
    td_loss = -jnp.mean(v_dist.log_prob(jax.lax.stop_gradient(q_values)))
    loss = td_weight * td_loss + (1 - td_weight) * mc_loss
    return loss, {
        "value_mean": jnp.mean(v_dist.mean()),
        "mc_loss": mc_loss,
        "td_loss": td_loss,
        "td_weight": td_weight,
    }


@at.typecheck
def train_q_step(
    config: OnlineTrainConfig,
    rng: at.KeyArrayLike,
    q_state: training_utils.TrainState,
    value_state: training_utils.TrainState,
    batch: CriticBatch,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    del rng
    q_model = nnx.merge(q_state.model_def, q_state.params)
    q_model.train()
    value_model = create_critic(value_state, config)
    value_model.eval()
    assert isinstance(config.rl, BestofNLearnerConfig)
    step = q_state.step // config.rl.num_critic_updates_per_batch

    @at.typecheck
    def loss_fn(
        critic_model: StateActionCritic, target_value_model: StateValue
    ) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:
        return _q_loss_and_aux(config, critic_model, target_value_model, batch, step)

    diff_state = nnx.DiffState(0, nnx.Param)
    (loss, aux_data), grads = nnx.value_and_grad(
        loss_fn, has_aux=True, argnums=diff_state
    )(q_model, value_model)
    new_state = _update_train_state(q_state, q_model, grads)
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": _kernel_param_norm(q_model),
    } | aux_data
    return new_state, info


@at.typecheck
def evaluate_q_loss(
    config: OnlineTrainConfig,
    q_state: training_utils.TrainState,
    value_state: training_utils.TrainState,
    batch: CriticBatch,
) -> dict[str, at.Array]:
    q_model = nnx.merge(q_state.model_def, q_state.params)
    q_model.train()
    value_model = create_critic(value_state, config)
    value_model.eval()
    assert isinstance(config.rl, BestofNLearnerConfig)
    step = q_state.step // config.rl.num_critic_updates_per_batch

    loss, aux_data = _q_loss_and_aux(config, q_model, value_model, batch, step)
    return {"loss": loss} | aux_data


@at.typecheck
def train_value_step(
    config: OnlineTrainConfig,
    rng: at.KeyArrayLike,
    value_state: training_utils.TrainState,
    q_state: training_utils.TrainState,
    batch: CriticBatch,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    del rng
    assert isinstance(config.rl, BestofNLearnerConfig)
    step = value_state.step // config.rl.num_critic_updates_per_batch
    value_model = nnx.merge(value_state.model_def, value_state.params)
    value_model.train()

    q_model = create_critic(q_state, config)
    q_model.eval()

    @at.typecheck
    def loss_fn(
        critic_model: StateValue, target_q_model: StateActionCritic
    ) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:
        return _value_loss_and_aux(config, critic_model, target_q_model, batch, step)

    diff_state = nnx.DiffState(0, nnx.Param)
    (loss, aux_data), grads = nnx.value_and_grad(
        loss_fn, has_aux=True, argnums=diff_state
    )(value_model, q_model)
    new_state = _update_train_state(value_state, value_model, grads)
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": _kernel_param_norm(value_model),
    } | aux_data
    return new_state, info


@at.typecheck
def evaluate_value_loss(
    config: OnlineTrainConfig,
    value_state: training_utils.TrainState,
    q_state: training_utils.TrainState,
    batch: CriticBatch,
) -> dict[str, at.Array]:
    assert isinstance(config.rl, BestofNLearnerConfig)
    step = value_state.step // config.rl.num_critic_updates_per_batch
    value_model = nnx.merge(value_state.model_def, value_state.params)
    value_model.train()

    q_model = create_critic(q_state, config)
    q_model.eval()

    loss, aux_data = _value_loss_and_aux(config, value_model, q_model, batch, step)
    return {"loss": loss} | aux_data
