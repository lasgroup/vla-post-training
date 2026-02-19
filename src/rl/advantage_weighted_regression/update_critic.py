# ruff: noqa: F722
import dataclasses
from collections.abc import Callable, Sequence
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
from src.training.config import OnlineTrainConfig
from src.rl.networks.rl_networks import ObsType, ActionType, StateActionCritic, StateValue

CriticBatch = tuple[
    ObsType,
    _model.Actions,
    ObsType,
    at.Float[at.Array, " b"],
    at.Float[at.Array, " b"],
]

StateActionCriticDef = Callable[[ObsType, ActionType, nnx.Rngs], StateActionCritic]
StateValueDef = Callable[[ObsType, nnx.Rngs], StateValue]


def _use_ema_critic(config: OnlineTrainConfig) -> bool:
    rl_config = getattr(config, "rl", None)
    return bool(getattr(rl_config, "use_ema_critic", False))


def _critic_ema_decay(config: OnlineTrainConfig) -> float | None:
    rl_config = getattr(config, "rl", None)
    if rl_config is not None and hasattr(rl_config, "critic_ema_decay"):
        critic_ema_decay = getattr(rl_config, "critic_ema_decay")
        return None if critic_ema_decay is None else float(critic_ema_decay)
    return config.ema_decay


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
def summarize_critic_values(critic_values: at.ArrayLike) -> at.Float[at.Array, " b"]:
    critic_values = jnp.asarray(critic_values, dtype=jnp.float32)
    # using an ensemble of critics
    if critic_values.ndim > 2:
        # Take min across the ensemble members
        # TODO: Add different summarization options such as sampling or mean
        critic_values = jnp.min(critic_values, axis=0)
    return _as_scalar_batch(critic_values)


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
    tx = _optimizer.create_optimizer(
        config.optimizer, config.lr_schedule, weight_decay_mask=None
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
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )
    train_state = jax.jit(
        init,
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(dummy_obs, dummy_act, init_rng)
    return train_state, state_sharding


def init_state_value_train_state(
    config: OnlineTrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    critic_def: StateValueDef,
    dummy_obs: ObsType,
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(
        config.optimizer, config.lr_schedule, weight_decay_mask=None
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
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )
    train_state = jax.jit(
        init,
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(dummy_obs, init_rng)
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
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old
                + (1 - state.ema_decay) * new,
                state.ema_params,
                new_params,
            ),
        )
    return new_state


def _kernel_param_norm(model: nnx.Module) -> at.Float[at.Array, ""]:
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(
                nnx_utils.PathRegex(
                    ".*/(bias|scale|pos_embedding|input_embedding)"
                )
            ),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    return optax.global_norm(kernel_params)


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

    observation, actions, next_observation, reward, discount = batch
    reward = _as_scalar_batch(reward)
    discount = _as_scalar_batch(discount)
    actions = flatten_action_horizon(actions)

    @at.typecheck
    def loss_fn(
        critic_model: StateActionCritic,
        observation: ObsType,
        actions: _model.Actions,
        next_observation: ObsType,
        reward: at.Float[at.ArrayLike, " b"],
        discount: at.Float[at.ArrayLike, " b"],
        target_value_model: StateValue,
    ) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:

        q_values = summarize_critic_values(critic_model(observation, actions))
        bootstrapped_values = summarize_critic_values(
            target_value_model(next_observation)
        )
        td_targets = reward +  discount * jax.lax.stop_gradient(bootstrapped_values)
        td_errors = q_values - td_targets
        loss = jnp.mean(jnp.square(td_errors))
        return loss, {
            "td_error_mean": jnp.mean(td_errors),
            "q_value_mean": jnp.mean(q_values),
            "td_target_mean": jnp.mean(td_targets),
        }

    diff_state = nnx.DiffState(0, nnx.Param)
    (loss, aux_data), grads = nnx.value_and_grad(
        loss_fn, has_aux=True, argnums=diff_state
    )(
        q_model,
        observation,
        actions,
        next_observation,
        reward,
        discount,
        value_model,
    )
    new_state = _update_train_state(q_state, q_model, grads)
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": _kernel_param_norm(q_model),
    } | aux_data
    return new_state, info


@at.typecheck
def train_value_step(
    config: OnlineTrainConfig,
    rng: at.KeyArrayLike,
    value_state: training_utils.TrainState,
    q_state: training_utils.TrainState,
    batch: CriticBatch,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    del rng
    value_model = nnx.merge(value_state.model_def, value_state.params)
    value_model.train()
    q_model = create_critic(q_state, config)
    q_model.eval()

    observation, actions, _, _, _, _ = batch

    @at.typecheck
    def loss_fn(
        critic_model: StateValue,
        observation: ObsType,
        actions: _model.Actions,
        target_q_model: StateActionCritic,
    ) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:
        values = summarize_critic_values(critic_model(observation))
        q_values = summarize_critic_values(target_q_model(observation, actions))
        q_targets = jax.lax.stop_gradient(q_values)
        errors = values - q_targets
        loss = jnp.mean(jnp.square(errors))
        return loss, {
            "value_mean": jnp.mean(values),
            "q_target_mean": jnp.mean(q_targets),
            "value_error_mean": jnp.mean(errors),
        }

    diff_state = nnx.DiffState(0, nnx.Param)
    (loss, aux_data), grads = nnx.value_and_grad(
        loss_fn, has_aux=True, argnums=diff_state
    )(value_model, observation, actions, q_model)
    new_state = _update_train_state(value_state, value_model, grads)
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": _kernel_param_norm(value_model),
    } | aux_data
    return new_state, info
