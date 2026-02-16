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

CriticBatch = tuple[
    _model.Observation,
    _model.Actions,
    at.Float[at.Array, "b s"],
    at.Float[at.Array, " b"],
    at.Float[at.Array, " b"],
]


def _use_ema_critic(config: OnlineTrainConfig) -> bool:
    rl_config = getattr(config, "rl", None)
    return bool(getattr(rl_config, "use_ema_critic", False))


def _critic_ema_decay(config: OnlineTrainConfig) -> float | None:
    rl_config = getattr(config, "rl", None)
    if rl_config is not None and hasattr(rl_config, "critic_ema_decay"):
        critic_ema_decay = getattr(rl_config, "critic_ema_decay")
        return None if critic_ema_decay is None else float(critic_ema_decay)
    return config.ema_decay


def critic_hidden_dims(config: OnlineTrainConfig) -> tuple[int, ...]:
    rl_config = getattr(config, "rl", None)
    hidden_dims = getattr(rl_config, "critic_hidden_dims", (256, 256))
    if isinstance(hidden_dims, int):
        hidden_dims = (hidden_dims,)
    hidden_dims = tuple(int(dim) for dim in hidden_dims)
    if not hidden_dims:
        raise ValueError("critic_hidden_dims must have at least one hidden layer.")
    return hidden_dims


def create_critic(
    critic_state: training_utils.TrainState,
    config: OnlineTrainConfig,
) -> nnx.Module:
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


class _MLPRegressor(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        *,
        rngs: nnx.Rngs,
    ):
        layers = []
        prev_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            hidden_dim = int(hidden_dim)
            layers.append(nnx.Linear(prev_dim, hidden_dim, rngs=rngs))
            prev_dim = hidden_dim
        self.layers = layers
        self.output = nnx.Linear(prev_dim, 1, rngs=rngs)

    @at.typecheck
    def __call__(
        self, inputs: at.Float[at.ArrayLike, "b d"]
    ) -> at.Float[at.Array, " b"]:
        x = jnp.asarray(inputs, dtype=jnp.float32)
        for layer in self.layers:
            x = jax.nn.silu(layer(x))
        x = self.output(x)
        return jnp.squeeze(x, axis=-1)


class StateValueCritic(nnx.Module):
    def __init__(
        self,
        *,
        state_dim: int,
        hidden_dims: Sequence[int],
        rngs: nnx.Rngs,
    ):
        self.mlp = _MLPRegressor(state_dim, hidden_dims, rngs=rngs)

    @at.typecheck
    def __call__(
        self, observation: _model.Observation
    ) -> at.Float[at.Array, " b"]:
        return self.mlp(jnp.asarray(observation.state, dtype=jnp.float32))


class StateActionValueCritic(nnx.Module):
    def __init__(
        self,
        *,
        state_dim: int,
        action_horizon: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        rngs: nnx.Rngs,
    ):
        self.mlp = _MLPRegressor(
            state_dim + action_horizon * action_dim,
            hidden_dims,
            rngs=rngs,
        )

    @at.typecheck
    def __call__(
        self, observation: _model.Observation, actions: _model.Actions
    ) -> at.Float[at.Array, " b"]:
        state = jnp.asarray(observation.state, dtype=jnp.float32)
        actions = jnp.asarray(actions, dtype=jnp.float32)
        flattened_actions = actions.reshape((actions.shape[0], -1))
        return self.mlp(jnp.concatenate([state, flattened_actions], axis=-1))


def init_critic_train_state(
    config: OnlineTrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    critic_factory: Callable[[at.KeyArrayLike], nnx.Module],
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(
        config.optimizer, config.lr_schedule, weight_decay_mask=None
    )
    ema_decay = _critic_ema_decay(config)

    def init(rng: at.KeyArrayLike) -> training_utils.TrainState:
        critic = critic_factory(rng)
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

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=False)
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )
    train_state = jax.jit(
        init,
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng)
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
def _next_observation(
    observation: _model.Observation,
    next_state: at.Float[at.ArrayLike, "b s"],
) -> _model.Observation:
    return dataclasses.replace(
        observation,
        state=jnp.asarray(next_state, dtype=jnp.float32),
    )


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

    observation, actions, next_state, reward, discount = batch
    reward = _as_scalar_batch(reward)
    discount = _as_scalar_batch(discount)

    @at.typecheck
    def loss_fn(
        critic_model: nnx.Module,
        observation: _model.Observation,
        actions: _model.Actions,
        next_state: at.Float[at.ArrayLike, "b s"],
        reward: at.Float[at.ArrayLike, " b"],
        discount: at.Float[at.ArrayLike, " b"],
        target_value_model: nnx.Module,
    ) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:
        q_values = _as_scalar_batch(critic_model(observation, actions))
        bootstrapped_values = _as_scalar_batch(
            target_value_model(_next_observation(observation, next_state))
        )
        td_targets = reward + discount * jax.lax.stop_gradient(bootstrapped_values)
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
        next_state,
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

    observation, actions, _, _, _ = batch

    @at.typecheck
    def loss_fn(
        critic_model: nnx.Module,
        observation: _model.Observation,
        actions: _model.Actions,
        target_q_model: nnx.Module,
    ) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:
        values = _as_scalar_batch(critic_model(observation))
        q_values = _as_scalar_batch(target_q_model(observation, actions))
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
