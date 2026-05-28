# ruff: noqa: F722
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

import openpi.shared.array_typing as at
import openpi.training.utils as training_utils
from src.training.config import OnlineTrainConfig, AdvantageWeightedSFTLearnerConfig
from src.rl.advantage_weighted_sft.update_critic import (
    CriticBatch,
    _as_scalar_batch,
    _kernel_param_norm,
    create_critic,
    flatten_action_horizon,
)
from src.rl.critic_utils import _update_train_state, _bro_pessimistic_reduce


@at.typecheck
def train_bronet_q_step(
    config: OnlineTrainConfig,
    rng: at.KeyArrayLike,
    q_state: training_utils.TrainState,
    value_state: training_utils.TrainState,
    batch: CriticBatch,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    del rng
    assert isinstance(config.rl, AdvantageWeightedSFTLearnerConfig)
    step = q_state.step // config.rl.critic.num_updates_per_batch

    q_model = nnx.merge(q_state.model_def, q_state.params)
    q_model.train()
    value_model = create_critic(value_state, config)  # EMA-V, no gradients
    value_model.eval()

    observation, actions, next_observation, reward, discount, mc_return = batch
    reward    = _as_scalar_batch(reward)
    discount  = _as_scalar_batch(discount)
    mc_return = _as_scalar_batch(mc_return)
    actions   = flatten_action_horizon(actions)

    # Bootstrap target from V ensemble — computed once, outside loss_fn
    next_v   = value_model(next_observation)                                   # (num_vs, B)
    target_v = _bro_pessimistic_reduce(next_v, config.rl.critic.bronet_pessimism)     # (B,)

    def loss_fn(q_model, observation, actions):
        td_weight = jnp.clip(config.rl.critic.td_weight_schedule.create()(step), 0.0, 1.0)
        mc_weight = 1.0 - td_weight

        q_values   = q_model(observation, actions)                             # (num_qs, B)
        td_targets = jax.lax.stop_gradient(reward + discount * target_v)      # (B,)

        td_losses = jnp.mean((q_values - td_targets[None]) ** 2, axis=1)      # (num_qs,)
        mc_losses = jnp.mean((q_values - mc_return[None])   ** 2, axis=1)     # (num_qs,)
        td_loss   = jnp.mean(td_losses)
        mc_loss   = jnp.mean(mc_losses)

        loss = (
            jax.lax.cond(td_weight > 0.0, lambda: td_weight * td_loss, lambda: jnp.zeros(()))
            + jax.lax.cond(mc_weight > 0.0, lambda: mc_weight * mc_loss, lambda: jnp.zeros(()))
        )
        return loss, {
            "value_mean": jnp.mean(q_values),
            "td_loss":    td_loss,
            "mc_loss":    mc_loss,
            "td_weight":  td_weight,
        }

    diff_state = nnx.DiffState(0, nnx.Param)
    (loss, aux_data), grads = nnx.value_and_grad(
        loss_fn, has_aux=True, argnums=diff_state
    )(q_model, observation, actions)

    new_state = _update_train_state(q_state, q_model, grads)
    info = {
        "loss":           loss,
        "grad_norm":      optax.global_norm(grads),
        "param_norm":     _kernel_param_norm(q_model),
        "mc_return_mean": jnp.mean(mc_return),
        "mc_return_std":  jnp.std(mc_return),
    } | aux_data
    return new_state, info


@at.typecheck
def train_bronet_value_step(
    config: OnlineTrainConfig,
    rng: at.KeyArrayLike,
    value_state: training_utils.TrainState,
    q_state: training_utils.TrainState,
    batch: CriticBatch,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    del rng
    assert isinstance(config.rl, AdvantageWeightedSFTLearnerConfig)
    step = value_state.step // config.rl.critic.num_updates_per_batch

    value_model = nnx.merge(value_state.model_def, value_state.params)
    value_model.train()
    q_model = create_critic(q_state, config)  # EMA-Q, no gradients
    q_model.eval()

    observation, actions, _, _, _, mc_return = batch
    actions   = flatten_action_horizon(actions)
    mc_return = _as_scalar_batch(mc_return)

    # TD target from Q ensemble — computed once, outside loss_fn
    q_values = q_model(observation, actions)                                   # (num_qs, B)
    target_q = _bro_pessimistic_reduce(q_values, config.rl.critic.bronet_pessimism)  # (B,)

    def loss_fn(value_model, observation):
        td_weight = jnp.clip(config.rl.critic.td_weight_schedule.create()(step), 0.0, 1.0)
        mc_weight = 1.0 - td_weight

        v_values   = value_model(observation)                                  # (num_vs, B)
        td_targets = jax.lax.stop_gradient(target_q)                          # (B,)

        td_losses = jnp.mean((v_values - td_targets[None]) ** 2, axis=1)      # (num_vs,)
        mc_losses = jnp.mean((v_values - mc_return[None])   ** 2, axis=1)     # (num_vs,)
        td_loss   = jnp.mean(td_losses)
        mc_loss   = jnp.mean(mc_losses)

        loss = (
            jax.lax.cond(td_weight > 0.0, lambda: td_weight * td_loss, lambda: jnp.zeros(()))
            + jax.lax.cond(mc_weight > 0.0, lambda: mc_weight * mc_loss, lambda: jnp.zeros(()))
        )
        return loss, {
            "value_mean": jnp.mean(v_values),
            "td_loss":    td_loss,
            "mc_loss":    mc_loss,
            "td_weight":  td_weight,
        }

    diff_state = nnx.DiffState(0, nnx.Param)
    (loss, aux_data), grads = nnx.value_and_grad(
        loss_fn, has_aux=True, argnums=diff_state
    )(value_model, observation)

    new_state = _update_train_state(value_state, value_model, grads)
    info = {
        "loss":           loss,
        "grad_norm":      optax.global_norm(grads),
        "param_norm":     _kernel_param_norm(value_model),
        "mc_return_mean": jnp.mean(mc_return),
        "mc_return_std":  jnp.std(mc_return),
    } | aux_data
    return new_state, info
