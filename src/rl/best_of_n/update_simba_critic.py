# ruff: noqa: F722
"""Categorical TD update logic for SimbaV2 critics in best-of-N.

Mirrors the AWR SimbaV2 update with the following adaptations:
  - Asserts BestofNLearnerConfig (not AdvantageWeightedSFTLearnerConfig)
  - Imports helpers from src.rl.best_of_n.update_critic
  - Reads simba_* fields from BestofNLearnerConfig
"""
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

import openpi.shared.array_typing as at
import openpi.training.utils as training_utils

from src.training.config import OnlineTrainConfig, BestofNLearnerConfig
from src.rl.best_of_n.update_critic import (
    CriticBatch,
    flatten_action_horizon,
    _as_scalar_batch,
    create_critic,
    _kernel_param_norm,
)
from src.rl.simba_update_utils import (
    categorical_td_loss,
    categorical_cross_entropy,
    scalar_to_hl_gauss,
    select_ensemble_log_probs,
    update_simba_train_state,
)


# ---------------------------------------------------------------------------
# Q-critic update
# ---------------------------------------------------------------------------

@at.typecheck
def train_simba_q_step(
    config: OnlineTrainConfig,
    rng: at.KeyArrayLike,
    q_state: training_utils.TrainState,
    value_state: training_utils.TrainState,
    batch: CriticBatch,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Update the SimbaV2 Q-critic with categorical TD + MC loss.

    TD target: Bellman projection of EMA-V's distribution through (reward, discount).
    MC target: two-hot encoding of the pre-computed MC return.
    """
    del rng
    assert isinstance(config.rl, BestofNLearnerConfig)
    step = q_state.step // config.rl.num_critic_updates_per_batch
    td_weight_schedule = config.rl.td_weight_schedule
    num_bins = config.rl.simba_num_bins
    min_v = config.rl.simba_min_v
    max_v = config.rl.simba_max_v

    q_model = nnx.merge(q_state.model_def, q_state.params)
    q_model.train()
    value_model = create_critic(value_state, config)  # EMA-V, no gradients
    value_model.eval()

    observation, actions, next_observation, reward, discount, mc_return = batch
    reward = _as_scalar_batch(reward)
    discount = _as_scalar_batch(discount)
    mc_return = _as_scalar_batch(mc_return)
    actions = flatten_action_horizon(actions)

    # EMA-V distribution at next_observation — computed once outside loss_fn
    next_v_values, next_v_log_probs = value_model(next_observation)
    # (num_vs, B), (num_vs, B, num_bins)
    target_v_log_probs = select_ensemble_log_probs(
        next_v_values, next_v_log_probs, reduction=config.rl.simba_ensemble_reduction
    )  # (B, num_bins)

    mc_target_probs = scalar_to_hl_gauss(mc_return, num_bins, min_v, max_v)  # (B, num_bins)

    def loss_fn(
        q_model,
        observation,
        actions,
    ) -> tuple[jnp.ndarray, dict]:
        td_weight = jnp.clip(td_weight_schedule.create()(step), 0.0, 1.0)

        q_values, q_log_probs = q_model(observation, actions)
        # q_values:    (num_qs, B)
        # q_log_probs: (num_qs, B, num_bins)

        # TD loss: shift V's distribution through the Bellman operator
        td_losses = jax.vmap(
            lambda lp: categorical_td_loss(
                lp, target_v_log_probs, reward, discount, num_bins, min_v, max_v
            ),
            in_axes=0,
        )(q_log_probs)  # (num_qs,)

        # MC loss: direct regression to discounted return
        mc_losses = jax.vmap(
            lambda lp: categorical_cross_entropy(lp, mc_target_probs),
            in_axes=0,
        )(q_log_probs)  # (num_qs,)

        td_loss = jnp.mean(td_losses)
        mc_loss = jnp.mean(mc_losses)
        loss = td_weight * td_loss + (1 - td_weight) * mc_loss

        q_probs = jnp.exp(q_log_probs)
        boundary_mass = q_probs[..., 0] + q_probs[..., -1]               # (num_qs, B)
        entropy = -jnp.sum(q_log_probs * q_probs, axis=-1)               # (num_qs, B)
        return loss, {
            "value_mean": jnp.mean(q_values),
            "td_loss": td_loss,
            "mc_loss": mc_loss,
            "td_weight": td_weight,
            "boundary_bin_pct": jnp.mean(boundary_mass) * 100,
            "entropy": jnp.mean(entropy),
        }

    diff_state = nnx.DiffState(0, nnx.Param)
    (loss, aux_data), grads = nnx.value_and_grad(
        loss_fn, has_aux=True, argnums=diff_state
    )(q_model, observation, actions)

    new_state = update_simba_train_state(q_state, q_model, grads)
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": _kernel_param_norm(q_model),
        "mc_return_mean": jnp.mean(mc_return),
        "mc_return_std": jnp.std(mc_return),
    } | aux_data
    return new_state, info


# ---------------------------------------------------------------------------
# V-critic update
# ---------------------------------------------------------------------------

@at.typecheck
def train_simba_value_step(
    config: OnlineTrainConfig,
    rng: at.KeyArrayLike,
    value_state: training_utils.TrainState,
    q_state: training_utils.TrainState,
    batch: CriticBatch,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Update the SimbaV2 V-critic with categorical TD + MC loss.

    TD target: distribution of EMA-Q at (obs, buffer_action) — no Bellman shift,
               V is trained to match Q at the same state.
    MC target: two-hot encoding of the pre-computed MC return.
    """
    del rng
    assert isinstance(config.rl, BestofNLearnerConfig)
    step = value_state.step // config.rl.num_critic_updates_per_batch
    td_weight_schedule = config.rl.td_weight_schedule
    num_bins = config.rl.simba_num_bins
    min_v = config.rl.simba_min_v
    max_v = config.rl.simba_max_v

    value_model = nnx.merge(value_state.model_def, value_state.params)
    value_model.train()
    q_model = create_critic(q_state, config)  # EMA-Q, no gradients
    q_model.eval()

    observation, actions, _, _, _, mc_return = batch
    actions = flatten_action_horizon(actions)
    mc_return = _as_scalar_batch(mc_return)

    # EMA-Q distribution at (obs, buffer_actions) — computed once outside loss_fn
    q_values, q_log_probs = q_model(observation, actions)
    # (num_qs, B), (num_qs, B, num_bins)
    target_q_log_probs = select_ensemble_log_probs(
        q_values, q_log_probs, reduction=config.rl.simba_ensemble_reduction
    )  # (B, num_bins)
    target_q_probs = jnp.exp(target_q_log_probs)  # (B, num_bins)

    mc_target_probs = scalar_to_hl_gauss(mc_return, num_bins, min_v, max_v)  # (B, num_bins)

    def loss_fn(
        value_model,
        observation,
    ) -> tuple[jnp.ndarray, dict]:
        td_weight = jnp.clip(td_weight_schedule.create()(step), 0.0, 1.0)

        v_values, v_log_probs = value_model(observation)
        # v_values:    (num_vs, B)
        # v_log_probs: (num_vs, B, num_bins)

        # TD loss: match the Q distribution (no Bellman shift — V ≈ Q at same state)
        td_losses = jax.vmap(
            lambda lp: categorical_cross_entropy(lp, target_q_probs),
            in_axes=0,
        )(v_log_probs)  # (num_vs,)

        # MC loss
        mc_losses = jax.vmap(
            lambda lp: categorical_cross_entropy(lp, mc_target_probs),
            in_axes=0,
        )(v_log_probs)  # (num_vs,)

        td_loss = jnp.mean(td_losses)
        mc_loss = jnp.mean(mc_losses)
        loss = td_weight * td_loss + (1 - td_weight) * mc_loss

        v_probs = jnp.exp(v_log_probs)
        boundary_mass = v_probs[..., 0] + v_probs[..., -1]               # (num_vs, B)
        entropy = -jnp.sum(v_log_probs * v_probs, axis=-1)               # (num_vs, B)
        return loss, {
            "value_mean": jnp.mean(v_values),
            "td_loss": td_loss,
            "mc_loss": mc_loss,
            "td_weight": td_weight,
            "boundary_bin_pct": jnp.mean(boundary_mass) * 100,
            "entropy": jnp.mean(entropy),
        }

    diff_state = nnx.DiffState(0, nnx.Param)
    (loss, aux_data), grads = nnx.value_and_grad(
        loss_fn, has_aux=True, argnums=diff_state
    )(value_model, observation)

    new_state = update_simba_train_state(value_state, value_model, grads)
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": _kernel_param_norm(value_model),
        "mc_return_mean": jnp.mean(mc_return),
        "mc_return_std": jnp.std(mc_return),
    } | aux_data
    return new_state, info
