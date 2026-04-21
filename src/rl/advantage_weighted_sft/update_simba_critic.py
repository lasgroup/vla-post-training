# ruff: noqa: F722
"""Categorical TD update logic for SimbaV2 critics in AWR.

Ported from the SimbaV2 reference implementation with the following adaptations:
  - No actor / entropy term  (AWR is not entropy-regularised)
  - Pre-computed `discount` replaces explicit gamma * (1 - terminated)
  - Q bootstraps from EMA-V; V bootstraps from EMA-Q  (AWR cross-bootstrap)
  - Mixed TD + MC loss for both Q and V via td_weight schedule
  - L2 weight normalisation applied after gradient step, before EMA update
"""
import dataclasses

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.utils as training_utils

from src.training.config import OnlineTrainConfig, AdvantageWeightedSFTLearnerConfig
from src.rl.advantage_weighted_sft.update_critic import (
    CriticBatch,
    flatten_action_horizon,
    _as_scalar_batch,
    create_critic,
    _kernel_param_norm,
)

EPS = 1e-8


# ---------------------------------------------------------------------------
# Core distributional losses
# ---------------------------------------------------------------------------

def categorical_td_loss(
    pred_log_probs: jnp.ndarray,   # (B, num_bins)
    target_log_probs: jnp.ndarray, # (B, num_bins) — distribution over next-state values
    reward: jnp.ndarray,           # (B,)
    discount: jnp.ndarray,         # (B,) — pre-computed, already gamma^k * (1 - done)
    num_bins: int,
    min_v: float,
    max_v: float,
) -> jnp.ndarray:
    """Categorical Bellman projection + cross-entropy for one ensemble member.

    Shifts the target distribution's support by (reward + discount * z_i), clips
    to [min_v, max_v], bilinearly interpolates back onto the fixed grid, then
    computes cross-entropy against the predicted log-probs.

    Adapted from SimbaV2: entropy term removed; uses AWR pre-computed discount.
    """
    reward = reward.reshape(-1, 1)
    discount = discount.reshape(-1, 1)

    bin_values = jnp.linspace(min_v, max_v, num_bins).reshape(1, -1)   # (1, num_bins)
    target_bin_values = reward + discount * bin_values                   # (B, num_bins)
    target_bin_values = jnp.clip(target_bin_values, min_v, max_v)

    # Bilinear interpolation onto the fixed support
    b = (target_bin_values - min_v) / ((max_v - min_v) / (num_bins - 1))
    l = jnp.floor(b)
    u = jnp.ceil(b)
    l_mask = jax.nn.one_hot(l.reshape(-1), num_bins).reshape(-1, num_bins, num_bins)
    u_mask = jax.nn.one_hot(u.reshape(-1), num_bins).reshape(-1, num_bins, num_bins)

    target_probs = jnp.exp(target_log_probs)                            # (B, num_bins)
    m_l = (target_probs * (u + (l == u).astype(jnp.float32) - b)).reshape(-1, num_bins, 1)
    m_u = (target_probs * (b - l)).reshape(-1, num_bins, 1)
    projected = jax.lax.stop_gradient(
        jnp.sum(m_l * l_mask + m_u * u_mask, axis=1)
    )  # (B, num_bins)

    return -jnp.mean(jnp.sum(projected * pred_log_probs, axis=1))


def _categorical_cross_entropy(
    pred_log_probs: jnp.ndarray,  # (B, num_bins)
    target_probs: jnp.ndarray,    # (B, num_bins)
) -> jnp.ndarray:
    """Cross-entropy against a fixed target distribution (no Bellman shift)."""
    return -jnp.mean(jnp.sum(jax.lax.stop_gradient(target_probs) * pred_log_probs, axis=1))


def scalar_to_two_hot(
    scalars: jnp.ndarray,  # (B,)
    num_bins: int,
    min_v: float,
    max_v: float,
) -> jnp.ndarray:          # (B, num_bins)
    """Project scalar values onto the bin support via two-hot encoding."""
    scalars = jnp.clip(scalars, min_v, max_v)
    b = (scalars - min_v) / ((max_v - min_v) / (num_bins - 1))
    l = jnp.floor(b).astype(jnp.int32)
    u = jnp.ceil(b).astype(jnp.int32)
    lower_w = u.astype(jnp.float32) - b + (l == u).astype(jnp.float32)
    upper_w = b - l.astype(jnp.float32)
    return (
        jax.nn.one_hot(l, num_bins) * lower_w[:, None]
        + jax.nn.one_hot(u, num_bins) * upper_w[:, None]
    )


# ---------------------------------------------------------------------------
# Ensemble helpers
# ---------------------------------------------------------------------------

def _select_min_member_log_probs(
    values: jnp.ndarray,    # (num_members, B)
    log_probs: jnp.ndarray, # (num_members, B, num_bins)
) -> jnp.ndarray:           # (B, num_bins)
    """Return log_probs of the ensemble member with the minimum expected value per batch element."""
    min_indices = jnp.argmin(values, axis=0)  # (B,)
    # vmap over B: each lp is (num_members, num_bins), idx is scalar
    return jax.vmap(lambda lp, idx: lp[idx], in_axes=(1, 0))(log_probs, min_indices)


# ---------------------------------------------------------------------------
# L2 weight normalisation  (SimbaV2 regularisation)
# ---------------------------------------------------------------------------

def _l2_normalize_kernel(x: jnp.ndarray) -> jnp.ndarray:
    """Normalise a weight matrix along its input dimension.

    2-D kernels (in_features, out_features): normalise along axis 0.
    3-D kernels (num_members, in_features, out_features): normalise along axis 1
    (the leading member axis comes from nnx.vmap stacking).
    """
    if x.ndim == 2:
        axis = 0
    elif x.ndim == 3:
        axis = 1
    else:
        return x
    return x / jnp.maximum(jnp.linalg.norm(x, axis=axis, keepdims=True), EPS)


def _l2_normalize_critic(model: nnx.Module) -> None:
    """In-place L2 normalisation of all linear kernel params.  Biases and
    LayerNorm scale/bias are left untouched."""
    kernel_state = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim >= 2,
        ),
    )
    normalized = jax.tree.map(_l2_normalize_kernel, kernel_state)
    nnx.update(model, normalized)


# ---------------------------------------------------------------------------
# Train-state update  (gradient step → L2 norm → EMA)
# ---------------------------------------------------------------------------

def _update_simba_train_state(
    state: training_utils.TrainState,
    model: nnx.Module,
    grads: nnx.State,
) -> training_utils.TrainState:
    """Gradient update with L2 normalisation applied before the EMA step."""
    params = nnx.filter_state(state.params, nnx.Param)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params_pre_norm = optax.apply_updates(params, updates)
    nnx.update(model, new_params_pre_norm)

    _l2_normalize_critic(model)
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
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                new_params,
            ),
        )
    return new_state


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
    assert isinstance(config.rl, AdvantageWeightedSFTLearnerConfig)
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
    target_v_log_probs = _select_min_member_log_probs(next_v_values, next_v_log_probs)
    # (B, num_bins) — pessimistic (min) V member

    mc_target_probs = scalar_to_two_hot(mc_return, num_bins, min_v, max_v)  # (B, num_bins)

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
            lambda lp: _categorical_cross_entropy(lp, mc_target_probs),
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

    new_state = _update_simba_train_state(q_state, q_model, grads)
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
    assert isinstance(config.rl, AdvantageWeightedSFTLearnerConfig)
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
    target_q_log_probs = _select_min_member_log_probs(q_values, q_log_probs)
    target_q_probs = jnp.exp(target_q_log_probs)  # (B, num_bins)

    mc_target_probs = scalar_to_two_hot(mc_return, num_bins, min_v, max_v)  # (B, num_bins)

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
            lambda lp: _categorical_cross_entropy(lp, target_q_probs),
            in_axes=0,
        )(v_log_probs)  # (num_vs,)

        # MC loss
        mc_losses = jax.vmap(
            lambda lp: _categorical_cross_entropy(lp, mc_target_probs),
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

    new_state = _update_simba_train_state(value_state, value_model, grads)
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": _kernel_param_norm(value_model),
        "mc_return_mean": jnp.mean(mc_return),
        "mc_return_std": jnp.std(mc_return),
    } | aux_data
    return new_state, info
