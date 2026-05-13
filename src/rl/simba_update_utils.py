"""Shared SimbaV2 update utilities.

Used by both AWR and best-of-N critic updates. Contains:
  - Categorical distributional losses (Bellman projection, cross-entropy, two-hot)
  - Ensemble reduction helpers
  - L2 weight normalisation (SimbaV2 regularisation)
  - Train-state update with L2 norm applied before EMA
"""
import dataclasses

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

import openpi.shared.nnx_utils as nnx_utils
import openpi.training.utils as training_utils

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


def categorical_cross_entropy(
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
    eps=1e-5,
) -> jnp.ndarray:          # (B, num_bins)
    """Project scalar values onto the bin support via two-hot encoding."""
    scalars = jnp.clip(scalars, min_v, max_v)
    b = (scalars - min_v) / ((max_v - min_v) / (num_bins - 1))
    l = jnp.floor(b).astype(jnp.int32)
    u = jnp.ceil(b).astype(jnp.int32)
    lower_w = u.astype(jnp.float32) - b + (l == u).astype(jnp.float32)
    upper_w = b - l.astype(jnp.float32)

    encoded = (
        jax.nn.one_hot(l, num_bins) * lower_w[:, None]
        + jax.nn.one_hot(u, num_bins) * upper_w[:, None]
        + eps
    )
    encoded = jnp.divide(encoded, jnp.sum(encoded, axis=-1, keepdims=True))

    return encoded


# ---------------------------------------------------------------------------
# Ensemble helpers
# ---------------------------------------------------------------------------

def select_min_member_log_probs(
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


def l2_normalize_critic(model: nnx.Module) -> None:
    """In-place L2 normalisation of all linear kernel params. Biases and
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

def update_simba_train_state(
    state: training_utils.TrainState,
    model: nnx.Module,
    grads: nnx.State,
) -> training_utils.TrainState:
    """Gradient update with L2 normalisation applied before the EMA step."""
    params = nnx.filter_state(state.params, nnx.Param)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params_pre_norm = optax.apply_updates(params, updates)
    nnx.update(model, new_params_pre_norm)

    l2_normalize_critic(model)
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
