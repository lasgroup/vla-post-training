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
from src.rl.networks.rl_networks import (
    ObsType,
    ActionType,
    StateActionCritic,
)

CriticBatch = tuple[
    ObsType,
    _model.Actions,
    ObsType,
    at.Float[at.Array, " b"],
    at.Float[at.Array, " b"],
]

StateActionCriticDef = Callable[[ObsType, ActionType, nnx.Rngs], StateActionCritic]


@at.typecheck
def _as_scalar_batch(values: at.ArrayLike) -> at.Float[at.Array, " b"]:
    values = jnp.asarray(values, dtype=jnp.float32)
    if values.ndim == 0:
        return values[jnp.newaxis]
    if values.ndim > 1:
        return values.reshape((values.shape[0], -1))[:, 0]
    return values


@at.typecheck
def flatten_action_horizon(values: ActionType) -> at.Float[at.Array, "b a"]:
    return values.reshape((values.shape[0], -1))


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
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                new_params,
            ),
        )
    return new_state


@at.typecheck
def summarize_critic_values(critic_values: at.ArrayLike) -> at.Float[at.Array, " b"]:
    critic_values = jnp.asarray(critic_values, dtype=jnp.float32)
    if critic_values.ndim > 1:
        critic_values = jnp.min(critic_values, axis=0)
    return _as_scalar_batch(critic_values)


@at.typecheck
def flatten_action_horizon(values: ActionType) -> at.Float[at.Array, "b a"]:
    return values.reshape((values.shape[0], -1))


def _ensure_rngs(rng: at.KeyArrayLike | nnx.Rngs) -> nnx.Rngs:
    if isinstance(rng, nnx.Rngs):
        return rng
    return nnx.Rngs(rng)

def _critic_ema_decay(config: OnlineTrainConfig) -> float | None:
    rl_config = getattr(config, "rl", None)
    if rl_config is not None and hasattr(rl_config, "critic_ema_decay"):
        critic_ema_decay = getattr(rl_config, "critic_ema_decay")
        return None if critic_ema_decay is None else float(critic_ema_decay)
    return config.ema_decay

def init_state_action_critic_train_state(
    config: OnlineTrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh | None,
    *,
    critic_def: StateActionCriticDef,
    dummy_obs: ObsType,
    dummy_act: ActionType,
    use_sharding: bool = True,
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(
        config.optimizer, config.lr_schedule, weight_decay_mask=None
    )
    ema_decay = _critic_ema_decay(config)

    # Normalize dummy inputs for shape inference / init.
    dummy_obs = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), dummy_obs)
    dummy_act = jnp.asarray(dummy_act, dtype=jnp.float32)
    if dummy_act.ndim == 1:
        dummy_act = dummy_act[None, ...]

    # flatten the array across the action dim
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

    if not use_sharding:
        # Local init without OpenPI fsdp sharding / mesh assumptions.
        train_state = init(dummy_obs, dummy_act, init_rng)
        return train_state, None

    if mesh is None:
        raise ValueError("mesh must be provided when use_sharding=True")

    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=False)
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    train_state = jax.jit(
        init,
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(dummy_obs, dummy_act, init_rng)
    return train_state, state_sharding


# ---------------------------------------------------------------------------
# SAC-style Q-function update
# ---------------------------------------------------------------------------
@at.typecheck
def train_q_step(
    config: OnlineTrainConfig,
    rng: at.KeyArrayLike,
    q_state: training_utils.TrainState,
    policy_state: training_utils.TrainState,
    batch: CriticBatch,
    alpha: at.Float[at.ArrayLike, ""] | float,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    # ── online Q model (will receive gradients) ──────────────────────────
    q_model = nnx.merge(q_state.model_def, q_state.params)
    q_model.train()

    # ── target Q model (EMA / Polyak-averaged, no gradients) ─────────────
    target_params = (q_state.ema_params if q_state.ema_params is not None else q_state.params)
    q_target_model = nnx.merge(q_state.model_def, target_params)
    q_target_model.eval()

    # Use current policy state directly (policy(obs) -> distribution)
    policy_params = policy_state.ema_params if policy_state.ema_params is not None else policy_state.params
    policy_model = nnx.merge(policy_state.model_def, policy_params)
    policy_model.eval()

    # ── unpack batch ─────────────────────────────────────────────────────
    observation, actions, next_observation, reward, discount = batch
    reward = _as_scalar_batch(reward)
    discount = _as_scalar_batch(discount)
    actions = flatten_action_horizon(actions)

    # ── sample next actions from policy distribution (no gradient through actor params here) ──
    actor_rng, rng = jax.random.split(rng)

    # actor_model is the policy module here: policy(obs) -> tfd.Distribution
    dist = policy_model(next_observation)
    next_actions = dist.sample(seed=actor_rng)
    next_log_probs = dist.log_prob(next_actions)

    # flatten action horizon if policy outputs (B, H, D)
    if next_actions.ndim > 2:
        next_actions = flatten_action_horizon(next_actions)
    next_log_probs = _as_scalar_batch(next_log_probs)

    # ── entropy coefficient ──────────────────────────────────────────────
    alpha = jnp.asarray(alpha, dtype=jnp.float32)

    # ── compute TD targets (everything stop-gradiented) ──────────────────
    target_q_next = summarize_critic_values(q_target_model(next_observation, next_actions))
    entropy_bonus = -alpha * next_log_probs
    td_targets = jax.lax.stop_gradient(reward + discount * (target_q_next + entropy_bonus))

    # ── loss only differentiates through q_model ─────────────────────────
    @at.typecheck
    def loss_fn(
        critic_model: StateActionCritic,
        observation: ObsType,
        actions: _model.Actions,
        td_targets: at.Float[at.Array, " b"],
    ) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:
        q_values = summarize_critic_values(critic_model(observation, actions))
        td_errors = q_values - td_targets
        loss = jnp.mean(jnp.square(td_errors))
        return loss, {
            "alpha": alpha,
            "td_error_mean": jnp.mean(td_errors),
            "q_value_mean": jnp.mean(q_values),
            "td_target_mean": jnp.mean(td_targets),
            "entropy_bonus_mean": jnp.mean(entropy_bonus),
            "next_log_prob_mean": jnp.mean(next_log_probs),
        }

    diff_state = nnx.DiffState(0, nnx.Param)
    (loss, aux_data), grads = nnx.value_and_grad(
        loss_fn, has_aux=True, argnums=diff_state
    )(
        q_model,
        observation,
        actions,
        td_targets,
    )
    new_state = _update_train_state(q_state, q_model, grads)
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
    } | aux_data
    return new_state, info
