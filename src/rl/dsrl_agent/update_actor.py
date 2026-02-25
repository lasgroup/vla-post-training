# ruff: noqa: F722
"""SAC actor update for DSRL.

The actor minimises:  E_s[ α · log π(a|s) − Q(s, a) ]
where a ~ π(·|s) via the reparameterisation trick (tanh-squashed Gaussian).

Compared to the AWR actor (which re-weights a supervised loss by advantage):
  - AWR:  min  Σ softmax(A/β) · ‖π(s) − a_data‖²
  - SAC:  min  E_{a~π}[ α log π(a|s) − Q(s, a) ]

The SAC actor does *not* need demonstration actions — it optimises purely
through the Q-function gradient (reparameterisation trick).
"""
from __future__ import annotations

import dataclasses
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.utils as training_utils
from src.training.config import OnlineTrainConfig
from src.rl.dsrl_agent.update_critic import (
    flatten_action_horizon,
    summarize_critic_values,
)
from collections.abc import Callable
from src.rl.networks.rl_networks import ObsType, ActionType, Policy
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding

PolicyDef = Callable[[ObsType, ActionType, nnx.Rngs], Policy]


def _ensure_rngs(rng: at.KeyArrayLike | nnx.Rngs) -> nnx.Rngs:
    if isinstance(rng, nnx.Rngs):
        return rng
    return nnx.Rngs(rng)

def init_policy_state(
    config: OnlineTrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh | None,
    *,
    policy_def: PolicyDef,
    dummy_obs: ObsType,
    dummy_act: ActionType,
    use_sharding: bool = True,
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(
        config.optimizer, config.lr_schedule, weight_decay_mask=None
    )
    ema_decay = None  # or pull from config if you want EMA on the actor

    dummy_obs = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), dummy_obs)
    dummy_act = jnp.asarray(dummy_act, dtype=jnp.float32)
    if dummy_act.ndim == 1:
        dummy_act = dummy_act[None, ...]

    def init(obs, act, rng) -> training_utils.TrainState:
        policy = policy_def(obs, act, _ensure_rngs(rng))
        params = nnx.state(policy)
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(policy),
            tx=tx,
            opt_state=tx.init(nnx.filter_state(params, nnx.Param)),
            ema_decay=ema_decay,
            ema_params=None if ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, dummy_obs, dummy_act, init_rng)

    if not use_sharding:
        train_state = init(dummy_obs, dummy_act, init_rng)
        return train_state, None

    if mesh is None:
        raise ValueError("mesh must be provided when use_sharding=True")

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


def _update_actor_state(
    state: training_utils.TrainState,
    model: nnx.Module,
    grads: nnx.State,
) -> training_utils.TrainState:
    """Apply gradients and optionally update EMA params."""
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


# ---------------------------------------------------------------------------
# SAC actor train step
# ---------------------------------------------------------------------------

@at.typecheck
def train_actor_step(
    config: OnlineTrainConfig,
    rng: at.KeyArrayLike,
    actor_state: training_utils.TrainState,
    q_state: training_utils.TrainState,
    batch: ObsType,
    alpha: at.Float[at.ArrayLike, ""] | float,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    alpha = jnp.asarray(alpha, dtype=jnp.float32)

    # ── rebuild models ───────────────────────────────────────────────────
    actor = nnx.merge(actor_state.model_def, actor_state.params)
    actor.train()
    q_params = q_state.ema_params if q_state.ema_params is not None else q_state.params
    q_model_def = q_state.model_def

    # ── loss: E_{a~π}[ α log π(a|s) − Q(s,a) ] ─────────────────────────
    @at.typecheck
    def loss_fn(
        actor_model: nnx.Module,
        observation: ObsType,
        rng: at.KeyArrayLike,
    ) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:
        dist = actor_model(observation)              # policy(obs) -> distribution
        actions = dist.sample(seed=rng)
        log_probs = dist.log_prob(actions)

        # If log_prob is per-dimension, reduce over action dim.
        if jnp.asarray(log_probs).ndim > 1:
            log_probs = jnp.sum(log_probs, axis=-1)
        # Flatten if actor outputs (B, H, D) to match critic input
        flat_actions = actions
        if actions.ndim > 2:
            flat_actions = flatten_action_horizon(actions)
            
        q_model = nnx.merge(q_model_def, q_params)
        q_model.eval()
        q_values = summarize_critic_values(q_model(observation, flat_actions))

        entropy_term = alpha * log_probs        # (B,)
        actor_loss = jnp.mean(entropy_term - q_values)

        return actor_loss, {
            "alpha": alpha,
            "entropy_term_mean": jnp.mean(entropy_term),
            "log_prob_mean": jnp.mean(log_probs),
            "q_value_mean": jnp.mean(q_values),
            "action_mean": jnp.mean(jnp.abs(actions)),
            "action_std": jnp.std(actions),
        }

    sample_rng = jax.random.fold_in(rng, actor_state.step)

    diff_state = nnx.DiffState(0, nnx.Param)
    (loss, aux_data), grads = nnx.value_and_grad(
        loss_fn, has_aux=True, argnums=diff_state
    )(
        actor,
        batch,
        sample_rng,
    )

    new_state = _update_actor_state(actor_state, actor, grads)

    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
    } | aux_data
    return new_state, info
