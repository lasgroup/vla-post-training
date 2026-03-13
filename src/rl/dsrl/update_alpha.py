# ruff: noqa: F722
from __future__ import annotations

import dataclasses
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

import openpi.shared.array_typing as at
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
from src.training.config import OnlineTrainConfig


class Temperature(nnx.Module):
    """Learnable SAC entropy coefficient: alpha = exp(log_alpha)."""

    def __init__(self, init_alpha: float):
        init_alpha = max(1e-6, float(init_alpha))
        self.log_alpha = nnx.Param(jnp.asarray(jnp.log(init_alpha), dtype=jnp.float32))

    def __call__(self) -> jax.Array:
        return jnp.exp(jnp.clip(self.log_alpha.value, -20.0, 2.0))


def alpha_autotune_enabled(config: OnlineTrainConfig) -> bool:
    rl = getattr(config, "rl", None)
    return bool(getattr(rl, "autotune_alpha", True))


def resolve_target_entropy(config: OnlineTrainConfig, action_dim: int) -> float:
    rl = getattr(config, "rl", None)
    target_entropy = getattr(rl, "target_entropy", "auto")
    if target_entropy in (None, "auto"):
        return -float(action_dim) / 2.0
    return float(target_entropy)


def _get_init_alpha(config: OnlineTrainConfig) -> float:
    rl = getattr(config, "rl", None)
    return max(1e-6, float(getattr(rl, "init_alpha", 1.0)))


def _get_alpha_lr(config: OnlineTrainConfig) -> float:
    rl = getattr(config, "rl", None)
    return max(1e-8, float(getattr(rl, "alpha_lr", 3e-4)))


def init_alpha_state(
    config: OnlineTrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh | None,
    *,
    use_sharding: bool = True,
) -> tuple[training_utils.TrainState, Any]:
    tx = optax.adam(_get_alpha_lr(config))
    init_alpha = _get_init_alpha(config)

    def init(rng) -> training_utils.TrainState:
        del rng
        temperature = Temperature(init_alpha=init_alpha)
        params = nnx.state(temperature)
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(temperature),
            tx=tx,
            opt_state=tx.init(nnx.filter_state(params, nnx.Param)),
            ema_decay=None,
            ema_params=None,
        )

    train_state_shape = jax.eval_shape(init, init_rng)

    if not use_sharding:
        train_state = init(init_rng)
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
    )(init_rng)
    return train_state, state_sharding


def alpha_value(alpha_state: training_utils.TrainState) -> jax.Array:
    temperature = nnx.merge(alpha_state.model_def, alpha_state.params)
    temperature.eval()
    return temperature()


def _update_alpha_state(
    state: training_utils.TrainState,
    model: nnx.Module,
    grads: nnx.State,
) -> training_utils.TrainState:
    params = nnx.filter_state(state.params, nnx.Param)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    return dataclasses.replace(
        state,
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
    )


@at.typecheck
def train_alpha_step(
    config: OnlineTrainConfig,
    rng: at.KeyArrayLike,
    alpha_state: training_utils.TrainState,
    entropy: at.Float[at.ArrayLike, ""],
    target_entropy: at.Float[at.ArrayLike, ""],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    del config, rng
    temperature = nnx.merge(alpha_state.model_def, alpha_state.params)
    temperature.train()

    entropy = jax.lax.stop_gradient(jnp.asarray(entropy, dtype=jnp.float32))
    target_entropy = jax.lax.stop_gradient(jnp.asarray(target_entropy, dtype=jnp.float32))

    # Match dsrl_reference objective:
    #   J(alpha) = E[ alpha * (entropy - target_entropy) ].
    @at.typecheck
    def loss_fn(
        model: Temperature,
        entropy_value: at.Float[at.ArrayLike, ""],
        target: at.Float[at.ArrayLike, ""],
    ) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:
        alpha = model()
        alpha_loss = alpha * (entropy_value - target)
        return alpha_loss, {
            "alpha": alpha,
            "alpha_loss": alpha_loss,
            "entropy": entropy_value,
            "log_prob_mean": -entropy_value,
            "target_entropy": target,
        }

    diff_state = nnx.DiffState(0, nnx.Param)
    (loss, aux_data), grads = nnx.value_and_grad(
        loss_fn, has_aux=True, argnums=diff_state
    )(
        temperature,
        entropy,
        target_entropy,
    )
    new_state = _update_alpha_state(alpha_state, temperature, grads)
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
    } | aux_data
    return new_state, info
