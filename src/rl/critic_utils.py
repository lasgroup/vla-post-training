"""Shared critic utilities used across RL algorithms."""
import dataclasses

import flax.nnx as nnx
import jax.numpy as jnp
import optax

import openpi.training.utils as training_utils


def _update_train_state(
    state: training_utils.TrainState,
    model: nnx.Module,
    grads: nnx.State,
) -> training_utils.TrainState:
    """NNX-correct train-state update that EMAs only nnx.Param leaves.

    BatchStat (BatchNorm running stats) and RngState (PRNG keys) are
    non-arithmetic — they are passed through as-is from new_params.
    """
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


def _bro_pessimistic_reduce(values: jnp.ndarray, pessimism: float) -> jnp.ndarray:
    """BRO ensemble reduction: mean - pessimism * half-range.

    For a 2-member ensemble with pessimism=1 this equals min(v1, v2).
    """
    mean = jnp.mean(values, axis=0)
    spread = (jnp.max(values, axis=0) - jnp.min(values, axis=0)) / 2
    return mean - pessimism * spread
