from functools import partial
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from src.rl_training.types import DataType, Params

DatasetDict = Dict[str, DataType]


# Helps to minimize CPU to GPU transfer.
def _unpack(batch):
    # Assuming that if next_observation is missing, it's combined with observation:
    obs_pixels = batch["observations"]["pixels"][..., :-1]
    next_obs_pixels = batch["observations"]["pixels"][..., 1:]

    obs = batch["observations"].copy(add_or_replace={"pixels": obs_pixels})
    next_obs = batch["next_observations"].copy(
        add_or_replace={"pixels": next_obs_pixels}
    )

    batch = batch.copy(
        add_or_replace={"observations": obs, "next_observations": next_obs}
    )

    return batch


@partial(jax.jit, static_argnames="actor_apply_fn")
def eval_log_prob_jit(
    actor_apply_fn: Callable[..., Any],
    actor_params: Params,
    actor_batch_stats: Any,
    batch: DatasetDict,
) -> float:
    # batch = _unpack(batch)
    input_collections = {"params": actor_params}
    if actor_batch_stats is not None:
        input_collections["batch_stats"] = actor_batch_stats
    dist = actor_apply_fn(
        input_collections, batch["observations"], training=False, mutable=False
    )
    log_probs = dist.log_prob(batch["actions"])
    return log_probs.mean()


@partial(jax.jit, static_argnames="actor_apply_fn")
def eval_mse_jit(
    actor_apply_fn: Callable[..., Any],
    actor_params: Params,
    actor_batch_stats: Any,
    batch: DatasetDict,
) -> float:
    # batch = _unpack(batch)
    input_collections = {"params": actor_params}
    if actor_batch_stats is not None:
        input_collections["batch_stats"] = actor_batch_stats
    dist = actor_apply_fn(
        input_collections, batch["observations"], training=False, mutable=False
    )
    mse = (dist.loc - batch["actions"]) ** 2
    return mse.mean()


def eval_reward_function_jit(
    actor_apply_fn: Callable[..., Any],
    actor_params: Params,
    actor_batch_stats: Any,
    batch: DatasetDict,
) -> float:
    # batch = _unpack(batch)
    input_collections = {"params": actor_params}
    if actor_batch_stats is not None:
        input_collections["batch_stats"] = actor_batch_stats
    dist = actor_apply_fn(
        input_collections, batch["observations"], training=False, mutable=False
    )
    pred = dist.mode().reshape(-1)
    loss = -(
        batch["rewards"] * jnp.log(1.0 / (1.0 + jnp.exp(-pred)))
        + (1.0 - batch["rewards"]) * jnp.log(1.0 - 1.0 / (1.0 + jnp.exp(-pred)))
    )
    return loss.mean()


@partial(jax.jit, static_argnames="actor_apply_fn")
def eval_actions_jit(
    actor_apply_fn: Callable[..., Any],
    actor_params: Params,
    observations: np.ndarray,
    actor_batch_stats: Any,
) -> jnp.ndarray:
    input_collections = {"params": actor_params}
    if actor_batch_stats is not None:
        input_collections["batch_stats"] = actor_batch_stats
    dist = actor_apply_fn(
        input_collections, observations, training=False, mutable=False
    )
    return dist.mode()


@partial(jax.jit, static_argnames="actor_apply_fn")
def sample_actions_jit(
    rng: jax.random.PRNGKey,
    actor_apply_fn: Callable[..., Any],
    actor_params: Params,
    observations: np.ndarray,
    actor_batch_stats: Any,
) -> Tuple[jax.random.PRNGKey, jnp.ndarray]:
    input_collections = {"params": actor_params}
    if actor_batch_stats is not None:
        input_collections["batch_stats"] = actor_batch_stats
    dist = actor_apply_fn(input_collections, observations)
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)
