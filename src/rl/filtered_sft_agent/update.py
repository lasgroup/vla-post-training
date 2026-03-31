from src.training.config import OnlineTrainConfig, FilteredSFTLearnerConfig
import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.utils as training_utils


def _tree_abs_stats(tree) -> tuple[jax.Array, jax.Array]:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return zero, zero

    abs_sums = jnp.stack([jnp.sum(jnp.abs(leaf)) for leaf in leaves])
    abs_maxes = jnp.stack([jnp.max(jnp.abs(leaf)) for leaf in leaves])
    total_count = sum(leaf.size for leaf in leaves)
    mean_abs = jnp.sum(abs_sums) / jnp.asarray(total_count, dtype=jnp.float32)
    max_abs = jnp.max(abs_maxes)
    return mean_abs, max_abs


@at.typecheck
def train_step(
    config: OnlineTrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()
    assert isinstance(config.rl, FilteredSFTLearnerConfig)
    reset_period = config.rl.reset_policy_params_to_ema_period

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = nnx.filter_state(state.params, config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )
        if reset_period:
            step = new_state.step

            def keep_state(state):
                return state

            def revert_to_ema(state):
                return state.replace(params=jax.tree.map(lambda x: x, state.ema_params))

            new_state = jax.lax.cond(step % reset_period == 0, revert_to_ema, keep_state, new_state)

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )

    learning_rate = config.lr_schedule.create()(state.step)
    grad_norm = optax.global_norm(grads)
    update_norm = optax.global_norm(updates)
    param_norm = optax.global_norm(kernel_params)
    grad_abs_mean, grad_abs_max = _tree_abs_stats(grads)
    update_abs_mean, update_abs_max = _tree_abs_stats(updates)
    param_abs_mean, param_abs_max = _tree_abs_stats(kernel_params)
    norm_denom = jnp.maximum(param_norm, jnp.asarray(1e-12, dtype=param_norm.dtype))

    info = {
        "train/loss": loss,
        "train/learning_rate": learning_rate,
        "train/grad_norm": grad_norm,
        "train/update_norm": update_norm,
        "train/param_norm": param_norm,
        "train/grad_abs_mean": grad_abs_mean,
        "train/grad_abs_max": grad_abs_max,
        "train/update_abs_mean": update_abs_mean,
        "train/update_abs_max": update_abs_max,
        "train/param_abs_mean": param_abs_mean,
        "train/param_abs_max": param_abs_max,
        "train/grad_to_param_norm": grad_norm / norm_denom,
        "train/update_to_param_norm": update_norm / norm_denom,
    }
    return new_state, info
