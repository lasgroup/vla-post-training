# ruff: noqa: F722
from src.training.config import OnlineTrainConfig, AdvantageWeightedSFTLearnerConfig
from src.rl.advantage_weighted_sft.update_critic import (
    create_critic,
    flatten_action_horizon,
    summarize_critic_values,
)
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
import dataclasses

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.utils as training_utils
from src.rl.networks.rl_networks import ObsType


@at.typecheck
def _awr_beta(config: OnlineTrainConfig) -> float:
    assert isinstance(config.rl, AdvantageWeightedSFTLearnerConfig)
    beta = config.rl.beta
    return max(beta, 1e-6)


@at.typecheck
def train_step(
    config: OnlineTrainConfig,
    rng: at.KeyArrayLike,
    policy_state: training_utils.TrainState,
    state_action_critic_state: training_utils.TrainState,
    value_state: training_utils.TrainState,
    batch: tuple[_model.Observation, ObsType, _model.Actions],
    mc_return: at.Array | None = None,
    is_success: at.Float[at.Array, " b"] | None = None,
    scale: at.Array | float = 1.0,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    policy_observation, critic_observation, actions = batch

    policy = nnx.merge(policy_state.model_def, policy_state.params)
    policy.train()

    assert isinstance(config.rl, AdvantageWeightedSFTLearnerConfig)
    reset_period = config.rl.policy.reset_params_to_ema_period
    normalizer_config = config.rl.normalizer_config

    if config.rl.use_mc_returns:
        assert mc_return is not None, "mc_return must be provided when use_mc_returns=True"
        value_critic = create_critic(value_state, config)
        value_critic.eval()
        value = summarize_critic_values(value_critic(critic_observation), config)  # (B,)
        advantage = mc_return - value  # (B, )
    else:
        state_action_critic = create_critic(state_action_critic_state, config)
        state_action_critic.eval()
        value_critic = create_critic(value_state, config)
        value_critic.eval()
        # 2. Compute the advantage weights OUTSIDE the value_and_grad trace
        critic_actions = flatten_action_horizon(actions)
        value = summarize_critic_values(value_critic(critic_observation), config)  # (B,)
        q_value = summarize_critic_values(
            state_action_critic(critic_observation, critic_actions), config
        )  # (B,)
        advantage = q_value - value  # (B, )
    score = advantage / scale
    score = score / _awr_beta(config)
    assert isinstance(config.rl, AdvantageWeightedSFTLearnerConfig)
    score = jnp.minimum(score, config.rl.weight_clip)  # Clipping

    score = jnp.exp(score)
    score = score / config.rl.advantage_scale  # Normalize advantage w.r.t scale
    score = jnp.clip(score, min=1e-6)

    score = jax.lax.stop_gradient(score)  # Explicitly cut gradients

    assert isinstance(config.rl, AdvantageWeightedSFTLearnerConfig)
    # Read weight outside loss_fn so the `if` becomes a Python-level branch that
    # JAX sees as a compile-time constant — zero overhead when the feature is off.
    filtered_sft_weight = config.rl.filtered_sft_weight

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        policy_observation: _model.Observation,
        actions: _model.Actions,
        score: jnp.ndarray,
    ) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:
        chunked_loss = model.compute_loss(rng, policy_observation, actions, train=True)
        while score.ndim < chunked_loss.ndim:
            score = score[..., jnp.newaxis]
        awr_loss = jnp.mean(score * chunked_loss)
        aux_data = {"chunked_loss": jnp.mean(chunked_loss), "awr_loss": awr_loss}

        if filtered_sft_weight > 0.0 and is_success is not None:
            _is_success = jax.lax.stop_gradient(is_success)
            # Broadcast (B,) mask to match chunked_loss which may be (B, T, ...).
            while _is_success.ndim < chunked_loss.ndim:
                _is_success = _is_success[..., jnp.newaxis]
            sft_loss = jnp.mean(_is_success * chunked_loss)
            aux_data["sft_loss"] = sft_loss
            total_loss = awr_loss + filtered_sft_weight * sft_loss
        else:
            total_loss = awr_loss

        return total_loss, aux_data

    train_rng = jax.random.fold_in(rng, policy_state.step)

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, aux_data), grads = nnx.value_and_grad(
        loss_fn, has_aux=True, argnums=diff_state
    )(
        policy,
        train_rng,
        policy_observation,
        actions,
        score,
    )

    params = nnx.filter_state(policy_state.params, config.trainable_filter)
    updates, new_opt_state = policy_state.tx.update(
        grads, policy_state.opt_state, params
    )
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(policy, new_params)
    new_params = nnx.state(policy)

    new_state = dataclasses.replace(
        policy_state,
        step=policy_state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
    )
    if policy_state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: policy_state.ema_decay * old
                + (1 - policy_state.ema_decay) * new,
                policy_state.ema_params,
                new_params,
            ),
        )
        if reset_period:
            step = new_state.step

            def keep_state(state):
                return state

            def revert_to_ema(state):
                return state.replace(params=jax.tree.map(lambda x: x, state.ema_params))

            new_state = jax.lax.cond(
                step % reset_period == 0, revert_to_ema, keep_state, new_state
            )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        policy,
        nnx.All(
            nnx.Param,
            nnx.Not(
                nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")
            ),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        "advantage_mean": jnp.mean(advantage),
        "advantage_max": jnp.max(advantage),
        "advantage_min": jnp.min(advantage),
        "advantage_std": jnp.std(advantage),
        "advantage_q_up": jnp.quantile(advantage, normalizer_config.q_up),
        "advantage_median": jnp.median(advantage),  # or jnp.quantile(advantage, 0.50)
        "advantage_q_low": jnp.quantile(advantage, normalizer_config.q_low),
    } | aux_data
    return new_state, info
