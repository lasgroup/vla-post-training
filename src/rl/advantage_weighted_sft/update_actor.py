# ruff: noqa: F722
from src.training.config import OnlineTrainConfig, AdvantageWeightedSFTLearnerConfig
from src.rl.advantage_weighted_sft.update_critic import (
    create_critic,
    critic_values_per_head,
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


def _per_group_stats(
    advantage: at.Float[at.Array, " b"],
    task_id: at.Int[at.Array, " b"],
    num_groups: int,
    config: OnlineTrainConfig,
) -> dict[str, at.Array]:
    """Advantage location/spread per task group, for the next EMA update.

    Everything is a masked reduction over a fixed (num_groups, b) layout so the
    shapes stay static under jit. Groups absent from this batch report zeros and
    are filtered out by `count > 0` when the EMA is applied.
    """
    normalizer_config = config.rl.normalizer_config
    onehot = jax.nn.one_hot(task_id, num_groups)  # (b, g)
    count = jnp.sum(onehot, axis=0)  # (g,)
    present = count > 0
    denom = jnp.maximum(count, 1.0)

    mean = jnp.sum(onehot * advantage[:, jnp.newaxis], axis=0) / denom
    var = jnp.sum(onehot * (advantage[:, jnp.newaxis] - mean[jnp.newaxis, :]) ** 2, axis=0) / denom
    std = jnp.sqrt(var)

    # NaN out non-members so the quantiles see only this group's samples.
    masked = jnp.where(onehot.T > 0, advantage[jnp.newaxis, :], jnp.nan)  # (g, b)
    zero = jnp.zeros_like(count)
    q_low = jnp.where(present, jnp.nanquantile(masked, normalizer_config.q_low, axis=-1), zero)
    q_up = jnp.where(present, jnp.nanquantile(masked, normalizer_config.q_up, axis=-1), zero)
    a_min = jnp.where(present, jnp.nanmin(masked, axis=-1), zero)
    a_max = jnp.where(present, jnp.nanmax(masked, axis=-1), zero)
    mean = jnp.where(present, mean, zero)
    std = jnp.where(present, std, zero)

    if normalizer_config.method == "quantile":
        bias, scale = q_low, q_up - q_low
    elif normalizer_config.method == "standard_normal":
        bias, scale = mean, std
    elif normalizer_config.method == "min_max":
        bias, scale = a_min, a_max - a_min
    elif normalizer_config.method is None:
        bias, scale = zero, jnp.ones_like(count)
    else:
        raise NotImplementedError(f"normalizer method {normalizer_config.method!r}")

    return {
        "group_count": count,
        "group_batch_bias": bias,
        "group_batch_scale": jnp.clip(scale, min=normalizer_config.min_scale),
        "group_advantage_mean": mean,
        "group_advantage_std": std,
        "group_advantage_q_low": q_low,
        "group_advantage_q_up": q_up,
    }


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
    bias: at.Array | float = 0.0,
    task_id: at.Int[at.Array, " b"] | None = None,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    policy_observation, critic_observation, actions = batch

    policy = nnx.merge(policy_state.model_def, policy_state.params)
    policy.train()

    assert isinstance(config.rl, AdvantageWeightedSFTLearnerConfig)
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
        if config.rl.advantage_combination == "conservative":
            # (4)+(5) per-critic A_i = Q_i - V_i, combined as max(min_i A_i, 0) + min(0, max_i A_i).
            assert config.rl.critic.num_qs == config.rl.critic.num_vs, "conservative advantage needs num_qs == num_vs"
            adv_heads = (
                critic_values_per_head(state_action_critic(critic_observation, critic_actions), config)
                - critic_values_per_head(value_critic(critic_observation), config)
            )  # (n, B)
            advantage = jnp.maximum(jnp.min(adv_heads, axis=0), 0.0) + jnp.minimum(jnp.max(adv_heads, axis=0), 0.0)
        else:
            value = summarize_critic_values(value_critic(critic_observation), config)  # (B,)
            q_value = summarize_critic_values(
                state_action_critic(critic_observation, critic_actions), config
            )  # (B,)
            advantage = q_value - value  # (B, )
    # Per-task baseline taken from THIS batch, not from the EMA. Two things
    # need removing and the batch's own statistics remove both exactly:
    #   - the per-task offset in Q - V (the critic's level error), which decides
    #     which task wins the weight mass in a multi-task batch;
    #   - the common-mode level, which moves by several value units between
    #     consecutive actor updates because the critic takes update_interval
    #     gradient steps in between. An EMA at ema_weight=0.99 cannot track
    #     that, and exponentiating the residual swings the mean weight (and so
    #     the effective lr) by orders of magnitude from one step to the next.
    # The EMA is still updated from these numbers, but only for logging.
    group_stats = (
        _per_group_stats(advantage, task_id, int(jnp.asarray(scale).shape[0]), config)
        if task_id is not None
        else None
    )
    if group_stats is not None and config.rl.normalize_advantages:
        adv_bias = group_stats["group_batch_bias"][task_id]
        adv_scale = group_stats["group_batch_scale"][task_id]
    else:
        adv_bias, adv_scale = bias, scale
    normalized_advantage = (advantage - adv_bias) / adv_scale

    # (1) relu: non-exponentiated max(adv, 0) weights; exp: standard AWR weights.
    if config.rl.advantage_weight_type == "relu":
        score = jax.nn.relu(normalized_advantage)
    else:
        score = normalized_advantage
        score = score / _awr_beta(config)
        score = jnp.minimum(score, config.rl.weight_clip)  # Clipping
        score = jnp.exp(score)
        score = score / config.rl.advantage_scale  # Normalize advantage w.r.t scale
        score = jnp.clip(score, min=1e-6)
    score = jax.lax.stop_gradient(score)  # Explicitly cut gradients

    assert isinstance(config.rl, AdvantageWeightedSFTLearnerConfig)
    # Read weight outside loss_fn so the `if` becomes a Python-level branch that
    # JAX sees as a compile-time constant — zero overhead when the feature is off.
    filtered_sft_weight = config.rl.filtered_sft_weight
    awr_loss_weight = config.rl.awr_loss_weight

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
        awr_loss = awr_loss_weight * jnp.mean(score * chunked_loss)
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
        "weight_mean": jnp.mean(score),
        "weight_max": jnp.max(score),
        # Effective sample size of the weights as a fraction of the batch. 1.0
        # means uniform (plain BC); near 1/b means the loss is an argmax.
        "weight_ess_frac": jnp.square(jnp.sum(score)) / (advantage.shape[0] * jnp.sum(jnp.square(score))),
    } | aux_data

    if group_stats is not None:
        num_groups = int(jnp.asarray(scale).shape[0])
        group_info = dict(group_stats)
        onehot = jax.nn.one_hot(task_id, num_groups)
        weight_sum = jnp.sum(onehot * score[:, jnp.newaxis], axis=0)
        # Share of the batch's total weight mass going to each task -- the
        # direct measure of one task crowding the others out of the gradient.
        group_info["group_weight_share"] = weight_sum / jnp.maximum(jnp.sum(score), 1e-12)
        group_info["group_weight_mean"] = weight_sum / jnp.maximum(group_info["group_count"], 1.0)
        group_info["group_batch_share"] = group_info["group_count"] / advantage.shape[0]
        info = info | group_info

    return new_state, info
