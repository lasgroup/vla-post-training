# ruff: noqa: F722
import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.utils as training_utils
from src.rl.advantage_weighted_sft.update_critic import (
    create_critic,
    flatten_action_horizon,
    summarize_critic_values,
)
from src.rl.networks.rl_networks import ObsType
from src.training.config import GroupedMPOWeightedSFTLearnerConfig, OnlineTrainConfig


@at.typecheck
def train_step(
    config: OnlineTrainConfig,
    rng: at.KeyArrayLike,
    policy_state: training_utils.TrainState,
    state_action_critic_state: training_utils.TrainState,
    value_state: training_utils.TrainState,
    batch: tuple[_model.Observation, ObsType, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    assert isinstance(config.rl, GroupedMPOWeightedSFTLearnerConfig)
    policy_observation, critic_observation, buffer_actions = batch

    policy = nnx.merge(policy_state.model_def, policy_state.params)
    policy.train()

    state_action_critic = create_critic(state_action_critic_state, config)
    state_action_critic.eval()

    value_critic = create_critic(value_state, config)
    value_critic.eval()

    group_size = max(config.rl.group_size, 1)
    beta = max(config.rl.beta, 1e-6)
    weight_clip = config.rl.weight_clip
    num_steps = config.rl.num_steps
    noise_level = config.rl.noise_level
    sft_anchor_coef = config.rl.sft_anchor_coef
    min_advantage_std = config.rl.min_advantage_std
    use_buffer_actions_for_loss = config.rl.use_buffer_actions_for_loss

    def expand_and_flatten(x):
        return jnp.repeat(x, repeats=group_size, axis=0)

    expanded_policy_obs = jax.tree.map(expand_and_flatten, policy_observation)
    expanded_critic_obs = jax.tree.map(expand_and_flatten, critic_observation)

    train_rng = jax.random.fold_in(rng, policy_state.step)
    sample_rng, loss_rng, noise_rng = jax.random.split(train_rng, 3)

    total_expanded = expanded_policy_obs.state.shape[0]
    noise = jax.random.normal(
        noise_rng,
        (
            total_expanded,
            policy.action_horizon,
            policy.action_dim,
        ),
    )

    # Deterministic anchor: zero initial noise for the first sample in each group.
    if config.rl.use_deterministic_anchor and group_size > 1:
        anchor_indices = jnp.arange(0, total_expanded, group_size)
        noise = noise.at[anchor_indices].set(0.0)

    sampled_actions = policy.sample_actions(
        rng=sample_rng,
        observation=expanded_policy_obs,
        noise=noise,
        num_steps=num_steps,
        noise_level=noise_level,
        return_info_dict=False,
        return_prefix_rep=False,
    )

    critic_actions = flatten_action_horizon(sampled_actions)
    value = summarize_critic_values(
        value_critic(expanded_critic_obs),
        critic_reduction=config.rl.critic_reduction,
    )
    q_value = summarize_critic_values(
        state_action_critic(expanded_critic_obs, critic_actions),
        critic_reduction=config.rl.critic_reduction,
    )
    advantage = jax.lax.stop_gradient(q_value - value)

    advantage_scale = config.rl.advantage_scale

    if group_size > 1:
        total_batch_size = advantage.shape[0]
        assert (
            total_batch_size % group_size == 0
        ), f"Batch/group mismatch: total_batch_size={total_batch_size}, group_size={group_size}"
        base_batch_size = total_batch_size // group_size
        score = advantage.reshape(base_batch_size, group_size) / beta
        if weight_clip is not None:
            score = jnp.minimum(score, weight_clip)
        score = jnp.exp(score)
        score = score / advantage_scale
        score = jnp.clip(score, min=1e-6)

        # Drop low-diversity groups: replace with uniform weights when
        # the within-group advantage std is below the threshold.
        if config.rl.drop_low_diversity_groups:
            group_adv = advantage.reshape(base_batch_size, group_size)
            group_std = jnp.std(group_adv, axis=-1, keepdims=True)
            low_div = group_std < config.rl.diversity_threshold
            uniform = jnp.ones_like(score) / group_size
            score = jnp.where(low_div, uniform, score)

        score_stats = score
        score = jax.lax.stop_gradient(score[..., jnp.newaxis])
    else:
        score = advantage / beta
        if weight_clip is not None:
            score = jnp.minimum(score, weight_clip)
        score = jnp.exp(score)
        score = score / advantage_scale
        score = jnp.clip(score, min=1e-6)
        score_stats = score
        score = jax.lax.stop_gradient(score)

    # Global advantage gating: compute std of GROUP MEANS to filter out
    # intra-group noise from the stochastic sampling. This measures whether
    # the critic can truly distinguish between different states.
    if group_size > 1:
        group_means = jnp.mean(advantage.reshape(base_batch_size, group_size), axis=-1)
        global_adv_std = jnp.std(group_means)
    else:
        global_adv_std = jnp.std(advantage)
    skip_policy_update = global_adv_std < min_advantage_std

    # Capture original (non-expanded) batch for SFT anchor.
    _anchor_obs = policy_observation
    _anchor_actions = buffer_actions

    # When use_buffer_actions_for_loss=True, compute a state-level score from
    # group-sampled advantages, but train the policy on buffer actions.
    # This breaks the self-reinforcing loop where the policy trains on its own
    # outputs while keeping the group-sampling benefit for advantage estimation.
    if use_buffer_actions_for_loss and group_size > 1:
        state_score = jnp.mean(score_stats, axis=-1)  # (B,) mean score per state
        state_score = jax.lax.stop_gradient(state_score)

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        policy_observation: _model.Observation,
        actions: _model.Actions,
        score: at.Array,
    ) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:
        if use_buffer_actions_for_loss and group_size > 1:
            # Grounded mode: train on buffer actions weighted by state-level advantage.
            chunked_loss = model.compute_loss(
                rng, _anchor_obs, _anchor_actions, train=True,
            )
            s = state_score
            while s.ndim < chunked_loss.ndim:
                s = s[..., jnp.newaxis]
            grouped_loss = jnp.mean(s * chunked_loss)
        else:
            # Default mode: train on sampled actions with per-sample scores.
            chunked_loss = model.compute_loss(
                rng, policy_observation, actions, train=True,
            )
            if group_size > 1:
                grouped_chunked_loss = chunked_loss.reshape(base_batch_size, group_size, -1)
                grouped_loss = jnp.mean(score * grouped_chunked_loss)
            else:
                s = score
                while s.ndim < chunked_loss.ndim:
                    s = s[..., jnp.newaxis]
                grouped_loss = jnp.mean(s * chunked_loss)

        # SFT anchor: computed on the original (non-expanded) batch inside the
        # same gradient tape so that gradients are properly combined.
        anchor_loss = jnp.asarray(0.0, dtype=grouped_loss.dtype)
        if sft_anchor_coef > 0.0:
            anchor_rng = jax.random.fold_in(jax.random.PRNGKey(0), 1)
            anchor_loss = jnp.mean(model.compute_loss(
                anchor_rng, _anchor_obs, _anchor_actions, train=True,
            ))

        loss = (1.0 - sft_anchor_coef) * grouped_loss + sft_anchor_coef * anchor_loss

        info = {
            "grouped_loss": grouped_loss,
            "anchor_loss": anchor_loss,
            "q_mean": jnp.mean(q_value),
            "value_mean": jnp.mean(value),
            "advantage_mean": jnp.mean(advantage),
            "advantage_max": jnp.max(advantage),
            "advantage_min": jnp.min(advantage),
            "advantage_std": global_adv_std,
            "score_mean": jnp.mean(score_stats),
            "score_max": jnp.max(score_stats),
            "policy_update_skipped": skip_policy_update.astype(jnp.float32),
        }
        return loss, info

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, aux_data), grads = nnx.value_and_grad(
        loss_fn, has_aux=True, argnums=diff_state
    )(
        policy,
        loss_rng,
        expanded_policy_obs,
        sampled_actions,
        score,
    )

    aux_data = aux_data | {"loss": loss}

    # Global advantage gating: zero out gradients when advantage_std is too low.
    # Guard with Python-level check to avoid tracing both cond branches over the
    # entire gradient tree when gating is disabled (min_advantage_std == 0).
    if min_advantage_std > 0.0:
        grads = jax.lax.cond(
            skip_policy_update,
            lambda g: jax.tree.map(jnp.zeros_like, g),
            lambda g: g,
            grads,
        )

    params = nnx.filter_state(policy_state.params, config.trainable_filter)
    updates, new_opt_state = policy_state.tx.update(
        grads, policy_state.opt_state, params
    )
    new_params = optax.apply_updates(params, updates)

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
        reset_period = config.rl.reset_policy_params_to_ema_period
        if reset_period is not None:
            def _ema_reset(s):
                s = s.replace(params=jax.tree.map(lambda x: x, s.ema_params))
                if config.rl.reset_optimizer_on_ema_reset:
                    s = s.replace(opt_state=jax.tree.map(jnp.zeros_like, s.opt_state))
                return s
            new_state = jax.lax.cond(
                new_state.step % reset_period == 0,
                _ema_reset,
                lambda s: s,
                new_state,
            )

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
    } | aux_data
    return new_state, info
