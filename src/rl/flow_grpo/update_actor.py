# ruff: noqa: F722
from src.training.config import OnlineTrainConfig, FlowGRPOSFTLearnerConfig
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


def _compute_rollout_log_probs(
    *,
    model: _model.BaseModel,
    policy_observation: _model.Observation,
    rollout_info: dict[str, at.Array],
    num_steps: int,
    noise_level: float,
) -> at.Array:
    if not hasattr(model, "get_dist_and_log_prob"):
        raise AttributeError(
            "Flow-GRPO requires a model with get_dist_and_log_prob()."
        )

    rollout_x = jax.lax.stop_gradient(rollout_info["x"])
    rollout_x_next = jax.lax.stop_gradient(rollout_info["x_next"])
    rollout_time = jax.lax.stop_gradient(rollout_info["time"])
    dt = jnp.asarray(-1.0 / max(num_steps, 1), dtype=rollout_x.dtype)

    def step_log_prob(
        x_t: at.Array,
        x_next: at.Array,
        time: at.Array,
    ) -> at.Array:
        time = jnp.broadcast_to(
            jnp.asarray(time, dtype=rollout_x.dtype), (x_t.shape[0],)
        )
        log_prob, _ = model.get_dist_and_log_prob(
            x_t=x_t,
            sample=x_next,
            time=time,
            observation=policy_observation,
            dt=dt,
            noise_level=noise_level,
        )
        return log_prob

    policy_log_probs = jax.vmap(step_log_prob, in_axes=(0, 0, 0))(
        rollout_x,
        rollout_x_next,
        rollout_time,
    )
    return jnp.moveaxis(policy_log_probs, 0, -1)


def _compute_reference_log_probs(
    *,
    model_def: nnx.GraphDef[_model.BaseModel],
    reference_params: nnx.State | None,
    policy_observation: _model.Observation,
    rollout_info: dict[str, at.Array],
    num_steps: int,
    noise_level: float,
) -> at.Array | None:
    """Compute reference log-probs outside the gradient tape."""
    if reference_params is None or noise_level <= 0.0:
        return None

    reference_model = nnx.merge(model_def, reference_params)
    reference_model.eval()
    if not hasattr(reference_model, "get_dist_and_log_prob"):
        raise AttributeError(
            "Flow-GRPO KL regularization requires a model with get_dist_and_log_prob()."
        )
    return _compute_rollout_log_probs(
        model=reference_model,
        policy_observation=policy_observation,
        rollout_info=rollout_info,
        num_steps=num_steps,
        noise_level=noise_level,
    )


@at.typecheck
def train_step(
    config: OnlineTrainConfig,
    rng: at.KeyArrayLike,
    policy_state: training_utils.TrainState,
    state_action_critic_state: training_utils.TrainState,
    value_state: training_utils.TrainState,
    batch: tuple[_model.Observation, ObsType, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    assert isinstance(config.rl, FlowGRPOSFTLearnerConfig)
    policy_observation, critic_observation, buffer_actions = batch

    policy_model = nnx.merge(policy_state.model_def, policy_state.params)
    policy_model.train()

    state_action_critic = create_critic(state_action_critic_state, config)
    state_action_critic.eval()

    value_critic = create_critic(value_state, config)
    value_critic.eval()

    assert isinstance(config.rl, FlowGRPOSFTLearnerConfig)
    group_size = config.rl.group_size
    normalize_adv = config.rl.normalize_adv
    use_mpo_advantage_weight = config.rl.use_mpo_advantage_weight
    weight_clip = config.rl.weight_clip
    beta = max(config.rl.beta, 1e-6)
    num_steps = config.rl.num_steps
    noise_level = config.rl.noise_level
    kl_coef = config.rl.kl_coef
    drop_low_diversity = config.rl.drop_low_diversity_groups
    diversity_threshold = config.rl.diversity_threshold
    sft_anchor_coef = config.rl.sft_anchor_coef
    min_advantage_std = config.rl.min_advantage_std

    def expand_and_flatten(x):
        return jnp.repeat(x, repeats=group_size, axis=0)

    expanded_policy_obs = jax.tree.map(expand_and_flatten, policy_observation)
    expanded_critic_obs = jax.tree.map(expand_and_flatten, critic_observation)

    train_rng = jax.random.fold_in(rng, policy_state.step)
    sample_rng, noise_rng = jax.random.split(train_rng)

    total_expanded = expanded_policy_obs.state.shape[0]
    noise = jax.random.normal(
        noise_rng,
        (
            total_expanded,
            policy_model.action_horizon,
            policy_model.action_dim,
        ),
    )

    # Deterministic anchor: zero initial noise for the first sample in each group.
    if config.rl.use_deterministic_anchor and group_size > 1:
        anchor_indices = jnp.arange(0, total_expanded, group_size)
        noise = noise.at[anchor_indices].set(0.0)

    sampled_actions, rollout_info = policy_model.sample_actions(
        rng=sample_rng,
        observation=expanded_policy_obs,
        noise=noise,
        num_steps=num_steps,
        noise_level=noise_level,
        return_info_dict=True,
    )

    value = summarize_critic_values(
        value_critic(expanded_critic_obs),
        critic_reduction=config.rl.critic_reduction,
    )
    q_value = summarize_critic_values(
        state_action_critic(
            expanded_critic_obs, flatten_action_horizon(sampled_actions)
        ),
        critic_reduction=config.rl.critic_reduction,
    )
    advantage = jax.lax.stop_gradient(q_value - value)

    # Compute reference log-probs OUTSIDE the gradient tape.
    # The reference model params are fixed, so no gradients needed.
    reference_log_probs = None
    if kl_coef > 0.0:
        reference_log_probs = _compute_reference_log_probs(
            model_def=policy_state.model_def,
            reference_params=policy_state.ema_params,
            policy_observation=expanded_policy_obs,
            rollout_info=rollout_info,
            num_steps=num_steps,
            noise_level=noise_level,
        )
    if reference_log_probs is not None:
        reference_log_probs = jax.lax.stop_gradient(reference_log_probs)

    # Global advantage gating: track whether advantage_std is too low.
    global_adv_std = jnp.std(advantage)
    skip_policy_update = global_adv_std < min_advantage_std

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        policy_observation: _model.Observation,
        rollout_info: dict[str, at.Array],
        advantage: at.Array,
        q_value: at.Array,
        reference_log_probs: at.Array | None,
    ) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:
        policy_log_probs = _compute_rollout_log_probs(
            model=model,
            policy_observation=policy_observation,
            rollout_info=rollout_info,
            num_steps=num_steps,
            noise_level=noise_level,
        )
        log_probs = policy_log_probs

        # KL against EMA reference, using pre-computed reference log-probs.
        kl_loss = jnp.asarray(0.0, dtype=policy_log_probs.dtype)
        ref_log_prob_mean = jnp.asarray(0.0, dtype=policy_log_probs.dtype)
        if kl_coef > 0.0 and reference_log_probs is not None:
            kl_loss = jnp.mean(policy_log_probs - reference_log_probs)
            ref_log_prob_mean = jnp.mean(reference_log_probs)

        advantage_scale = config.rl.advantage_scale

        if use_mpo_advantage_weight:
            if group_size > 1:
                total_batch_size = advantage.shape[0]
                assert (
                    total_batch_size % group_size == 0
                ), f"Batch/group mismatch: total_batch_size={total_batch_size}, group_size={config.rl.group_size}"
                B = total_batch_size // group_size
                score = advantage.reshape(B, group_size) / beta
                if weight_clip is not None:
                    score = jnp.minimum(score, weight_clip)
                score = jnp.exp(score)
                score = score / advantage_scale
                score = jnp.clip(score, min=1e-6)

                # Drop low-diversity groups: use uniform weights when
                # within-group advantage std is below the threshold.
                if drop_low_diversity:
                    group_adv = advantage.reshape(B, group_size)
                    group_std = jnp.std(group_adv, axis=-1, keepdims=True)
                    low_div = group_std < diversity_threshold
                    uniform = jnp.ones_like(score) / group_size
                    score = jnp.where(low_div, uniform, score)

                score = score[:, jnp.newaxis, :]
                log_probs = jnp.swapaxes(
                    log_probs.reshape(B, group_size, -1), 1, 2
                )
            else:
                # Flow-MPO path (group_size=1): global exp/scale.
                if normalize_adv:
                    advantage = (advantage - jnp.mean(advantage)) / (jnp.std(advantage) + 1e-6)
                score = advantage / beta
                if weight_clip is not None:
                    score = jnp.minimum(score, weight_clip)
                score = jnp.exp(score)
                score = score / advantage_scale
                score = jnp.clip(score, min=1e-6)
            score = jax.lax.stop_gradient(score)
        else:
            adv = advantage
            if normalize_adv and group_size > 1:
                total_batch_size = adv.shape[0]
                assert (
                    total_batch_size % group_size == 0
                ), f"Batch/group mismatch: total_batch_size={total_batch_size}, group_size={config.rl.group_size}"
                B = total_batch_size // group_size
                # Reshape to (B, G) for Group Relative calculations.
                # This works because 'repeat' groups copies together, and 'reshape' reads row-major.
                adv = jnp.swapaxes(adv.reshape(B, group_size, -1), 1, 2)
                log_probs = jnp.swapaxes(
                    log_probs.reshape(B, group_size, -1), 1, 2
                )
                group_mean = jnp.mean(adv, axis=-1, keepdims=True)
                group_std = jnp.std(adv, axis=-1, keepdims=True)
                adv = (adv - group_mean) / jnp.maximum(group_std, 1e-6)

                # Drop low-diversity groups: zero out normalized advantages
                # so these groups contribute no gradient.
                if drop_low_diversity:
                    low_div = group_std < diversity_threshold
                    adv = jnp.where(low_div, 0.0, adv)
            if weight_clip is not None:
                adv = jnp.clip(adv, -weight_clip, weight_clip)
            score = jax.lax.stop_gradient(adv)

        # Expand score (B*G,) to (B*G, 1, 1) to broadcast with log_probs (B*G, action_horizon, num_steps)
        if score.ndim == 1:
            score = score[:, jnp.newaxis, jnp.newaxis]

        flow_loss = -jnp.mean(score * log_probs)
        loss = (1.0 - sft_anchor_coef) * flow_loss + kl_coef * kl_loss

        info = {
            "flow_loss": flow_loss,
            "kl_loss": kl_loss,
            "kl_coef": jnp.asarray(kl_coef, dtype=policy_log_probs.dtype),
            "q_mean": jnp.mean(q_value),
            "advantage_std": global_adv_std,
            "score_mean": jnp.mean(score),
            "log_prob_mean": jnp.mean(policy_log_probs),
            "reference_log_prob_mean": ref_log_prob_mean,
            "policy_update_skipped": skip_policy_update.astype(jnp.float32),
        }

        return loss, info

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (flow_total_loss, flow_aux), flow_grads = nnx.value_and_grad(
        loss_fn, has_aux=True, argnums=diff_state
    )(
        policy_model,
        expanded_policy_obs,
        rollout_info,
        advantage,
        q_value,
        reference_log_probs,
    )

    # SFT anchor loss: computed in a SEPARATE gradient tape to avoid OOM.
    # Uses the original (non-expanded) batch — anchor is plain SFT, no grouping needed.
    anchor_loss_val = jnp.asarray(0.0)
    if sft_anchor_coef > 0.0:
        def anchor_loss_fn(
            model: _model.BaseModel,
            policy_observation: _model.Observation,
            actions: _model.Actions,
        ):
            anchor_rng = jax.random.fold_in(jax.random.PRNGKey(0), 1)
            loss = sft_anchor_coef * jnp.mean(model.compute_loss(
                anchor_rng, policy_observation, actions, train=True,
            ))
            return loss, {"anchor_loss": loss}

        anchor_diff_state = nnx.DiffState(0, config.trainable_filter)
        (anchor_loss_val, _anchor_aux), anchor_grads = nnx.value_and_grad(
            anchor_loss_fn, has_aux=True, argnums=anchor_diff_state,
        )(policy_model, policy_observation, buffer_actions)

        # Combine gradients: flow_grads + anchor_grads.
        flow_grads = jax.tree.map(lambda f, a: f + a, flow_grads, anchor_grads)

    grads = flow_grads

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

    # Merge info from both losses.
    loss = flow_total_loss + anchor_loss_val
    aux_data = flow_aux | {
        "loss": loss,
        "policy_loss": loss,
        "anchor_loss": anchor_loss_val,
    }

    params = nnx.filter_state(policy_state.params, config.trainable_filter)
    updates, new_opt_state = policy_state.tx.update(
        grads, policy_state.opt_state, params
    )
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(policy_model, new_params)
    new_params = nnx.state(policy_model)

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
            new_state = jax.lax.cond(
                new_state.step % reset_period == 0,
                lambda s: s.replace(params=jax.tree.map(lambda x: x, s.ema_params)),
                lambda s: s,
                new_state,
            )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        policy_model,
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
