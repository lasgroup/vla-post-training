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
    policy_model.eval()  # eval mode for sampling; switched to train() before gradient tape

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
    use_buffer_actions_for_loss = config.rl.use_buffer_actions_for_loss

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

    # Advantage gating: for grouped methods, measure WITHIN-GROUP spread (can the
    # critic rank candidate actions for the same state?).  For non-grouped, use
    # global std.  Computed on RAW advantages.
    if group_size > 1:
        _B = advantage.shape[0] // group_size
        within_group_stds = jnp.std(advantage.reshape(_B, group_size), axis=-1)
        global_adv_std = jnp.mean(within_group_stds)
    else:
        global_adv_std = jnp.std(advantage)
    skip_policy_update = global_adv_std < min_advantage_std

    # Capture original (non-expanded) batch for SFT anchor.
    _anchor_obs = policy_observation
    _anchor_actions = buffer_actions

    # --- Score computation (OUTSIDE gradient tape) ---
    # All derived from stop_gradient'd advantage — no gradients needed.
    advantage_scale = config.rl.advantage_scale
    _needs_group_reshape = False

    if use_mpo_advantage_weight:
        if group_size > 1:
            _B = advantage.shape[0] // group_size
            if normalize_adv:
                _adv = (advantage - jnp.mean(advantage)) / (jnp.std(advantage) + 1e-6)
            else:
                _adv = advantage
            score = _adv.reshape(_B, group_size) / beta
            if weight_clip is not None:
                score = jnp.minimum(score, weight_clip)
            score = jnp.exp(score)
            # Normalize within each group so weights sum to 1 per state.
            score = score / jnp.maximum(jnp.sum(score, axis=-1, keepdims=True), 1e-8)

            if drop_low_diversity:
                group_adv = advantage.reshape(_B, group_size)
                group_std = jnp.std(group_adv, axis=-1, keepdims=True)
                low_div = group_std < diversity_threshold
                uniform = jnp.ones_like(score) / group_size
                score = jnp.where(low_div, uniform, score)

            score = score[:, jnp.newaxis, :]  # (B, 1, G)
            _needs_group_reshape = True
        else:
            # Flow-MPO path (group_size=1): global exp/scale.
            if normalize_adv:
                _adv = (advantage - jnp.mean(advantage)) / (jnp.std(advantage) + 1e-6)
            else:
                _adv = advantage
            score = _adv / beta
            if weight_clip is not None:
                score = jnp.minimum(score, weight_clip)
            score = jnp.exp(score)
            score = score / advantage_scale
            score = jnp.clip(score, min=1e-6)
    else:
        # GRPO path
        _adv = advantage
        if normalize_adv and group_size > 1:
            _B = advantage.shape[0] // group_size
            _adv = _adv.reshape(_B, group_size)
            group_mean = jnp.mean(_adv, axis=-1, keepdims=True)
            group_std = jnp.std(_adv, axis=-1, keepdims=True)
            _adv = (_adv - group_mean) / jnp.maximum(group_std, 1e-6)

            if drop_low_diversity:
                low_div = group_std < diversity_threshold
                _adv = jnp.where(low_div, 0.0, _adv)

            score = _adv[:, jnp.newaxis, :]  # (B, 1, G)
            _needs_group_reshape = True
        else:
            score = _adv
        if weight_clip is not None:
            score = jnp.clip(score, -weight_clip, weight_clip)

    score = jax.lax.stop_gradient(score)

    
    if score.ndim == 1:
        score = score[:, jnp.newaxis, jnp.newaxis]

    if use_buffer_actions_for_loss:
        if group_size > 1:
            _B_buf = advantage.shape[0] // group_size
            raw_group_mean_adv = jnp.mean(advantage.reshape(_B_buf, group_size), axis=-1)  # (B,)
            if normalize_adv:
                raw_group_mean_adv = (raw_group_mean_adv - jnp.mean(raw_group_mean_adv)) / (jnp.std(raw_group_mean_adv) + 1e-6)
            state_score = raw_group_mean_adv / beta
            if weight_clip is not None:
                state_score = jnp.minimum(state_score, weight_clip)
            state_score = jnp.exp(state_score)
            state_score = state_score / advantage_scale
            state_score = jnp.clip(state_score, min=1e-6)
        else:
            state_score = score.reshape(-1)  # (B,)
        state_score = jax.lax.stop_gradient(state_score)

    score_mean_for_log = jnp.mean(score)

    # --- loss_fn: only things that need gradients ---
    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        policy_observation: _model.Observation,
        rollout_info: dict[str, at.Array],
        reference_log_probs: at.Array | None,
    ) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:
        policy_log_probs = _compute_rollout_log_probs(
            model=model,
            policy_observation=policy_observation,
            rollout_info=rollout_info,
            num_steps=num_steps,
            noise_level=noise_level,
        )

        # KL against EMA reference.
        kl_loss = jnp.asarray(0.0, dtype=policy_log_probs.dtype)
        ref_log_prob_mean = jnp.asarray(0.0, dtype=policy_log_probs.dtype)
        if kl_coef > 0.0 and reference_log_probs is not None:
            kl_loss = jnp.mean(policy_log_probs - reference_log_probs)
            ref_log_prob_mean = jnp.mean(reference_log_probs)

        if use_buffer_actions_for_loss:
            anchor_rng = jax.random.fold_in(jax.random.PRNGKey(0), 1)
            buffer_loss = model.compute_loss(
                anchor_rng, _anchor_obs, _anchor_actions, train=True,
            )
            _ss = state_score
            while _ss.ndim < buffer_loss.ndim:
                _ss = _ss[..., jnp.newaxis]
            flow_loss = jnp.mean(_ss * buffer_loss)
        else:
            log_probs = policy_log_probs
            if _needs_group_reshape:
                _B = log_probs.shape[0] // group_size
                log_probs = jnp.swapaxes(
                    log_probs.reshape(_B, group_size, -1), 1, 2
                )  # (B, AH*NS, G)
            flow_loss = -jnp.mean(score * log_probs)

        # SFT anchor.
        anchor_loss = jnp.asarray(0.0, dtype=policy_log_probs.dtype)
        if sft_anchor_coef > 0.0:
            anchor_rng = jax.random.fold_in(jax.random.PRNGKey(0), 1)
            anchor_loss = jnp.mean(model.compute_loss(
                anchor_rng, _anchor_obs, _anchor_actions, train=True,
            ))

        loss = (1.0 - sft_anchor_coef) * flow_loss + sft_anchor_coef * anchor_loss + kl_coef * kl_loss

        info = {
            "flow_loss": flow_loss,
            "anchor_loss": anchor_loss,
            "kl_loss": kl_loss,
            "kl_coef": jnp.asarray(kl_coef, dtype=policy_log_probs.dtype),
            "q_mean": jnp.mean(q_value),
            "advantage_std": global_adv_std,
            "score_mean": score_mean_for_log,
            "log_prob_mean": jnp.mean(policy_log_probs),
            "reference_log_prob_mean": ref_log_prob_mean,
            "policy_update_skipped": skip_policy_update.astype(jnp.float32),
        }

        return loss, info

    policy_model.train()  # switch to train mode for gradient computation
    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, aux_data), grads = nnx.value_and_grad(
        loss_fn, has_aux=True, argnums=diff_state
    )(
        policy_model,
        expanded_policy_obs,
        rollout_info,
        reference_log_probs,
    )

    # zero out gradients when advantage_std is too low, we can set a threshold and use it instead of 0.0 later
    if min_advantage_std > 0.0:
        grads = jax.lax.cond(
            skip_policy_update,
            lambda g: jax.tree.map(jnp.zeros_like, g),
            lambda g: g,
            grads,
        )

    aux_data = aux_data | {
        "loss": loss,
        "policy_loss": loss,
    }

    params = nnx.filter_state(policy_state.params, config.trainable_filter)
    updates, new_opt_state = policy_state.tx.update(
        grads, policy_state.opt_state, params
    )
    new_params = optax.apply_updates(params, updates)
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
