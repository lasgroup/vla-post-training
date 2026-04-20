# ruff: noqa: F722
from src.training.config import OnlineTrainConfig, FlowMPOSFTLearnerConfig, NormalizerState
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

# calc. log-probs of a previous trajectory under new model
def _compute_rollout_log_probs(
    *,
    model: _model.BaseModel,
    policy_observation: _model.Observation,
    rollout_info: dict[str, at.Array],
    num_steps: int,
    noise_level: float,
) -> at.Array:
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

    # [num_steps, batch, horizon] -> [batch, horizon, num_steps]
    return jnp.moveaxis(policy_log_probs, 0, -1)
    

@at.typecheck
def train_step(
    config: OnlineTrainConfig,
    rng: at.KeyArrayLike,
    policy_state: training_utils.TrainState,
    state_action_critic_state: training_utils.TrainState,
    value_state: training_utils.TrainState,
    batch: tuple[_model.Observation, ObsType, _model.Actions],
    mc_return: at.Array | None = None,
    normalizer_state: NormalizerState | None = None,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    assert isinstance(config.rl, FlowMPOSFTLearnerConfig)
    policy_observation, critic_observation, actions = batch

    reset_period = config.rl.reset_policy_params_to_ema_period
    group_size = config.rl.group_size # should be one, mpo does global batch weighting not group relative
    assert group_size == 1
    
    weight_clip = config.rl.weight_clip
    beta = max(config.rl.beta, 1e-6)
    num_steps = config.rl.num_steps
    noise_level = config.rl.noise_level
    use_ema_for_sampling = config.rl.use_ema_for_sampling
    clip_epsilon = config.rl.clip_epsilon # this is for ratio clipping 

    #group_size = 1

    # build two models one for sampling, one ema
    # current policy for gradient computatioms
    policy_model = nnx.merge(policy_state.model_def, policy_state.params)
    policy_model.train()
    # use ema one if it is enabled with a flag and fall back to current one if not avlb
    if use_ema_for_sampling and policy_state.ema_params is not None:
        sampling_model = nnx.merge(policy_state.model_def, policy_state.ema_params)
        sampling_model.eval()
    else:
        sampling_model = nnx.merge(policy_state.model_def, policy_state.params)
        sampling_model.eval()

    state_action_critic = create_critic(state_action_critic_state, config)
    state_action_critic.eval()

    value_critic = create_critic(value_state, config)
    value_critic.eval()

    # define batch extension out of loss function
    def expand_and_flatten(x):
        return jnp.repeat(x, repeats=group_size, axis=0)

    expanded_policy_obs = jax.tree.map(expand_and_flatten, policy_observation)
    expanded_critic_obs = jax.tree.map(expand_and_flatten, critic_observation)

    train_rng = jax.random.fold_in(rng, policy_state.step)
    step_rng, noise_rng = jax.random.split(train_rng)

    # sample actions from ema
    noise = jax.random.normal(
        noise_rng,
        (
            expanded_policy_obs.state.shape[0],
            sampling_model.action_horizon,
            sampling_model.action_dim,
        ),
    )
    sampled_actions, rollout_info = sampling_model.sample_actions(
        rng=step_rng,
        observation=expanded_policy_obs,
        noise=noise,
        num_steps=num_steps,
        noise_level=noise_level,
        return_info_dict=True,
    )

    # log-probs under ema/old policy (shape [B*G, action_horizon, num_steps])
    old_log_probs = jnp.moveaxis(rollout_info["log_prob"], 0, -1)
    old_log_probs = jax.lax.stop_gradient(old_log_probs)

    # fix trajectory so the current policy is evaluated correctly
    rollout_info = jax.tree.map(jax.lax.stop_gradient, rollout_info)

    value = summarize_critic_values(
        value_critic(expanded_critic_obs),
        critic_reduction=config.rl.critic_reduction,
    )
    q_value = summarize_critic_values(
        state_action_critic(expanded_critic_obs, flatten_action_horizon(sampled_actions)),
        critic_reduction=config.rl.critic_reduction,
    )
    advantage = q_value - value

    # Use EMA-smoothed scale from the learner's NormalizerState to rescale
    # advantages before the exponential weighting.
    #
    # Rationale (DreamerV3, Hafner et al. 2023, Eq. 6-7):
    #   S = EMA(Per(R, q_up) - Per(R, q_low), decay)
    #   normalized = (R - V(s)) / max(min_scale, S)
    # We apply the same trick to A = Q(s,a) - V(s). Note we divide by scale only
    # and do NOT subtract bias: Dreamer's REINFORCE gradient is invariant to a
    # constant offset, but our exp(A/beta) weighting is not - subtracting a
    # bias would rescale all weights by exp(-bias/beta), which would push weights
    # uniformly above 1 and prevent downweighting of bad actions.
    if config.rl.use_adaptive_advantage_scale and normalizer_state is not None:
        # Cast to advantage dtype (e.g. bfloat16) to avoid silent dtype promotion.
        advantage_scale = normalizer_state.scale.astype(advantage.dtype)
        # Also clip at use time (DreamerV3 Eq. 6 applies max at use, not at EMA).
        # This is a safety net in case min_scale was set < 1.0 in NormalizerConfig.
        min_scale = jnp.asarray(
            config.rl.normalizer_config.min_scale, dtype=advantage.dtype
        )
        advantage_scale = jnp.maximum(min_scale, advantage_scale)
        normalized_advantage = advantage / advantage_scale
    else:
        # Legacy path: fixed divisor from config, applied AFTER exp().
        advantage_scale = jnp.asarray(config.rl.advantage_scale, dtype=advantage.dtype)
        normalized_advantage = advantage

    score = normalized_advantage / beta
    score = jnp.minimum(score, weight_clip)
    score = jnp.exp(score)
    if not config.rl.use_adaptive_advantage_scale:
        score = score / advantage_scale
    score = jnp.clip(score, min=1e-6)
    score = jax.lax.stop_gradient(score)

    if score.ndim == 1:
        score = score[:, jnp.newaxis, jnp.newaxis]

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        policy_observation: _model.Observation,
        critic_observation: ObsType,
        state_action_critic: nnx.Module,
        value_critic: nnx.Module,
    ) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:
         # evaluate current policy log-probs on the previously ema-trajectory.
        current_log_probs = _compute_rollout_log_probs(
            model=model,
            policy_observation=expanded_policy_obs,
            rollout_info=rollout_info,
            num_steps=num_steps,
            noise_level=noise_level,
        )

        # r_t(theta) = pi_theta / pi_theta_old  -> logr_t = log_pi_theta - log_pi_theta_old
        log_ratio = current_log_probs - old_log_probs
        ratio = jnp.exp(log_ratio) # ratio = exp(logr) = r

        # clip it with epsilon min(r * A, clip(r, 1-eps, 1+eps) * A)
        clipped_ratio = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        surr1 = ratio * score
        surr2 = clipped_ratio * score
        loss = -jnp.mean(jnp.minimum(surr1, surr2))
        
        info = {
            "loss": loss,
            "q_mean": jnp.mean(q_value),
            "v_mean": jnp.mean(value),
            "advantage_scale_used": advantage_scale,
            "score_mean": jnp.mean(score),
            "log_prob_mean": jnp.mean(current_log_probs),
            "old_log_prob_mean": jnp.mean(old_log_probs),
            "ratio_mean": jnp.mean(ratio),
            "ratio_clipped_frac": jnp.mean(
                (ratio < 1.0 - clip_epsilon) | (ratio > 1.0 + clip_epsilon)
            ),
        }

        return loss, info

    train_rng = jax.random.fold_in(rng, policy_state.step)

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, aux_data), grads = nnx.value_and_grad(
        loss_fn, has_aux=True, argnums=diff_state
    )(
        policy_model,
        train_rng,
        policy_observation,
        critic_observation,
        state_action_critic,
        value_critic,
    )

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
        policy_model,
        nnx.All(
            nnx.Param,
            nnx.Not(
                nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")
            ),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    # Advantage statistics for the host-side NormalizerState update.
    # These match the keys used in advantage_weighted_sft_learner so the same
    # normalizer code path can be reused across learners.
    normalizer_config = config.rl.normalizer_config
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        "advantage_mean": jnp.mean(advantage),
        "advantage_std": jnp.std(advantage),
        "advantage_max": jnp.max(advantage),
        "advantage_min": jnp.min(advantage),
        "advantage_median": jnp.median(advantage),
        "advantage_q_up": jnp.quantile(advantage, normalizer_config.q_up),
        "advantage_q_low": jnp.quantile(advantage, normalizer_config.q_low),
    } | aux_data
    return new_state, info