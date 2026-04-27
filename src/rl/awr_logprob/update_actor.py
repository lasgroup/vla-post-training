# ruff: noqa: F722
"""AWR actor train_step where the FM regression loss is swapped for a clipped
log-prob ratio against actions sampled from the EMA policy.

Loss (per (batch, horizon, flow-step)):
    weight = exp(A(s, a_sampled) / beta / scale)         # AWR weighting
    weight = clip-then-exp-then-divide(advantage_scale)  # AWR normalisation
    ratio  = exp(log pi_new(step) - log pi_ema(step))    # per flow-step ratio
    loss   = -E[ clip(ratio, 1-eps, 1+eps) * weight ]
             (or PPO's `min(r*w, clip(r)*w)` when use_pessimistic_clip=True)

Note: the ratio is taken elementwise -> more like token-level PPO over the SDE chain.

actions are sampled from EMA, not buffer (flow log-likelihoods aren't computable for arbitrary actions)
the actor gradient flows through per-step log-probs instead of the FM loss.
"""
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
from src.training.config import AWRLogProbLearnerConfig, OnlineTrainConfig


def _compute_rollout_log_probs(
    *,
    model: _model.BaseModel,
    policy_observation: _model.Observation,
    rollout_info: dict[str, at.Array],
    num_steps: int,
    noise_level: float,
) -> at.Array:
    """Per-flow-step log-probs for each (batch, action-dim).

    Returns shape [batch, action_horizon, num_flow_steps]. Each entry is the
    log-density of one denoising-step transition under model. 
    """
    rollout_x = jax.lax.stop_gradient(rollout_info["x"])
    rollout_x_next = jax.lax.stop_gradient(rollout_info["x_next"])
    rollout_time = jax.lax.stop_gradient(rollout_info["time"])
    dt = jnp.asarray(-1.0 / max(num_steps, 1), dtype=rollout_x.dtype)

    def step_log_prob(x_t, x_next, time):
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
        rollout_x, rollout_x_next, rollout_time,
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
    scale: at.Array | float = 1.0,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    assert isinstance(config.rl, AWRLogProbLearnerConfig)
    rl_config = config.rl
    policy_observation, critic_observation, _buffer_actions = batch  # buffer actions unused

    reset_period = rl_config.reset_policy_params_to_ema_period
    normalizer_config = rl_config.normalizer_config
    beta = max(rl_config.beta, 1e-6)
    weight_clip = rl_config.weight_clip
    num_steps = rl_config.num_steps
    noise_level = rl_config.noise_level
    use_ema_for_sampling = rl_config.use_ema_for_sampling
    clip_epsilon = rl_config.clip_epsilon
    use_pessimistic_clip = rl_config.use_pessimistic_clip

    policy_model = nnx.merge(policy_state.model_def, policy_state.params)
    policy_model.train()

    # Sampling model: EMA when available (matches flow_pg / flow_mpo).
    if use_ema_for_sampling and policy_state.ema_params is not None:
        sampling_model = nnx.merge(policy_state.model_def, policy_state.ema_params)
    else:
        sampling_model = nnx.merge(policy_state.model_def, policy_state.params)
    sampling_model.eval()

    state_action_critic = create_critic(state_action_critic_state, config)
    state_action_critic.eval()
    value_critic = create_critic(value_state, config)
    value_critic.eval()

    train_rng = jax.random.fold_in(rng, policy_state.step)
    sample_rng, noise_rng = jax.random.split(train_rng)

    batch_size = policy_observation.state.shape[0]
    noise = jax.random.normal(
        noise_rng,
        (batch_size, sampling_model.action_horizon, sampling_model.action_dim),
    )
    sampled_actions, rollout_info = sampling_model.sample_actions(
        rng=sample_rng,
        observation=policy_observation,
        noise=noise,
        num_steps=num_steps,
        noise_level=noise_level,
        return_info_dict=True,
    )
    rollout_info = jax.tree.map(jax.lax.stop_gradient, rollout_info)
    # [num_steps, batch, horizon] -> [batch, horizon, num_steps]
    old_log_probs = jax.lax.stop_gradient(
        jnp.moveaxis(rollout_info["log_prob"], 0, -1)
    )

    # --- AWR-style advantage weighting ---
    # `mc_return` is accepted to keep AWR's JIT signature but is unused:
    # advantages are always Q(s, a_sampled) - V(s) here. 
    # mc_return path is not meaningful when we use EMA-sampled action whose MC trajectory we don't have. 
    # `use_mc_returns=False` (the default) for this learner.
    del mc_return

    if rl_config.use_mc_returns:
        raise ValueError(
            "AWRLogProbLearner is incompatible with use_mc_returns=True. "
            "Use AdvantageWeightedSFTLearner if you want the MC-return path."
        )

    value = summarize_critic_values(
        value_critic(critic_observation),
        critic_reduction=rl_config.critic_reduction,
    )
    q_value = summarize_critic_values(
        state_action_critic(critic_observation, flatten_action_horizon(sampled_actions)),
        critic_reduction=rl_config.critic_reduction,
    )
    advantage = q_value - value  # (B,)

    # Mirror advantage_weighted_sft.update_actor exactly.
    score = advantage / scale
    score = score / beta
    score = jnp.minimum(score, weight_clip)             # clip in log space
    score = jnp.exp(score)
    score = score / rl_config.advantage_scale           # post-exp constant
    score = jnp.clip(score, min=1e-6)
    score = jax.lax.stop_gradient(score)                # gradients only through ratio

    # Broadcast (B,) -> (B, 1, 1) so it lines up with [B, horizon, num_steps].
    if score.ndim == 1:
        score = score[:, jnp.newaxis, jnp.newaxis]

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        policy_observation: _model.Observation,
    ) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:
        current_log_probs = _compute_rollout_log_probs(
            model=model,
            policy_observation=policy_observation,
            rollout_info=rollout_info,
            num_steps=num_steps,
            noise_level=noise_level,
        )

        # r = pi_new / pi_ema (per (batch, horizon, step), same conv as flow_mpo)
        log_ratio = current_log_probs - old_log_probs
        ratio = jnp.exp(log_ratio)

        # Surrogate. score >= 0 always.
        # use_pessimistic_clip=False (default): one-sided `clip(r) * w`. When
        #   ratio drifts to < 1-eps the gradient is zero -- we don't chase
        #   actions the new policy already disagrees with.
        # use_pessimistic_clip=True: standard PPO `min(r*w, clip(r)*w)`. Keeps
        #   a recovery gradient when ratio < 1-eps; safer if the policy can
        #   over-shoot away from high-advantage actions.
        clipped_ratio = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        if use_pessimistic_clip:
            loss = -jnp.mean(jnp.minimum(ratio * score, clipped_ratio * score))
        else:
            loss = -jnp.mean(clipped_ratio * score)

        info = {
            "pg_loss": loss,
            "log_prob_mean": jnp.mean(current_log_probs),
            "old_log_prob_mean": jnp.mean(old_log_probs),
            "ratio_mean": jnp.mean(ratio),
            "ratio_clipped_frac": jnp.mean(
                (ratio < 1.0 - clip_epsilon) | (ratio > 1.0 + clip_epsilon)
            ),
            "q_mean": jnp.mean(q_value),
            "v_mean": jnp.mean(value),
            "score_mean": jnp.mean(score),
        }
        return loss, info

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, aux_data), grads = nnx.value_and_grad(
        loss_fn, has_aux=True, argnums=diff_state
    )(policy_model, policy_observation)

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
        if reset_period:
            step = new_state.step

            def keep_state(state):
                return state

            def revert_to_ema(state):
                return state.replace(params=jax.tree.map(lambda x: x, state.ema_params))

            new_state = jax.lax.cond(
                step % reset_period == 0, revert_to_ema, keep_state, new_state
            )

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
        "advantage_mean": jnp.mean(advantage),
        "advantage_max": jnp.max(advantage),
        "advantage_min": jnp.min(advantage),
        "advantage_std": jnp.std(advantage),
        # Same keys as AWR so the existing normalizer host-update path works
        # without changes (advantage_weighted_sft_learner.update reads these).
        "advantage_q_up": jnp.quantile(advantage, normalizer_config.q_up),
        "advantage_median": jnp.median(advantage),
        "advantage_q_low": jnp.quantile(advantage, normalizer_config.q_low),
    } | aux_data
    return new_state, info
