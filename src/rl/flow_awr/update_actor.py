# ruff: noqa: F722
"""FlowAWR actor train_step.

score = exp((Q(s, a_buf) - V(s)) / beta / scale)
FM loss is replaced by a per-flow-step PPO surrogate of log p_new(rollout)/log p_old(rollout)
  ratio_t = exp(log p_new(x_next^t | x_t, time, s) - log p_collect(x_next^t | x_t, time, s))
  loss   = -E[ min(ratio * score, clip(ratio, 1-eps, 1+eps) * score) ]

both the action and the noise trajectory come from the replay buffer.
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
from src.training.config import FlowAWRSFTLearnerConfig, OnlineTrainConfig


def _compute_rollout_log_probs(
    *,
    model: _model.BaseModel,
    policy_observation: _model.Observation,
    rollout_info: dict[str, at.Array],
    num_steps: int,
    noise_level: float,
) -> at.Array:
    """Per-flow-step log-probs along a stored trajectory under `model`.

    `rollout_info` shape conventions (matching what is stored by the buffer):
        x        : (B, num_steps, H, A)
        x_next   : (B, num_steps, H, A)
        time     : (B, num_steps)
    Returns log-probs of shape `(B, H, num_steps)`. The leading flow-step axis
    is moved to the back so it lines up with the per-(batch, horizon, step)
    PPO ratio used by flow_mpo / flow_pg / awr_logprob.
    """
    # Move the flow-step axis to the front so we can vmap over it like
    # flow_mpo / flow_pg / awr_logprob do.
    rollout_x = jnp.swapaxes(jax.lax.stop_gradient(rollout_info["x"]), 0, 1)
    rollout_x_next = jnp.swapaxes(jax.lax.stop_gradient(rollout_info["x_next"]), 0, 1)
    rollout_time = jnp.swapaxes(jax.lax.stop_gradient(rollout_info["time"]), 0, 1)
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
    batch: tuple[
        _model.Observation,
        ObsType,
        _model.Actions,
        dict[str, at.Array],
    ],
    mc_return: at.Array | None = None,
    scale: at.Array | float = 1.0,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    assert isinstance(config.rl, FlowAWRSFTLearnerConfig)
    rl_config = config.rl
    policy_observation, critic_observation, actions, rollout_info = batch

    reset_period = rl_config.reset_policy_params_to_ema_period
    normalizer_config = rl_config.normalizer_config
    beta = max(rl_config.beta, 1e-6)
    weight_clip = rl_config.weight_clip
    num_steps = rl_config.num_steps
    noise_level = rl_config.noise_level
    clip_epsilon = rl_config.clip_epsilon
    use_pessimistic_clip = rl_config.use_pessimistic_clip

    policy_model = nnx.merge(policy_state.model_def, policy_state.params)
    policy_model.train()

    state_action_critic = create_critic(state_action_critic_state, config)
    state_action_critic.eval()
    value_critic = create_critic(value_state, config)
    value_critic.eval()

    del mc_return
    if rl_config.use_mc_returns:
        raise ValueError(
            "FlowAWRLearner is incompatible with use_mc_returns=True. "
            "Advantages are computed from Q(s, a_buffer) - V(s)."
        )

    critic_actions = flatten_action_horizon(actions)
    value = summarize_critic_values(
        value_critic(critic_observation),
        critic_reduction=rl_config.critic_reduction,
    )
    q_value = summarize_critic_values(
        state_action_critic(critic_observation, critic_actions),
        critic_reduction=rl_config.critic_reduction,
    )
    advantage = q_value - value  # (B,)

    # AWR weight (identical to advantage_weighted_sft/update_actor.py).
    score = advantage / scale
    score = score / beta
    score = jnp.minimum(score, weight_clip)
    score = jnp.exp(score)
    score = score / rl_config.advantage_scale
    score = jnp.clip(score, min=1e-6)
    score = jax.lax.stop_gradient(score)
    if score.ndim == 1:
        score = score[:, jnp.newaxis, jnp.newaxis]

    # log p_old came back from the buffer; reorient to (B, H, num_steps) so it
    # broadcasts against `score` and against `log_pi_new` from
    # `_compute_rollout_log_probs`.
    old_log_probs = jnp.moveaxis(
        jax.lax.stop_gradient(rollout_info["log_prob"]), 1, -1
    )

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

        log_ratio = current_log_probs - old_log_probs
        ratio = jnp.exp(log_ratio)
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
        # Same advantage stats AWR's normalizer host-update path expects.
        "advantage_mean": jnp.mean(advantage),
        "advantage_std": jnp.std(advantage),
        "advantage_max": jnp.max(advantage),
        "advantage_min": jnp.min(advantage),
        "advantage_median": jnp.median(advantage),
        "advantage_q_up": jnp.quantile(advantage, normalizer_config.q_up),
        "advantage_q_low": jnp.quantile(advantage, normalizer_config.q_low),
    } | aux_data
    return new_state, info
