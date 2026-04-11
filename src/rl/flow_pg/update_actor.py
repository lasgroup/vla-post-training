from src.training.config import OnlineTrainConfig, FlowPGSFTLearnerConfig
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
    """Evaluate log-probs of a pre-recorded trajectory under *model*."""
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
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    assert isinstance(config.rl, FlowPGSFTLearnerConfig)
    policy_observation, critic_observation, actions = batch

    reset_period = config.rl.reset_policy_params_to_ema_period
    weight_clip = config.rl.weight_clip
    beta = max(config.rl.beta, 1e-6)
    num_steps = config.rl.num_steps
    noise_level = config.rl.noise_level
    use_ema_for_sampling = config.rl.use_ema_for_sampling
    kl_coef = config.rl.kl_coef

    # build two models one for sampling, one ema
    # current policy for gradient computatioms
    policy_model = nnx.merge(policy_state.model_def, policy_state.params)
    policy_model.train()

    if use_ema_for_sampling and policy_state.ema_params is not None:
        sampling_model = nnx.merge(policy_state.model_def, policy_state.ema_params)
    else:
        sampling_model = nnx.merge(policy_state.model_def, policy_state.params)
    sampling_model.eval()

    state_action_critic = create_critic(state_action_critic_state, config)
    state_action_critic.eval()

    value_critic = create_critic(value_state, config)
    value_critic.eval()

     # sample actions from ema
    train_rng = jax.random.fold_in(rng, policy_state.step)
    step_rng, noise_rng = jax.random.split(train_rng)

    noise = jax.random.normal(
        noise_rng,
        (
            policy_observation.state.shape[0],
            sampling_model.action_horizon,
            sampling_model.action_dim,
        ),
    )
    sampled_actions, rollout_info = sampling_model.sample_actions(
        rng=step_rng,
        observation=policy_observation,
        noise=noise,
        num_steps=num_steps,
        noise_level=noise_level,
        return_info_dict=True,
    )

    # fix trajectory so the current policy is evaluated correctly
    rollout_info = jax.tree.map(jax.lax.stop_gradient, rollout_info)

    # compute mpo scores
    value = summarize_critic_values(
        value_critic(critic_observation),
        critic_reduction=config.rl.critic_reduction,
    )
    q_value = summarize_critic_values(
        state_action_critic(critic_observation, flatten_action_horizon(sampled_actions)),
        critic_reduction=config.rl.critic_reduction,
    )
    advantage = q_value - value

    score = advantage / beta
    score = jnp.minimum(score, weight_clip)
    score = jax.nn.softmax(score, axis=0)
    score = jax.lax.stop_gradient(score)

    if score.ndim == 1:
        score = score[:, jnp.newaxis, jnp.newaxis]

    # ref. log probs for kl computation
    reference_log_probs = None
    if kl_coef > 0.0 and noise_level > 0.0 and policy_state.ema_params is not None:
        ref_model = nnx.merge(policy_state.model_def, policy_state.ema_params)
        ref_model.eval()
        reference_log_probs = _compute_rollout_log_probs(
            model=ref_model,
            policy_observation=policy_observation,
            rollout_info=rollout_info,
            num_steps=num_steps,
            noise_level=noise_level,
        )
        reference_log_probs = jax.lax.stop_gradient(reference_log_probs)

    # score x log_probs + kl
    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        policy_observation: _model.Observation,
        critic_observation: ObsType,
        state_action_critic: nnx.Module,
        value_critic: nnx.Module,
    ) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:
        current_log_probs = _compute_rollout_log_probs(
            model=model,
            policy_observation=policy_observation,
            rollout_info=rollout_info,
            num_steps=num_steps,
            noise_level=noise_level,
        )

        # score × log-probs.
        pg_loss = -jnp.mean(score * current_log_probs)

        # kl against ema
        kl_loss = jnp.asarray(0.0, dtype=current_log_probs.dtype)
        ref_log_prob_mean = jnp.asarray(0.0, dtype=current_log_probs.dtype)
        if kl_coef > 0.0 and reference_log_probs is not None:
            kl_loss = jnp.mean(current_log_probs - reference_log_probs)
            ref_log_prob_mean = jnp.mean(reference_log_probs)

        loss = pg_loss + kl_coef * kl_loss

        info = {
            "loss": loss,
            "pg_loss": pg_loss,
            "kl_loss": kl_loss,
            "q_mean": jnp.mean(q_value),
            "score_mean": jnp.mean(score),
            "log_prob_mean": jnp.mean(current_log_probs),
            "reference_log_prob_mean": ref_log_prob_mean,
        }
        return loss, info

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
    } | aux_data
    return new_state, info
