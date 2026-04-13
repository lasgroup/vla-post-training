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
from src.training.config import MPOLearnerConfig, OnlineTrainConfig


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
    return jnp.moveaxis(policy_log_probs, 0, -1)


def _solve_e_step_dual(
    advantages_grouped: jnp.ndarray,
    epsilon: float,
    num_steps: int = 15,
    lr: float = 0.5,
) -> jnp.ndarray:
    """solve the E-step dual for optimal temperature eta (Eq. 9).

    min g(eta) = eta * epsilon + eta * mean_s[log(mean_a[exp(A(s,a) / eta)])]

    A(s,a) = Q(s,a) - V(s) are per-group advantages 
    epsilon is the KL budget parameterised as log(eta) so the optimisation is unconstrained.
    """
    G = jnp.float32(advantages_grouped.shape[-1])

    # Initialise at a scale proportional to the advantage spread.
    init_log_eta = jnp.log(jnp.maximum(jnp.std(advantages_grouped), 1e-4))

    def dual_fn(log_eta):
        eta = jnp.exp(log_eta)
        # logsumexp for numerical stability:
        #   log(mean_a[exp(A/eta)]) = logsumexp(A/eta, axis=-1) - log(G)
        scaled = advantages_grouped / jnp.maximum(eta, 1e-8)
        log_mean_exp = jax.scipy.special.logsumexp(scaled, axis=-1) - jnp.log(G)
        return eta * epsilon + eta * jnp.mean(log_mean_exp)

    def step_fn(log_eta, _):
        grad = jax.grad(dual_fn)(log_eta)
        return log_eta - lr * grad, None

    log_eta, _ = jax.lax.scan(step_fn, init_log_eta, None, length=num_steps)
    return jnp.clip(jnp.exp(log_eta), 1e-6, 1e6)


@at.typecheck
def train_step(
    config: OnlineTrainConfig,
    rng: at.KeyArrayLike,
    policy_state: training_utils.TrainState,
    state_action_critic_state: training_utils.TrainState,
    value_state: training_utils.TrainState,
    batch: tuple[_model.Observation, ObsType, _model.Actions],
    alpha_kl: at.Array | None = None,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    assert isinstance(config.rl, MPOLearnerConfig)
    policy_observation, critic_observation, buffer_actions = batch

    policy = nnx.merge(policy_state.model_def, policy_state.params)
    policy.eval()  # eval mode for sampling; switched to train() before gradient tape

    state_action_critic = create_critic(state_action_critic_state, config)
    state_action_critic.eval()

    value_critic = create_critic(value_state, config)
    value_critic.eval()

    group_size = max(config.rl.group_size, 1)
    beta = max(config.rl.beta, 1e-6)
    weight_clip = config.rl.weight_clip
    normalize_adv = config.rl.normalize_adv
    num_steps = config.rl.num_steps
    noise_level = config.rl.noise_level
    kl_coef = config.rl.kl_coef
    use_dual_eta = config.rl.use_dual_eta
    epsilon_e = config.rl.epsilon_e
    dual_eta_steps = config.rl.dual_eta_steps
    dual_eta_lr = config.rl.dual_eta_lr
    use_adaptive_kl = config.rl.use_adaptive_kl
    effective_kl_coef = alpha_kl if alpha_kl is not None else jnp.asarray(kl_coef, dtype=jnp.float32)
    _compute_kl = use_adaptive_kl or kl_coef > 0.0

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

    _need_rollout_info = _compute_kl
    _sample_result = policy.sample_actions(
        rng=sample_rng,
        observation=expanded_policy_obs,
        noise=noise,
        num_steps=num_steps,
        noise_level=noise_level,
        return_info_dict=_need_rollout_info,
        return_prefix_rep=False,
    )
    if _need_rollout_info:
        sampled_actions, rollout_info = _sample_result
    else:
        sampled_actions = _sample_result
        rollout_info = None

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

    # m-step trust region: use the sampling policy's own log-probs as reference.
    # The actions were sampled from the current (pre-update) policy, so its log-probs are already stored in rollout_info from sample_actions().
    sampling_log_probs = None
    if _compute_kl and rollout_info is not None:
        # rollout_info["log_prob"] has shape [num_steps, batch, horizon] move num_steps to last axis to match _compute_rollout_log_probs output.
        sampling_log_probs = jax.lax.stop_gradient(
            jnp.moveaxis(rollout_info["log_prob"], 0, -1)
        )

    # E-step: compute per-group advantage weights (paper Eq. 8).
    assert group_size > 1, "MPO requires group_size > 1 for per-state action reweighting."
    total_batch_size = advantage.shape[0]
    assert (
        total_batch_size % group_size == 0
    ), f"Batch/group mismatch: total_batch_size={total_batch_size}, group_size={group_size}"
    base_batch_size = total_batch_size // group_size

    adv_grouped = advantage.reshape(base_batch_size, group_size)

    if use_dual_eta:
        # Solve the convex dual for optimal temperature eta* (Eq. 9).
        eta = _solve_e_step_dual(
            adv_grouped, epsilon_e, dual_eta_steps, dual_eta_lr,
        )
    else:
        # Fallback: fixed temperature from config.
        if normalize_adv:
            adv_normed = (advantage - jnp.mean(advantage)) / (jnp.std(advantage) + 1e-6)
            adv_grouped = adv_normed.reshape(base_batch_size, group_size)
        eta = jnp.asarray(beta, dtype=adv_grouped.dtype)

    score = adv_grouped / jnp.maximum(eta, 1e-8)
    if weight_clip is not None:
        score = jnp.minimum(score, weight_clip)
    # Per-group softmax: q(a|s) proportional to pi(a|s) exp(Q(s,a)/eta*) (Eq. 8)
    score = jax.nn.softmax(score, axis=-1)

    score_stats = score
    score = jax.lax.stop_gradient(score[..., jnp.newaxis])

    # M-step: weighted maximum likelihood policy fitting (paper Eq. 10).
    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        policy_observation: _model.Observation,
        actions: _model.Actions,
        score: at.Array,
    ) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:
        chunked_loss = model.compute_loss(
            rng, policy_observation, actions, train=True,
        )
        grouped_chunked_loss = chunked_loss.reshape(base_batch_size, group_size, -1)
        # score sums to 1 per group after softmax, so sum over group axis (not mean)
        loss = jnp.mean(jnp.sum(score * grouped_chunked_loss, axis=1))

        info = {
            "q_mean": jnp.mean(q_value),
            "value_mean": jnp.mean(value),
            "advantage_mean": jnp.mean(advantage),
            "advantage_max": jnp.max(advantage),
            "advantage_min": jnp.min(advantage),
            "advantage_std": jnp.mean(
                jnp.std(adv_grouped, axis=-1)
            ),
            "score_mean": jnp.mean(score_stats),
            "score_max": jnp.max(score_stats),
            "eta": eta,
        }
        return loss, info

    policy.train()  # switch to train mode for gradient computation
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

    # M-step KL constraint (Eq. 12) penalise divergence from the sampling (pre-update) policy.  
    # seperate gradient tape to avoid OOM. 
    #
    # KL(pi_old || pi_new) = E_old[log pi_old - log pi_new] -> penalises the policy for moving away from pi_old
    # sampling_log_probs = log pi_old; recompute log pi_new under the current (being-updated) policy.

    kl_loss_val = jnp.asarray(0.0)
    kl_divergence = jnp.asarray(0.0)
    if _compute_kl and rollout_info is not None and sampling_log_probs is not None:
        def kl_loss_fn(model, policy_observation, rollout_info, sampling_log_probs):
            policy_log_probs = _compute_rollout_log_probs(
                model=model,
                policy_observation=policy_observation,
                rollout_info=rollout_info,
                num_steps=num_steps,
                noise_level=noise_level,
            )
            # KL(pi_old || pi_new) = E_old[log pi_old - log pi_new]
            kl = jnp.mean(sampling_log_probs - policy_log_probs)
            return effective_kl_coef * kl, {"kl_divergence": kl}

        kl_diff_state = nnx.DiffState(0, config.trainable_filter)
        (kl_loss_val, kl_aux), kl_grads = nnx.value_and_grad(
            kl_loss_fn, has_aux=True, argnums=kl_diff_state,
        )(policy, expanded_policy_obs, rollout_info, sampling_log_probs)

        grads = jax.tree.map(lambda g, k: g + k, grads, kl_grads)
        aux_data = aux_data | kl_aux
        kl_divergence = kl_aux["kl_divergence"]

    loss = loss + kl_loss_val
    aux_data = aux_data | {
        "loss": loss,
        "kl_loss": kl_loss_val,
        "kl_divergence": kl_divergence,
        "effective_kl_coef": effective_kl_coef,
    }

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
