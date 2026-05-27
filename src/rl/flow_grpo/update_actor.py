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
    assert isinstance(config.rl, FlowGRPOSFTLearnerConfig)
    policy_observation, critic_observation, actions = batch

    policy_model = nnx.merge(policy_state.model_def, policy_state.params)
    policy_model.train()

    state_action_critic = create_critic(state_action_critic_state, config)
    state_action_critic.eval()

    value_critic = create_critic(value_state, config)
    value_critic.eval()

    assert isinstance(config.rl, FlowGRPOSFTLearnerConfig)
    reset_period = config.rl.policy.reset_params_to_ema_period
    group_size = config.rl.group_size
    normalize_adv = config.rl.normalize_adv
    use_mpo_advantage_weight = config.rl.use_mpo_advantage_weight
    weight_clip = config.rl.weight_clip
    beta = max(config.rl.beta, 1e-6)
    num_steps = config.rl.num_steps
    noise_level = config.rl.noise_level
    # Read outside loss_fn so the `if` is a compile-time Python branch.
    filtered_sft_weight = config.rl.filtered_sft_weight
    K = config.rl.grad_accumulation_steps

    if use_mpo_advantage_weight:
        group_size = 1

    def loss_fn(
        model,
        rng,
        policy_obs,
        critic_obs,
        micro_actions,
        micro_is_success,
        state_action_critic,
        value_critic,
    ):
        step_rng, noise_rng = jax.random.split(rng)

        def expand_and_flatten(x):
            return jnp.repeat(x, repeats=group_size, axis=0)

        # Repeat each observation G times for group evaluation, [B_micro * G, ...]
        expanded_policy_obs = jax.tree.map(expand_and_flatten, policy_obs)
        expanded_critic_obs = jax.tree.map(expand_and_flatten, critic_obs)

        # Sample noise vector x_1, [B_micro * G, T, dim_A]
        noise = jax.random.normal(
            noise_rng,
            (
                expanded_policy_obs.state.shape[0],
                model.action_horizon,
                model.action_dim,
            ),
        )
        # Sample actions for expanded states [B_micro * G, dim_A]
        sampled_actions, outs = model.sample_actions(
            rng=step_rng,
            observation=expanded_policy_obs,
            noise=noise,
            num_steps=num_steps,
            noise_level=noise_level,
            return_info_dict=True,
        )

        # Stack log-prob of per step generation [B_micro * G, ..., T].
        log_probs = jnp.moveaxis(outs["log_prob"], 0, -1)

        # Compute advantage weights.
        value = summarize_critic_values(
            value_critic(expanded_critic_obs),
            config,
            critic_reduction=config.rl.critic.reduction,
        )
        q_value = summarize_critic_values(
            state_action_critic(expanded_critic_obs, flatten_action_horizon(sampled_actions)),
            config,
            critic_reduction=config.rl.critic.reduction,
        )
        advantage = q_value - value
        if use_mpo_advantage_weight:
            score = advantage / beta
            score = jnp.minimum(score, weight_clip)  # Clipping
            score = jax.nn.softmax(score, axis=0)  # (B_micro,)
            score = jax.lax.stop_gradient(score)  # Explicitly cut gradients
        else:
            adv = advantage
            if normalize_adv and group_size > 1:
                total_batch_size = adv.shape[0]
                assert (
                    total_batch_size % group_size == 0
                ), f"Batch/group mismatch: total_batch_size={total_batch_size}, group_size={config.rl.group_size}"
                B_micro = total_batch_size // group_size
                # Reshape to (B_micro, G) for group-relative normalisation.
                adv = jnp.swapaxes(adv.reshape(B_micro, group_size, -1), 1, 2)
                log_probs = jnp.swapaxes(log_probs.reshape(B_micro, group_size, -1), 1, 2)
                group_mean = jnp.mean(adv, axis=-1, keepdims=True)
                group_std = jnp.std(adv, axis=-1, keepdims=True)
                adv = (adv - group_mean) / jnp.maximum(group_std, 1e-6)
            if weight_clip is not None:
                adv = jnp.clip(adv, -weight_clip, weight_clip)
            score = jax.lax.stop_gradient(adv)

        # Expand score to broadcast with log_probs (B_micro*G, action_horizon, num_steps)
        if score.ndim == 1:
            score = score[:, jnp.newaxis, jnp.newaxis]

        grpo_loss = -jnp.mean(score * log_probs)

        info = {
            "grpo_loss": grpo_loss,
            "q_mean": jnp.mean(q_value),
            "score_mean": jnp.mean(score),
            "log_prob_mean": jnp.mean(log_probs),
        }

        if filtered_sft_weight > 0.0 and micro_is_success is not None:
            _is_success = jax.lax.stop_gradient(micro_is_success)
            # Use buffer actions on the original B_micro observations —
            # not the expanded B_micro*G used for the GRPO loss.
            sft_rng = jax.random.fold_in(rng, 1)
            sft_chunked_loss = model.compute_loss(sft_rng, policy_obs, micro_actions, train=True)
            while _is_success.ndim < sft_chunked_loss.ndim:
                _is_success = _is_success[..., jnp.newaxis]
            sft_loss = jnp.mean(_is_success * sft_chunked_loss)
            info["sft_loss"] = sft_loss
            total_loss = grpo_loss + filtered_sft_weight * sft_loss
        else:
            total_loss = grpo_loss

        return total_loss, info

    train_rng = jax.random.fold_in(rng, policy_state.step)

    # Gradient accumulation: split B distinct observations into K micro-batches.
    # Group normalisation operates within each group (single observation × G samples),
    # so splitting at observation boundaries preserves correct normalisation.
    B = policy_observation.state.shape[0]
    assert B % K == 0, f"Batch size {B} must be divisible by grad_accumulation_steps {K}"
    micro_B = B // K

    accumulated_grads = None
    total_loss = jnp.zeros(())
    total_aux: dict = {}

    for k in range(K):
        start, end = k * micro_B, (k + 1) * micro_B
        micro_policy_obs = jax.tree.map(lambda x: x[start:end], policy_observation)
        micro_critic_obs = jax.tree.map(lambda x: x[start:end], critic_observation)
        micro_actions = actions[start:end]
        micro_is_success = is_success[start:end] if is_success is not None else None
        micro_rng = jax.random.fold_in(train_rng, k)

        diff_state = nnx.DiffState(0, config.trainable_filter)
        (micro_loss, micro_aux), micro_grads = nnx.value_and_grad(
            loss_fn, has_aux=True, argnums=diff_state
        )(
            policy_model,
            micro_rng,
            micro_policy_obs,
            micro_critic_obs,
            micro_actions,
            micro_is_success,
            state_action_critic,
            value_critic,
        )

        total_loss = total_loss + micro_loss / K
        total_aux = {
            key: total_aux.get(key, jnp.zeros(())) + val / K
            for key, val in micro_aux.items()
        }

        if accumulated_grads is None:
            accumulated_grads = micro_grads
        else:
            accumulated_grads = jax.tree.map(jnp.add, accumulated_grads, micro_grads)

    # Average accumulated gradients across K micro-batches.
    accumulated_grads = jax.tree.map(lambda g: g / K, accumulated_grads)

    params = nnx.filter_state(policy_state.params, config.trainable_filter)
    updates, new_opt_state = policy_state.tx.update(
        accumulated_grads, policy_state.opt_state, params
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
    info = {
        "loss": total_loss,
        "grad_norm": optax.global_norm(accumulated_grads),
        "param_norm": optax.global_norm(kernel_params),
    } | total_aux
    return new_state, info
