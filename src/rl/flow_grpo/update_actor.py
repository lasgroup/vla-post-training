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
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    assert isinstance(config.rl, FlowGRPOSFTLearnerConfig)
    policy_observation, critic_observation, actions = batch

    policy_model = nnx.merge(policy_state.model_def, policy_state.params)
    policy_model.train()

    state_action_critic = create_critic(state_action_critic_state, config)
    state_action_critic.eval()

    value_critic = create_critic(value_state, config)
    value_critic.eval()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        policy_observation: _model.Observation,
        critic_observation: ObsType,
        state_action_critic: nnx.Module,
        value_critic: nnx.Module,
    ) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:
        step_rng, noise_rng = jax.random.split(rng)

        def expand_and_flatten(x):
            return jnp.repeat(x, repeats=config.rl.group_size, axis=0)

        # Repeat action G times to get a group evaluation, [B * G, ...]
        expanded_policy_obs = jax.tree.map(expand_and_flatten, policy_observation)
        expanded_critic_obs = jax.tree.map(expand_and_flatten, critic_observation)

        # Sample noise vector x_1, [B * G, T, dim_A]
        noise = jax.random.normal(
            noise_rng,
            (
                expanded_policy_obs.state.shape[0],
                model.action_horizon,
                model.action_dim,
            ),
        )
        # Sample actions for expanded states [B * G, dim_A]
        actions, outs = model.sample_actions(
            rng=step_rng,
            observation=expanded_policy_obs,
            noise=noise,
            num_steps=config.rl.num_steps,
            noise_level=config.rl.noise_level,
            return_info_dict=True,
        )

        # Stack log-prob of per step generation [B * G, ..., T].
        log_probs = jnp.moveaxis(outs["log_prob"], 0, -1)

        # 2. Compute the advantage weights
        value = summarize_critic_values(
            value_critic(expanded_critic_obs),
            critic_reduction=config.rl.critic_reduction,
        )
        q_value = summarize_critic_values(
            state_action_critic(expanded_critic_obs, flatten_action_horizon(actions)),
            critic_reduction=config.rl.critic_reduction,
        )
        advantage = q_value - value

        adv = advantage
        if config.rl.normalize_adv and config.rl.group_size > 1:
            total_batch_size = adv.shape[0]
            assert (
                total_batch_size % config.rl.group_size == 0
            ), f"Batch/group mismatch: total_batch_size={total_batch_size}, group_size={config.rl.group_size}"
            B = total_batch_size // config.rl.group_size
            # 4. Reshape back to (B, G) for Group Relative calculations
            # This works because 'repeat' groups copies together, and 'reshape' reads row-major.
            # Get groups per sample in the batch and set the group to be the last dimension.
            # NOTE: jnp.transpose requires a full permutation for all dimensions;
            # swapaxes is the intended "swap last two dims" operation.
            adv = jnp.swapaxes(adv.reshape(B, config.rl.group_size, -1), 1, 2)
            log_probs = jnp.swapaxes(
                log_probs.reshape(B, config.rl.group_size, -1), 1, 2
            )
            group_mean = jnp.mean(adv, axis=-1, keepdims=True)
            group_std = jnp.std(adv, axis=-1, keepdims=True)
            adv = (adv - group_mean) / jnp.maximum(group_std, 1e-6)
        if config.rl.weight_clip is not None:
            adv = jnp.clip(adv, -config.rl.weight_clip, config.rl.weight_clip)

        loss = -jnp.mean(jax.lax.stop_gradient(adv) * log_probs)

        info = {
            "loss": loss,
            "q_mean": jnp.mean(q_value),
            "adv_mean": jnp.mean(adv),
            "log_prob_mean": jnp.mean(log_probs),
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
