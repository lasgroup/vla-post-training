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
    policy_observation, critic_observation, _ = batch

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

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        policy_observation: _model.Observation,
        critic_observation: ObsType,
    ) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:
        sample_rng, loss_rng, noise_rng = jax.random.split(rng, 3)

        def expand_and_flatten(x):
            return jnp.repeat(x, repeats=group_size, axis=0)

        expanded_policy_obs = jax.tree.map(expand_and_flatten, policy_observation)
        expanded_critic_obs = jax.tree.map(expand_and_flatten, critic_observation)

        noise = jax.random.normal(
            noise_rng,
            (
                expanded_policy_obs.state.shape[0],
                model.action_horizon,
                model.action_dim,
            ),
        )
        sampled_actions = model.sample_actions(
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
        advantage = q_value - value

        chunked_loss = model.compute_loss(
            loss_rng,
            expanded_policy_obs,
            sampled_actions,
            train=True,
        )

        if group_size > 1:
            total_batch_size = advantage.shape[0]
            assert (
                total_batch_size % group_size == 0
            ), f"Batch/group mismatch: total_batch_size={total_batch_size}, group_size={group_size}"
            base_batch_size = total_batch_size // group_size
            score = advantage.reshape(base_batch_size, group_size) / beta
            if weight_clip is not None:
                score = jnp.minimum(score, weight_clip)
            score = jax.nn.softmax(score, axis=-1)
            score = jax.lax.stop_gradient(score[..., jnp.newaxis])

            grouped_chunked_loss = chunked_loss.reshape(base_batch_size, group_size, -1)
            loss = jnp.mean(jnp.sum(score * grouped_chunked_loss, axis=1))
            score_stats = score[..., 0]
        else:
            score = advantage / beta
            if weight_clip is not None:
                score = jnp.minimum(score, weight_clip)
            score = jax.nn.softmax(score, axis=0)
            score = jax.lax.stop_gradient(score)
            while score.ndim < chunked_loss.ndim:
                score = score[..., jnp.newaxis]
            loss = jnp.sum(score * chunked_loss)
            score_stats = score[..., 0]

        info = {
            "loss": loss,
            "chunked_loss": jnp.mean(chunked_loss),
            "q_mean": jnp.mean(q_value),
            "value_mean": jnp.mean(value),
            "advantage_mean": jnp.mean(advantage),
            "advantage_max": jnp.max(advantage),
            "advantage_min": jnp.min(advantage),
            "advantage_std": jnp.std(advantage),
            "score_mean": jnp.mean(score_stats),
            "score_max": jnp.max(score_stats),
        }
        return loss, info

    train_rng = jax.random.fold_in(rng, policy_state.step)

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, aux_data), grads = nnx.value_and_grad(
        loss_fn, has_aux=True, argnums=diff_state
    )(
        policy,
        train_rng,
        policy_observation,
        critic_observation,
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
