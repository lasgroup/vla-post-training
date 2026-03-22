# ruff: noqa: F722
from src.training.config import OnlineTrainConfig, AdvantageWeightedSFTLearnerConfig
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
def _awr_beta(config: OnlineTrainConfig) -> float:
    assert isinstance(config.rl, AdvantageWeightedSFTLearnerConfig)
    beta = config.rl.beta
    return max(beta, 1e-6)


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
    policy_observation, critic_observation, actions = batch

    policy = nnx.merge(policy_state.model_def, policy_state.params)
    policy.train()

    assert isinstance(config.rl, AdvantageWeightedSFTLearnerConfig)

    if config.rl.use_mc_returns:
        assert mc_return is not None, "mc_return must be provided when use_mc_returns=True"
        value_critic = create_critic(value_state, config)
        value_critic.eval()
        value = summarize_critic_values(value_critic(critic_observation))  # (B,)
        advantage = mc_return - value  # (B, )
    else:
        state_action_critic = create_critic(state_action_critic_state, config)
        state_action_critic.eval()
        value_critic = create_critic(value_state, config)
        value_critic.eval()
        # 2. Compute the advantage weights OUTSIDE the value_and_grad trace
        critic_actions = flatten_action_horizon(actions)
        value = summarize_critic_values(value_critic(critic_observation))  # (B,)
        q_value = summarize_critic_values(
            state_action_critic(critic_observation, critic_actions)
        )  # (B,)
        advantage = q_value - value  # (B, )

    if config.rl.normalize_adv:
        advantage = (advantage - jnp.mean(advantage)) / (jnp.std(advantage) + 1e-6)

    score = advantage / _awr_beta(config)
    score = jnp.minimum(score, config.rl.weight_clip)
    score = jnp.exp(score)
    score = score / config.rl.advantage_scale
    score = jnp.clip(score, min=1e-6)
    score = jax.lax.stop_gradient(score)  # Explicitly cut gradients

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        policy_observation: _model.Observation,
        actions: _model.Actions,
        score: jnp.ndarray,
    ) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:
        # We up-weight terms that have high advantage
        chunked_loss = model.compute_loss(rng, policy_observation, actions, train=True)
        # TODO: Replce nasty while loop with assert on the dimension of the arrays
        # assert chunked_loss.shape == (B, 1)
        while score.ndim < chunked_loss.ndim:
            score = score[..., jnp.newaxis]
        aux_data = {
            "chunked_loss": jnp.mean(chunked_loss),
        }
        return jnp.mean(score * chunked_loss), aux_data

    train_rng = jax.random.fold_in(rng, policy_state.step)

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, aux_data), grads = nnx.value_and_grad(
        loss_fn, has_aux=True, argnums=diff_state
    )(
        policy,
        train_rng,
        policy_observation,
        actions,
        score,
    )

    params = nnx.filter_state(policy_state.params, config.trainable_filter)
    updates, new_opt_state = policy_state.tx.update(
        grads, policy_state.opt_state, params
    )
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
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
            new_state = jax.lax.cond(
                new_state.step % reset_period == 0,
                lambda s: s.replace(params=jax.tree.map(lambda x: x, s.ema_params)),
                lambda s: s,
                new_state,
            )

    # Filter out params that aren't kernels.
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
        "advantage_mean": jnp.mean(advantage),
        "advantage_max": jnp.max(advantage),
        "advantage_min": jnp.min(advantage),
        "advantage_std": jnp.std(advantage),
    } | aux_data
    return new_state, info
