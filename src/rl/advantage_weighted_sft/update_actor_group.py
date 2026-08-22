# ruff: noqa: F722
"""Group-relative AWR actor step.

Differs from `update_actor.train_step` in what gets scored and what gets
regressed onto. There, the buffer action is both the input to Q and the
endpoint of the flow-matching interpolant. Here neither is: for each state we
draw `group_size` fresh chunks from the current policy, score each with Q, and
take the per-state group mean as the baseline in place of V(s). The same
samples are the CFM targets, weighted by exp(A_i / beta).

The group mean is an exact per-state baseline, so every common-mode critic
error at that state -- the per-task Q-V offset, the level drift between actor
updates -- cancels inside the group. `normalize_advantages` has nothing left
to do and is ignored.

The cost is that Q is now queried at actions it was never trained on, and
exp(A/beta) is a soft-argmax over G, i.e. a directed search for the critic's
largest error at that state. Two knobs can bound that, both left at their
awr.sh values in the shipped config so the comparison stays clean:
`filtered_sft_weight` adds a fixed-data anchor to the loss, and
`rl.policy.training_start_step` holds the actor off until the critic is worth
querying. Watch actor/within_group_adv_std against the critic's TD error -- if
the within-group spread is the smaller of the two, the weights are noise.
"""

from src.training.config import OnlineTrainConfig, AdvantageWeightedSFTLearnerConfig
from src.rl.advantage_weighted_sft.update_actor import _per_group_stats
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
    return max(config.rl.beta, 1e-6)


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
    bias: at.Array | float = 0.0,
    task_id: at.Int[at.Array, " b"] | None = None,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    # V and the EMA normalizer bias are unused here; `scale` survives only as
    # the carrier of the task-group count for the diagnostics below.
    del mc_return, bias
    policy_observation, critic_observation, buffer_actions = batch

    assert isinstance(config.rl, AdvantageWeightedSFTLearnerConfig)
    group_size = config.rl.group_size
    assert group_size > 1, "group_advantage needs rl.group_size > 1"
    assert not config.rl.use_mc_returns, "group_advantage is incompatible with use_mc_returns"

    policy = nnx.merge(policy_state.model_def, policy_state.params)

    # --- 1. Draw the group, outside the value_and_grad trace ---------------
    # The samples are regression targets, not a differentiable path, so nothing
    # backprops through the ODE. eval() so the sampler sees inference-mode
    # dropout; switched back before the loss.
    policy.eval()
    sample_rng, train_rng = jax.random.split(jax.random.fold_in(rng, policy_state.step))

    def repeat_group(x):
        return jnp.repeat(x, repeats=group_size, axis=0)

    # (B*G, ...) with each state's G copies contiguous, so a later reshape to
    # (B, G) recovers the groups row-major.
    group_policy_obs = jax.tree.map(repeat_group, policy_observation)
    group_critic_obs = jax.tree.map(repeat_group, critic_observation)

    noise = jax.random.normal(
        jax.random.fold_in(sample_rng, 0),
        (group_policy_obs.state.shape[0], policy.action_horizon, policy.action_dim),
    )
    sampled_actions = policy.sample_actions(
        rng=sample_rng,
        observation=group_policy_obs,
        noise=noise,
        num_steps=config.rl.group_num_steps,
        noise_level=config.rl.group_noise_level,
    )
    sampled_actions = jax.lax.stop_gradient(sampled_actions)
    policy.train()

    # --- 2. Group-relative advantage ---------------------------------------
    state_action_critic = create_critic(state_action_critic_state, config)
    state_action_critic.eval()
    q_value = summarize_critic_values(
        state_action_critic(group_critic_obs, flatten_action_horizon(sampled_actions)),
        config,
    )  # (B*G,)

    q_grouped = q_value.reshape(-1, group_size)  # (B, G)
    group_mean = jnp.mean(q_grouped, axis=-1, keepdims=True)
    advantage = (q_grouped - group_mean).reshape(-1)  # (B*G,)

    score = advantage / _awr_beta(config)
    score = jnp.minimum(score, config.rl.weight_clip)
    score = jnp.exp(score)
    score = score / config.rl.advantage_scale
    score = jnp.clip(score, min=1e-6)
    score = jax.lax.stop_gradient(score)

    filtered_sft_weight = config.rl.filtered_sft_weight
    awr_loss_weight = config.rl.awr_loss_weight

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        group_policy_obs: _model.Observation,
        sampled_actions: _model.Actions,
        score: jnp.ndarray,
        policy_observation: _model.Observation,
        buffer_actions: _model.Actions,
    ) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:
        chunked_loss = model.compute_loss(rng, group_policy_obs, sampled_actions, train=True)
        while score.ndim < chunked_loss.ndim:
            score = score[..., jnp.newaxis]
        awr_loss = awr_loss_weight * jnp.mean(score * chunked_loss)
        aux_data = {"chunked_loss": jnp.mean(chunked_loss), "awr_loss": awr_loss}

        # The anchor. Self-samples alone have no fixed point pulling back to the
        # data, so the BC term runs on the un-repeated buffer batch.
        if filtered_sft_weight > 0.0 and is_success is not None:
            _is_success = jax.lax.stop_gradient(is_success)
            sft_rng = jax.random.fold_in(rng, 1)
            sft_chunked_loss = model.compute_loss(
                sft_rng, policy_observation, buffer_actions, train=True
            )
            while _is_success.ndim < sft_chunked_loss.ndim:
                _is_success = _is_success[..., jnp.newaxis]
            sft_loss = jnp.mean(_is_success * sft_chunked_loss)
            aux_data["sft_loss"] = sft_loss
            total_loss = awr_loss + filtered_sft_weight * sft_loss
        else:
            total_loss = awr_loss

        return total_loss, aux_data

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, aux_data), grads = nnx.value_and_grad(
        loss_fn, has_aux=True, argnums=diff_state
    )(
        policy,
        train_rng,
        group_policy_obs,
        sampled_actions,
        score,
        policy_observation,
        buffer_actions,
    )

    params = nnx.filter_state(policy_state.params, config.trainable_filter)
    updates, new_opt_state = policy_state.tx.update(grads, policy_state.opt_state, params)
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
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )

    # Whether the group has any signal at all: within-group spread of Q against
    # the critic's own error. If group_adv_std sits below critic/q_td_loss the
    # weights are ranking critic noise, not actions.
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        "within_group_adv_std": jnp.mean(jnp.std(q_grouped, axis=-1)),
        "within_group_adv_range": jnp.mean(
            jnp.max(q_grouped, axis=-1) - jnp.min(q_grouped, axis=-1)
        ),
        "sampled_q_mean": jnp.mean(q_value),
        "advantage_mean": jnp.mean(advantage),
        "advantage_max": jnp.max(advantage),
        "weight_mean": jnp.mean(score),
        "weight_max": jnp.max(score),
        # ESS/N of the exp weights: 1.0 is uniform, 1/G is a hard argmax.
        "weight_ess_frac": jnp.sum(score) ** 2 / (score.size * jnp.sum(score**2)),
    } | aux_data

    # Per-task diagnostics on the same layout as the non-group path, so
    # actor/weight_share/<task> stays comparable across the two runs.
    if task_id is not None:
        group_task_id = repeat_group(task_id)
        num_groups = int(jnp.asarray(scale).shape[0]) if hasattr(scale, "shape") else 1
        info = info | _per_group_stats(advantage, group_task_id, num_groups, config)
        onehot = jax.nn.one_hot(group_task_id, num_groups)
        info["group_weight_share"] = jnp.sum(onehot * score[:, jnp.newaxis], axis=0) / jnp.maximum(
            jnp.sum(score), 1e-12
        )

    return new_state, info
