# ruff: noqa: F722
#
# FMGRPO policy update.
#
# Unlike Flow-GRPO (which runs the SDE *inside* value_and_grad and stores
# O(num_steps) transformer activations), here the group actions are sampled
# with stop_gradient BEFORE the gradient tape.  The tape only sees a single
# FM forward pass (model.compute_loss), giving O(1) memory w.r.t. SDE steps.
#
# Algorithm per update:
#   1. Sample G actions per observation via ODE (no gradient).
#   2. Compute Q(s,a) and V(s) for all B×G pairs (no gradient).
#   3. Group-normalise advantages within each group of G.
#   4. Inside value_and_grad: loss = mean(score × FM_loss(obs, sampled_action)).
#   5. Optionally add filtered-SFT auxiliary loss on buffer actions.

from src.training.config import OnlineTrainConfig, FMGRPOLearnerConfig
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
    assert isinstance(config.rl, FMGRPOLearnerConfig)
    policy_observation, critic_observation, actions = batch

    policy_model = nnx.merge(policy_state.model_def, policy_state.params)
    policy_model.train()

    state_action_critic = create_critic(state_action_critic_state, config)
    state_action_critic.eval()
    value_critic = create_critic(value_state, config)
    value_critic.eval()

    reset_period = config.rl.policy.reset_params_to_ema_period
    group_size = config.rl.group_size
    normalize_adv = config.rl.normalize_adv
    weight_clip = config.rl.weight_clip
    filtered_sft_weight = config.rl.filtered_sft_weight
    sample_num_steps = config.rl.sample_num_steps

    def expand_and_flatten(x):
        return jnp.repeat(x, repeats=group_size, axis=0)

    expanded_policy_obs = jax.tree.map(expand_and_flatten, policy_observation)
    expanded_critic_obs = jax.tree.map(expand_and_flatten, critic_observation)

    sample_rng, train_rng = jax.random.split(rng)
    train_rng = jax.random.fold_in(train_rng, policy_state.step)

    # ---- Step 1: Sample B×G actions (stop_gradient — no SDE activation storage) ----
    # Deterministic ODE gives diverse samples across G different initial noises;
    # higher sample_num_steps improves action quality at no gradient-memory cost.
    sampled_actions = jax.lax.stop_gradient(
        policy_model.sample_actions(
            rng=sample_rng,
            observation=expanded_policy_obs,
            num_steps=sample_num_steps,
            noise_level=0.0,
        )
    )  # [B*G, action_horizon, action_dim]

    # ---- Step 2: Compute advantages (all stop_gradient) ----
    value = jax.lax.stop_gradient(
        summarize_critic_values(
            value_critic(expanded_critic_obs),
            config,
            critic_reduction=config.rl.critic.reduction,
        )
    )  # [B*G]
    q_value = jax.lax.stop_gradient(
        summarize_critic_values(
            state_action_critic(
                expanded_critic_obs, flatten_action_horizon(sampled_actions)
            ),
            config,
            critic_reduction=config.rl.critic.reduction,
        )
    )  # [B*G]
    advantage = q_value - value  # [B*G]

    # ---- Step 3: Group-normalise advantages ----
    adv = advantage
    if normalize_adv and group_size > 1:
        B_obs = adv.shape[0] // group_size
        assert (
            adv.shape[0] % group_size == 0
        ), f"Batch/group mismatch: {adv.shape[0]} % {group_size} != 0"
        adv_grouped = adv.reshape(B_obs, group_size)
        group_mean = jnp.mean(adv_grouped, axis=1, keepdims=True)
        group_std = jnp.std(adv_grouped, axis=1, keepdims=True)
        adv_grouped = (adv_grouped - group_mean) / jnp.maximum(group_std, 1e-6)
        adv = adv_grouped.reshape(B_obs * group_size)
    if weight_clip is not None:
        adv = jnp.clip(adv, -weight_clip, weight_clip)
    score = jax.lax.stop_gradient(adv)  # [B*G]

    # ---- Step 4: FM loss weighted by score (gradient flows here — one forward pass) ----
    # FM loss = E_t[||v_θ(x_t,t) - u_t||²].  model.compute_loss samples a random t
    # and returns per-token losses; no SDE loop → O(1) activation memory.
    def loss_fn(model, rng):
        fm_chunked_loss = model.compute_loss(
            rng, expanded_policy_obs, sampled_actions, train=True
        )
        # fm_chunked_loss: [B*G] or [B*G, action_horizon, ...]
        score_broadcast = score
        while score_broadcast.ndim < fm_chunked_loss.ndim:
            score_broadcast = score_broadcast[..., jnp.newaxis]
        fmgrpo_loss = jnp.mean(score_broadcast * fm_chunked_loss)

        info = {
            "fmgrpo_loss": fmgrpo_loss,
            "q_mean": jnp.mean(q_value),
            "score_mean": jnp.mean(score),
            "advantage_mean": jnp.mean(advantage),
            "advantage_std": jnp.std(advantage),
        }

        if filtered_sft_weight > 0.0 and is_success is not None:
            # SFT on buffer actions (original B obs, not the expanded B*G).
            sft_rng = jax.random.fold_in(rng, 1)
            sft_chunked_loss = model.compute_loss(
                sft_rng, policy_observation, actions, train=True
            )
            _is_success = jax.lax.stop_gradient(is_success)
            while _is_success.ndim < sft_chunked_loss.ndim:
                _is_success = _is_success[..., jnp.newaxis]
            sft_loss = jnp.mean(_is_success * sft_chunked_loss)
            info["sft_loss"] = sft_loss
            total_loss = fmgrpo_loss + filtered_sft_weight * sft_loss
        else:
            total_loss = fmgrpo_loss

        return total_loss, info

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, aux_data), grads = nnx.value_and_grad(
        loss_fn, has_aux=True, argnums=diff_state
    )(policy_model, train_rng)

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
                return state.replace(
                    params=jax.tree.map(lambda x: x, state.ema_params)
                )

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
