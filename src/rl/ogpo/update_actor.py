# ruff: noqa: F722
"""OGPO actor train step for the Pi05 flow policy.

Implements the canonical "vanilla" OGPO policy-extraction branch
(``ogpo/agents/ogpo.py:778-792``):

  1. Expand the batch to ``B*G`` copies (one chain per sample).
  2. Sample G SDE rollouts under the *old* (EMA) policy; capture
     ``old_lp`` from the closed-form Gaussian transitions.
  3. Q − V advantage on the sampled actions, then group-relative
     normalization across the G axis.
  4. Recompute the joint log-prob of the same chains under the *current*
     policy; take the clipped PPO surrogate against the group advantages.
  5. Add a CFM BC anchor on the actions stored in the *un-expanded* online
     batch (size B). No success filter, no static demos.

The PaliGemma backbone is expected to be frozen via ``config.freeze_filter``;
this train step does not assume anything about which parameters are
trainable beyond ``config.trainable_filter``.
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
from src.rl.ogpo.sampling import (
    sample_chain_with_logprob,
    score_chain_under_model,
    sum_log_prob,
)
from src.training.config import OnlineTrainConfig, OGPOSFTLearnerConfig


def _group_baseline(adv: jax.Array, B: int, G: int, strategy: str) -> jax.Array:
    """Return a per-state baseline ``[B, 1]`` that will be broadcast over G."""
    adv_g = adv.reshape(B, G)
    if strategy == "vanilla":
        return adv_g.mean(axis=1, keepdims=True)
    if strategy == "max":
        return adv_g.max(axis=1, keepdims=True)
    if strategy == "subtract_v":
        # advantage already contains the Q-V centering.
        return jnp.zeros((B, 1), dtype=adv.dtype)
    raise ValueError(f"Unknown adv_strategy: {strategy!r}")


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
    del mc_return, scale  # OGPO uses Q-V advantages, not MC normalization (v1).

    assert isinstance(config.rl, OGPOSFTLearnerConfig)
    rl = config.rl

    policy_observation, critic_observation, actions_demo = batch

    # Build the "old" (EMA) policy — purely for sampling, no gradient.
    if rl.use_ema_as_old_policy and policy_state.ema_params is not None:
        old_params = policy_state.ema_params
    else:
        old_params = policy_state.params
    old_model = nnx.merge(
        policy_state.model_def, jax.tree.map(jax.lax.stop_gradient, old_params)
    )
    old_model.eval()

    # Critics are frozen w.r.t. this train_step.
    state_action_critic = create_critic(state_action_critic_state, config)
    state_action_critic.eval()
    value_critic = create_critic(value_state, config)
    value_critic.eval()

    G = rl.group_num_samples
    num_steps = rl.num_sde_steps
    noise_level = rl.noise_level

    sample_rng, score_rng, bc_rng = jax.random.split(rng, 3)

    # Expand observations to B*G copies for group rollouts.
    def _expand(x):
        return jnp.repeat(x, repeats=G, axis=0)

    expanded_policy_obs = jax.tree.map(_expand, policy_observation)
    expanded_critic_obs = jax.tree.map(_expand, critic_observation)

    # --- 1. Sample G SDE chains from the OLD policy. --------------------
    chain_pack = sample_chain_with_logprob(
        old_model,
        expanded_policy_obs,
        rng=sample_rng,
        num_steps=num_steps,
        noise_level=noise_level,
    )
    sampled_actions = jax.lax.stop_gradient(chain_pack["actions"])      # [B*G, H, D]
    x_chain         = jax.lax.stop_gradient(chain_pack["x_chain"])      # [K, B*G, H, D]
    x_next_chain    = jax.lax.stop_gradient(chain_pack["x_next_chain"]) # [K, B*G, H, D]
    times           = jax.lax.stop_gradient(chain_pack["times"])        # [K, B*G]
    dt              = jax.lax.stop_gradient(chain_pack["dt"])           # scalar

    # Per-scalar-dim normalization of the summed log-prob, mirroring the
    # official OGPO knobs. Without this, `sum_log_prob` returns the joint
    # log-prob over K·H·D ≈ hundreds of dims, so a sub-1% per-dim policy
    # shift produces a log-ratio of tens and PPO's clip saturates.
    K, _, H, D = x_chain.shape
    log_prob_norm = jnp.float32(1.0)
    if rl.normalize_denoising_horizon:
        log_prob_norm = log_prob_norm * jnp.float32(K * H)
    if rl.normalize_act_space_dimension:
        log_prob_norm = log_prob_norm * jnp.float32(D)

    old_lp          = jax.lax.stop_gradient(
        sum_log_prob(chain_pack["log_prob_per_step"]) / log_prob_norm
    )  # [B*G]

    # --- 2. Q − V advantages, then group-relative centering. -----------
    q_value = summarize_critic_values(
        state_action_critic(
            expanded_critic_obs, flatten_action_horizon(sampled_actions)
        ),
        critic_reduction=rl.critic_reduction,
    )  # [B*G]
    v_value = summarize_critic_values(
        value_critic(expanded_critic_obs),
        critic_reduction=rl.critic_reduction,
    )  # [B*G]
    advantage_raw = q_value - v_value  # [B*G]

    B = jax.tree.leaves(policy_observation)[0].shape[0]
    baseline = _group_baseline(advantage_raw, B, G, rl.adv_strategy)  # [B, 1]
    advantage = advantage_raw.reshape(B, G) - baseline
    if rl.adv_clip_min is not None:
        advantage = jnp.maximum(advantage, rl.adv_clip_min)
    advantage = advantage.reshape(-1)  # [B*G]
    advantage = jax.lax.stop_gradient(advantage)

    # --- 3. PPO loss + BC anchor (gradient sink). ----------------------
    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        bc_rng: at.KeyArrayLike,
    ) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:
        new_log_prob_per_step = score_chain_under_model(
            model,
            expanded_policy_obs,
            x_chain=x_chain,
            x_next_chain=x_next_chain,
            times=times,
            dt=dt,
            noise_level=noise_level,
        )  # [K, B*G, H]
        # Same normalization as old_lp so the ratio is on a per-dim scale.
        new_lp = sum_log_prob(new_log_prob_per_step) / log_prob_norm  # [B*G]

        log_ratio = new_lp - old_lp                   # [B*G]
        ratio = jnp.exp(log_ratio)
        lower_bound = 1.0 - rl.clip_epsilon
        upper_bound = 1.0 + rl.clip_epsilon
        clipped_ratio = jnp.clip(ratio, lower_bound, upper_bound)
        pg_per_sample = jnp.minimum(ratio * advantage, clipped_ratio * advantage)
        pg_loss = -jnp.mean(pg_per_sample)
        # Unclipped surrogate for comparison — when clipfrac saturates, this
        # diverges from pg_loss and reveals how much signal the clip is
        # actually killing.
        pg_loss_unclipped = -jnp.mean(ratio * advantage)

        # BC on the un-expanded online batch (size B).
        bc_loss = jnp.float32(0.0)
        if rl.use_bc_regularization:
            chunked_bc = model.compute_loss(
                bc_rng, policy_observation, actions_demo, train=True
            )
            bc_loss = jnp.mean(chunked_bc)

        total = pg_loss + rl.bc_coeff * bc_loss
        # PPO's k3 approximation of KL(old || new); ratio_mean and log_ratio
        # together pin down a Gaussian fit on the log-ratio when needed.
        approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
        ratio_clipped_lower = jnp.mean(
            (ratio < lower_bound).astype(jnp.float32)
        )
        ratio_clipped_upper = jnp.mean(
            (ratio > upper_bound).astype(jnp.float32)
        )
        clipfrac = ratio_clipped_lower + ratio_clipped_upper
        # Alive-fraction proxy: fraction of samples whose ratio sits inside the
        # PPO clip window — these are the only samples carrying a non-clipped
        # PG gradient. Healthy runs should have this well above 0.5.
        alive_fraction = jnp.mean(
            ((ratio >= lower_bound) & (ratio <= upper_bound)).astype(jnp.float32)
        )

        aux = {
            "pg_loss": pg_loss,
            "pg_loss_unclipped": pg_loss_unclipped,
            "bc_loss": bc_loss,
            # Ratio distribution: mean/std collapse to a single point estimate
            # when the distribution is bimodal (mass near 0 + small heavy
            # tail); quantiles + min/max disambiguate that case.
            "ratio_mean": jnp.mean(ratio),
            "ratio_std":  jnp.std(ratio),
            "ratio_min":  jnp.min(ratio),
            "ratio_max":  jnp.max(ratio),
            "ratio_p05":  jnp.quantile(ratio, 0.05),
            "ratio_p50":  jnp.quantile(ratio, 0.50),
            "ratio_p95":  jnp.quantile(ratio, 0.95),
            # log_ratio is roughly Gaussian per sample even when ratio isn't;
            # std measures per-sample disagreement between current and EMA.
            "log_ratio_mean": jnp.mean(log_ratio),
            "log_ratio_std":  jnp.std(log_ratio),
            "log_ratio_min":  jnp.min(log_ratio),
            "log_ratio_max":  jnp.max(log_ratio),
            "approx_kl": approx_kl,
            "clipfrac": clipfrac,
            "clipfrac_upper": ratio_clipped_upper,
            "clipfrac_lower": ratio_clipped_lower,
            "alive_fraction": alive_fraction,
            "new_log_prob_mean": jnp.mean(new_lp),
            "old_log_prob_mean": jnp.mean(old_lp),
            "q_mean": jnp.mean(q_value),
            "v_mean": jnp.mean(v_value),
        }
        return total, aux

    policy_model = nnx.merge(policy_state.model_def, policy_state.params)
    policy_model.train()
    train_rng = jax.random.fold_in(bc_rng, policy_state.step)
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True, argnums=diff_state)(
        policy_model, train_rng
    )

    # --- 4. Optimizer step + EMA update (mirrors awr/flow_grpo). -------
    params = nnx.filter_state(policy_state.params, config.trainable_filter)
    updates, new_opt_state = policy_state.tx.update(grads, policy_state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(policy_model, new_params)
    new_full_params = nnx.state(policy_model)

    new_state = dataclasses.replace(
        policy_state,
        step=policy_state.step + 1,
        params=new_full_params,
        opt_state=new_opt_state,
    )
    if policy_state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: policy_state.ema_decay * old
                + (1.0 - policy_state.ema_decay) * new,
                policy_state.ema_params,
                new_full_params,
            ),
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
        "advantage_max":  jnp.max(advantage),
        "advantage_min":  jnp.min(advantage),
        "advantage_std":  jnp.std(advantage),
        "advantage_q_up":  jnp.quantile(advantage, 0.95),
        "advantage_q_low": jnp.quantile(advantage, 0.05),
        "advantage_median": jnp.median(advantage),
    } | aux
    return new_state, info
