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

The single ``train_step`` is split into sequential jit bodies
(``sample_and_advantage`` → ``loss_and_grad_pg`` → ``bc_grad_accumulate`` →
``optimizer_tail`` + ``policy_param_norm``) so each transient is bounded in its
own arena; the retained ``train_step`` composes them for the built-but-uncalled
base mono jit and the split-vs-mono equivalence test. The loss backward itself is
two passes (``loss_and_grad_pg`` for the PPO surrogate, ``bc_grad_accumulate`` for
the BC anchor accumulated into the donated PPO grads) so the two full fp32 weight-
gradient trees and their activation sets are never co-resident — the jit-2 arena
fix (docs/plans/ogpo-memory/analysis-jit2-forensics.md).

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
    critic_values_per_head,
    flatten_action_horizon,
    summarize_critic_values,
)
from src.rl.ema_utils import compose_full_params
from src.rl.networks.rl_networks import ObsType
from src.rl.ogpo.sampling import (
    sample_chain_with_logprob,
    score_chain_under_model,
    sum_log_prob,
)
from src.rl.prefix_embedding import PREFIX_EMBEDDING_NAME
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
def sample_and_advantage(
    config: OnlineTrainConfig,
    rng: at.KeyArrayLike,                                   # = policy_rng
    policy_state: training_utils.TrainState,               # UNDONATED (model_def, params, step)
    state_action_critic_state: training_utils.TrainState,  # UNDONATED
    value_state: training_utils.TrainState,                # UNDONATED
    policy_observation: _model.Observation,                # un-expanded (B); carries images
    critic_prefix: at.Float[at.Array, "b embed"] | None,   # None => recompute; WP-C sidecar otherwise
    ema: nnx.State,                                         # explicit _ema_sharding input (full OR trainable-only)
) -> tuple[
    # Outputs are at the EXPANDED batch (bg = B*G); `b` is reserved for the
    # un-expanded inputs above. With G=1 the two coincide, but G>1 must not
    # fail the typecheck.
    at.Float[at.Array, "k bg ah ad"],  # x_chain
    at.Float[at.Array, "k bg ah ad"],  # x_next_chain
    at.Float[at.Array, "k bg"],        # times
    at.Float[at.Array, ""],            # dt
    at.Float[at.Array, " bg"],         # old_lp (already / log_prob_norm)
    at.Float[at.Array, " bg"],         # advantage (stop-grad'd)
    dict[str, at.Array],               # sampler_aux
]:
    assert isinstance(config.rl, OGPOSFTLearnerConfig)
    rl = config.rl
    G = rl.group_num_samples
    num_steps = rl.num_sde_steps
    noise_level = rl.noise_level

    # Reproduce the mono jit's arity-3 split verbatim so sample_rng=[0] matches;
    # score_rng/bc_rng are jit-2's slots, kept here as dead slots so the keystream
    # is bit-identical (do NOT re-split to arity 2 — that would shift bc_rng, G1).
    sample_rng, score_rng, bc_rng = jax.random.split(rng, 3)
    del score_rng, bc_rng

    # The "old" (sampling) model composes the EMA trainable leaves over the live
    # frozen leaves: frozen params are optimizer fixed points, so sourcing them
    # from `policy_state.params` is bit-identical and lets the EMA drop its frozen
    # duplicate (compose_full_params tolerates a full or trainable-only ema).
    if rl.use_ema_as_old_policy:
        old_params = compose_full_params(policy_state.params, ema, config.trainable_filter)
    else:
        old_params = policy_state.params
    old_model = nnx.merge(
        policy_state.model_def, jax.tree.map(jax.lax.stop_gradient, old_params)
    )
    old_model.eval()

    # Critic observation. The recompute lives behind the `critic_prefix is None`
    # static branch so WP-C can supply the buffer's EMA-computed sidecar instead;
    # the None branch reproduces the mono jit's recompute under current params.
    if critic_prefix is None:
        current_model = nnx.merge(policy_state.model_def, policy_state.params)
        current_model.eval()
        prefix_rep = current_model.get_prefix_rep(policy_observation)
        prefix_rep = prefix_rep[0] if isinstance(prefix_rep, tuple) else prefix_rep
        prefix = jnp.mean(
            prefix_rep.reshape((prefix_rep.shape[0], -1, prefix_rep.shape[-1])), axis=1
        )
    else:
        prefix = critic_prefix
    critic_observation = {
        "state": policy_observation.state,
        PREFIX_EMBEDDING_NAME: prefix,
    }

    # Critics are frozen w.r.t. this train_step.
    state_action_critic = create_critic(state_action_critic_state, config)
    state_action_critic.eval()
    value_critic = create_critic(value_state, config)
    value_critic.eval()

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
    q_logits = state_action_critic(
        expanded_critic_obs, flatten_action_horizon(sampled_actions)
    )
    v_logits = value_critic(expanded_critic_obs)
    q_value = summarize_critic_values(
        q_logits, config, critic_reduction=rl.critic.reduction
    )  # [B*G]
    v_value = summarize_critic_values(
        v_logits, config, critic_reduction=rl.critic.reduction
    )  # [B*G]
    if rl.advantage_combination == "conservative":
        # Per-head A_i = Q_i - V_i, combined sign-unanimously (mirrors the AWR
        # actor's conservative branch): positive only if every head agrees it's
        # positive (take the smallest), negative only if every head agrees it's
        # negative (take the smallest magnitude), zero on sign disagreement.
        assert rl.critic.num_qs == rl.critic.num_vs, (
            "conservative advantage needs num_qs == num_vs"
        )
        adv_heads = (
            critic_values_per_head(q_logits, config)
            - critic_values_per_head(v_logits, config)
        )  # (n, B*G)
        advantage_raw = jnp.maximum(jnp.min(adv_heads, axis=0), 0.0) + jnp.minimum(
            jnp.max(adv_heads, axis=0), 0.0
        )  # [B*G]
    else:
        advantage_raw = q_value - v_value  # [B*G]

    B = jax.tree.leaves(policy_observation)[0].shape[0]
    baseline = _group_baseline(advantage_raw, B, G, rl.adv_strategy)  # [B, 1]
    advantage = advantage_raw.reshape(B, G) - baseline
    if rl.adv_clip_min is not None:
        advantage = jnp.maximum(advantage, rl.adv_clip_min)
    advantage = advantage.reshape(-1)  # [B*G]
    advantage = jax.lax.stop_gradient(advantage)

    # q_mean/v_mean/advantage_* live here because they reduce q_value/v_value/
    # advantage — kept out of loss_fn's aux (jit-2) and the old info dict.
    sampler_aux = {
        "q_mean": jnp.mean(q_value),
        "v_mean": jnp.mean(v_value),
        "advantage_mean": jnp.mean(advantage),
        "advantage_max":  jnp.max(advantage),
        "advantage_min":  jnp.min(advantage),
        "advantage_std":  jnp.std(advantage),
        "advantage_q_up":  jnp.quantile(advantage, 0.95),
        "advantage_q_low": jnp.quantile(advantage, 0.05),
        "advantage_median": jnp.median(advantage),
    }
    return x_chain, x_next_chain, times, dt, old_lp, advantage, sampler_aux


@at.typecheck
def loss_and_grad_pg(
    config: OnlineTrainConfig,
    policy_state: training_utils.TrainState,    # UNDONATED (model_def, params-current)
    policy_observation: _model.Observation,     # un-expanded (rescore expands internally)
    x_chain: at.Float[at.Array, "k bg ah ad"],
    x_next_chain: at.Float[at.Array, "k bg ah ad"],
    times: at.Float[at.Array, "k bg"],
    dt: at.Float[at.Array, ""],
    old_lp: at.Float[at.Array, " bg"],
    advantage: at.Float[at.Array, " bg"],
) -> tuple[nnx.State, at.Float[at.Array, ""], dict[str, at.Array]]:
    # jit-2a: the PPO surrogate ONLY (no BC anchor). Emits the scan-accumulated
    # weight gradients (grads_pg) plus pg_loss and the PPO aux. The BC anchor's
    # separate full fp32 grad tree — which co-resided with this scan accumulator
    # in the single-jit `loss_and_grad` and forced the ~78.7 GiB jit-2 need (see
    # docs/plans/ogpo-memory/analysis-jit2-forensics.md) — is now emitted in the
    # SEPARATE jit-2b (`bc_grad_accumulate`), which accumulates it INTO these
    # donated grads. The two backward passes' activation sets are therefore never
    # co-resident: this jit holds only the rescoring scan's activations, jit-2b
    # only the BC forward/backward's.
    #
    # No `rng`: the PPO path (score_chain_under_model + surrogate) reads no rng.
    # bc_rng is derived from the SAME policy_rng in jit-2b (arity-3 split, [2]),
    # so dropping rng here does not shift the BC keystream (G1).
    assert isinstance(config.rl, OGPOSFTLearnerConfig)
    rl = config.rl
    noise_level = rl.noise_level

    # Re-derive log_prob_norm from x_chain.shape (not passed across the boundary)
    # so new_lp/old_lp share the exact per-dim divisor.
    K, _, H, D = x_chain.shape
    log_prob_norm = jnp.float32(1.0)
    if rl.normalize_denoising_horizon:
        log_prob_norm = log_prob_norm * jnp.float32(K * H)
    if rl.normalize_act_space_dimension:
        log_prob_norm = log_prob_norm * jnp.float32(D)

    def _expand(x):
        return jnp.repeat(x, repeats=rl.group_num_samples, axis=0)

    expanded_policy_obs = jax.tree.map(_expand, policy_observation)

    # --- 3a. PPO surrogate (gradient sink; BC anchor lives in jit-2b). ---
    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
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

        # 20 PPO keys — NO bc_loss (jit-2b adds bc_loss + grad_norm so the
        # learner/composer info dict is the same 33-key schema as before).
        aux = {
            "pg_loss": pg_loss,
            "pg_loss_unclipped": pg_loss_unclipped,
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
        }
        return pg_loss, aux

    policy_model = nnx.merge(policy_state.model_def, policy_state.params)
    policy_model.train()
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (pg_loss, pg_aux), grads_pg = nnx.value_and_grad(
        loss_fn, has_aux=True, argnums=diff_state
    )(policy_model)
    return grads_pg, pg_loss, pg_aux


@at.typecheck
def bc_grad_accumulate(
    config: OnlineTrainConfig,
    grads_pg: nnx.State,                        # DONATED (scan-accumulated PPO grads from jit-2a)
    rng: at.KeyArrayLike,                       # = policy_rng (SAME key as jit-1/jit-2a)
    policy_state: training_utils.TrainState,    # UNDONATED (model_def, params-current, PRE-increment step)
    policy_observation: _model.Observation,     # un-expanded (BC uses this directly)
    actions_demo: _model.Actions,
    pg_loss: at.Float[at.Array, ""],
    pg_aux: dict[str, at.Array],
) -> tuple[nnx.State, at.Float[at.Array, ""], dict[str, at.Array]]:
    # jit-2b: the CFM BC anchor. Differentiate bc_coeff*bc_loss and accumulate
    # the resulting grads INTO the donated grads_pg accumulator (donate_argnums
    # aliases the sum in-place). The two full fp32 grad trees are the only large
    # tensors this jit builds — the rescoring scan's activations (jit-2a) are
    # already freed — so the co-residency that sized the single jit-2 arena is
    # broken structurally, not by remat (which was measured ineffective; see the
    # J-2-revert commit / AFTERJ2_B32 artifacts).
    #
    # RNG contract (G1): reproduce the mono derivation EXACTLY — bc_rng is the
    # arity-3 split of the SAME policy_rng at index [2], folded into the
    # PRE-increment policy_state.step (jit-3 does the increment). jit-2a consumed
    # no rng, so the model's rng state entering compute_loss here is the
    # freshly-merged initial state, identical to the mono's single merge.
    assert isinstance(config.rl, OGPOSFTLearnerConfig)
    rl = config.rl

    if rl.use_bc_regularization:
        _, _, bc_rng = jax.random.split(rng, 3)
        policy_model = nnx.merge(policy_state.model_def, policy_state.params)
        policy_model.train()
        train_rng = jax.random.fold_in(bc_rng, policy_state.step)

        @at.typecheck
        def bc_loss_fn(
            model: _model.BaseModel,
            bc_rng: at.KeyArrayLike,
        ) -> tuple[at.Float[at.Array, ""], at.Float[at.Array, ""]]:
            # BC on the un-expanded online batch (size B). Return the bc_coeff-
            # scaled loss as the differentiand so grads_bc == bc_coeff * d(bc)/dθ
            # — exactly the BC contribution the mono's value_and_grad over
            # (pg + bc_coeff*bc) produced. The raw bc_loss rides along as aux.
            bc_loss = jnp.mean(
                model.compute_loss(bc_rng, policy_observation, actions_demo, train=True)
            )
            return rl.bc_coeff * bc_loss, bc_loss

        diff_state = nnx.DiffState(0, config.trainable_filter)
        (_, bc_loss), grads_bc = nnx.value_and_grad(
            bc_loss_fn, has_aux=True, argnums=diff_state
        )(policy_model, train_rng)
        # Accumulate into the DONATED grads_pg: same 2-input per-leaf add the
        # mono did (grad_pg + bc_coeff*grad_bc), only reassociated across the jit
        # boundary — floating-point ulps, certified at atol=1e-6 by
        # tests/ogpo/test_split_equivalence.py.
        grads = jax.tree.map(lambda g_pg, g_bc: g_pg + g_bc, grads_pg, grads_bc)
    else:
        # use_bc_regularization=False: no BC term. grads pass through unchanged,
        # bc_loss=0, grad_norm over the PPO tree — matches the mono False path.
        bc_loss = jnp.float32(0.0)
        grads = grads_pg

    # Total loss = pg + bc_coeff*bc (bit-identical to the mono `loss`); grad_norm
    # is computed over the COMBINED grads (post-accumulation), not the PPO-only
    # tree. loss_aux restores the 22-key jit-2 aux schema (20 PPO + bc_loss +
    # grad_norm).
    loss = pg_loss + rl.bc_coeff * bc_loss
    loss_aux = pg_aux | {"bc_loss": bc_loss, "grad_norm": optax.global_norm(grads)}
    return grads, loss, loss_aux


def optimizer_tail(
    config: OnlineTrainConfig,
    policy_state: training_utils.TrainState,   # DONATED (params + opt_state + step)
    grads: nnx.State,                          # DONATED (trainable tree from jit-2)
) -> training_utils.TrainState:
    # --- 4. Optimizer step (mirrors awr/flow_grpo). --------------------
    trainable = nnx.filter_state(policy_state.params, config.trainable_filter)
    updates, new_opt_state = policy_state.tx.update(grads, policy_state.opt_state, trainable)
    new_trainable = optax.apply_updates(trainable, updates)

    # Splice replaces nnx.update(policy_model, new_params) + nnx.state(policy_model):
    # frozen + non-Param leaves are optimizer fixed points, so sourcing them from
    # the (donated) input params is bit-identical to re-extracting them via the
    # round-trip, and it lets XLA alias the frozen leaves in place instead of
    # rebuilding the full tree.
    frozen_and_rest = nnx.filter_state(policy_state.params, nnx.Not(config.trainable_filter))
    new_full_params = nnx.merge_state(frozen_and_rest, new_trainable)

    # ema_params/ema_decay are not passed to replace, so they pass through
    # unchanged from the donated policy_state (None in production; the EMA is
    # learner-managed by WP-B).
    return dataclasses.replace(
        policy_state,
        step=policy_state.step + 1,
        params=new_full_params,
        opt_state=new_opt_state,
    )


def policy_param_norm(
    config: OnlineTrainConfig, params: nnx.State
) -> at.Float[at.Array, ""]:
    # Hoisted into its own jit so the param tree's live range never re-enters the
    # binding optimizer tail. nnx.state(model, filter) == filter_state(nnx.state(
    # model), filter); `params` is already nnx.state, so this is equivalent.
    kernel_params = nnx.filter_state(
        params,
        nnx.All(
            nnx.Param,
            nnx.Not(
                nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")
            ),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    return optax.global_norm(kernel_params)


@at.typecheck
def train_step(
    config: OnlineTrainConfig,
    rng: at.KeyArrayLike,
    policy_state: training_utils.TrainState,
    state_action_critic_state: training_utils.TrainState,
    value_state: training_utils.TrainState,
    batch: tuple[_model.Observation, ObsType, _model.Actions],
    mc_return: at.Array | None = None,
    is_success: at.Array | None = None,
    scale: at.Array | float = 1.0,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    del mc_return, is_success, scale  # OGPO uses Q-V advantages; AWR-style MC normalization unused.

    policy_observation, critic_observation, actions_demo = batch
    # The prefix is already built upstream (_sft_batch_to_actor_batch); feed it as
    # the sidecar so the composer matches the pre-split behaviour (the recompute
    # happened OUTSIDE this function). ema_params passes through unchanged.
    critic_prefix = critic_observation[PREFIX_EMBEDDING_NAME]
    ema = policy_state.ema_params if policy_state.ema_params is not None else policy_state.params

    x_chain, x_next_chain, times, dt, old_lp, advantage, sampler_aux = sample_and_advantage(
        config, rng, policy_state, state_action_critic_state, value_state,
        policy_observation, critic_prefix, ema,
    )
    # Two-pass loss backward: jit-2a emits the PPO scan grads; jit-2b differentiates
    # the BC anchor and accumulates INTO those (donated) grads. grads_pg is never
    # co-resident with the BC backward's activations (the jit-2 arena fix).
    grads_pg, pg_loss, pg_aux = loss_and_grad_pg(
        config, policy_state, policy_observation,
        x_chain, x_next_chain, times, dt, old_lp, advantage,
    )
    grads, loss, loss_aux = bc_grad_accumulate(
        config, grads_pg, rng, policy_state, policy_observation, actions_demo,
        pg_loss, pg_aux,
    )
    new_state = optimizer_tail(config, policy_state, grads)
    param_norm = policy_param_norm(config, new_state.params)
    info = {"loss": loss, "param_norm": param_norm} | loss_aux | sampler_aux
    return new_state, info
