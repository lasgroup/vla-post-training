# ruff: noqa: F722
"""Split-vs-mono equivalence for the OGPO actor step (B.7).

The jit bodies (``sample_and_advantage`` → ``loss_and_grad_pg`` →
``bc_grad_accumulate`` → ``optimizer_tail`` + ``policy_param_norm``) and the
retained composed ``train_step`` must each reproduce the pre-split mono
``train_step`` (the loss backward is two passes: PPO surrogate grads, then the BC
anchor accumulated into them). The
reference is a VERBATIM copy of the pre-refactor ``update_actor.train_step``
(``_reference_mono_train_step``, additive-return-only) — a monolithic
implementation that cannot share the split's structural lift error, so it breaks
the split==composer circularity (B1). Both the split pipeline and the composer
are checked against it in two regimes: ``ema == params`` (leg a, the step-900
OOM state) and a trainable-perturbed ``ema != params`` (leg a', the compose
lock, B3). Plus: the splice is structurally identical to the ``nnx.update`` /
``nnx.state`` round-trip (leg b), the RNG re-split is arity-3 with ``[0]``/``[2]``
(leg c), and the ``critic_prefix is None`` recompute reproduces the original
prefix (leg d).
"""
import dataclasses
import functools
import types

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
from openpi.models import pi0_config

from src.rl.advantage_weighted_sft.update_critic import (
    create_critic,
    flatten_action_horizon,
    init_state_action_critic_train_state,
    init_state_value_train_state,
    summarize_critic_values,
)
from src.rl.ogpo.ogpo_learner import OGPOAgentLearner
from src.rl.ogpo.sampling import sample_chain_with_logprob, score_chain_under_model, sum_log_prob
from src.rl.ogpo.update_actor import (
    _group_baseline,
    bc_grad_accumulate,
    loss_and_grad_pg,
    optimizer_tail,
    policy_param_norm,
    sample_and_advantage,
    train_step,
)
from src.rl.prefix_embedding import PREFIX_EMBEDDING_NAME
from src.training.config import OGPOSFTLearnerConfig, get_config

_B = 2
_ATOL = 1e-6
# 33 keys today: loss + grad_norm + param_norm + 7 advantage_* + q_mean/v_mean +
# 21 ratio/kl/pg/bc/log_prob (validates B.6's "info == today's 33 keys exactly").
# Asserted on the REFERENCE monolith's info, which is frozen — this stays 33.
_N_INFO_KEYS = 33
# Keys the split emits that the frozen reference monolith predates. See
# docs/changes/2026-08-15-actor-pg-bc-grad-norms/. Diagnostics only: they are
# reductions over gradient trees the reference also built, it just never
# reported them separately.
_SPLIT_ONLY_INFO_KEYS = {"grad_norm_pg", "grad_norm_bc", "grad_cos_pg_bc"}


def _build_config():
    # subtract_v (the production adv_strategy) keeps advantage = q - v non-zero at
    # G=1; the vanilla group-mean baseline would collapse advantage to 0 there and
    # blind the PPO path. use_ema_as_old_policy=True makes leg (a') non-vacuous.
    base = get_config("pi05_libero_online_ogpo_sft")
    dummy_model = pi0_config.Pi0Config(
        paligemma_variant="dummy", action_expert_variant="dummy",
        action_dim=4, action_horizon=2, max_token_len=8, pi05=True,
    )
    critic = dataclasses.replace(
        base.rl.critic, use_bronet=True, bronet_hidden_dim=32, bronet_depth=1,
        num_qs=2, num_vs=2,
    )
    rl = dataclasses.replace(
        base.rl, critic=critic, group_num_samples=1, num_sde_steps=3,
        adv_strategy="subtract_v", use_ema_as_old_policy=True,
    )
    config = dataclasses.replace(
        base, model=dummy_model, rl=rl, batch_size=_B,
        freeze_filter=nnx_utils.PathRegex(".*PaliGemma/img.*"),
    )
    assert isinstance(config.rl, OGPOSFTLearnerConfig)
    assert config.rl.use_ema_as_old_policy is True
    return config


def _make_params(config, model):
    params = nnx.state(model)
    # Frozen (SigLIP) leaves cast to bf16, mirroring init_train_state (FSL:159-164).
    return nnx_utils.state_map(
        params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16))
    )


def _build_policy_state(config, model, params, *, ema_params):
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)
    opt_state = tx.init(nnx.filter_state(params, config.trainable_filter))
    return training_utils.TrainState(
        step=900, params=params, model_def=nnx.graphdef(model), tx=tx,
        opt_state=opt_state, ema_decay=None, ema_params=ema_params,
    )


def _build_critics(config, model, mesh, rng):
    fake_obs1 = config.model.fake_obs(batch_size=1)
    prefix_rep = model.get_prefix_rep(fake_obs1)[0]
    embed_shape = tuple(prefix_rep.shape[2:])
    dummy_obs = {"state": fake_obs1.state,
                 PREFIX_EMBEDDING_NAME: jnp.zeros((1, *embed_shape), dtype=jnp.float32)}
    dummy_act = config.model.fake_act(batch_size=1)
    from src.rl.networks.bronet_critic import BroNetStateActionCritic, BroNetStateValue
    hd, dep = config.rl.critic.bronet_hidden_dim, config.rl.critic.bronet_depth
    nq, nv = config.rl.critic.num_qs, config.rl.critic.num_vs

    def sa_def(observation, action, rngs):
        return BroNetStateActionCritic(observation=observation, action=action, hidden_dim=hd, depth=dep, num_qs=nq, rngs=rngs)

    def sv_def(observation, rngs):
        return BroNetStateValue(observation=observation, hidden_dim=hd, depth=dep, num_vs=nv, rngs=rngs)

    q_rng, v_rng = jax.random.split(rng, 2)
    q_state, _ = init_state_action_critic_train_state(config, q_rng, mesh, critic_def=sa_def, dummy_obs=dummy_obs, dummy_act=dummy_act)
    value_state, _ = init_state_value_train_state(config, v_rng, mesh, critic_def=sv_def, dummy_obs=dummy_obs)
    return q_state, value_state


def _original_recompute(model, policy_observation):
    # Transcribed from the ORIGINAL recompute (AWR:227-229 + _get_prefix_rep_with_model_fn
    # FSL:332-334), NOT the new jit-1 body — leg (d) checks jit-1 reproduces THIS.
    prefix_rep = model.get_prefix_rep(policy_observation)
    prefix_rep = prefix_rep[0] if isinstance(prefix_rep, tuple) else prefix_rep
    prefix = prefix_rep.reshape((prefix_rep.shape[0], -1, prefix_rep.shape[-1]))
    return jnp.mean(prefix, axis=1)


def _reference_mono_train_step(
    config, rng, policy_state, state_action_critic_state, value_state, batch,
    mc_return=None, is_success=None, scale=1.0,
):
    # VERBATIM copy of update_actor.train_step at 1f897a9 (the body of UA:60-302),
    # the independent monolithic baseline for B1. The ONLY deviation is additive:
    # the six intermediates are appended to the return tuple. No computation edit.
    del mc_return, is_success, scale  # OGPO uses Q-V advantages; AWR-style MC normalization unused.

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
        config,
        critic_reduction=rl.critic.reduction,
    )  # [B*G]
    v_value = summarize_critic_values(
        value_critic(expanded_critic_obs),
        config,
        critic_reduction=rl.critic.reduction,
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
    return new_state, info, x_chain, x_next_chain, times, dt, old_lp, advantage


@pytest.fixture(scope="module")
def fx():
    config = _build_config()
    model = config.model.create(jax.random.key(0))
    params = _make_params(config, model)
    mesh = sharding.make_mesh(1)
    q_state, value_state = _build_critics(config, model, mesh, jax.random.key(1))
    policy_observation = config.model.fake_obs(batch_size=_B)
    actions_demo = config.model.fake_act(batch_size=_B)
    # Recompute the fixed sidecar prefix from the SAME (frozen-bf16) params the
    # policy state carries, so jit-1's None-branch recompute matches it (leg d).
    model_cast = nnx.merge(nnx.graphdef(model), params)
    model_cast.eval()
    prefix = _original_recompute(model_cast, policy_observation)
    return types.SimpleNamespace(
        config=config, model=model, params=params, q_state=q_state, value_state=value_state,
        policy_observation=policy_observation, actions_demo=actions_demo, prefix=prefix,
        ref_jit=jax.jit(functools.partial(_reference_mono_train_step, config)),
        sa_jit=jax.jit(functools.partial(sample_and_advantage, config)),
        lg_pg_jit=jax.jit(functools.partial(loss_and_grad_pg, config)),
        bc_acc_jit=jax.jit(functools.partial(bc_grad_accumulate, config)),
        ot_jit=jax.jit(functools.partial(optimizer_tail, config)),
        pn_jit=jax.jit(functools.partial(policy_param_norm, config)),
        comp_jit=jax.jit(functools.partial(train_step, config)),
    )


def _assert_close(a, b, name):
    a, b = np.asarray(a), np.asarray(b)
    max_abs = float(np.max(np.abs(a - b))) if a.size else 0.0
    assert np.allclose(a, b, atol=_ATOL, rtol=0.0), f"{name}: max|Δ|={max_abs} exceeds atol={_ATOL}"


def _assert_tree_close(a, b, name):
    la, ta = jax.tree_util.tree_flatten(a)
    lb, tb = jax.tree_util.tree_flatten(b)
    assert ta == tb, f"{name}: tree structure mismatch"
    bad = [(i, float(np.max(np.abs(np.asarray(x) - np.asarray(y)))))
           for i, (x, y) in enumerate(zip(la, lb))
           if not np.allclose(np.asarray(x), np.asarray(y), atol=_ATOL, rtol=0.0)]
    assert not bad, f"{name}: {len(bad)} leaf(s) exceed atol={_ATOL} (flag for review): {bad[:6]}"


def _assert_info_close(got, ref, name):
    # The reference is a VERBATIM copy of the pre-split monolith (see module
    # docstring) and is deliberately never edited, so keys added to the split
    # after the refactor legitimately have no counterpart in it. Every reference
    # key must still be present and numerically equal; the extras must be
    # EXACTLY the known set, so an unintended third addition still fails here.
    missing = set(ref) - set(got)
    assert not missing, f"{name}: info keys missing from split: {missing}"
    extra = set(got) - set(ref)
    assert extra == _SPLIT_ONLY_INFO_KEYS, (
        f"{name}: unexpected split-only info keys: {extra ^ _SPLIT_ONLY_INFO_KEYS}"
    )
    for k in ref:
        _assert_close(got[k], ref[k], f"{name}:info[{k}]")


def _run_split(fx, policy_state, ema):
    rng = jax.random.key(7)
    x_chain, x_next_chain, times, dt, old_lp, advantage, sampler_aux = fx.sa_jit(
        rng, policy_state, fx.q_state, fx.value_state, fx.policy_observation, fx.prefix, ema,
    )
    # Two-pass loss backward: jit-2a (PPO scan grads) -> jit-2b (BC anchor accumulated
    # into grads_pg). Routes the certification through the real Option-2 structure.
    grads_pg, pg_loss, pg_aux = fx.lg_pg_jit(
        policy_state, fx.policy_observation,
        x_chain, x_next_chain, times, dt, old_lp, advantage,
    )
    grads, loss, loss_aux = fx.bc_acc_jit(
        grads_pg, rng, policy_state, fx.policy_observation, fx.actions_demo,
        pg_loss, pg_aux,
    )
    new_state = fx.ot_jit(policy_state, grads)
    param_norm = fx.pn_jit(new_state.params)
    info = {"loss": loss, "param_norm": param_norm} | loss_aux | sampler_aux
    return types.SimpleNamespace(
        new_state=new_state, info=info, grads=grads,
        intermediates=(x_chain, x_next_chain, times, dt, old_lp, advantage),
    )


def _run_reference(fx, policy_state):
    rng = jax.random.key(7)
    critic_observation = {"state": fx.policy_observation.state, PREFIX_EMBEDDING_NAME: fx.prefix}
    batch = (fx.policy_observation, critic_observation, fx.actions_demo)
    new_state, info, *intermediates = fx.ref_jit(rng, policy_state, fx.q_state, fx.value_state, batch)
    return types.SimpleNamespace(new_state=new_state, info=info, intermediates=tuple(intermediates))


def _run_composer(fx, policy_state):
    rng = jax.random.key(7)
    critic_observation = {"state": fx.policy_observation.state, PREFIX_EMBEDDING_NAME: fx.prefix}
    batch = (fx.policy_observation, critic_observation, fx.actions_demo)
    new_state, info = fx.comp_jit(rng, policy_state, fx.q_state, fx.value_state, batch)
    return types.SimpleNamespace(new_state=new_state, info=info)


def _compare_all(fx, ema, policy_state):
    ref = _run_reference(fx, policy_state)
    assert len(ref.info) == _N_INFO_KEYS
    split = _run_split(fx, policy_state, ema)
    comp = _run_composer(fx, policy_state)

    names = ("x_chain", "x_next_chain", "times", "dt", "old_lp", "advantage")
    for got, exp, name in zip(split.intermediates, ref.intermediates, names):
        _assert_close(got, exp, f"split:{name}")
    _assert_info_close(split.info, ref.info, "split")
    _assert_tree_close(split.new_state.params, ref.new_state.params, "split:params")

    # The composer returns (new_state, info) only; its intermediates surface
    # transitively through info + params (it calls the same four bodies).
    _assert_info_close(comp.info, ref.info, "composer")
    _assert_tree_close(comp.new_state.params, ref.new_state.params, "composer:params")


def test_leg_a_split_and_composer_match_reference_ema_equals_params(fx):
    # Step-900 OOM regime: ema == params. Any operand order/filter polarity yields
    # `params` here, so this leg is blind to a compose swap — leg (a') supplies that.
    ema = fx.params
    policy_state = _build_policy_state(fx.config, fx.model, fx.params, ema_params=ema)
    _compare_all(fx, ema, policy_state)


def test_leg_a_prime_split_and_composer_match_reference_perturbed_ema(fx):
    # ema' perturbs ONLY the trainable leaves, frozen == params: the reference builds
    # its old model by raw nnx.merge(model_def, ema'), so reference and the composed
    # split (frozen-from-params) agree by construction only when ema'.frozen == params.
    # A swapped/inverted compose gives the split the CURRENT trainable weights ->
    # x_chain/old_lp diverge from the reference -> caught. The 1e-3 magnitude keeps
    # the correct split bit-clean against the reference (no reduction-order drift in
    # the off-policy grad_norm aggregate) while a swap still diverges by ~1.7e-2.
    tf = fx.config.trainable_filter
    trainable = nnx.filter_state(fx.params, tf)
    trainable = jax.tree.map(lambda x: x + jnp.asarray(1e-3, dtype=x.dtype), trainable)
    ema_prime = nnx.merge_state(nnx.filter_state(fx.params, nnx.Not(tf)), trainable)
    policy_state = _build_policy_state(fx.config, fx.model, fx.params, ema_params=ema_prime)
    _compare_all(fx, ema_prime, policy_state)


def test_leg_b_splice_structural_identity(fx):
    ema = fx.params
    policy_state = _build_policy_state(fx.config, fx.model, fx.params, ema_params=ema)
    split = _run_split(fx, policy_state, ema)

    # Reconstruct the pre-split nnx.update(policy_model, new_params) + nnx.state path.
    tf = fx.config.trainable_filter
    policy_model = nnx.merge(policy_state.model_def, policy_state.params)
    trainable = nnx.filter_state(policy_state.params, tf)
    updates, _ = policy_state.tx.update(split.grads, policy_state.opt_state, trainable)
    new_trainable = optax.apply_updates(trainable, updates)
    nnx.update(policy_model, new_trainable)
    mono_params = nnx.state(policy_model)

    assert jax.tree_util.tree_structure(split.new_state.params) == jax.tree_util.tree_structure(mono_params)
    split_flat = dict(split.new_state.params.flat_state())
    mono_flat = dict(mono_params.flat_state())
    assert set(split_flat) == set(mono_flat), "spliced key set differs from the round-trip"
    for k in mono_flat:
        assert type(split_flat[k]) is type(mono_flat[k]), f"VariableState type mismatch at {k}"
    # The splice claim is only load-bearing when the frozen complement is non-empty.
    frozen = nnx.filter_state(policy_state.params, nnx.Not(tf))
    assert len(list(frozen.flat_state())) > 0


def test_leg_c_rng_arity_three_and_indexing(fx):
    # jit-1 draws sample_rng from split(rng, 3)[0]; jit-2 folds bc_rng =
    # split(rng, 3)[2] into the pre-increment step. threefry makes split(rng, 3)[0]
    # == split(rng, 2)[0], so the arity-2 partial break (dropping the dead score_rng
    # slot) is invisible in sample_rng and surfaces only in bc_rng ([2] shifting to
    # [1]) — the real discriminator, also caught by leg (a)'s bc_loss comparison.
    ema = fx.params
    policy_state = _build_policy_state(fx.config, fx.model, fx.params, ema_params=ema)
    split = _run_split(fx, policy_state, ema)
    x_chain_split = split.intermediates[0]
    rng = jax.random.key(7)
    rl = fx.config.rl

    old_model = nnx.merge(policy_state.model_def, jax.tree.map(jax.lax.stop_gradient, fx.params))
    old_model.eval()
    chain3 = sample_chain_with_logprob(
        old_model, fx.policy_observation, rng=jax.random.split(rng, 3)[0],
        num_steps=rl.num_sde_steps, noise_level=rl.noise_level,
    )
    _assert_close(chain3["x_chain"], x_chain_split, "jit1 uses split(rng,3)[0]")

    # jit-2 folds split(rng,3)[2]; the arity-2 break (bc_rng=split(rng,2)[1]) must NOT
    # reproduce the same BC loss. Rebuild the model per call (compute_loss consumes
    # the merged model's RNG state) so each reads the same initial state.
    def _bc(bc_rng):
        pm = nnx.merge(policy_state.model_def, policy_state.params)
        pm.train()
        train_rng = jax.random.fold_in(bc_rng, policy_state.step)
        return jnp.mean(pm.compute_loss(train_rng, fx.policy_observation, fx.actions_demo, train=True))

    _assert_close(_bc(jax.random.split(rng, 3)[2]), split.info["bc_loss"], "jit2 folds split(rng,3)[2] pre-increment")
    assert not np.allclose(np.asarray(_bc(jax.random.split(rng, 2)[1])), np.asarray(split.info["bc_loss"]), atol=_ATOL), \
        "arity-2 bc_rng=[1] must NOT reproduce the BC loss (the dead score_rng slot is load-bearing)"


def test_leg_d_none_branch_recompute_matches_original(fx):
    # jit-1's critic_prefix=None branch must reproduce the original recompute
    # (_original_recompute above, transcribed from AWR:227-229 + FSL:332-334): feeding
    # that exact prefix as the sidecar must give the same advantage/q_mean as None.
    ema = fx.params
    policy_state = _build_policy_state(fx.config, fx.model, fx.params, ema_params=ema)
    rng = jax.random.key(7)
    out_none = fx.sa_jit(rng, policy_state, fx.q_state, fx.value_state, fx.policy_observation, None, ema)
    out_side = fx.sa_jit(rng, policy_state, fx.q_state, fx.value_state, fx.policy_observation, fx.prefix, ema)
    _assert_close(out_none[5], out_side[5], "None-branch advantage == recompute-sidecar advantage")
    _assert_close(out_none[6]["q_mean"], out_side[6]["q_mean"], "None-branch q_mean == recompute-sidecar q_mean")


def test_g_stored_prefix_threading(fx):
    # WP-C Phase G: the OGPO 3-tuple override threads the buffer's stored prefix as a sidecar past
    # Observation.from_dict, and jit-1 actually consumes it (the one intended non-bit-identical
    # change). The override never touches self, so call it unbound with a stub self.
    obs_dict = fx.policy_observation.to_dict()
    known_prefix = fx.prefix + 2.0
    obs_dict[PREFIX_EMBEDDING_NAME] = known_prefix
    online_batch = {"observation": obs_dict, "actions": fx.actions_demo}
    _, _, sidecar = OGPOAgentLearner._online_batch_to_sft_batch(None, online_batch)
    assert sidecar is known_prefix, "override must return the stored prefix array as the third element"

    obs_no_prefix = fx.policy_observation.to_dict()
    _, _, absent = OGPOAgentLearner._online_batch_to_sft_batch(
        None, {"observation": obs_no_prefix, "actions": fx.actions_demo}
    )
    assert absent is None, "override returns None when the prefix key is absent (store_prefix_rep off)"

    # The sidecar drives the critic observation: a perturbed prefix must change q_mean/advantage vs
    # the None-recompute path (guards jit-1 against silently dropping arg5). leg (d) separately pins
    # that the EXACT recompute value reproduces the None-path outputs.
    ema = fx.params
    policy_state = _build_policy_state(fx.config, fx.model, fx.params, ema_params=ema)
    rng = jax.random.key(7)
    out_none = fx.sa_jit(rng, policy_state, fx.q_state, fx.value_state, fx.policy_observation, None, ema)
    out_perturbed = fx.sa_jit(
        rng, policy_state, fx.q_state, fx.value_state, fx.policy_observation, fx.prefix + 1.0, ema
    )
    assert not np.allclose(np.asarray(out_none[6]["q_mean"]), np.asarray(out_perturbed[6]["q_mean"]), atol=_ATOL), \
        "perturbed sidecar must change q_mean vs the None-recompute path"
    assert not np.allclose(np.asarray(out_none[5]), np.asarray(out_perturbed[5]), atol=_ATOL), \
        "perturbed sidecar must change advantage vs the None-recompute path"
