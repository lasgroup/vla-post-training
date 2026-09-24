# ruff: noqa: F722
"""SDE chain sampling and rescoring helpers for OGPO on a Pi05 flow policy.

OGPO requires two coupled operations on the flow policy:

1. ``sample_chain_with_logprob``: draw an SDE rollout from a fixed (old) set
   of weights, returning the clean action chunk together with the entire
   noising trajectory and the per-step Gaussian log-prob under those weights.
2. ``score_chain_under_model``: re-evaluate the log-prob of a frozen chain
   under a *different* (current) set of weights. The ratio of these two
   log-probs is the PPO importance weight.

Both functions speak Pi05's flow head idioms: dt = -1/num_steps (time runs
from 1 → 0), and the per-step transition is a closed-form
``MultivariateNormalDiag`` whose mean is the v_t-conditioned Euler step and
whose scale_diag is ``noise_level * sqrt(t/(1-t)) * sqrt(|dt|)``.
"""
from typing import Any

import einops
import jax
import jax.numpy as jnp

import openpi.models.model as _model
import openpi.models.pi0 as _pi0
import openpi.shared.array_typing as at
from openpi.models import gemma as _gemma
from openpi.models.pi0 import make_attn_mask


# ---------------------------------------------------------------------------
# Prefix-cache helpers.
#
# These mirror what would otherwise be ``Pi0`` methods, but live here in the
# parent repo so the openpi submodule stays clean. They reach into Pi0's
# public surface (embed_prefix/embed_suffix/PaliGemma.llm/action_out_proj)
# plus the convention-private ``_get_sde_dist``; numerically identical to
# the in-method version.
# ---------------------------------------------------------------------------

def compute_prefix_cache(
    model: _pi0.Pi0,
    observation: _model.Observation,
) -> tuple[_gemma.KVCache, at.Bool[at.Array, "b _p"]]:
    """Run a single prefix forward and return the per-layer KV cache.

    Output can be reused across many suffix-only forwards (e.g. the 10 SDE
    rescoring steps in OGPO's PPO surrogate), avoiding redundant PaliGemma
    evaluations and the activation-memory blow-up they cause. The prefix
    mask is returned alongside because suffix-side attention masks need to
    know which prefix positions are valid.

    Gradients flow through ``kv_cache`` exactly as they would through a
    normal forward pass — so if the PaliGemma backbone is unfrozen later,
    the cached path is still numerically equivalent.
    """
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    (_, _), kv_cache = model.PaliGemma.llm(
        [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
    )
    return kv_cache, prefix_mask


def get_dist_and_log_prob_with_cache(
    model: _pi0.Pi0,
    *,
    x_t: at.Float[at.Array, "batch horizon action_dim"],
    sample: at.Float[at.Array, "batch horizon action_dim"],
    time: at.Float[at.Array, " batch"],
    observation: _model.Observation,
    kv_cache: _gemma.KVCache,
    prefix_mask: at.Bool[at.Array, "b _p"],
    dt: at.Float[at.Array, ""],
    noise_level: float = 0.7,
):
    """Same as Pi0.get_dist_and_log_prob but reusing a precomputed prefix.

    The KV cache and prefix mask come from a prior ``compute_prefix_cache``
    call on the same observation under the same model parameters. Only the
    action-expert / suffix side of the model is re-run here — the PaliGemma
    prefix path is replayed from the cache.
    """
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
        observation, x_t, time
    )
    suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_attn_mask_b = einops.repeat(
        prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
    )
    full_attn_mask = jnp.concatenate(
        [prefix_attn_mask_b, suffix_attn_mask], axis=-1
    )
    positions = (
        jnp.sum(prefix_mask, axis=-1)[:, None]
        + jnp.cumsum(suffix_mask, axis=-1)
        - 1
    )

    (prefix_out, suffix_out), _ = model.PaliGemma.llm(
        [None, suffix_tokens],
        mask=full_attn_mask,
        positions=positions,
        kv_cache=kv_cache,
        adarms_cond=[None, adarms_cond],
    )
    assert prefix_out is None
    v_t = model.action_out_proj(suffix_out[:, -model.action_horizon:])
    dist = model._get_sde_dist(
        x_t=x_t, v_t=v_t, time=time, dt=dt, noise_level=noise_level
    )
    return dist.log_prob(sample), dist


def get_sde_dist_with_cache(
    model: _pi0.Pi0,
    *,
    x_t: at.Float[at.Array, "batch horizon action_dim"],
    time: at.Float[at.Array, " batch"],
    observation: _model.Observation,
    kv_cache: _gemma.KVCache,
    prefix_mask: at.Bool[at.Array, "b _p"],
    dt: at.Float[at.Array, ""],
    noise_level: float = 0.7,
):
    """SDE transition distribution at (x_t, time), reusing a precomputed prefix.

    The suffix-side forward of ``get_dist_and_log_prob_with_cache`` without the
    final ``log_prob(sample)`` — used by the group-deduplicated sampler, which
    needs the distribution to SAMPLE from (then scores the draw itself).
    """
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
        observation, x_t, time
    )
    suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_attn_mask_b = einops.repeat(
        prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
    )
    full_attn_mask = jnp.concatenate(
        [prefix_attn_mask_b, suffix_attn_mask], axis=-1
    )
    positions = (
        jnp.sum(prefix_mask, axis=-1)[:, None]
        + jnp.cumsum(suffix_mask, axis=-1)
        - 1
    )
    (prefix_out, suffix_out), _ = model.PaliGemma.llm(
        [None, suffix_tokens],
        mask=full_attn_mask,
        positions=positions,
        kv_cache=kv_cache,
        adarms_cond=[None, adarms_cond],
    )
    assert prefix_out is None
    v_t = model.action_out_proj(suffix_out[:, -model.action_horizon:])
    return model._get_sde_dist(
        x_t=x_t, v_t=v_t, time=time, dt=dt, noise_level=noise_level
    )


def repeat_prefix_cache(
    kv_cache: _gemma.KVCache,
    prefix_mask: at.Bool[at.Array, "b _p"],
    group: int,
) -> tuple[_gemma.KVCache, at.Bool[at.Array, "bg _p"]]:
    """Tile a batch-B prefix cache to batch B*G, matching jnp.repeat(obs, G, 0).

    KVCache leaves are [layers, batch, tokens, k, h] — batch is AXIS 1; the
    prefix mask is [batch, tokens] — axis 0. jnp.repeat keeps group members
    adjacent ([s0,s0,...,s1,s1,...]), the layout `_group_baseline`'s
    reshape(B, G) assumes. Valid because attention is strictly per-sequence:
    repeat(prefix(obs)) == prefix(repeat(obs)) with no cross-batch coupling
    (no batch norm; deterministic/no-dropout forward — same argument as the
    Phase-J scan-ify).
    """
    kv_rep = jax.tree.map(lambda x: jnp.repeat(x, group, axis=1), kv_cache)
    mask_rep = jnp.repeat(prefix_mask, group, axis=0)
    return kv_rep, mask_rep


# ---------------------------------------------------------------------------
# Sampling: draw an SDE rollout under (typically) the EMA / "old" policy.
# ---------------------------------------------------------------------------

def sample_chain_with_logprob(
    model: _pi0.Pi0,
    observation: _model.Observation,
    rng: at.KeyArrayLike,
    *,
    num_steps: int,
    noise_level: float,
    noise: at.Float[at.Array, "b ah ad"] | None = None,
) -> dict[str, jax.Array]:
    """Run Pi05's stochastic flow sampler and unpack the trajectory.

    Returns a dict with:
        actions      : [B, H, D]            -- final clean action chunk (x_0)
        x_chain      : [num_steps, B, H, D] -- x_t at the *start* of each step
        x_next_chain : [num_steps, B, H, D] -- x_t at the *end*   of each step
        times        : [num_steps, B]       -- start time of each step
        dt           : scalar               -- -1.0 / num_steps
        log_prob_per_step : [num_steps, B, H]  -- Gaussian log-prob under
            the *sampling* (=`model`) weights for each step. Sum across the
            chain dim and the chunk dim H to get the joint log-prob of the
            rollout under these weights.
    """
    if noise_level <= 0.0:
        raise ValueError(
            f"OGPO requires stochastic sampling (noise_level > 0); got {noise_level}."
        )

    if noise is None:
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(
            rng, (batch_size, model.action_horizon, model.action_dim)
        )

    actions, outs = model.sample_actions(
        rng=rng,
        observation=observation,
        noise=noise,
        num_steps=num_steps,
        noise_level=noise_level,
        return_info_dict=True,
    )

    dt = jnp.asarray(-1.0 / num_steps, dtype=noise.dtype)
    return {
        "actions": actions,
        "x_chain": outs["x"],
        "x_next_chain": outs["x_next"],
        "times": outs["time"],
        "dt": dt,
        "log_prob_per_step": outs["log_prob"],
    }


def sample_chain_with_logprob_grouped(
    model: _pi0.Pi0,
    observation: _model.Observation,
    rng: at.KeyArrayLike,
    *,
    group: int,
    num_steps: int,
    noise_level: float,
) -> dict[str, jax.Array]:
    """Group-deduplicated ``sample_chain_with_logprob``.

    Semantically identical to ``sample_chain_with_logprob(model,
    jnp.repeat(observation, group, 0), rng, ...)`` — same preprocessing, same
    noise key, same per-step RNG split structure, same scan — except the
    PaliGemma prefix (SigLIP + Gemma over images/text) runs ONCE at batch B and
    its KV cache is tiled to B*group, instead of running ``group`` redundant
    times on identical observations. Pure performance: no algorithmic change
    (and MEMORY-load-bearing once the backbone carries trainable LoRA adapters
    -- dedup-off pays ``group`` backbone backwards per update; see
    docs/changes/2026-08-29-backbone-lora/).
    Bit-parity caveat: batch-size-dependent XLA tiling can introduce float-ulp
    differences in v_t, so equality is allclose (certified by
    tests/ogpo/test_group_dedup.py), not bitwise.
    """
    if noise_level <= 0.0:
        raise ValueError(
            f"OGPO requires stochastic sampling (noise_level > 0); got {noise_level}."
        )
    # Mirror sample_actions: preprocess (resize/mask-fill, train=False) — a
    # per-sample op, so preprocess-then-repeat == repeat-then-preprocess.
    observation = _model.preprocess_observation(None, observation, train=False)
    B = observation.state.shape[0]
    bg = B * group
    # SAME key and shape as the un-deduplicated path (which draws noise at the
    # already-expanded batch), so the noise — and therefore the chains — match.
    noise = jax.random.normal(rng, (bg, model.action_horizon, model.action_dim))

    # ONE prefix forward at B; tile cache + mask to B*group. The observation is
    # also tiled for embed_suffix, which reads only state/adarms inputs — the
    # repeated image leaves are dead code inside the jit and XLA removes them.
    kv_cache, prefix_mask = compute_prefix_cache(model, observation)
    kv_rep, mask_rep = repeat_prefix_cache(kv_cache, prefix_mask, group)
    obs_rep = jax.tree.map(lambda x: jnp.repeat(x, group, axis=0), observation)

    dt = -1.0 / num_steps

    def stochastic_step(carry, _):
        x_t, time, step_rng = carry
        dist = get_sde_dist_with_cache(
            model,
            x_t=x_t,
            time=time,
            observation=obs_rep,
            kv_cache=kv_rep,
            prefix_mask=mask_rep,
            dt=dt,
            noise_level=noise_level,
        )
        # EXACT split structure of pi0.sample_actions' stochastic_step:
        # sample with the first output, carry the second.
        step_rng, key = jax.random.split(step_rng)
        x_next = jax.lax.stop_gradient(dist.sample(seed=step_rng))
        log_prob = dist.log_prob(x_next)
        t_next = time + dt
        out = {
            "x_next": x_next,
            "x": x_t,
            "time_next": t_next,
            "time": time,
            "log_prob": log_prob,
        }
        return (x_next, t_next, key), out

    initial_time = jnp.ones((bg,), dtype=noise.dtype)
    (x_0, _, _), outs = jax.lax.scan(
        stochastic_step, (noise, initial_time, rng), xs=None, length=num_steps
    )
    return {
        "actions": x_0,
        "x_chain": outs["x"],
        "x_next_chain": outs["x_next"],
        "times": outs["time"],
        "dt": jnp.asarray(dt, dtype=noise.dtype),
        "log_prob_per_step": outs["log_prob"],
    }


# ---------------------------------------------------------------------------
# Rescoring: log-prob of a frozen chain under a different set of weights.
# ---------------------------------------------------------------------------

def score_chain_under_model(
    model: _pi0.Pi0,
    observation: _model.Observation,
    *,
    x_chain: at.Float[at.Array, "k b ah ad"],
    x_next_chain: at.Float[at.Array, "k b ah ad"],
    times: at.Float[at.Array, "k b"],
    dt: at.Float[at.Array, ""],
    noise_level: float,
    prefix_cache: tuple[_gemma.KVCache, at.Bool[at.Array, "b _p"]] | None = None,
) -> at.Float[at.Array, "k b ah"]:
    """Recompute per-step Gaussian log-prob of ``x_next_chain`` under ``model``.

    The chain (``x_chain``, ``x_next_chain``, ``times``) is treated as frozen
    data — only ``model`` carries gradients. This is the IS-ratio numerator
    in OGPO's PPO surrogate.

    Implementation: compute the PaliGemma prefix **once** here (OUTSIDE the
    scan) and reuse the resulting KV cache across all ``K`` suffix-only
    forwards, which run under a single ``lax.scan`` over the rescoring
    steps. Scanning (not a Python loop) keeps activation memory at ~1× the
    prefix forward AND lets reverse-mode AD accumulate the K weight-gradient
    contributions into one carry accumulator instead of K materialized fp32
    partials — the jit-2 memory fix (see
    docs/plans/ogpo-memory/analysis-jit2-forensics.md; the naive unrolled
    path sized a >100 GiB grad arena that OOM'd on the target GPU).

    Returns ``[K, B, H]``. Sum across the first axis to get the joint
    log-prob per (B, H) entry, then sum across H (or leave as-is and let the
    loss broadcast).
    """
    # ONE prefix forward, computed OUTSIDE the scan and closed over its body:
    # the KV-cache cotangent is therefore summed across the K steps FIRST (one
    # small [L,B,T,K,H] accumulator) and mapped to the Gemma prefix-weight
    # gradient by a SINGLE backward through compute_prefix_cache — instead of K
    # full Gemma-weight cotangents. This is the jit-2 arena fix; see
    # docs/plans/ogpo-memory/analysis-jit2-forensics.md.
    #
    # prefix_cache: an already-computed (and possibly group-tiled) cache from
    # the caller. Group dedup passes prefix at batch B tiled to B*G here, so
    # the redundant per-group-member prefix forward (and, when the backbone is
    # unfrozen, its backward) is skipped. Gradients still flow through the
    # provided cache identically — jnp.repeat is linear, so the group members'
    # cache cotangents sum before the single prefix backward, exactly equal to
    # summing G separate prefix backwards.
    if prefix_cache is None:
        kv_cache, prefix_mask = compute_prefix_cache(model, observation)
    else:
        kv_cache, prefix_mask = prefix_cache

    # We scan (NOT a Python loop) over the K rescoring steps so reverse-mode AD
    # accumulates the K weight-gradient contributions in ONE carry instead of
    # materializing 11 full fp32 stacked Gemma-weight cotangents simultaneously
    # — the ~99 GiB batch-invariant jit-2 temp arena that OOM'd Phase-I
    # acceptance (docs/plans/ogpo-memory/analysis-jit2-forensics.md;
    # benchmark-results.md).
    #
    # The old "nnx hidden mutable state" objection does not apply: this forward
    # reads no rng and writes no nnx state — dropout=0.0 (gemma.py:295 -> no
    # Dropout module), the suffix llm runs at its default deterministic=True
    # (gemma.py:398; the rescoring path threads no ``deterministic``), and
    # self.deterministic is set-but-never-read. So ``model`` is a pure read-only
    # constant of the scan, exactly as in pi0.Pi0.sample_actions' lax.scan over
    # stochastic_step (pi0.py:408-448).
    def _score_step(carry, step):
        x_t, x_next, time = step
        lp_k, _ = get_dist_and_log_prob_with_cache(
            model,
            x_t=x_t,
            sample=x_next,
            time=time,
            observation=observation,
            kv_cache=kv_cache,
            prefix_mask=prefix_mask,
            dt=dt,
            noise_level=noise_level,
        )
        return carry, lp_k

    # Empty carry: there is no sequential dependence between rescoring steps.
    # The memory win is in the BACKWARD pass — differentiating this lax.scan
    # accumulates the K per-step cotangents w.r.t. the closed-over weights into
    # ONE carry accumulator (collapsing the 11x fp32 grad partials the Python
    # loop materialized).
    _, per_step = jax.lax.scan(
        _score_step, init=None, xs=(x_chain, x_next_chain, times)
    )
    return per_step  # [K, B, H] — identical shape/dtype to jnp.stack(..., axis=0)


# ---------------------------------------------------------------------------
# Reductions over the per-step log-prob tensor.
# ---------------------------------------------------------------------------

def sum_log_prob(
    log_prob_per_step: at.Float[at.Array, "k b ah"],
    *,
    ft_last_k: int | None = None,
) -> at.Float[at.Array, " b"]:
    """Sum a [num_steps, B, H] log-prob tensor down to [B].

    When ``ft_last_k`` is given, only the last ``ft_last_k`` SDE steps
    contribute (matches OGPO's ``ft_flow_steps`` knob); otherwise all steps
    are summed.
    """
    if ft_last_k is not None:
        log_prob_per_step = log_prob_per_step[-ft_last_k:]
    # Sum over (steps, chunk) → per-batch scalar.
    return jnp.sum(log_prob_per_step, axis=(0, 2))
