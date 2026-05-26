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
) -> at.Float[at.Array, "k b ah"]:
    """Recompute per-step Gaussian log-prob of ``x_next_chain`` under ``model``.

    The chain (``x_chain``, ``x_next_chain``, ``times``) is treated as frozen
    data — only ``model`` carries gradients. This is the IS-ratio numerator
    in OGPO's PPO surrogate.

    Implementation: compute the PaliGemma prefix **once** here and reuse the
    resulting KV cache for all ``num_steps`` suffix-only forwards. This
    keeps activation memory at ~1× the prefix forward instead of
    ``num_steps``× — critical for OGPO's PPO update to fit in GPU memory
    (the naive 10× unrolled-prefix path blows up to >100 GB on B=256).

    Returns ``[num_steps, B, H]``. Sum across the first axis to get the
    joint log-prob per (B, H) entry, then sum across H (or leave as-is and
    let the loss broadcast).
    """
    num_steps = x_chain.shape[0]

    # ONE prefix forward; gradients still flow through the cache when the
    # PaliGemma backbone is unfrozen.
    kv_cache, prefix_mask = compute_prefix_cache(model, observation)

    # We unroll the loop in Python rather than ``jax.lax.scan`` because nnx
    # modules carry hidden mutable state that scan does not handle cleanly,
    # and because ``num_steps`` is small (10 by default). XLA still
    # de-duplicates the shared prefix-side activations because they all
    # come from a single ``compute_prefix_cache`` call upstream.
    per_step = []
    for k in range(num_steps):
        lp_k, _ = get_dist_and_log_prob_with_cache(
            model,
            x_t=x_chain[k],
            sample=x_next_chain[k],
            time=times[k],
            observation=observation,
            kv_cache=kv_cache,
            prefix_mask=prefix_mask,
            dt=dt,
            noise_level=noise_level,
        )
        per_step.append(lp_k)
    return jnp.stack(per_step, axis=0)


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
