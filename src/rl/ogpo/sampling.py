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

import jax
import jax.numpy as jnp

import openpi.models.model as _model
import openpi.models.pi0 as _pi0
import openpi.shared.array_typing as at


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

    Returns ``[num_steps, B, H]``. Sum across the first axis to get the
    joint log-prob per (B, H) entry, then sum across H (or leave as-is and
    let the loss broadcast).
    """
    num_steps = x_chain.shape[0]
    # We unroll the loop in Python rather than ``jax.lax.scan`` because nnx
    # modules carry hidden mutable state that scan does not handle cleanly,
    # and because ``num_steps`` is small (10 by default).
    per_step = []
    for k in range(num_steps):
        lp_k, _ = model.get_dist_and_log_prob(
            x_t=x_chain[k],
            sample=x_next_chain[k],
            time=times[k],
            observation=observation,
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
