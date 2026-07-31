"""Correctness checks for the OGPO chain sampling / rescoring helpers.

The key invariant: if we sample a chain under a given model and then rescore
that same chain under the *same* model, the per-step Gaussian log-probs must
match exactly (they are computed from the same v_t and the same Gaussian).

If this test fails, the PPO ratio in ``update_actor.train_step`` is broken
no matter how the rest of the loss is wired.
"""
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0_config
from src.rl.ogpo.sampling import (
    compute_prefix_cache,
    get_dist_and_log_prob_with_cache,
    sample_chain_with_logprob,
    score_chain_under_model,
    sum_log_prob,
)


@pytest.fixture(scope="module")
def tiny_pi0():
    """Build a minimal Pi05 model with the dummy gemma variants for speed."""
    cfg = pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=4,
        action_horizon=2,
        max_token_len=8,
        pi05=True,
    )
    rng = jax.random.key(0)
    model = cfg.create(rng)
    obs = cfg.fake_obs(batch_size=2)
    return model, cfg, obs


def test_sampling_returns_expected_shapes(tiny_pi0):
    model, cfg, obs = tiny_pi0
    num_steps = 4
    pack = sample_chain_with_logprob(
        model, obs, rng=jax.random.key(1),
        num_steps=num_steps, noise_level=0.3,
    )
    B = obs.state.shape[0]
    H = cfg.action_horizon
    D = cfg.action_dim
    assert pack["actions"].shape == (B, H, D)
    assert pack["x_chain"].shape == (num_steps, B, H, D)
    assert pack["x_next_chain"].shape == (num_steps, B, H, D)
    assert pack["times"].shape == (num_steps, B)
    assert pack["log_prob_per_step"].shape == (num_steps, B, H)


def test_rescoring_under_same_model_matches_sampling_logprob(tiny_pi0):
    """The roundtrip invariant: score_chain_under_model under the sampling
    model must return the same per-step log-probs as sample_chain_with_logprob.
    """
    model, _cfg, obs = tiny_pi0
    num_steps = 4
    noise_level = 0.3

    pack = sample_chain_with_logprob(
        model, obs, rng=jax.random.key(7),
        num_steps=num_steps, noise_level=noise_level,
    )
    rescored = score_chain_under_model(
        model, obs,
        x_chain=pack["x_chain"],
        x_next_chain=pack["x_next_chain"],
        times=pack["times"],
        dt=pack["dt"],
        noise_level=noise_level,
    )

    # Both tensors are [num_steps, B, H]; should match to numerical tol.
    np.testing.assert_allclose(
        np.asarray(rescored),
        np.asarray(pack["log_prob_per_step"]),
        atol=1e-4, rtol=1e-4,
    )

    # And the joint log-prob (the OGPO old_lp/new_lp scalar) must match too.
    new_lp  = sum_log_prob(rescored)
    old_lp  = sum_log_prob(pack["log_prob_per_step"])
    np.testing.assert_allclose(np.asarray(new_lp), np.asarray(old_lp), atol=1e-3, rtol=1e-4)


def test_sum_log_prob_respects_ft_last_k(tiny_pi0):
    """sum_log_prob(..., ft_last_k=k) drops everything except the last k steps."""
    rng = np.random.default_rng(0)
    arr = jnp.asarray(rng.normal(size=(6, 3, 2)))  # [K=6, B=3, H=2]
    full = sum_log_prob(arr)
    tail = sum_log_prob(arr, ft_last_k=2)
    tail_manual = jnp.sum(arr[-2:], axis=(0, 2))
    np.testing.assert_allclose(np.asarray(tail), np.asarray(tail_manual), atol=1e-6)
    # The full sum should not equal the tail sum (sanity).
    assert not np.allclose(np.asarray(full), np.asarray(tail))


def test_ratio_is_one_when_old_equals_new(tiny_pi0):
    """A direct check of the PPO-relevant quantity: ratio = exp(new_lp - old_lp)
    must be exactly 1 when both come from the same model on the same chain.
    """
    model, _cfg, obs = tiny_pi0
    pack = sample_chain_with_logprob(
        model, obs, rng=jax.random.key(11),
        num_steps=5, noise_level=0.2,
    )
    old_lp = sum_log_prob(pack["log_prob_per_step"])
    new_lp = sum_log_prob(score_chain_under_model(
        model, obs,
        x_chain=pack["x_chain"],
        x_next_chain=pack["x_next_chain"],
        times=pack["times"],
        dt=pack["dt"],
        noise_level=0.2,
    ))
    ratio = jnp.exp(new_lp - old_lp)
    np.testing.assert_allclose(np.asarray(ratio), np.ones_like(ratio), atol=5e-3)


# The four tests above check only the FORWARD log-prob. The GRADIENT that
# score_chain_under_model produces — the entire point of the Phase-J scan-ify —
# is certified below by test_scanned_rescorer_grad_matches_unrolled.


def _score_chain_unrolled(
    model,
    observation,
    *,
    x_chain,
    x_next_chain,
    times,
    dt,
    noise_level,
):
    """Pre-J Python-unrolled ``score_chain_under_model``, vendored VERBATIM.

    An op-for-op copy of the production body as it stood before Phase J
    (``sampling.py`` ``score_chain_under_model`` at HEAD ``b34cc24``, lines
    203-228): same single ``compute_prefix_cache``, same ``for k in
    range(num_steps)`` over the UNCHANGED ``get_dist_and_log_prob_with_cache``,
    same ``jnp.stack``. Only the enclosing name differs. This monolithic
    baseline cannot share the scan's structural error, so it is the independent
    reference the scanned rescorer's gradient is checked against below.
    """
    num_steps = x_chain.shape[0]
    kv_cache, prefix_mask = compute_prefix_cache(model, observation)
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


def _activate_adarms_gates(model):
    """Return an ISOLATED copy of ``model`` with its adaRMS gates activated.

    Why this is required (and not a workaround): pi05's adaptive RMSNorm
    zero-inits its modulation Dense (``gemma.py`` ``RMSNorm``: ``kernel_init=
    zeros`` and a zero bias), so on a FRESH-init fixture ``gate == 0`` and
    ``_gated_residual`` returns ``x + y*0 == x`` — the ENTIRE attention/MLP
    contribution is gated out. On the untouched fixture every Gemma-2B backbone
    weight-grad (including the prefix ``kv_einsum``, whose cotangent flows ONLY
    through the closed-over ``kv_cache``) is EXACTLY zero, so the
    kv-cache-cotangent path this test exists to certify would not be exercised
    at all. Perturbing only the adaRMS conditioning Dense params
    (``*_norm_1.Dense_0``) makes the gates nonzero and attention go live,
    turning the backbone grad real (prefix ``kv_einsum`` |g| ~ O(10)) so the N1
    floor assertion below is both satisfiable and truthful. The perturbation is
    confined to this returned copy; the caller's model is untouched (nnx.split
    → merge, no in-place mutation).
    """
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    rng = np.random.default_rng(0)
    new_flat = []
    for path, var in nnx.to_flat_state(params):
        path_str = "/".join(str(p) for p in path)
        if "Dense_0" in path_str and "norm_1" in path_str:  # adaRMS modulation
            arr = np.asarray(var.value)
            delta = jnp.asarray(rng.standard_normal(arr.shape).astype(arr.dtype) * 0.5)
            new_flat.append((path, var.replace(value=var.value + delta)))
        else:
            new_flat.append((path, var))
    return nnx.merge(graphdef, nnx.from_flat_state(new_flat), rest)


def test_scanned_rescorer_grad_matches_unrolled():
    """The scan-ified ``score_chain_under_model`` must reproduce the Python-
    unrolled loop's GRADIENT w.r.t. model params, not merely its forward
    log-prob — the one property no other test covers. ``test_split_equivalence``'s
    reference AND split both call the live (now scanned) helper, so any change
    moves both sides identically; ``test_sampling``'s other four check only the
    forward. A broken diff-connection (model captured as a disconnected const)
    gives an identical forward with a zero/wrong grad; this is the gate (R-J1).

    Two deviations from the amendment's literal J-1.3 fixture, both forced by
    the tiny model's structure and both making the test do what J-1.3 SAYS it
    does:

    * adaRMS gates are activated (see ``_activate_adarms_gates``). Fresh-init
      ``tiny_pi0`` zero-gates the attention/MLP, so the kv-cache-cotangent path
      N1 targets carries EXACTLY zero grad and would certify nothing.
    * the model is built in FLOAT32. The scanned (while-loop) and unrolled
      graphs are OP-IDENTICAL, but they are two different compiled graphs; in
      bf16 with active 776-token attention their forwards already diverge
      (~2e-2) — compilation-drift class, not a diff-connection break — which
      swamps the reassociation signal. float32 makes the forward op-identical
      again (Δ~1e-6), so ``rtol=1e-3`` cleanly separates benign gradient
      reassociation (ULP/compilation-drift scale, observed max rel Δ ~5e-3 on
      small-grad leaves with abs Δ within ``atol``) from a broken connection
      (~100% relative / all-zero).
    """
    cfg = pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=4,
        action_horizon=2,
        max_token_len=8,
        pi05=True,
        dtype="float32",
    )
    model = _activate_adarms_gates(cfg.create(jax.random.key(0)))
    obs = cfg.fake_obs(batch_size=2)
    num_steps, noise_level = 3, 0.3
    pack = sample_chain_with_logprob(
        model, obs, rng=jax.random.key(3), num_steps=num_steps, noise_level=noise_level
    )

    def _loss(m, scorer):
        lp = scorer(
            m,
            obs,
            x_chain=pack["x_chain"],
            x_next_chain=pack["x_next_chain"],
            times=pack["times"],
            dt=pack["dt"],
            noise_level=noise_level,
        )
        return jnp.sum(sum_log_prob(lp))

    # Same diff mechanism on both sides (nnx.grad over all params); only the
    # scorer differs — so any mismatch is attributable to the scan alone.
    g_scan = nnx.grad(lambda m: _loss(m, score_chain_under_model))(model)
    g_unroll = nnx.grad(lambda m: _loss(m, _score_chain_unrolled))(model)

    scan_wp = jax.tree_util.tree_leaves_with_path(g_scan)
    unroll_wp = jax.tree_util.tree_leaves_with_path(g_unroll)
    assert [jax.tree_util.keystr(p) for p, _ in scan_wp] == [
        jax.tree_util.keystr(p) for p, _ in unroll_wp
    ], "scanned/unrolled grad trees have different structure"

    # Belt-and-suspenders whole-tree floor: the unrolled baseline must carry
    # real signal, and the scanned grad must be non-zero overall (a
    # disconnected-const scan zeroes it).
    assert any(float(jnp.max(jnp.abs(x))) > 0 for _, x in unroll_wp)
    assert (
        sum(float(jnp.sum(x**2)) for _, x in scan_wp) ** 0.5 > 0
    ), "scanned grad is all-zero (disconnected scan)"

    # N1 — kv-cache-cotangent PATH floor (the production-critical, novel-risk
    # path). The prefix Gemma-2B (expert-0) attention/MLP weights are never run
    # inside the suffix body; their gradient arrives ONLY through the closed-over
    # kv_cache. A partial disconnect (an accidental stop_gradient on the cache,
    # or the cache lowered as a literal) would sever exactly these leaves while
    # the action-expert leaves stay correct — invisible to the whole-tree norm
    # above. Assert the prefix kv_einsum grad clears the floor in the unrolled
    # baseline AND is nonzero in the scanned tree, on the exact leaf that severs.
    atol = 1e-5
    kv_unroll = [x for p, x in unroll_wp if "['kv_einsum']" in jax.tree_util.keystr(p)]
    kv_scan = [x for p, x in scan_wp if "['kv_einsum']" in jax.tree_util.keystr(p)]
    assert kv_unroll, "prefix kv_einsum grad leaf not found (fixture structure changed)"
    for x in kv_unroll:
        assert float(jnp.max(jnp.abs(x))) > 10 * atol, (
            "prefix kv_einsum grad is at the floor — the kv_cache path is not "
            "exercised, so N1 certifies nothing (are the adaRMS gates active?)"
        )
    for x in kv_scan:
        assert float(jnp.max(jnp.abs(x))) > 0, "scanned prefix kv_einsum grad severed"

    # Leaf-for-leaf equality. The scanned and unrolled rescorers are OP-IDENTICAL;
    # any residual forward difference is compilation-drift class (while-loop vs
    # unrolled graph), inside existing tolerances. The gradient is LEGITIMATELY
    # reassociated (sequential scan carry vs add_any tree, sign-off S1), so rtol
    # admits ULP reassociation while a broken diff-connection (~100% / all-zero)
    # fails cleanly. Do NOT loosen below without confirming any miss is
    # reassociation-scale (<<1), not disconnection-scale (~1) — S1 policy.
    for (_, a), (_, b) in zip(scan_wp, unroll_wp):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-3, atol=atol)
