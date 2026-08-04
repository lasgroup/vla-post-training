# ruff: noqa: F722
"""Group-prefix-dedup equivalence (performance-only change).

``dedup_group_prefix`` must be a pure optimization: with the SAME rng, the
grouped sampler must reproduce the expanded-batch sampler's chains/log-probs,
and the cached rescorer must reproduce the uncached rescorer's outputs AND
gradients. Tolerances are allclose (1e-5): batch-size-dependent XLA tiling can
introduce float-ulp differences, but nothing beyond.
"""
import functools

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import openpi.models.model as _model
import openpi.shared.nnx_utils as nnx_utils
from openpi.models import pi0_config

from src.rl.ogpo.sampling import (
    compute_prefix_cache,
    repeat_prefix_cache,
    sample_chain_with_logprob,
    sample_chain_with_logprob_grouped,
    score_chain_under_model,
)

_B = 2
_G = 4
_K = 3
_NOISE = 0.05
_ATOL = 1e-5


@pytest.fixture(scope="module")
def fx():
    config = pi0_config.Pi0Config(
        paligemma_variant="dummy", action_expert_variant="dummy",
        action_dim=4, action_horizon=2, max_token_len=8, pi05=True,
    )
    model = config.create(jax.random.key(0))
    model.eval()
    obs = config.fake_obs(batch_size=_B)
    return model, obs


def _expand(tree, g):
    return jax.tree.map(lambda x: jnp.repeat(x, g, axis=0), tree)


def test_sampler_grouped_matches_expanded(fx):
    model, obs = fx
    rng = jax.random.key(7)
    ref = sample_chain_with_logprob(
        model, _expand(obs, _G), rng=rng, num_steps=_K, noise_level=_NOISE
    )
    got = sample_chain_with_logprob_grouped(
        model, obs, rng=rng, group=_G, num_steps=_K, noise_level=_NOISE
    )
    for k in ("actions", "x_chain", "x_next_chain", "times", "dt", "log_prob_per_step"):
        a, b = np.asarray(ref[k]), np.asarray(got[k])
        assert a.shape == b.shape, f"{k}: shape {a.shape} vs {b.shape}"
        assert np.allclose(a, b, atol=_ATOL), (
            f"{k}: max|Δ|={np.max(np.abs(a - b))} exceeds atol={_ATOL}"
        )


def test_repeat_cache_matches_expanded_cache(fx):
    model, obs = fx
    kv_b, mask_b = compute_prefix_cache(model, _model.preprocess_observation(None, obs, train=False))
    kv_rep, mask_rep = repeat_prefix_cache(kv_b, mask_b, _G)
    obs_g = _model.preprocess_observation(None, _expand(obs, _G), train=False)
    kv_g, mask_g = compute_prefix_cache(model, obs_g)
    assert np.array_equal(np.asarray(mask_rep), np.asarray(mask_g))
    for a, b in zip(jax.tree.leaves(kv_rep), jax.tree.leaves(kv_g)):
        assert a.shape == b.shape, f"cache leaf shape {a.shape} vs {b.shape} (batch must be axis 1)"
        # Cache leaves may be bf16; cast for numpy comparison.
        a32 = np.asarray(jnp.asarray(a, dtype=jnp.float32))
        b32 = np.asarray(jnp.asarray(b, dtype=jnp.float32))
        assert np.allclose(a32, b32, atol=_ATOL)


def test_rescorer_cached_matches_uncached_forward_and_grad(fx):
    model, obs = fx
    rng = jax.random.key(11)
    obs_g = _expand(obs, _G)
    chain = sample_chain_with_logprob(
        model, obs_g, rng=rng, num_steps=_K, noise_level=_NOISE
    )
    kwargs = dict(
        x_chain=chain["x_chain"], x_next_chain=chain["x_next_chain"],
        times=chain["times"], dt=chain["dt"], noise_level=_NOISE,
    )

    def lp_sum(m, cached):
        # Mirror production exactly: loss_and_grad_pg passes the raw
        # (unpreprocessed) expanded observation to score_chain_under_model,
        # and the dedup branch computes the cache from the raw B-batch obs.
        pc = None
        if cached:
            kv_b, mask_b = compute_prefix_cache(m, obs)
            pc = repeat_prefix_cache(kv_b, mask_b, _G)
        return jnp.sum(score_chain_under_model(m, obs_g, **kwargs, prefix_cache=pc))

    # Forward equivalence.
    f_ref = lp_sum(model, cached=False)
    f_got = lp_sum(model, cached=True)
    assert np.allclose(np.asarray(f_ref), np.asarray(f_got), atol=_ATOL), (
        f"forward: {f_ref} vs {f_got}"
    )

    # Gradient equivalence w.r.t. ALL params (harder than trainable-only;
    # covers the unfrozen-backbone case where the prefix carries gradient).
    g_ref = nnx.grad(functools.partial(lp_sum, cached=False))(model)
    g_got = nnx.grad(functools.partial(lp_sum, cached=True))(model)
    ref_leaves = jax.tree.leaves(g_ref)
    got_leaves = jax.tree.leaves(g_got)
    assert len(ref_leaves) == len(got_leaves)
    bad = []
    for i, (a, b) in enumerate(zip(ref_leaves, got_leaves)):
        a, b = np.asarray(a), np.asarray(b)
        if not np.allclose(a, b, atol=_ATOL):
            bad.append((i, float(np.max(np.abs(a - b)))))
    assert not bad, f"{len(bad)} grad leaf(s) exceed atol={_ATOL}: {bad[:5]}"
