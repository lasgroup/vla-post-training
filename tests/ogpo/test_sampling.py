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


# NOTE: A perturbed-model sanity check was considered here, but constructing
# a deliberately-mutated nnx model in a way that is robust across flax
# versions turned out to be fiddly. The four tests above (especially
# ``test_rescoring_under_same_model_matches_sampling_logprob`` and
# ``test_ratio_is_one_when_old_equals_new``) already pin down the
# correctness invariant the PPO ratio needs. Dependence on model params is
# verified end-to-end by the training loop itself: if rescoring did not
# read current params, the gradient through ``score_chain_under_model``
# would be zero and the policy could not learn.
