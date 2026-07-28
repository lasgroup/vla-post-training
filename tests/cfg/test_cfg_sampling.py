"""Classifier-free guidance on the language conditioning of pi0/pi05 sampling."""

import dataclasses
import functools

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import openpi.models.pi0_config as _pi0_config

# float32 matmuls fall back to TF32 on GPU (~1e-3 relative error), which varies with
# the batch shape and would swamp a fused-vs-two-pass comparison.
jax.config.update("jax_default_matmul_precision", "highest")


def _ungate_adarms(model, rng):
    """Make a freshly-initialized pi05 action expert sensitive to its conditioning.

    Adaptive RMSNorm computes its scale/shift/gate from a zero-initialized Dense
    (`gemma.py`), so at init every gate is 0 and `_gated_residual` drops the whole
    attention + FFN contribution. A random-init pi05 model is therefore constant in
    images, prompt and state alike, which would make any conditioning test vacuous.
    Trained checkpoints have non-zero gates; we emulate that by randomizing the
    modulation biases.
    """
    state = nnx.state(model, nnx.Param)
    flat = state.flat_state()
    keys = jax.random.split(rng, len(flat))
    for (path, var), key in zip(flat, keys):
        name = "/".join(str(part) for part in path)
        if "norm_1/Dense_0/bias" in name:
            var.value = jax.random.normal(key, var.value.shape, dtype=var.value.dtype)
    nnx.update(model, state)
    return model


@functools.lru_cache(maxsize=None)
def _make_model_and_inputs(*, pi05: bool = True, batch_size: int = 2, seed: int = 0):
    config = _pi0_config.Pi0Config(
        # float32 so the fused CFG pass (one batch-doubled forward) and the reference
        # two-pass computation agree beyond bf16 granularity.
        dtype="float32",
        pi05=pi05,
        discrete_state_input=False,
        action_dim=4,
        action_horizon=6,
        max_token_len=8,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    model = config.create(jax.random.key(seed))
    if pi05:
        model = _ungate_adarms(model, jax.random.key(seed + 100))
    obs_spec, action_spec = config.inputs_spec(batch_size=batch_size)
    rng = jax.random.key(seed + 1)

    def fill(spec, key):
        if spec.dtype == jnp.int32:
            return jax.random.randint(key, spec.shape, 0, 100)
        if spec.dtype == bool:
            return jnp.ones(spec.shape, dtype=bool)
        return jax.random.normal(key, spec.shape)

    leaves, treedef = jax.tree.flatten(obs_spec)
    keys = jax.random.split(rng, len(leaves))
    obs = jax.tree.unflatten(treedef, [fill(s, k) for s, k in zip(leaves, keys)])
    noise = jax.random.normal(rng, action_spec.shape)
    return model, obs, noise


def _sample(model, obs, noise, **kwargs):
    """One Euler step, so x_0 = noise - v and the velocity is recoverable."""
    return np.asarray(
        model.sample_actions(rng=jax.random.key(0), observation=obs, noise=noise, num_steps=1, **kwargs)
    )


def _drop_prompt(obs):
    return dataclasses.replace(obs, tokenized_prompt_mask=jnp.zeros_like(obs.tokenized_prompt_mask))


@pytest.mark.parametrize("pi05", [True, False])
def test_cfg_scale_one_is_a_noop(pi05):
    model, obs, noise = _make_model_and_inputs(pi05=pi05)
    np.testing.assert_array_equal(_sample(model, obs, noise), _sample(model, obs, noise, cfg_scale=1.0))


@pytest.mark.parametrize("pi05", [True, False])
def test_cfg_matches_manual_two_pass_combination(pi05):
    model, obs, noise = _make_model_and_inputs(pi05=pi05)

    v_cond = noise - _sample(model, obs, noise)
    v_uncond = noise - _sample(model, _drop_prompt(obs), noise)
    assert not np.allclose(v_cond, v_uncond), "prompt masking had no effect on the velocity"

    for scale in (0.0, 1.5, 3.0):
        expected = noise - (v_uncond + scale * (v_cond - v_uncond))
        np.testing.assert_allclose(_sample(model, obs, noise, cfg_scale=scale), expected, atol=1e-4, rtol=1e-4)


def test_cfg_multi_step_still_guides():
    """Guidance is applied inside the denoising loop, so it must survive num_steps > 1."""
    model, obs, noise = _make_model_and_inputs()
    sample = functools.partial(model.sample_actions, rng=jax.random.key(0), observation=obs, noise=noise)
    baseline = np.asarray(sample(num_steps=4))
    guided = np.asarray(sample(num_steps=4, cfg_scale=3.0))
    assert not np.allclose(baseline, guided)
    np.testing.assert_array_equal(baseline, np.asarray(sample(num_steps=4, cfg_scale=1.0)))


# That the unconditional branch drops language *only* is covered by the scale=0.0 case
# of test_cfg_matches_manual_two_pass_combination: its reference pass masks the prompt
# while leaving the images untouched. It cannot be tested through the images directly —
# the "dummy" SigLIP tower emits a constant regardless of its input.


def test_prompt_dropout_changes_the_training_loss():
    """Premise of the training-side conditioning dropout in filtered_sft_agent/update.py:
    masking tokenized_prompt_mask is what makes a sample unconditional."""
    model, obs, noise = _make_model_and_inputs()
    actions = jnp.zeros_like(noise)
    loss = functools.partial(model.compute_loss, jax.random.key(0), actions=actions, train=False)
    conditional = np.asarray(loss(obs))
    unconditional = np.asarray(loss(_drop_prompt(obs)))
    assert not np.allclose(conditional, unconditional)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_cfg_preserves_output_shape(batch_size):
    model, obs, noise = _make_model_and_inputs(batch_size=batch_size)
    assert _sample(model, obs, noise, cfg_scale=2.0).shape == noise.shape
