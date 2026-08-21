# ruff: noqa: F722
"""The actor's combined ``grad_norm`` decomposes into its PG and BC parts.

``bc_grad_accumulate`` reports ``grad_norm`` over the COMBINED gradient tree,
after the BC anchor has been accumulated into the donated PPO grads. The two
component norms — ``grad_norm_pg`` (jit-2a) and ``grad_norm_bc`` (jit-2b) —
exist so that figure can be split.

These tests pin the *interpretation*, not just the plumbing:

  * with the advantage zeroed the PG gradient vanishes exactly, so
    ``grad_norm_pg == 0`` and ``grad_norm == grad_norm_bc``. This is the
    warmstart regime (``training_steps < pg_start_step`` multiplies the
    advantage by 0.0), which is how ``‖g_BC‖`` was read off existing runs
    before these keys existed — the test certifies that reading.
  * with ``bc_coeff = 0`` the anchor vanishes, so ``grad_norm_bc == 0`` and
    ``grad_norm == grad_norm_pg``.
  * in general the three satisfy the triangle inequality.
  * ``grad_cos_pg_bc`` — the angle between the two gradient directions — is in
    [-1, 1], satisfies the law of cosines against the three norms, and is NaN
    (undefined, not orthogonal) whenever either gradient is exactly zero.

Dummy PaliGemma/action-expert variants throughout, so this is CPU-only and
runs in seconds. No critics are built: the chain is drawn directly from
``sampling.sample_chain_with_logprob``, which is what jit-1 does minus the
Q/V evaluation, and the advantage is supplied by the test.
"""
import dataclasses
import functools
import types

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import openpi.models.pi0_config as pi0_config
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.optimizer as _optimizer
import openpi.training.utils as training_utils

from src.rl.ogpo.sampling import sample_chain_with_logprob, sum_log_prob
from src.rl.ogpo.update_actor import bc_grad_accumulate, loss_and_grad_pg
from src.training.config import OGPOSFTLearnerConfig, get_config

_B = 2
# Exact-zero legs are structural (a zero gradient tree gives global_norm 0.0
# exactly), but the "combined == component" legs go through a leaf-wise add of
# that zero tree, which XLA is free to fuse or reassociate. 1e-6 absolute is the
# same tolerance the split-equivalence suite uses for the identical add.
_ATOL = 1e-6


def _build_config(*, bc_coeff=1.0):
    base = get_config("pi05_libero_online_ogpo_sft")
    dummy_model = pi0_config.Pi0Config(
        paligemma_variant="dummy", action_expert_variant="dummy",
        action_dim=4, action_horizon=2, max_token_len=8, pi05=True,
    )
    rl = dataclasses.replace(
        base.rl, group_num_samples=1, num_sde_steps=3, noise_level=0.3,
        adv_strategy="subtract_v", bc_coeff=bc_coeff, use_bc_regularization=True,
    )
    config = dataclasses.replace(
        base, model=dummy_model, rl=rl, batch_size=_B,
        # Freeze only the vision tower so the action expert carries a real
        # gradient in both the PG and BC paths (mirrors test_split_equivalence).
        freeze_filter=nnx_utils.PathRegex(".*PaliGemma/img.*"),
    )
    assert isinstance(config.rl, OGPOSFTLearnerConfig)
    return config


def _make_params(config, model):
    params = nnx.state(model)
    return nnx_utils.state_map(
        params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16))
    )


def _build_policy_state(config, model, params):
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)
    opt_state = tx.init(nnx.filter_state(params, config.trainable_filter))
    return training_utils.TrainState(
        step=900, params=params, model_def=nnx.graphdef(model), tx=tx,
        opt_state=opt_state, ema_decay=None, ema_params=None,
    )


@pytest.fixture(scope="module")
def fx():
    config = _build_config()
    model = config.model.create(jax.random.key(0))
    params = _make_params(config, model)
    policy_state = _build_policy_state(config, model, params)
    policy_observation = config.model.fake_obs(batch_size=_B)
    actions_demo = config.model.fake_act(batch_size=_B)

    # Draw a real SDE chain under the (eval-mode) policy — the same call jit-1
    # makes, minus the critic evaluation. G=1, so B*G == _B.
    old_model = nnx.merge(nnx.graphdef(model), params)
    old_model.eval()
    chain = sample_chain_with_logprob(
        old_model, policy_observation, rng=jax.random.key(3),
        num_steps=config.rl.num_sde_steps, noise_level=config.rl.noise_level,
    )
    K, _, H, D = chain["x_chain"].shape
    log_prob_norm = jnp.float32(K * H) * jnp.float32(D)
    old_lp = sum_log_prob(chain["log_prob_per_step"]) / log_prob_norm

    return types.SimpleNamespace(
        config=config, policy_state=policy_state,
        policy_observation=policy_observation, actions_demo=actions_demo,
        x_chain=chain["x_chain"], x_next_chain=chain["x_next_chain"],
        times=chain["times"], dt=chain["dt"], old_lp=old_lp,
        pg_jit=jax.jit(functools.partial(loss_and_grad_pg, config)),
        bc_jit=jax.jit(functools.partial(bc_grad_accumulate, config)),
    )


def _run(fx, advantage, *, bc_coeff=None):
    """jit-2a -> jit-2b for a given advantage, returning the merged aux."""
    pg_jit, bc_jit = fx.pg_jit, fx.bc_jit
    if bc_coeff is not None:
        cfg = _build_config(bc_coeff=bc_coeff)
        pg_jit = jax.jit(functools.partial(loss_and_grad_pg, cfg))
        bc_jit = jax.jit(functools.partial(bc_grad_accumulate, cfg))
    grads_pg, pg_loss, pg_aux = pg_jit(
        fx.policy_state, fx.policy_observation,
        fx.x_chain, fx.x_next_chain, fx.times, fx.dt, fx.old_lp, advantage,
    )
    _, _, loss_aux = bc_jit(
        grads_pg, jax.random.key(11), fx.policy_state,
        fx.policy_observation, fx.actions_demo, pg_loss, pg_aux,
    )
    return loss_aux


def _nonzero_advantage():
    # Group-centered advantages are signed and O(1); use a fixed asymmetric
    # vector so neither the upper nor the lower PPO clip branch is exercised
    # exclusively.
    return jnp.asarray([1.0, -0.5], dtype=jnp.float32)


def test_cosine_is_in_range_and_finite(fx):
    aux = _run(fx, _nonzero_advantage())
    assert "grad_cos_pg_bc" in aux
    cos = float(np.asarray(aux["grad_cos_pg_bc"]))
    assert np.isfinite(cos), f"cosine is not finite with both gradients non-zero: {cos}"
    assert -1.0 - _ATOL <= cos <= 1.0 + _ATOL, f"cosine out of range: {cos}"


def test_cosine_satisfies_the_law_of_cosines(fx):
    # The identity that ties all four emitted numbers together:
    #   ‖g_pg + g_bc‖² = ‖g_pg‖² + ‖g_bc‖² + 2·cos·‖g_pg‖·‖g_bc‖
    # This validates the cosine numerically against the three norms without the
    # test needing access to the gradient trees — if the dot product were
    # computed over a mismatched tree traversal, or the norms paired wrongly,
    # this fails.
    aux = _run(fx, _nonzero_advantage())
    total = float(np.asarray(aux["grad_norm"]))
    pg = float(np.asarray(aux["grad_norm_pg"]))
    bc = float(np.asarray(aux["grad_norm_bc"]))
    cos = float(np.asarray(aux["grad_cos_pg_bc"]))
    predicted = pg**2 + bc**2 + 2.0 * cos * pg * bc
    # fp32 accumulation over the whole trainable tree, squared — 1e-4 relative.
    np.testing.assert_allclose(
        total**2, predicted, rtol=1e-4, atol=0.0,
        err_msg="grad_cos_pg_bc is inconsistent with the three reported norms",
    )


def test_cosine_is_nan_when_either_gradient_is_zero(fx):
    # Undefined, not orthogonal. exp.py reduces with jnp.nanmean, so an
    # undefined value drops out of the window rather than reading as 0.
    muted_pg = _run(fx, jnp.zeros((_B,), dtype=jnp.float32))
    assert np.isnan(np.asarray(muted_pg["grad_cos_pg_bc"])), (
        "cosine must be NaN when the PG gradient is exactly zero (warmstart)"
    )
    muted_bc = _run(fx, _nonzero_advantage(), bc_coeff=0.0)
    assert np.isnan(np.asarray(muted_bc["grad_cos_pg_bc"])), (
        "cosine must be NaN when the BC gradient is exactly zero"
    )


def test_component_norms_are_present_and_finite(fx):
    aux = _run(fx, _nonzero_advantage())
    for k in ("grad_norm", "grad_norm_pg", "grad_norm_bc"):
        assert k in aux, f"{k} missing from the actor aux dict"
        v = float(np.asarray(aux[k]))
        assert np.isfinite(v), f"{k} is not finite: {v}"
        assert v >= 0.0, f"{k} is a norm but is negative: {v}"
    # The anchor and the surrogate must both actually be contributing here,
    # or the remaining legs would pass vacuously.
    assert float(np.asarray(aux["grad_norm_pg"])) > 0.0
    assert float(np.asarray(aux["grad_norm_bc"])) > 0.0


def test_zero_advantage_leaves_only_the_bc_gradient(fx):
    # The warmstart regime: ogpo_learner.update() multiplies the advantage by
    # 0.0 while training_steps < pg_start_step. min(r*0, clip(r)*0) == 0 with a
    # zero derivative in r, so grads_pg is an exact zero tree.
    aux = _run(fx, jnp.zeros((_B,), dtype=jnp.float32))
    assert float(np.asarray(aux["grad_norm_pg"])) == 0.0
    assert float(np.asarray(aux["pg_loss"])) == 0.0
    np.testing.assert_allclose(
        np.asarray(aux["grad_norm"]), np.asarray(aux["grad_norm_bc"]),
        atol=_ATOL, rtol=0.0,
        err_msg="with PG muted the combined norm must equal the BC-only norm",
    )
    assert float(np.asarray(aux["grad_norm_bc"])) > 0.0


def test_zero_bc_coeff_leaves_only_the_pg_gradient(fx):
    # bc_grad_accumulate differentiates bc_coeff * bc_loss, so bc_coeff = 0
    # yields an exact zero BC gradient tree.
    aux = _run(fx, _nonzero_advantage(), bc_coeff=0.0)
    assert float(np.asarray(aux["grad_norm_bc"])) == 0.0
    np.testing.assert_allclose(
        np.asarray(aux["grad_norm"]), np.asarray(aux["grad_norm_pg"]),
        atol=_ATOL, rtol=0.0,
        err_msg="with the anchor off the combined norm must equal the PG-only norm",
    )
    assert float(np.asarray(aux["grad_norm_pg"])) > 0.0


def test_combined_norm_obeys_the_triangle_inequality(fx):
    # ‖g_pg + g_bc‖ ∈ [ |‖g_pg‖ - ‖g_bc‖| , ‖g_pg‖ + ‖g_bc‖ ] — the bound that
    # makes the two component keys usable for attributing the combined figure.
    aux = _run(fx, _nonzero_advantage())
    total = float(np.asarray(aux["grad_norm"]))
    pg = float(np.asarray(aux["grad_norm_pg"]))
    bc = float(np.asarray(aux["grad_norm_bc"]))
    assert abs(pg - bc) - _ATOL <= total <= pg + bc + _ATOL, (
        f"grad_norm={total} outside [{abs(pg - bc)}, {pg + bc}]"
    )


def test_bc_norm_carries_the_coefficient(fx):
    # grads_bc is the gradient of bc_coeff * bc_loss, so grad_norm_bc scales
    # linearly with bc_coeff. This is the property that makes the key directly
    # comparable against grad_norm_pg (it is the anchor's actual contribution
    # to the sum, not the raw BC gradient).
    zero_adv = jnp.zeros((_B,), dtype=jnp.float32)
    bc_1 = float(np.asarray(_run(fx, zero_adv, bc_coeff=1.0)["grad_norm_bc"]))
    bc_3 = float(np.asarray(_run(fx, zero_adv, bc_coeff=3.0)["grad_norm_bc"]))
    assert bc_1 > 0.0
    # Exact in exact arithmetic, but the backward runs through the bf16 frozen
    # vision tower and the dummy Gemma stack, so a 3x seed cotangent is not a
    # bitwise-3x result: measured relative error ~4e-4 here. 5e-3 leaves ~12x
    # headroom and still fails loudly if the coefficient were dropped entirely
    # (that would show ~200%, not 0.04%).
    np.testing.assert_allclose(
        bc_3, 3.0 * bc_1, rtol=5e-3, atol=0.0,
        err_msg="grad_norm_bc must scale linearly with bc_coeff",
    )
