# ruff: noqa: F722
"""Gradient-flow and metric tests for backbone LoRA
(docs/changes/2026-08-29-backbone-lora/).

What this file certifies, on a dummy-width LoRA model under the PRODUCTION
freeze filter (``_make_ogpo_freeze_filter(allow_lora=True)``):

  * adapters receive nonzero PG and BC gradients, and the frozen stack-0 base
    leaves are absent from the gradient tree entirely;
  * the group-dedup prefix-cache path carries the same adapter gradients as the
    expanded-batch path (the ``dedup_group_prefix`` equivalence claim, now
    tested with a trainable backbone);
  * ``grad_norm² == grad_norm_lora² + grad_norm_rest²`` (disjoint filters), and
    ``grad_norm_lora == 0.0`` exactly on a lora-less tree;
  * jit-1's ``critic_prefix is None`` recompute is invariant to the adapter
    values (R1: the critic's V is computed on unadapted-backbone features),
    while the sampled chain does depend on them (non-vacuousness control).

Scaffold cloned from ``test_grad_norm_decomposition.py``; the dummy-lora
variant comes from a monkeypatched ``gemma.get_config`` as in
``test_backbone_lora_init.py``.
"""
import dataclasses
import functools
import importlib
import types

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import openpi.models.gemma as gemma
import openpi.models.lora as lora
import openpi.models.pi0_config as pi0_config
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils

from src.rl.lora_utils import LORA_FILTER
from src.rl.ogpo.sampling import sample_chain_with_logprob, sum_log_prob
from src.rl.ogpo.update_actor import (
    bc_grad_accumulate,
    loss_and_grad_pg,
    sample_and_advantage,
)
from src.training.config import (
    OGPOSFTLearnerConfig,
    _make_ogpo_freeze_filter,
    get_config,
)

# Same cross-module import pattern as test_per_task_critics.py (conftest puts
# tests/ on sys.path as the `ogpo` package root).
_split = importlib.import_module("ogpo.test_split_equivalence")

_B = 2
_ATOL = 1e-6


def _patched_get_config(orig):
    def patched(variant):
        if variant == "dummy_lora":
            return dataclasses.replace(
                orig("dummy"),
                lora_configs={
                    "attn": lora.LoRAConfig(rank=4, alpha=4.0),
                    "ffn": lora.LoRAConfig(rank=4, alpha=4.0),
                },
            )
        return orig(variant)

    return patched


def _build_config(*, variant="dummy_lora", group_num_samples=1, dedup=False):
    base = get_config("pi05_libero_online_ogpo_sft")
    dummy_model = pi0_config.Pi0Config(
        paligemma_variant=variant, action_expert_variant="dummy",
        action_dim=4, action_horizon=2, max_token_len=8, pi05=True,
    )
    critic = dataclasses.replace(
        base.rl.critic, use_bronet=True, bronet_hidden_dim=32, bronet_depth=1,
        num_qs=2, num_vs=2,
    )
    rl = dataclasses.replace(
        base.rl, critic=critic, group_num_samples=group_num_samples,
        num_sde_steps=3, noise_level=0.3, adv_strategy="subtract_v",
        use_ema_as_old_policy=False, dedup_group_prefix=dedup,
    )
    # The PRODUCTION lora-aware filter — unlike the sibling scaffolds, which
    # freeze only SigLIP. Trainable: action expert + heads + adapters.
    config = dataclasses.replace(
        base, model=dummy_model, rl=rl, batch_size=_B,
        freeze_filter=_make_ogpo_freeze_filter(allow_lora=True),
    )
    assert isinstance(config.rl, OGPOSFTLearnerConfig)
    return config


def _unblock_adarms_gates(params):
    """Randomize the zero-initialized adaRMS gate heads of the action expert.

    pi05 gates every suffix block's attention/MLP contribution through an
    adaRMS head (``.*norm*_1/Dense_0``) that is ZERO at init — so at random
    init the suffix stream never reads attention, and v_t is bit-independent
    of the KV cache, the images, and the LoRA adapters (verified empirically
    on the dummy AND gemma_300m variants). Trained models have nonzero gates.
    Without this, every adapter-gradient assertion below is vacuously false.
    """
    import openpi.shared.nnx_utils as nnx_utils

    gate_filter = nnx_utils.PathRegex(".*norm.*Dense_0.*")
    gates = nnx.filter_state(params, gate_filter)
    assert len(gates.flat_state()) > 0, "no adaRMS gate heads found — fixture stale?"
    keys = jax.random.split(jax.random.key(42), len(gates.flat_state()))
    flat = [
        (p, leaf.replace(0.5 * jax.random.normal(k, leaf.value.shape, leaf.value.dtype)))
        for (p, leaf), k in zip(gates.flat_state(), keys)
    ]
    randomized = nnx.State.from_flat_path(flat)
    return nnx.merge_state(nnx.filter_state(params, nnx.Not(gate_filter)), randomized)


def _build_policy_state(config, model):
    params = _unblock_adarms_gates(nnx.state(model))
    tx = _optimizer.create_optimizer(
        config.optimizer, config.lr_schedule, weight_decay_mask=None
    )
    opt_state = tx.init(nnx.filter_state(params, config.trainable_filter))
    return training_utils.TrainState(
        step=900, params=params, model_def=nnx.graphdef(model), tx=tx,
        opt_state=opt_state, ema_decay=None, ema_params=None,
    )


@pytest.fixture(scope="module")
def fx():
    orig = gemma.get_config
    gemma.get_config = _patched_get_config(orig)
    try:
        config = _build_config()
        model = config.model.create(jax.random.key(0))
        policy_state = _build_policy_state(config, model)
        policy_observation = config.model.fake_obs(batch_size=_B)
        actions_demo = config.model.fake_act(batch_size=_B)

        old_model = nnx.merge(nnx.graphdef(model), policy_state.params)
        old_model.eval()
        chain = sample_chain_with_logprob(
            old_model, policy_observation, rng=jax.random.key(3),
            num_steps=config.rl.num_sde_steps, noise_level=config.rl.noise_level,
        )
        K, _, H, D = chain["x_chain"].shape
        log_prob_norm = jnp.float32(K * H) * jnp.float32(D)
        old_lp = sum_log_prob(chain["log_prob_per_step"]) / log_prob_norm

        mesh = sharding.make_mesh(1)
        q_state, value_state = _split._build_critics(
            config, model, mesh, jax.random.key(1)
        )

        yield types.SimpleNamespace(
            config=config, model=model, policy_state=policy_state,
            policy_observation=policy_observation, actions_demo=actions_demo,
            x_chain=chain["x_chain"], x_next_chain=chain["x_next_chain"],
            times=chain["times"], dt=chain["dt"], old_lp=old_lp,
            q_state=q_state, value_state=value_state,
            pg_jit=jax.jit(functools.partial(loss_and_grad_pg, config)),
            bc_jit=jax.jit(functools.partial(bc_grad_accumulate, config)),
            sa_jit=jax.jit(functools.partial(sample_and_advantage, config)),
        )
    finally:
        gemma.get_config = orig
        # See test_backbone_lora_init.py: keep the suite under the sbatch
        # runner's 32G cap — this module's jits are large (jit-1 + jit-2a/2b).
        jax.clear_caches()


def _nonzero_advantage(n=_B):
    return jnp.asarray([1.0, -0.5] * (n // 2), dtype=jnp.float32)


def _run_both_jits(fx, advantage):
    grads_pg, pg_loss, pg_aux = fx.pg_jit(
        fx.policy_state, fx.policy_observation,
        fx.x_chain, fx.x_next_chain, fx.times, fx.dt, fx.old_lp, advantage,
    )
    grads, _, loss_aux = fx.bc_jit(
        grads_pg, jax.random.key(11), fx.policy_state,
        fx.policy_observation, fx.actions_demo, pg_loss, pg_aux,
    )
    return grads_pg, grads, loss_aux


def test_adapters_receive_nonzero_pg_gradient(fx):
    grads_pg, _, _ = _run_both_jits(fx, _nonzero_advantage())
    lora_grads = nnx.filter_state(grads_pg, LORA_FILTER)
    assert len(lora_grads.flat_state()) > 0, "no adapter leaves in the PG grad tree"
    assert float(optax.global_norm(lora_grads)) > 0.0
    # The frozen set is absent from the grad tree entirely (DiffState over
    # trainable_filter): nothing in grads matches the freeze filter.
    frozen_in_grads = nnx.filter_state(grads_pg, fx.config.freeze_filter)
    assert len(frozen_in_grads.flat_state()) == 0


def test_adapters_receive_nonzero_bc_gradient(fx):
    # Zero advantage mutes PG exactly (the warmstart regime), so the combined
    # grads out of jit-2b are the BC gradient alone.
    _, grads, loss_aux = _run_both_jits(fx, jnp.zeros((_B,), dtype=jnp.float32))
    assert float(np.asarray(loss_aux["grad_norm_pg"])) == 0.0
    assert float(optax.global_norm(nnx.filter_state(grads, LORA_FILTER))) > 0.0


def test_grad_norm_lora_and_rest_are_pythagorean(fx):
    _, _, loss_aux = _run_both_jits(fx, _nonzero_advantage())
    total = float(np.asarray(loss_aux["grad_norm"]))
    lora_n = float(np.asarray(loss_aux["grad_norm_lora"]))
    rest_n = float(np.asarray(loss_aux["grad_norm_rest"]))
    assert lora_n > 0.0 and rest_n > 0.0
    # Disjoint filters over the same tree — exact in exact arithmetic; fp32
    # accumulation over the trainable tree, squared, is the same 1e-4-relative
    # situation as the law-of-cosines leg in test_grad_norm_decomposition.
    np.testing.assert_allclose(
        total**2, lora_n**2 + rest_n**2, rtol=1e-4, atol=0.0,
        err_msg="grad_norm_lora/rest do not decompose grad_norm",
    )


def test_grad_norm_lora_is_zero_without_adapters():
    # The plain dummy variant under the sibling scaffold's SigLIP-only filter:
    # no adapter leaves exist, so grad_norm_lora is 0.0 by construction and
    # grad_norm_rest carries the whole norm.
    mod = importlib.import_module("ogpo.test_grad_norm_decomposition")
    cfg = mod._build_config()
    model = cfg.model.create(jax.random.key(0))
    params = mod._make_params(cfg, model)
    policy_state = mod._build_policy_state(cfg, model, params)
    obs = cfg.model.fake_obs(batch_size=_B)
    acts = cfg.model.fake_act(batch_size=_B)

    old_model = nnx.merge(nnx.graphdef(model), params)
    old_model.eval()
    chain = sample_chain_with_logprob(
        old_model, obs, rng=jax.random.key(3),
        num_steps=cfg.rl.num_sde_steps, noise_level=cfg.rl.noise_level,
    )
    K, _, H, D = chain["x_chain"].shape
    old_lp = sum_log_prob(chain["log_prob_per_step"]) / (
        jnp.float32(K * H) * jnp.float32(D)
    )
    grads_pg, pg_loss, pg_aux = jax.jit(functools.partial(loss_and_grad_pg, cfg))(
        policy_state, obs, chain["x_chain"], chain["x_next_chain"],
        chain["times"], chain["dt"], old_lp, _nonzero_advantage(),
    )
    _, _, loss_aux = jax.jit(functools.partial(bc_grad_accumulate, cfg))(
        grads_pg, jax.random.key(11), policy_state, obs, acts, pg_loss, pg_aux,
    )
    assert float(np.asarray(loss_aux["grad_norm_lora"])) == 0.0
    np.testing.assert_allclose(
        np.asarray(loss_aux["grad_norm_rest"]), np.asarray(loss_aux["grad_norm"]),
        atol=0.0, rtol=0.0,
        err_msg="with no adapters, grad_norm_rest must equal grad_norm exactly",
    )


def test_adapter_gradients_match_with_dedup_on_and_off():
    # dedup_group_prefix tiles one prefix forward+backward at B instead of
    # scoring at B*G — "allclose-equivalent" per its design note, previously
    # only certified with a frozen backbone. With adapters trainable the cache
    # cotangent path must deliver the same adapter gradients.
    # No fx: one shared model/chain, two jits — the login node's RSS cap is
    # tight and per-variant models/samplers OOM'd the CPU backend.
    jax.clear_caches()
    orig = gemma.get_config
    gemma.get_config = _patched_get_config(orig)
    try:
        G = 2
        cfg_off = _build_config(group_num_samples=G, dedup=False)
        cfg_on = _build_config(group_num_samples=G, dedup=True)
        model = cfg_off.model.create(jax.random.key(0))
        policy_state = _build_policy_state(cfg_off, model)
        obs = cfg_off.model.fake_obs(batch_size=_B)

        expanded_obs = jax.tree.map(lambda x: jnp.repeat(x, repeats=G, axis=0), obs)
        old_model = nnx.merge(nnx.graphdef(model), policy_state.params)
        old_model.eval()
        chain = sample_chain_with_logprob(
            old_model, expanded_obs, rng=jax.random.key(3),
            num_steps=cfg_off.rl.num_sde_steps, noise_level=cfg_off.rl.noise_level,
        )
        K, _, H, D = chain["x_chain"].shape
        old_lp = sum_log_prob(chain["log_prob_per_step"]) / (
            jnp.float32(K * H) * jnp.float32(D)
        )
        results = {}
        for name, cfg in (("off", cfg_off), ("on", cfg_on)):
            grads_pg, _, _ = jax.jit(functools.partial(loss_and_grad_pg, cfg))(
                policy_state, obs, chain["x_chain"], chain["x_next_chain"],
                chain["times"], chain["dt"], old_lp, _nonzero_advantage(_B * G),
            )
            lora_grads = nnx.filter_state(grads_pg, LORA_FILTER)
            results[name] = [
                np.asarray(leaf.value) for _, leaf in lora_grads.flat_state()
            ]
            del grads_pg, lora_grads
        assert len(results["off"]) == len(results["on"]) > 0
        # NOT the split-equivalence 1e-6: the two paths run the same math with
        # different reassociation (per-sample backward at B*G vs one tiled-cache
        # backward at B) through bf16 activations, and the adapter cotangents
        # compound it over the layer scan — measured max|Δ| 2.4e-4 (~3% on the
        # affected small elements) on this fixture. The failure modes this test
        # exists for (a missing cache cotangent → zeros; a dropped tiling
        # factor → 2x) are orders of magnitude larger, and the aggregate norm
        # check below is tight.
        for a, b in zip(results["off"], results["on"]):
            np.testing.assert_allclose(a, b, atol=1e-3, rtol=0.05)
        n_off = float(np.sqrt(sum(float(np.sum(a**2)) for a in results["off"])))
        n_on = float(np.sqrt(sum(float(np.sum(b**2)) for b in results["on"])))
        assert n_off > 0.0
        np.testing.assert_allclose(n_on, n_off, rtol=0.01, atol=0.0)
    finally:
        gemma.get_config = orig


def test_critic_prefix_recompute_is_invariant_to_adapter_values(fx):
    # R1 at the jit-1 level: with critic_prefix=None the recompute zeroes the
    # adapters, so V(critic_obs) cannot depend on them — while the sampled
    # chain (drawn through the full model) does, which is the control proving
    # the adapters were actually engaged.
    def perturbed_state(transform):
        params = fx.policy_state.params
        adapters = jax.tree.map(transform, nnx.filter_state(params, LORA_FILTER))
        new_params = nnx.merge_state(
            nnx.filter_state(params, nnx.Not(LORA_FILTER)), adapters
        )
        return dataclasses.replace(fx.policy_state, params=new_params)

    ema = nnx.filter_state(fx.policy_state.params, fx.config.trainable_filter)
    rng = jax.random.key(5)
    outs = []
    for state in (
        perturbed_state(lambda x: x * 0.5),
        perturbed_state(lambda x: x * 2.0 + 0.01),
    ):
        outs.append(
            fx.sa_jit(
                rng, state, fx.q_state, fx.value_state,
                fx.policy_observation, None, ema,
            )
        )
    (x1, *_, aux1), (x2, *_, aux2) = outs
    np.testing.assert_array_equal(
        np.asarray(aux1["v_mean"]), np.asarray(aux2["v_mean"]),
        err_msg="V saw the adapters — the jit-1 recompute is not adapter-free",
    )
    assert not np.array_equal(np.asarray(x1), np.asarray(x2)), (
        "sampled chains identical across different adapters — adapters inert, "
        "the invariance above is vacuous"
    )
