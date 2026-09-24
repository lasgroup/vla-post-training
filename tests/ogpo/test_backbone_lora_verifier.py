# ruff: noqa: F722
"""INDEPENDENT VERIFIER probes for backbone LoRA (2026-08-29-backbone-lora).

Written by the step-3 verifier agent, not by the implementing session. These
are adversarial: each one either breaks a claim in the change record or fails
to break it after an honest attempt. Naming mirrors
``tests/ogpo/test_per_task_critics_verifier.py`` (the precedent for a
verifier-owned file).

Coverage map (claim letters are the verifier brief's):
  A  inertness         -> V1, V2, V5
  B  guard coverage    -> V3  (demonstrates the hole; xfail-free, it ASSERTS
                               the hole exists so a future fix flips it)
  D  R3 / dead adapter -> V4, V6
  E  adaRMS discovery  -> V7, V8, V9
  F  metrics plumbing  -> V5
"""

import dataclasses
import functools
import importlib

import flax.linen as nn
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import openpi.models.gemma as gemma
import openpi.models.lora as lora
import openpi.models.pi0_config as pi0_config
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.optimizer as _optimizer
import openpi.training.utils as training_utils

import src.training.config as _config
from src.rl.lora_utils import LORA_FILTER, zero_lora_b_params, zero_lora_params
from src.rl.ogpo.sampling import sample_chain_with_logprob, sum_log_prob
from src.rl.ogpo.update_actor import loss_and_grad_pg
from src.training.config import _make_ogpo_freeze_filter

_B = 2
_LEAF = nnx.VariableState(nnx.Param, 0.0)
_GATE_FILTER = nnx_utils.PathRegex(".*norm.*Dense_0.*")


# --------------------------------------------------------------------------
# V1/V2 — A: `_zero_matching` really is `is`-identity, and shape-preserving.
# --------------------------------------------------------------------------


class _Deep(nnx.Module):
    """A deliberately nested, lora-less tree: an empty filter result must not
    come back as a tree of empty sub-States (which would make `not matched`
    False and rebuild the tree)."""

    def __init__(self):
        self.a = nnx.Dict(b=nnx.Dict(c=nnx.Param(jnp.ones((2, 3)))))
        self.d = nnx.Param(jnp.ones((4,)))


def test_v1_zero_matching_is_identity_on_a_nested_lora_less_state():
    state = nnx.state(_Deep())
    matched = nnx.filter_state(state, LORA_FILTER)
    assert len(matched.flat_state()) == 0
    assert not matched, "an empty filter_state must be falsy or _zero_matching rebuilds"
    assert zero_lora_params(state) is state
    assert zero_lora_b_params(state) is state


class _Lora(nnx.Module):
    def __init__(self):
        self.base = nnx.Param(jnp.ones((2, 3), dtype=jnp.float32))
        self.q_einsum_lora_a = nnx.Param(jnp.ones((3, 2), dtype=jnp.float32))
        self.q_einsum_lora_b = nnx.Param(jnp.ones((2, 4), dtype=jnp.bfloat16))


def test_v2_zeroing_preserves_the_eval_shape_tree_exactly():
    # R3 rides inside `init_train_state`'s inner `init`, which is first run
    # under `jax.eval_shape` and whose OUTPUT drives fsdp_sharding + the jitted
    # init's out_shardings. If the zeroing changed any leaf's shape/dtype or
    # the tree structure, those two would disagree.
    def build(scale):
        state = nnx.state(_Lora())
        return jax.tree.map(lambda x: x * scale, state)

    plain = jax.eval_shape(build, 1.0)
    zeroed_b = jax.eval_shape(lambda s: zero_lora_b_params(build(s)), 1.0)
    zeroed_all = jax.eval_shape(lambda s: zero_lora_params(build(s)), 1.0)
    for other, label in ((zeroed_b, "lora_b"), (zeroed_all, "lora")):
        got = {p: (v.value.shape, v.value.dtype) for p, v in other.flat_state()}
        want = {p: (v.value.shape, v.value.dtype) for p, v in plain.flat_state()}
        assert got == want, f"{label} zeroing changed the shape tree: {got} != {want}"


# --------------------------------------------------------------------------
# V3 — B: the __post_init__ guard probes ONE literal adapter path.
# --------------------------------------------------------------------------

_PROBE_PATH = ("PaliGemma", "llm", "layers", "attn", "q_einsum", "lora_a")
_MLP_ADAPTER = ("PaliGemma", "llm", "layers", "mlp", "gating_einsum_lora_a")


def test_v3_guard_probe_catches_a_partial_adapter_freeze():
    """Originally this test DEMONSTRATED the hole (verifier finding F2,
    VERIFICATION.md): the guard probed one path (``attn/q_einsum/lora_a``), so
    a filter freezing only the FFN adapters constructed silently with 4 of the
    10 adapter leaves randomly initialized, never loaded and never trained.
    The guard now probes all 10 adapter paths, so the same partial filter must
    raise — this test was inverted to pin the widened guard.
    """
    partial = nnx.Any(
        _make_ogpo_freeze_filter(allow_lora=True),
        nnx_utils.PathRegex(".*mlp.*lora.*"),
    )
    freezes = nnx.filterlib.to_predicate(partial)
    assert not freezes(_PROBE_PATH, _LEAF), "attn probe path must be trainable here"
    assert freezes(_MLP_ADAPTER, _LEAF), "ffn adapters frozen — the partial freeze"

    base = _config.get_config("pi05_libero_online_ogpo_sft")
    with pytest.raises(ValueError, match="--backbone_lora"):
        dataclasses.replace(
            base,
            model=dataclasses.replace(base.model, paligemma_variant="gemma_2b_lora"),
            freeze_filter=partial,
        )
    # And a lora_b-only freeze — the other single-factor variant of the same
    # trap — is caught too.
    b_only = nnx.Any(
        _make_ogpo_freeze_filter(allow_lora=True),
        nnx_utils.PathRegex(".*lora_b.*"),
    )
    with pytest.raises(ValueError, match="--backbone_lora"):
        dataclasses.replace(
            base,
            model=dataclasses.replace(base.model, paligemma_variant="gemma_2b_lora"),
            freeze_filter=b_only,
        )


# --------------------------------------------------------------------------
# V4 — D: the dead-adapter arithmetic the R3 rationale rests on.
# --------------------------------------------------------------------------


def _einsum_grads(params):
    mod = lora.Einsum(
        shape=(4, 5),
        init_fn=nn.initializers.normal(stddev=1.0),
        lora_config=lora.LoRAConfig(rank=2, alpha=2.0),
    )
    x = jax.random.normal(jax.random.key(0), (3, 4))
    target = jax.random.normal(jax.random.key(1), (3, 5))

    def loss(p):
        y = mod.apply({"params": p}, "bd,dh->bh", x)
        return jnp.mean((y - target) ** 2)

    return jax.grad(loss)(params)


def test_v4_zeroing_both_factors_is_permanently_dead_zeroing_b_is_not():
    mod = lora.Einsum(
        shape=(4, 5),
        init_fn=nn.initializers.normal(stddev=1.0),
        lora_config=lora.LoRAConfig(rank=2, alpha=2.0),
    )
    x = jax.random.normal(jax.random.key(0), (3, 4))
    p0 = dict(mod.init(jax.random.key(1), "bd,dh->bh", x)["params"])

    both_zero = dict(p0)
    both_zero["lora_a"] = jnp.zeros_like(p0["lora_a"])
    both_zero["lora_b"] = jnp.zeros_like(p0["lora_b"])
    g = _einsum_grads(both_zero)
    assert float(jnp.abs(g["lora_a"]).max()) == 0.0
    assert float(jnp.abs(g["lora_b"]).max()) == 0.0, "init_fn=zeros: dead forever"

    b_zero = dict(p0)
    b_zero["lora_b"] = jnp.zeros_like(p0["lora_b"])
    g = _einsum_grads(b_zero)
    # Exactly the standard-LoRA-init picture: at step 0 only lora_b moves
    # (dL/da is proportional to b, which is 0), and once b is nonzero a is live.
    assert float(jnp.abs(g["lora_a"]).max()) == 0.0
    assert float(jnp.abs(g["lora_b"]).max()) > 0.0
    stepped = dict(b_zero)
    stepped["lora_b"] = b_zero["lora_b"] - 0.1 * g["lora_b"]
    g2 = _einsum_grads(stepped)
    assert float(jnp.abs(g2["lora_a"]).max()) > 0.0, "adapter did not come alive"


# --------------------------------------------------------------------------
# V5 — A/F: the two new metric keys on a tree with no adapters.
# --------------------------------------------------------------------------


def test_v5_global_norm_of_an_empty_filter_is_an_exact_float_zero():
    state = nnx.state(_Deep())
    empty = nnx.filter_state(state, LORA_FILTER)
    n = optax.global_norm(empty)
    assert float(np.asarray(n)) == 0.0
    assert np.asarray(n).dtype == np.float32, np.asarray(n).dtype
    # ...and the complement carries the whole norm, bit-exactly.
    rest = optax.global_norm(nnx.filter_state(state, nnx.Not(LORA_FILTER)))
    assert float(np.asarray(rest)) == float(np.asarray(optax.global_norm(state)))


# --------------------------------------------------------------------------
# V6..V9 — the dummy LoRA model legs.
# --------------------------------------------------------------------------

_MODEL_KW = dict(action_dim=4, action_horizon=2, max_token_len=8, pi05=True)


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


@pytest.fixture(scope="module")
def dummy_lora():
    orig = gemma.get_config
    gemma.get_config = _patched_get_config(orig)
    try:
        cfg = pi0_config.Pi0Config(
            paligemma_variant="dummy_lora", action_expert_variant="dummy", **_MODEL_KW
        )
        model = cfg.create(jax.random.key(0))
        yield cfg, model, nnx.graphdef(model), nnx.state(model), cfg.fake_obs(batch_size=_B)
    finally:
        gemma.get_config = orig
        jax.clear_caches()


def _perturb_adapters(state, fn):
    adapters = jax.tree.map(fn, nnx.filter_state(state, LORA_FILTER))
    return nnx.merge_state(nnx.filter_state(state, nnx.Not(LORA_FILTER)), adapters)


def _chain(graphdef, state, obs):
    m = nnx.merge(graphdef, state)
    m.eval()
    return sample_chain_with_logprob(
        m, obs, rng=jax.random.key(3), num_steps=3, noise_level=0.3
    )


def test_v6_adarms_gate_heads_are_zero_at_init(dummy_lora):
    # Structural, not fixture-specific: gemma.RMSNorm's adaptive branch builds
    # the modulation head with kernel_init=zeros (gemma.py:128), and
    # _gated_residual multiplies the attention/FFN branch by that gate.
    _, _, _, state, _ = dummy_lora
    gates = nnx.filter_state(state, _GATE_FILTER)
    assert len(gates.flat_state()) > 0
    for path, leaf in gates.flat_state():
        assert not np.any(np.asarray(leaf.value)), path
    # Every match is on the ACTION EXPERT branch (`_1`), never the backbone.
    for path, _ in gates.flat_state():
        assert any(str(part).endswith("_1") for part in path), path


def test_v7_zero_gates_make_the_suffix_stream_adapter_independent(dummy_lora):
    # Confirms the DIFF's fixture discovery independently: at random init the
    # sampled chain is BIT-identical under two very different adapter settings,
    # so any adapter-gradient assertion taken without _unblock_adarms_gates
    # would be measuring a zero.
    _, _, graphdef, state, obs = dummy_lora
    a = np.asarray(_chain(graphdef, _perturb_adapters(state, lambda x: x * 0.5), obs)["x_chain"])
    b = np.asarray(_chain(graphdef, _perturb_adapters(state, lambda x: x * 2.0 + 0.01), obs)["x_chain"])
    np.testing.assert_array_equal(a, b)

    # ...and the fixture's randomization is what restores the dependence.
    gates = nnx.filter_state(state, _GATE_FILTER)
    keys = jax.random.split(jax.random.key(42), len(gates.flat_state()))
    flat = [
        (p, leaf.replace(0.5 * jax.random.normal(k, leaf.value.shape, leaf.value.dtype)))
        for (p, leaf), k in zip(gates.flat_state(), keys)
    ]
    live = nnx.merge_state(
        nnx.filter_state(state, nnx.Not(_GATE_FILTER)), nnx.State.from_flat_path(flat)
    )
    a2 = np.asarray(_chain(graphdef, _perturb_adapters(live, lambda x: x * 0.5), obs)["x_chain"])
    b2 = np.asarray(_chain(graphdef, _perturb_adapters(live, lambda x: x * 2.0 + 0.01), obs)["x_chain"])
    assert not np.array_equal(a2, b2)


def test_v8_without_the_gate_fixture_the_adapter_pg_gradient_is_exactly_zero():
    # The load-bearing half of the discovery: run the REAL jit-2a on the REAL
    # production lora filter with the params as initialized (gates zero) and
    # show the adapter gradient is an exact 0. So `_unblock_adarms_gates` is
    # not cosmetics — without it test_backbone_lora_grads.py's three adapter
    # assertions would FAIL (they would not silently pass), and the pre-existing
    # suite's "trainable backbone" coverage at random init is indeed weaker
    # than it looks.
    grads_mod = importlib.import_module("ogpo.test_backbone_lora_grads")
    orig = gemma.get_config
    gemma.get_config = _patched_get_config(orig)
    try:
        cfg = grads_mod._build_config()
        model = cfg.model.create(jax.random.key(0))
        params = nnx.state(model)  # gates NOT unblocked
        tx = _optimizer.create_optimizer(cfg.optimizer, cfg.lr_schedule, weight_decay_mask=None)
        policy_state = training_utils.TrainState(
            step=900, params=params, model_def=nnx.graphdef(model), tx=tx,
            opt_state=tx.init(nnx.filter_state(params, cfg.trainable_filter)),
            ema_decay=None, ema_params=None,
        )
        obs = cfg.model.fake_obs(batch_size=_B)
        m = nnx.merge(nnx.graphdef(model), params)
        m.eval()
        chain = sample_chain_with_logprob(
            m, obs, rng=jax.random.key(3),
            num_steps=cfg.rl.num_sde_steps, noise_level=cfg.rl.noise_level,
        )
        K, _, H, D = chain["x_chain"].shape
        old_lp = sum_log_prob(chain["log_prob_per_step"]) / (jnp.float32(K * H) * jnp.float32(D))
        grads_pg, _, _ = jax.jit(functools.partial(loss_and_grad_pg, cfg))(
            policy_state, obs, chain["x_chain"], chain["x_next_chain"],
            chain["times"], chain["dt"], old_lp,
            jnp.asarray([1.0, -0.5], dtype=jnp.float32),
        )
        lora_grads = nnx.filter_state(grads_pg, LORA_FILTER)
        assert len(lora_grads.flat_state()) > 0
        assert float(optax.global_norm(lora_grads)) == 0.0, (
            "adapter PG gradient is nonzero with zero adaRMS gates — the DIFF's "
            "fixture rationale would be wrong"
        )
    finally:
        gemma.get_config = orig
        jax.clear_caches()


def test_v9_dedup_grad_tolerance_is_not_masking_a_scale_bug():
    # E: the DIFF relaxes the dedup adapter-grad equivalence to atol 1e-3 /
    # rtol 5% elementwise. Two things must hold for that to be honest:
    #   (1) the AGGREGATE norms must agree far tighter than the elementwise
    #       bound (reassociation noise on small elements, not a scale error);
    #   (2) a deliberately broken variant (cache cotangent scaled by 2) must
    #       still be REJECTED by both of the shipped tolerances.
    grads_mod = importlib.import_module("ogpo.test_backbone_lora_grads")
    orig = gemma.get_config
    gemma.get_config = _patched_get_config(orig)
    jax.clear_caches()
    try:
        G = 2
        cfg_off = grads_mod._build_config(group_num_samples=G, dedup=False)
        cfg_on = grads_mod._build_config(group_num_samples=G, dedup=True)
        model = cfg_off.model.create(jax.random.key(0))
        policy_state = grads_mod._build_policy_state(cfg_off, model)
        obs = cfg_off.model.fake_obs(batch_size=_B)
        expanded = jax.tree.map(lambda x: jnp.repeat(x, repeats=G, axis=0), obs)
        m = nnx.merge(nnx.graphdef(model), policy_state.params)
        m.eval()
        chain = sample_chain_with_logprob(
            m, expanded, rng=jax.random.key(3),
            num_steps=cfg_off.rl.num_sde_steps, noise_level=cfg_off.rl.noise_level,
        )
        K, _, H, D = chain["x_chain"].shape
        old_lp = sum_log_prob(chain["log_prob_per_step"]) / (jnp.float32(K * H) * jnp.float32(D))
        adv = jnp.asarray([1.0, -0.5] * ((_B * G) // 2), dtype=jnp.float32)
        out = {}
        for name, cfg in (("off", cfg_off), ("on", cfg_on)):
            grads_pg, _, _ = jax.jit(functools.partial(loss_and_grad_pg, cfg))(
                policy_state, obs, chain["x_chain"], chain["x_next_chain"],
                chain["times"], chain["dt"], old_lp, adv,
            )
            lg = nnx.filter_state(grads_pg, LORA_FILTER)
            out[name] = [np.asarray(l.value) for _, l in lg.flat_state()]
            del grads_pg, lg
        a_all = np.concatenate([a.ravel() for a in out["off"]])
        b_all = np.concatenate([b.ravel() for b in out["on"]])
        n_off = float(np.linalg.norm(a_all))
        n_on = float(np.linalg.norm(b_all))
        max_abs = float(np.abs(a_all - b_all).max())
        rel_norm = abs(n_on - n_off) / n_off
        cos = float(a_all @ b_all / (n_off * n_on))
        print(
            f"\n[V9] adapter grads: n_off={n_off:.6g} n_on={n_on:.6g} "
            f"|Δnorm|/norm={rel_norm:.3e} max|Δ|={max_abs:.3e} cos={cos:.9f}"
        )
        # (1) aggregate agreement is ~2 orders tighter than the shipped 1% bound
        assert rel_norm < 1e-3, rel_norm
        assert 1.0 - cos < 1e-5, cos
        # (2) a 2x cotangent scaling is rejected by BOTH shipped tolerances
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(a_all, 2.0 * b_all, atol=1e-3, rtol=0.05)
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(2.0 * n_on, n_off, rtol=0.01, atol=0.0)
        # ...and so is a dropped cotangent (adapters live only in the prefix,
        # so losing it zeroes them outright).
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(a_all, np.zeros_like(b_all), atol=1e-3, rtol=0.05)
    finally:
        gemma.get_config = orig
        jax.clear_caches()
