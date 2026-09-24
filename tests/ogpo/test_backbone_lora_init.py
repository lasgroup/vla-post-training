"""LoRA surgery + init-identity tests (docs/changes/2026-08-29-backbone-lora/).

The two properties this file certifies:

* R1 mechanism — ``zero_lora_params`` turns an adapted model into the base
  model exactly: the prefix representation the critic consumes is bit-identical
  to a lora-less model with the same base weights, and invariant to whatever
  the adapters hold.
* R3 mechanism — ``zero_lora_b_params`` makes the freshly-initialized LoRA
  model identical to the non-LoRA twin (identity at init), while ``lora_a``
  stays nonzero so adapter gradients are live.

Fixture: there is no ``dummy_lora`` variant upstream and ``Pi0.__init__``
hardcodes ``gemma.get_config``, so the fixture monkeypatches ``get_config`` to
serve a dummy-width config with rank-4 adapters — no submodule edit. The
lora-less twin gets its weights *copied* from the LoRA model (never re-seeded:
linen's per-param RNG derivation is scope/counter based, so equal seeds do not
guarantee equal base weights across different module trees).
"""

import dataclasses

import flax.linen as nn
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import openpi.models.gemma as gemma
import openpi.models.lora as lora
import openpi.shared.nnx_utils as nnx_utils
from openpi.models import pi0_config
from src.rl.lora_utils import (
    LORA_B_FILTER,
    LORA_FILTER,
    zero_lora_b_params,
    zero_lora_params,
)

_MODEL_KW = dict(action_dim=4, action_horizon=2, max_token_len=8, pi05=True)


@pytest.fixture(scope="module")
def lora_pair():
    """(lora model, base twin, graphdefs, obs) with identical base weights."""
    orig_get_config = gemma.get_config

    def patched(variant):
        if variant == "dummy_lora":
            return dataclasses.replace(
                orig_get_config("dummy"),
                lora_configs={
                    "attn": lora.LoRAConfig(rank=4, alpha=4.0),
                    "ffn": lora.LoRAConfig(rank=4, alpha=4.0),
                },
            )
        return orig_get_config(variant)

    gemma.get_config = patched
    try:
        lora_cfg = pi0_config.Pi0Config(
            paligemma_variant="dummy_lora", action_expert_variant="dummy", **_MODEL_KW
        )
        base_cfg = pi0_config.Pi0Config(
            paligemma_variant="dummy", action_expert_variant="dummy", **_MODEL_KW
        )
        model_lora = lora_cfg.create(jax.random.key(0))
        model_base = base_cfg.create(jax.random.key(1))

        lora_state = nnx.state(model_lora)
        base_state = nnx.state(model_base)
        lora_paths = {p for p, _ in lora_state.flat_state()}
        base_paths = {p for p, _ in base_state.flat_state()}
        adapter_paths = lora_paths - base_paths
        # Schema assertion: the two trees differ by exactly the adapter leaves.
        assert base_paths <= lora_paths
        assert adapter_paths
        assert all(any("lora" in str(part) for part in p) for p in adapter_paths)

        # Copy the shared (base) weights from the LoRA model into the twin.
        nnx.update(model_base, nnx.filter_state(lora_state, nnx.Not(LORA_FILTER)))

        obs = lora_cfg.fake_obs(batch_size=2)
        yield model_lora, model_base, obs
    finally:
        gemma.get_config = orig_get_config
        # The full suite runs under the sbatch runner's 32G cap and this module
        # precedes the heavy split-equivalence compiles alphabetically — drop
        # this module's compiled executables so they don't stack on top.
        jax.clear_caches()


def _prefix(model, obs):
    rep = model.get_prefix_rep(obs)
    return rep[0] if isinstance(rep, tuple) else rep


def test_zero_lora_params_is_identity_without_adapters(lora_pair):
    _, model_base, _ = lora_pair
    state = nnx.state(model_base)
    assert zero_lora_params(state) is state
    assert zero_lora_b_params(state) is state


def test_zero_lora_params_zeroes_every_adapter_leaf(lora_pair):
    model_lora, _, _ = lora_pair
    state = nnx.state(model_lora)
    zeroed = zero_lora_params(state)
    n_adapters = 0
    for path, leaf in zeroed.flat_state():
        orig = dict(state.flat_state())[path]
        if any("lora" in str(part) for part in path):
            n_adapters += 1
            assert not np.any(np.asarray(leaf.value)), path
        else:
            assert np.array_equal(np.asarray(leaf.value), np.asarray(orig.value)), path
    assert n_adapters > 0


def test_zeroed_lora_prefix_equals_the_base_model_prefix(lora_pair):
    # THE R1 key property: the critic-prefix forward through the adapter-zeroed
    # model is the base model's forward, exactly.
    model_lora, model_base, obs = lora_pair
    graphdef = nnx.graphdef(model_lora)
    zeroed_model = nnx.merge(graphdef, zero_lora_params(nnx.state(model_lora)))
    got = np.asarray(_prefix(zeroed_model, obs))
    want = np.asarray(_prefix(model_base, obs))
    np.testing.assert_array_equal(got, want)


def test_prefix_is_invariant_to_adapter_values(lora_pair):
    model_lora, _, obs = lora_pair
    graphdef = nnx.graphdef(model_lora)
    state = nnx.state(model_lora)

    def with_adapters(transform):
        # Rescale/shift the existing (nonzero, normal(0.01)) adapter leaves —
        # two different transforms give two genuinely different adapter
        # settings without any RNG plumbing. Base leaves untouched.
        adapters = jax.tree.map(transform, nnx.filter_state(state, LORA_FILTER))
        return nnx.merge_state(nnx.filter_state(state, nnx.Not(LORA_FILTER)), adapters)

    s1 = with_adapters(lambda x: x * 0.5)
    s2 = with_adapters(lambda x: x * 2.0 + 0.01)
    z1 = np.asarray(_prefix(nnx.merge(graphdef, zero_lora_params(s1)), obs))
    z2 = np.asarray(_prefix(nnx.merge(graphdef, zero_lora_params(s2)), obs))
    np.testing.assert_array_equal(z1, z2)

    # Negative control: without the zeroing, different adapters must change the
    # prefix — otherwise the invariance above is vacuous.
    u1 = np.asarray(_prefix(nnx.merge(graphdef, s1), obs))
    u2 = np.asarray(_prefix(nnx.merge(graphdef, s2), obs))
    assert not np.array_equal(u1, u2)


def test_lora_b_zeroing_makes_the_model_identical_to_the_non_lora_twin(lora_pair):
    # The R3 differential: after zero_lora_b_params the freshly-initialized
    # LoRA model IS the SFT policy (here: the base twin) — the full sampled SDE
    # chain matches, not just one forward.
    from src.rl.ogpo.sampling import sample_chain_with_logprob

    model_lora, model_base, obs = lora_pair
    graphdef = nnx.graphdef(model_lora)
    init_model = nnx.merge(graphdef, zero_lora_b_params(nnx.state(model_lora)))

    kw = dict(rng=jax.random.key(3), num_steps=3, noise_level=0.3)
    pack_lora = sample_chain_with_logprob(init_model, obs, **kw)
    pack_base = sample_chain_with_logprob(model_base, obs, **kw)
    for key in ("actions", "x_chain", "x_next_chain", "log_prob_per_step"):
        np.testing.assert_array_equal(
            np.asarray(pack_lora[key]), np.asarray(pack_base[key]), err_msg=key
        )
    # The chain comparison alone is weak at init: pi05's adaRMS gate heads are
    # zero-initialized, so the suffix stream bypasses attention and the chain
    # would match even with wrong adapters. The prefix stream has no adaRMS —
    # it is live at init (this file's negative control proves it) — so its
    # equality is the decisive leg of the differential.
    np.testing.assert_array_equal(
        np.asarray(_prefix(init_model, obs)), np.asarray(_prefix(model_base, obs))
    )


def test_zeroing_only_lora_b_keeps_lora_a_nonzero(lora_pair):
    # Pins the dead-adapter trap: init_fn=zeros would zero BOTH factors and
    # each factor's gradient is proportional to the other — permanently dead.
    model_lora, _, _ = lora_pair
    zeroed = zero_lora_b_params(nnx.state(model_lora))
    lora_a = nnx.filter_state(zeroed, nnx.All(LORA_FILTER, nnx.Not(LORA_B_FILTER)))
    lora_b = nnx.filter_state(zeroed, LORA_B_FILTER)
    assert len(lora_a.flat_state()) == len(lora_b.flat_state())
    for path, leaf in lora_a.flat_state():
        assert np.any(np.asarray(leaf.value)), f"lora_a zeroed at {path}"
    for path, leaf in lora_b.flat_state():
        assert not np.any(np.asarray(leaf.value)), f"lora_b nonzero at {path}"


def test_lora_einsum_is_an_exact_identity_at_lora_b_zero():
    # Unit-level exactness, independent of Pi0 fusion: Einsum with lora_b == 0
    # equals the plain einsum bit-for-bit (the adapter term is an exact zero).
    cfg = lora.LoRAConfig(rank=2, alpha=2.0)
    mod = lora.Einsum(
        shape=(4, 5), init_fn=nn.initializers.normal(stddev=1.0), lora_config=cfg
    )
    x = jax.random.normal(jax.random.key(0), (2, 4))
    variables = mod.init(jax.random.key(1), "bd,dh->bh", x)
    zeroed = {"params": dict(variables["params"])}
    zeroed["params"]["lora_b"] = jnp.zeros_like(variables["params"]["lora_b"])

    plain = lora.Einsum(shape=(4, 5), init_fn=nn.initializers.normal(stddev=1.0))
    got = mod.apply(zeroed, "bd,dh->bh", x)
    want = plain.apply({"params": {"w": variables["params"]["w"]}}, "bd,dh->bh", x)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_state_map_would_not_have_zeroed_lora_b(lora_pair):
    # Regression pin on the reason lora_utils avoids nnx_utils.state_map: under
    # flax 0.10.6 it is a silent no-op (bare path tuples tested against a set
    # of (path, value) pairs). If this test ever FAILS, state_map got fixed —
    # only then may lora_utils be "simplified" onto it.
    model_lora, _, _ = lora_pair
    state = nnx.state(model_lora)
    out = nnx_utils.state_map(
        state, LORA_B_FILTER, lambda p: p.replace(jnp.zeros_like(p.value))
    )
    b_leaves = nnx.filter_state(out, LORA_B_FILTER)
    assert any(np.any(np.asarray(leaf.value)) for _, leaf in b_leaves.flat_state())
