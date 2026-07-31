# ruff: noqa: F722
"""Direct behavioral lock on ``compose_full_params`` (B3).

At the step-900 OOM site ``ema == params``, so every operand order and filter
polarity produces the same tree — the split/composer equivalence fixture is
blind to a swapped or inverted compose there. This test perturbs ``canonical``
and ``ema`` on BOTH a frozen and a trainable leaf and pins the wiring at the
helper level: the composed output must take the trainable leaf from ``ema`` and
the frozen leaf from ``canonical``. Run once with a full-tree ``ema`` (locks the
tolerant-form guarantee that a drifted frozen EMA leaf cannot win the merge) and
once with a trainable-only ``ema`` (the post-slice production shape).
"""
import flax.nnx as nnx
import jax
import jax.numpy as jnp

import openpi.shared.nnx_utils as nnx_utils

from src.rl.ema_utils import compose_full_params


class _TwoLeaf(nnx.Module):
    """Minimal module with one frozen-by-path leaf and one trainable leaf."""

    def __init__(self):
        self.frozen = nnx.Param(jnp.ones((2, 3), dtype=jnp.float32))
        self.trainable = nnx.Param(2.0 * jnp.ones((2, 3), dtype=jnp.float32))


def _leaf(state: nnx.State, token: str):
    for path, var in state.flat_state():
        if token in "/".join(str(k) for k in path):
            return var.value
    return None


def _assert_wiring(canonical, ema, trainable_filter):
    composed = compose_full_params(canonical, ema, trainable_filter)
    # Trainable leaf comes from the EMA; frozen leaf from canonical (never the EMA).
    assert jnp.allclose(_leaf(composed, "trainable"), _leaf(ema, "trainable"))
    assert jnp.allclose(_leaf(composed, "frozen"), _leaf(canonical, "frozen"))
    ema_frozen = _leaf(ema, "frozen")
    if ema_frozen is not None:
        # Full-tree EMA: the drifted frozen EMA leaf must NOT win the merge_state
        # union — the tolerant form filters it out so frozen stays from canonical.
        assert not jnp.allclose(_leaf(composed, "frozen"), ema_frozen)


def test_compose_takes_trainable_from_ema_frozen_from_canonical():
    module = _TwoLeaf()
    canonical = nnx.state(module)
    # The exact production filter algebra (OCFG:552): trainable = Params not on a
    # frozen path. ".*frozen.*" matches the ("frozen",) attribute path.
    trainable_filter = nnx.All(nnx.Param, nnx.Not(nnx_utils.PathRegex(".*frozen.*")))

    # Perturb BOTH leaves so the frozen-from-canonical assert has teeth.
    ema_full = jax.tree.map(lambda x: x + 10.0, canonical)

    # Regime 1: full-tree EMA — a swapped/inverted compose would let the drifted
    # frozen EMA leaf win; the tolerant form must source frozen from canonical.
    _assert_wiring(canonical, ema_full, trainable_filter)

    # Regime 2: trainable-only EMA — the post-slice production shape.
    ema_trainable_only = nnx.filter_state(ema_full, trainable_filter)
    _assert_wiring(canonical, ema_trainable_only, trainable_filter)
