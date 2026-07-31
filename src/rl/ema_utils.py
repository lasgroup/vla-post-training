# ruff: noqa: F722
"""Compose a full param tree from a (possibly trainable-only) EMA.

``self._ema`` is stored as the TRAINABLE-ONLY subset (~11.3 GiB) so the frozen
SigLIP bf16 duplicate is off-device and the tree can be host-resident. Frozen
(and non-Param) leaves are bit-invariant across the run — the optimizer only
touches trainable leaves — so every full-model consumer recomposes them from
the current params. Filtering BOTH operands keeps them disjoint whether the EMA
is full (WP-A interim) or already trainable-only (post-slice); sourcing frozen
leaves from ``canonical_params`` is byte-identical at step 900 and drops the
frozen bf16 EMA drift at steps >=910.
"""
import flax.nnx as nnx


def compose_full_params(
    canonical_params: nnx.State,
    ema: nnx.State,
    trainable_filter: nnx.filterlib.Filter,
) -> nnx.State:
    # Take frozen + non-Param leaves from the live params and trainable leaves
    # from the EMA. Filtering the EMA too (not just the canonical params) is what
    # makes this tolerant: the two operands are disjoint whether ``ema`` is
    # full-tree or already trainable-only, so a drifted frozen EMA leaf can never
    # win the merge_state union — the frozen half always comes from canonical.
    frozen_and_rest = nnx.filter_state(canonical_params, nnx.Not(trainable_filter))
    trainable_ema = nnx.filter_state(ema, trainable_filter)
    return nnx.merge_state(frozen_and_rest, trainable_ema)
