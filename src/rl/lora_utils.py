"""LoRA adapter surgery on a param state.

R1 (docs/changes/2026-08-29-backbone-lora/): the critic's prefix embedding must
come from the UNADAPTED backbone — value and policy must not share the adapting
representation, and the buffer's stored prefixes must stay a pure function of
the observation so the existing (frozen-backbone) arms remain comparable and
the staleness problem cannot arise.

``zero_lora_params`` makes that structural. With both factors zeroed the
adapter term is EXACTLY zero — ``lora.Einsum`` adds ``lora * scaling_value``
(openpi lora.py:59-63) and ``lora.FeedForward._dot`` adds ``x @ a @ b``
(lora.py:144-148) — so the forward is bit-identical to a lora-less model with
the same base weights. Everything ``Pi0.get_prefix_rep`` reads is then frozen
(SigLIP under ``.*PaliGemma/img.*``; the stack-0 LLM under ``.*llm.*`` minus
``_1``), which additionally makes the critic prefix invariant to params-vs-EMA
and to the training step.

``zero_lora_b_params`` is the fresh-init fix: openpi's ``LoRAConfig`` has a
SINGLE ``init_fn`` for both factors (lora.py:20, :51-52, :113-120), so as
shipped ``lora_b`` is normal(0.01) and the model at step 0 is NOT the loaded
SFT policy. ``init_fn=zeros`` is not the fix — each factor's gradient is
proportional to the other, so zeroing both dead-ends the adapters forever.
a-random / b-zero gives identity at init AND live gradients.

Deliberately NOT ``nnx_utils.state_map``: that helper is a silent no-op under
flax 0.10.6 (it tests bare path tuples against a set of ``(path, value)``
pairs, openpi/src/openpi/shared/nnx_utils.py:66-69). These helpers use the
``filter_state`` / ``merge_state`` primitives ``src/rl/ema_utils.py`` already
relies on.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import openpi.shared.nnx_utils as nnx_utils

# On a gemma_2b_lora Pi0 tree these match exactly the 10 / 5 adapter leaves
# (enumerated in docs/changes/2026-08-29-backbone-lora/BLAST-RADIUS.md); no
# base param name contains the substring.
LORA_FILTER = nnx_utils.PathRegex(".*lora.*")
LORA_B_FILTER = nnx_utils.PathRegex(".*lora_b.*")


def _zero_matching(params: nnx.State, filter_) -> nnx.State:
    matched = nnx.filter_state(params, filter_)
    if not matched:
        # Exact identity on a lora-less tree: no rebuild, no cost. This is what
        # lets every call site stay unconditional for non-LoRA configs.
        return params
    return nnx.merge_state(
        nnx.filter_state(params, nnx.Not(filter_)),
        jax.tree.map(jnp.zeros_like, matched),
    )


def zero_lora_params(params: nnx.State) -> nnx.State:
    """Params with every ``.*lora.*`` leaf replaced by zeros — the base model."""
    return _zero_matching(params, LORA_FILTER)


def zero_lora_b_params(params: nnx.State) -> nnx.State:
    """Params with every ``.*lora_b.*`` leaf zeroed — standard LoRA init.

    Fresh-init only; never apply on a resume path (the trained ``lora_b`` is
    exactly what a resume must preserve).
    """
    return _zero_matching(params, LORA_B_FILTER)
