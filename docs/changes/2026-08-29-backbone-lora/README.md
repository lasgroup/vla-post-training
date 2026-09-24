# Backbone LoRA for OGPO (LLM adapters only, SigLIP stays frozen)

**Tier:** 2 — changes `trainable_filter` (which feeds the EMA slicing, the jit
in/out shardings, and both `nnx.DiffState` grad paths), adds a registered config,
and invalidates the frozen-backbone contract stated in `update_actor.py:27-29`.
**Status:** discovery pass complete. **No source touched. No plan approved.**
Next step is the plan phase (plan mode), which must build `PLAN.md` from this
record and `BLAST-RADIUS.md`.
**Date:** 2026-08-29 · **Branch at time of writing:** `pranav_ref_ogpo`

---

## Why

The strongest prior in the repo says the frozen backbone is leaving performance on
the table: in the libero44 ledger (`docs/ogpo_experiment_notes_libero44.md:13-22`)
every unfrozen-backbone arm ran healthy PPO (alive ~0.5–0.9) and arm (iv) hit
100% @10k, while every frozen arm shows "PPO dead (clip)". The full-unfrozen arm
was retired for memory (the entire host-EMA/jit-split architecture in commit
`4818b57` was built to make it fit), and the shipped compromise —
`scripts/ws_bcbb_pipeline.sh`, backbone moved in a BC-only stage A then re-frozen
for PPO — exists precisely because a backbone that moves *during* PPO breaks the
stored-prefix contract the critic depends on.

LoRA is the middle point: adapters on the PaliGemma LLM only (+27.9M trainable
params, +6.5%), SigLIP and the base 2B weights stay frozen, the action expert and
action heads stay fully trainable as today. Steady-state memory cost is ~0.5 GB;
the open cost is the newly-live backward through the backbone (see
`BLAST-RADIUS.md` §Memory).

## What is being built

A new registered OGPO config (working name `pi05_libero_online_ogpo_lora`) with:

- `model=Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False,
  paligemma_variant="gemma_2b_lora")` — rank-16, α=16 adapters on q/kv/attn_vec
  einsums + both FFN matmuls, all 18 stack-0 layers. Action expert stays
  `gemma_300m` (no adapters).
- A lora-aware freeze filter: today's `_make_ogpo_freeze_filter` plus
  `nnx.Not(PathRegex(".*lora.*"))` on the LLM branch. Verified: exactly the 19
  current trainable leaves + the 10 adapter leaves become trainable; SigLIP and
  the 2B base stay frozen.
- **`lora_b` zeroed at init, repo-side.** openpi initializes *both* LoRA factors
  `normal(0.01)` (a measured 2–6% per-einsum perturbation of the loaded policy at
  step 0), and `LoRAConfig` has a single `init_fn` for both factors — passing
  `zeros` would zero `lora_a` too and permanently kill adapter gradients. The fix
  is zeroing the five `.*lora_b.*` leaves after model init, which also makes
  "LoRA model at init ≡ SFT policy" a true, testable property.
- A `__post_init__` guard raising when a lora model variant meets a freeze filter
  that freezes any adapter (the CLI trap in `BLAST-RADIUS.md` §Gotchas is live
  today and produces a randomly-perturbed, permanently-untrainable backbone with
  every health metric looking normal).
- A recipe knob (`LORA=1` selecting the config name) in the OGPO recipes.

## Settled by the maintainer (inputs, not open questions)

| # | Decision |
|---|---|
| S1 | LoRA on the **VLM backbone (PaliGemma LLM stack 0) only**. |
| S2 | **SigLIP stays frozen** — explicitly not training the vision encoder. |
| S3 | Action expert + action heads stay **fully trainable** (no adapters there). |

## Headline open decisions for the plan phase

Full list in `BLAST-RADIUS.md` §Residual decisions.

1. **Stored-prefix staleness policy** — the critic consumes backbone features
   cached in the replay buffer at collection time; a training backbone makes them
   drift under the critic for up to the full run (nothing is evicted at current
   capacity). This is the research blast radius, not an implementation detail.
2. **One blocker to fix**: `create_trained_policy` at learner construction loads
   the checkpoint through the strict `BaseModelConfig.load` path and raises on
   the missing lora keys — the first LoRA run dies before training without a fix.
3. Whether the global grad clip (norm 1.0 over the whole trainable tree) stays
   shared between action expert and adapters.

## Non-goals

- No SigLIP training, no action-expert LoRA, no Molmo LoRA
  (`make_base_molmo_config` takes no kwargs).
- No behavior change to any existing config: every current name keeps its exact
  filter and param tree, bit-identical.
- No openpi submodule edits (the `lora_b` fix is repo-side; the upstream FFN
  scaling omission is documented, not fixed).
- No warm-start from a non-LoRA OGPO checkpoint (new-run only, enforced loudly
  by orbax).
- Fixing the pre-existing `state_map` bf16 no-op is **separate triage**, not this
  change (see `BLAST-RADIUS.md` §Standing defects).
