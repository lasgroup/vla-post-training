# Blast radius & change spec

## Intent

Add `grad_norm_pg` (jit-2a aux) and `grad_norm_bc` (jit-2b aux) so the actor's
combined `grad_norm` can be decomposed. Behavior-neutral.

## Files to touch

| File | Change |
|---|---|
| `src/rl/ogpo/update_actor.py` | jit-2a: add `"grad_norm_pg"` to `pg_aux` (20 → 21 keys). jit-2b: add `"grad_norm_bc"` to `loss_aux` (22 → 23 keys). Update the two in-source key-count comments (`:425`, `:538`). |
| `tests/ogpo/test_split_equivalence.py` | `_assert_info_close` must tolerate keys present in the split but absent from the frozen reference monolith. Add a `test_grad_norm_decomposition` leg. |
| `docs/code/rl-ogpo.md` | `:169` "20-key aux" → 21; `:381` 33-key assertion note. |
| `docs/code/tests.md` | `:244` 33-key schema note. |

## Duplication sweep — result: NO CLONES

`grep -rn "loss_and_grad_pg\|bc_grad_accumulate" --include=*.py .` (excluding
the `openpi/` and `molmospaces/` submodules) returns only:

- `src/rl/ogpo/update_actor.py` — the definitions and the `train_step` composer
  at `:621-627`, which calls both and merges `loss_aux` into `info`. **Picks up
  both keys automatically; no edit needed.**
- `src/rl/ogpo/ogpo_learner.py:174,198,563` — the two jit builds and the
  production call. **No edit needed** (see sharding note below).
- `tests/ogpo/test_split_equivalence.py:401-402` — jits both bodies.
- `tests/ogpo/test_group_dedup.py:98` — comment reference only.
- `scripts/exp_ogpo_unfrozen_backbone_memdiag.py:330` — wraps
  `_bc_grad_accumulate_jitted` in a memory tracer; key-agnostic. **No edit.**

None of the five clone families named in `CLAUDE.md` involve `src/rl/ogpo/`.

## Inheritance sweep — result: NO SUBCLASSES REACHED

`loss_and_grad_pg` / `bc_grad_accumulate` are module-level functions, not
methods, and `OGPOAgentLearner` has no subclasses. The AWR/MPO/FlowGRPO/BofN/
FSFT siblings never import them — they use
`advantage_weighted_sft/update_actor.py`. `OGPOAgentLearner.update()` is an
override that owns its own EMA advance (OQ-10), and is unchanged here.

## Sharding / donation / RNG — result: NO TIER-2 TRIGGER

- **Shardings unchanged.** `pg_aux` and the jit-2b `aux` each map to a *single*
  `self._replicated_sharding` in `out_shardings` (`ogpo_learner.py:190,209`),
  and `pg_aux` likewise as an `in_shardings` entry at `:204`. A scalar sharding
  broadcasts over the whole pytree, so adding dict keys needs no signature edit.
- **Donation unaffected.** jit-2a has `donate_argnums=()`; `grads_pg` is an
  *output* there, so reading it for a norm is an ordinary traced read. jit-2b
  donates argnum 0 (`grads_pg`); the new norm there is taken over `grads_bc`,
  a locally-produced tree, not over the donated input. Invariant G2 (a donated
  value is never read after its final use) is untouched.
- **RNG unchanged.** Neither addition draws or splits a key. jit-2a still takes
  no rng; jit-2b's arity-3 split of `policy_rng` with `bc_rng = [2]` is
  untouched, so G1 holds bit-for-bit.

Per `CLAUDE.md` these are the three Tier-2 triggers and none fire. **Tier 1.**

## Gotchas checked

- `docs/code/rl-ogpo.md` Gotchas: the RNG-arity contract and the
  donated-argument rule both apply to this file and are respected above.
- The `train_step` composer at `update_actor.py:594` is the subject of the
  split-vs-mono equivalence test and is retained; it inherits the new keys
  through `loss_aux` with no edit.

## The one real consequence: the equivalence test

`tests/ogpo/test_split_equivalence.py:426` asserts
`set(got) == set(ref)` where `ref` is a **verbatim copy of the pre-refactor
monolith** (`:142`). That copy is the artifact that breaks the split==composer
circularity and **must not be edited** — so the split will legitimately carry
two keys the reference does not.

Fix: `_assert_info_close` gains an explicit allowlist of split-only keys and
asserts (a) the reference's keys are all present and numerically equal, and
(b) the extra keys are *exactly* the expected two — so an accidental third
addition still fails the test. `_N_INFO_KEYS = 33` is asserted on `ref.info`
(`:473`), i.e. on the monolith, and therefore stays 33.

## Memory

Both norms are reductions over trees that are already materialized at that
point in their respective jits (`grads_pg` is jit-2a's output; `grads_bc` is
live for the `jax.tree.map` add on the next line). No new large transient, no
change to the live range that sized either arena.

## Expected behavior after

Two new keys in the actor info dict, surfacing as `actor/grad_norm_pg` and
`actor/grad_norm_bc`. Every other metric, and every parameter after an update,
byte-identical. `exp.py` NaN-fills the keys on critic-only steps as it already
does for the rest of the actor block.

In the `use_bc_regularization=False` branch `grad_norm_bc` is `0.0` — not a
placeholder, that branch's BC gradient genuinely is zero — keeping the key
schema static across both config branches.

## Verification plan

1. `pytest tests/ogpo` — 15 existing tests must stay green (the equivalence
   legs prove the split still reproduces the monolith numerically).
2. A new test asserting the decomposition is real, not just plumbed:
   - the two new keys exist and are finite;
   - `grad_norm_bc == 0` when the advantage is zeroed... **no** — that is the
     PG side. Concretely: with `advantage = 0` the PG gradient vanishes, so
     `grad_norm_pg == 0` and `grad_norm == grad_norm_bc`. This is exactly the
     warmstart regime the run logs were read through, so the test pins the
     interpretation those numbers were given.
   - with `bc_coeff = 0`, `grad_norm_bc == 0` and `grad_norm == grad_norm_pg`.
   - triangle inequality on the combined norm in the general case.
3. Cannot be verified locally: nothing here needs a GPU or π0.5 weights — the
   dummy PaliGemma variants in `tests/ogpo/` exercise the real code path, so
   this change is fully testable on CPU. No honest-fallback needed.
