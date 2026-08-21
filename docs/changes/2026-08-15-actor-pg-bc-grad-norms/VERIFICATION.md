# Verification

> Two passes: the initial `grad_norm_pg` / `grad_norm_bc` change, then an
> amendment adding `grad_cos_pg_bc`. Results below are from the amended tree
> unless a section says otherwise.

## Scope caveat — the independent verifier agent was NOT run

`CLAUDE.md` step 3 calls for a fresh-context agent (Opus 4.8, xhigh) to receive
the change spec + diff and adversarially verify. This session carries a standing
instruction not to spawn agents unless the user asks, so **the implementing
session wrote and ran the tests itself**. That is a weaker signal than the
workflow specifies — the tests below were written by the same reasoning that
wrote the change. Offered to the user; not run.

Everything else in step 3 was done: pytest-native plain asserts, dummy PaliGemma
variants, module-scoped fixture, explicit tolerance with a stated reason. No
honest-fallback was needed — this change needs no GPU and no π0.5 weights.

## Result: `pytest tests/ogpo` — 18 passed, 5 failed (992s)

```
..........FF....F...FF.                                                  [100%]
FAILED test_group_dedup.py::test_repeat_cache_matches_expanded_cache
FAILED test_group_dedup.py::test_rescorer_cached_matches_uncached_forward_and_grad
FAILED test_sampling.py::test_scanned_rescorer_grad_matches_unrolled
FAILED test_split_equivalence.py::test_leg_c_rng_arity_three_and_indexing
FAILED test_split_equivalence.py::test_leg_d_none_branch_recompute_matches_original
```

**All 8 new tests passed** (positions 2-9). No failure is attributable to this
change; see the two subsections below.

### The suite is not a stable signal on this machine

The first pass (before the cosine) failed a DIFFERENT set:
`test_group_dedup` ×2 and `test_leg_d`. The amended run added
`test_scanned_rescorer_grad_matches_unrolled` and `test_leg_c` — both of which
had passed minutes earlier. Isolating them:

| | `test_scanned_rescorer_grad_matches_unrolled` | `test_leg_c_rng_arity...` |
|---|---|---|
| patched tree, isolated | **pass** | **pass** |
| clean tree, isolated (stashed) | **FAIL** | pass |
| patched tree, full suite | FAIL | FAIL |

One of them fails on the CLEAN tree and passes on the PATCHED tree in the same
isolated configuration — the opposite of a regression. These two are
**nondeterministic on this CPU backend**, flipping with run and with suite
ordering. Same root cause as the three hard failures: tight `atol` (1e-5/1e-6)
against XLA CPU reductions whose accumulation order is not pinned.

`test_leg_c` failing intermittently deserves a call-out because it is the RNG
arity contract that `CLAUDE.md` names as a silent correctness invariant. It
passes in isolation on the patched tree, and this change draws no rng and
alters no split — but a flaky guard on that contract is worth fixing on its own.

### The 3 hard failures are pre-existing (unchanged from the first pass)

Verified by `git stash push -- src/rl/ogpo/update_actor.py tests/ogpo/test_split_equivalence.py`
and re-running exactly those three on the clean tree:

```
FFF                                                     [100%]
FAILED tests/ogpo/test_group_dedup.py::test_repeat_cache_matches_expanded_cache
FAILED tests/ogpo/test_group_dedup.py::test_rescorer_cached_matches_uncached_forward_and_grad
FAILED tests/ogpo/test_split_equivalence.py::test_leg_d_none_branch_recompute_matches_original
3 failed, 13 warnings in 227.36s (0:03:47)
```

Stash popped and the diff confirmed restored afterwards.

### The 3 pre-existing failures, verbatim

1. `test_repeat_cache_matches_expanded_cache` (`test_group_dedup.py:82`) —
   `assert np.allclose(a32, b32, atol=1e-05)` fails on a KV-cache leaf.
2. `test_rescorer_cached_matches_uncached_forward_and_grad`
   (`test_group_dedup.py:126`) —
   `AssertionError: 2 grad leaf(s) exceed atol=1e-05: [(38, 0.296875), (39, 0.2509765625)]`
   (the forward leg passes; only the gradient leg fails).
3. `test_leg_d_none_branch_recompute_matches_original`
   (`test_split_equivalence.py:582` clean / `:598` patched) —
   `None-branch advantage == recompute-sidecar advantage: max|Δ|=0.008434295654296875 exceeds atol=1e-06`
   (`a = [2.9283168, 3.560798]` vs `b = [2.935904, 3.5692322]`).

All three compare a **B-batch** prefix forward against a **B·G-batch** one, or an
in-jit recompute against an out-of-jit one. That is the exact hazard
`sampling.py:256-258` documents — "batch-size-dependent XLA tiling can shift
`v_t` by float ulps, so equality is `allclose`, not bitwise". On this CPU backend
the drift is ~1e-2, three to four orders of magnitude above the tolerances those
tests chose. Not diagnosed further; out of scope for this change.

**Consequence for the working agreement:** `CLAUDE.md` states `pytest tests/ogpo`
passes and runs in "seconds". On this machine it does neither — 4 fail, and the
suite takes ~20 minutes (3 tests alone took 227s). Flagged to the user; **not
edited unilaterally**, per the instruction to record new inconsistencies rather
than decide them.

### The 4th failure was mine, and is fixed

`test_bc_norm_carries_the_coefficient` initially failed at `rtol=1e-5`:

```
Not equal to tolerance rtol=1e-05, atol=0
grad_norm_bc must scale linearly with bc_coeff
Max relative difference: 0.00037865
 x: array(36.667694)     # bc_coeff = 3
 y: array(36.653815)     # 3 * (bc_coeff = 1)
```

Not a defect in the change: the linearity is exact in exact arithmetic, but the
backward runs through the bf16 frozen vision tower and the dummy Gemma stack, so
a 3× seed cotangent is not a bitwise-3× result. Tolerance raised to `rtol=5e-3`
with the reason recorded in-source — ~12× headroom over the measured 4e-4, while
still failing loudly (~200%) if the coefficient were dropped entirely.

Re-run after the fix:

```
tests/ogpo/test_grad_norm_decomposition.py  .....  [100%]
5 passed, 13 warnings in 197.28s (0:03:17)
```

After the cosine amendment, the file was re-run in isolation to confirm the new
tests are not themselves flaky on this backend:

```
tests/ogpo/test_grad_norm_decomposition.py  ........  [100%]
8 passed, 13 warnings in 232.65s (0:03:52)
```

Twice green in isolation plus green inside the full suite; the law-of-cosines
leg at `rtol=1e-4` was the one to watch and did not move.

## What the passing tests establish

- **The split-equivalence suite still holds with the two new keys.** Legs a,
  a', b (splice), c (RNG arity), and stored-prefix threading all pass; the
  loosened `_assert_info_close` did not weaken the numeric comparison, which
  still runs over every reference key. The only failing leg (d) fails identically
  without the change.
- **`grad_norm_pg == 0` exactly** when the advantage is zeroed, and
  `grad_norm == grad_norm_bc` there. This certifies the reading taken from the
  existing `mt4_*` runs, where warmstart `actor/grad_norm` was interpreted as
  `‖g_BC‖`.
- **`grad_norm_bc == 0` exactly** at `bc_coeff = 0`, with
  `grad_norm == grad_norm_pg`.
- **Triangle inequality** holds on the combined norm, which is what makes the
  two keys usable for attribution.
- **The cosine is consistent with the three norms** via the law of cosines
  `‖g‖² = ‖g_pg‖² + ‖g_bc‖² + 2·cos·‖g_pg‖·‖g_bc‖` at `rtol=1e-4`. This is the
  leg that would catch a dot product summed over a mismatched tree traversal or
  a wrongly paired denominator — neither of which the range check would see.
- **The cosine is NaN, not 0**, whenever either gradient is exactly zero.
- **`grad_norm_bc` scales linearly with `bc_coeff`**, pinning it as the
  post-coefficient contribution rather than the raw BC gradient.

## What could NOT be verified

- **No GPU run.** Memory neutrality is argued structurally (both norms reduce
  trees already materialized at that point) and was not measured. The claim to
  be skeptical of is that adding `optax.global_norm(grads_pg)` inside jit-2a
  does not extend `grads_pg`'s live range — believed safe because it is that
  jit's output regardless, but not confirmed against a device-bytes gate.
- **No end-to-end run.** The keys were not observed arriving in W&B under
  `actor/grad_norm_pg` / `actor/grad_norm_bc`; the `actor/` prefixing is
  inherited plumbing (`ogpo_learner.py:610`) that this change does not touch.
- **The `use_bc_regularization=False` branch** is not covered by a test. Its
  `grad_norm_bc = 0.0` / `grad_cos_pg_bc = NaN` is a three-line static branch,
  but it is untested.
- **The NaN convention end-to-end.** `exp.py` reduces with `jnp.nanmean`
  (`exp.py:179`), so a window where every policy step has an undefined cosine
  logs NaN rather than a misleading 0. That reduction path was read, not
  executed — no run was launched.

## Gotchas discovered

- `pytest tests/ogpo` takes ~20 minutes on this machine, not "seconds". Piping
  it through `tail` swallows all progress until exit, and a 900s timeout kills
  it mid-suite — run it to a log file unbuffered.
- `CLAUDE.md` and `docs/code/` are now in `.gitignore` (`.gitignore:24-25`), so
  edits to them are local-only and invisible to `git diff`. `docs/changes/` is
  still tracked.
