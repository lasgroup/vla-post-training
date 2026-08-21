# What actually changed

> Amended after the initial pass to add `grad_cos_pg_bc` (the angle between the
> two gradient trees) at the user's request. Sections below cover both.

## `src/rl/ogpo/update_actor.py`

1. **jit-2a `loss_and_grad_pg`** — after `nnx.value_and_grad` returns, fold
   `optax.global_norm(grads_pg)` into the aux:
   ```python
   pg_aux = pg_aux | {"grad_norm_pg": optax.global_norm(grads_pg)}
   ```
   The norm cannot be computed inside `loss_fn` (the gradients do not exist
   there yet), hence the fold at the return rather than a new aux entry in the
   dict literal. The `20 PPO keys` comment above that literal now says why the
   key is absent from it.
2. **jit-2b `bc_grad_accumulate`** — `grad_norm_bc = optax.global_norm(grads_bc)`
   taken immediately before the accumulating `jax.tree.map`, while `grads_bc` is
   still a distinct tree; `jnp.float32(0.0)` in the `use_bc_regularization=False`
   branch (not a placeholder — that branch's BC gradient genuinely is zero, and
   emitting it keeps the key schema static across both config branches). Added
   to `loss_aux`.
3. **jit-2b, cosine** — `grad_cos_pg_bc = _tree_cosine(grads_pg, grads_bc,
   pg_aux["grad_norm_pg"], grad_norm_bc)`, also before the accumulation. Reuses
   the norm jit-2a already shipped in `pg_aux` rather than recomputing it, and
   reads the same two trees the accumulating map reads, so residency is
   unchanged. `NaN` in the `use_bc_regularization=False` branch.
4. **New module-level helper `_tree_cosine`** — sits with
   `_grpo_conservative_advantage` / `_group_baseline`. Sums per-leaf
   `sum(a*b)` via `jax.tree.map`, then one `jnp.sum(jnp.stack(...))` so the
   summation order does not depend on tree traversal.

   **The one judgment call in this change:** it returns **NaN**, not `0.0`, when
   either tree is identically zero. The angle to a zero vector is undefined, and
   `0.0` would be read as "orthogonal" — a different and wrong claim. `exp.py`
   reduces the info dict with `jnp.nanmean`, so an undefined value drops out of
   the logged window instead of dragging the mean toward zero, which is what
   would otherwise happen across the entire PG warmstart (where `grads_pg` is an
   exact zero tree). Flagged to the user as the sentinel choice; overrule to
   `0.0` if a NaN in W&B is worse for their tooling than a wrong zero.

   Key-count comments now read 25 for the jit-2 aux (21 in from jit-2a + 4 added
   here) and 36 for the learner info dict, was 33.

Nothing else in the file changed. No loss, gradient, optimizer, sharding,
donation, or RNG change — both additions are reductions over trees that are
already materialized at that point.

## `tests/ogpo/test_split_equivalence.py`

`_assert_info_close` previously asserted `set(got) == set(ref)` against the
frozen verbatim monolith, which the two new keys necessarily violate. It now
asserts (a) no reference key is missing from the split, and (b) the split-only
keys are **exactly** `_SPLIT_ONLY_INFO_KEYS = {"grad_norm_pg", "grad_norm_bc"}`
— so an unintended third addition still fails. Numeric comparison over the
reference's keys is unchanged.

`_N_INFO_KEYS = 33` is asserted on `ref.info` (the monolith) and stays 33; a
comment now says so explicitly, since the number is easy to misread as the
split's schema.

## `tests/ogpo/test_grad_norm_decomposition.py` (new, 8 tests)

Covered in `VERIFICATION.md`. Builds no critics — the chain comes from
`sampling.sample_chain_with_logprob` and the advantage is supplied by the test,
so the fixture stays ~40 lines.

## Docs

- `docs/code/rl-ogpo.md` — "20-key aux" → 21 with the reason the key is folded
  in at the return; the `test_split_equivalence` 33-key note now says it is the
  *reference's* schema and names the allowlist; new bullet for the new test file.
- `docs/code/tests.md` — module-map row, a full section for the new file, and
  the 33-key/allowlist note.
- Test count `15 → 20` (verified by `pytest --collect-only -q tests/ogpo`) in
  `tests.md`, `rl-ogpo.md`, `README.md`, `rl-learners.md`, `rl-dsrl.md`,
  `scripts.md`, `envs.md`, `training.md`, `STYLE.md`, and `CLAUDE.md`.

## Divergence from PLAN.md

- The plan said four asserts in the new test file; it landed as **eight** — a
  `bc_coeff` linearity leg was added because it is what actually pins
  `grad_norm_bc` as the *post-coefficient* contribution (without it the key
  could be the raw BC gradient and every other leg would still pass), and three
  more came in with the cosine. The strongest of those is the **law of
  cosines**, `‖g‖² = ‖g_pg‖² + ‖g_bc‖² + 2·cos·‖g_pg‖·‖g_bc‖`, which
  cross-validates the cosine against the three reported norms without the test
  needing access to the gradient trees — it catches a dot product taken over a
  mismatched traversal or a wrongly paired denominator.
- `grad_cos_pg_bc` was listed as an explicit **non-goal** in `README.md`; the
  user asked for it in a follow-up, so it landed in the same change record
  rather than a new one. The non-goal line in `README.md` is left as written
  history with this note superseding it.
- The plan did not anticipate the repo-wide `15 tests` count. Ten files assert
  it. Updating exactly the one number this change invalidates is not the
  "repo-wide cite sweep" `CLAUDE.md` step 4 prohibits, and leaving ten
  documented falsehoods behind seemed clearly worse.
