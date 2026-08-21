# Split the actor's `grad_norm` into its PG and BC components

**Tier 1.** Date: 2026-08-15.

## What

Emit two new actor metrics alongside the existing combined `actor/grad_norm`:

- `actor/grad_norm_pg` — `‖∇θ pg_loss‖` from jit-2a, before the BC anchor is
  accumulated into it.
- `actor/grad_norm_bc` — `‖∇θ (bc_coeff · bc_loss)‖` from jit-2b, before the
  accumulation. Note this is the **post-coefficient** tree, i.e. the actual
  contribution to the sum, not the raw BC gradient.

## Why

`actor/grad_norm` is computed in jit-2b *after* `grads_bc` has been accumulated
into the donated `grads_pg` (`update_actor.py:542`), so it reports only the
combined norm. There is currently no way to read off how much of the actor's
update comes from the policy gradient versus the BC anchor.

The only existing handle is an accident of the warmstart: while
`training_steps < pg_start_step` the advantage is multiplied by exactly `0.0`,
so `pg_loss` and its gradient are identically zero and `actor/grad_norm`
degenerates to `‖g_BC‖`. Reading `run_store/wandb` for `mt4_v0_s0`
(`offline-run-20260811_161927-xqr360bw`), that gives:

| phase | `actor/grad_norm` (median) |
|---|---|
| warmstart 900–20k (= `‖g_BC‖`) | 0.0406 |
| full PG 25–50k (combined) | 0.4440 |
| full PG 80–100k (combined) | 0.3087 |

which bounds `‖g_PG‖ ∈ [0.404, 0.484]` at 25–50k by the triangle inequality —
roughly 10–12× the BC anchor. That estimate is only available during the
warmstart, is a bound rather than a measurement, and disappears entirely for
any arm run with `PG_START=0`.

This makes `bc_coeff` tunable against a measured quantity instead of an
inferred one, and gives the stability study a direct read on whether the BC
anchor is a meaningful brake or a rounding error once PG is at full strength.

## Non-goals

- Cosine similarity between the two gradient trees. It is the more informative
  diagnostic (norms cannot tell you whether the anchor *opposes* the policy
  gradient) and both trees are live in jit-2b, but it is a separate metric with
  a separate cost and is not part of this change.
- Changing any training behavior. Both additions are pure reductions over trees
  that already exist; no loss, gradient, optimizer state, or RNG draw changes.

See [BLAST-RADIUS.md](./BLAST-RADIUS.md) for the change spec.
