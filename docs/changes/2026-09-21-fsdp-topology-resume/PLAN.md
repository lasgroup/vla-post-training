# Cross-topology (`fsdp_devices`) checkpoint resume

## Context

The maintainer runs OGPO multitask training on a shared cluster where GPU
availability fluctuates — they want to be able to resume a run under a
*different* `fsdp_devices` than it was checkpointed with (e.g. checkpointed
on 1 GPU, resumed on 2 GPUs when more become available, or vice versa).

This is currently broken. `init_train_state` already computes the correct
target sharding for the *current* run's mesh
(`src/rl/filtered_sft_agent/filtered_sft_learner.py:199`), but the restore
call at `:318-324` only forwards the unsharded state, never the sharding —
so orbax falls back to rebuilding the *saved* device-id sharding from the
checkpoint's metadata file. Result: a 2→1 GPU resume hard-errors inside
orbax (already hit for real: job 10324589, 2026-09-05,
`docs/changes/2026-09-07-candidate-q-spread-rollouts/BLAST-RADIUS.md:83-97`);
a 1→2 GPU resume also fails, loudly, the first time a jit touches a
≥2-device-mesh leaf under the stale sharding. Neither direction silently
corrupts anything — both fail loudly — but neither works.

The RL critic/value checkpoint path (`AdvantageWeightedSFTLearner`) already
does this correctly: it passes a concrete, current-mesh target into its
`StandardCheckpointer.restore()` call, so it reshards automatically. This
change brings the main policy `train_state` restore up to the same pattern.

Full investigation trail: `docs/changes/2026-09-21-fsdp-topology-resume/README.md`
and `BLAST-RADIUS.md` (Tier-2 discovery pass, source-verified, includes CPU
probes proving the mechanism).

## Decisions

- **Placement: `src/`-only**, not the `openpi` submodule. `openpi` is
  currently a clean submodule (`best_practices.md:316-318`: helpers that
  would naturally be submodule methods stay in this repo so the submodule
  stays clean). Bonus: leaving `openpi.training.checkpoints.restore_state`
  untouched gives the mandatory differential test (CLAUDE.md step 3) a free
  verbatim pre-refactor reference to diff against in the same process.
- **Params/EMA item sharding: mirror the save-side split** (not a uniform
  FSDP target). `save_checkpoint` writes this item *mixed* — replicated for
  trainable/EMA leaves, FSDP-sharded for frozen leaves
  (`filtered_sft_learner.py:757-767`, `src/rl/ema_utils.py:16-28`). Requesting
  that same mixed layout on restore keeps same-topology (`M == N`) resumes
  byte- and placement-identical to today, with zero memory-profile change.
  A uniform FSDP target would be simpler but would change even today's
  working single-GPU resume path (an unmeasured memory spike from an
  FSDP→all-gather-to-replicated round trip on the ~11.3 GiB EMA, in a
  codebase where `del ema_dev` is already load-bearing to stay under a
  ~34.7 GiB floor) — rejected.
- **`scripts/probe_critic_action_sensitivity.py:130`**: excluded. Same bug,
  but a throwaway analysis probe off the training path with no mesh at all —
  fixing it is a separate small change. Record as knowingly-not-fixed in
  `DIFF.md`.
- **`dsrl_env.py`'s `init_train_state` clone**: does NOT have this bug (no
  `resume` parameter, no short-circuit branch) — leave untouched, do not
  "sync" it, that would *import* the bug.
- **`fsdp_devices` is not added to the resume manifest.** Nothing validates
  it today and nothing needs to once restore is topology-agnostic.

## Implementation

All changes in `src/rl/filtered_sft_agent/filtered_sft_learner.py`.

1. **Imports** (third-party group, ~line 10-16): add
   `import orbax.checkpoint as ocp` and
   `from orbax.checkpoint.checkpoint_utils import construct_restore_args`.

2. **New module-level helper `_params_item_sharding`** (placed after
   `init_train_state`, ~line 217): given the current-mesh sharding tree,
   compose the mixed replicated/FSDP layout for the `params` item, reusing
   `compose_full_params` (`src/rl/ema_utils.py:16-28`) — the *same* function
   `save_checkpoint` uses to build the item in the first place, so save/restore
   stay symmetric by construction. `nnx.filter_state`/`merge_state` on a
   sharding-valued `nnx.State` is already an established pattern in this file
   (`:361` does exactly this for jit `in_shardings`).

3. **New module-level helper `_restore_state_sharded`** (right after): a local
   reimplementation of openpi's `restore_state`
   (`openpi/src/openpi/training/checkpoints.py:89-107`) that adds explicit
   `restore_args`. Reaches two openpi privates, `_checkpoints._split_params`/
   `_merge_params`, to split the shape tree and the sharding tree the same
   way, and calls `construct_restore_args(target, sharding_tree)` per item.
   Runs inside `at.disable_typechecking()` — `dataclasses.replace` inside
   `_split_params` re-invokes `TrainState`'s beartype-checked `__init__`,
   which rejects a `NamedSharding` for the `step: at.Int[...]` field (openpi's
   own `_split_params` call is guarded the same way, `checkpoints.py:97`).
   Restore via `checkpoint_manager.restore(step, args=ocp.args.Composite(train_state=ocp.args.PyTreeRestore(...), params=ocp.args.PyTreeRestore(...)))`
   — verified this is the same underlying object as the legacy
   `items=`/`restore_kwargs=` form in the pinned orbax 0.11.13
   (`orbax/checkpoint/args.py:36`), and `Composite` is what this manager's
   multi-item mode requires.

4. **Call site** (`:318-324`): replace the `_checkpoints.restore_state(...)`
   call with `_restore_state_sharded(self._checkpoint_manager, self._train_state,
   self._train_state_sharding, trainable_filter=self._config.trainable_filter,
   replicated_sharding=self._replicated_sharding, step=self._resume_state.step)`.
   `self._replicated_sharding` already exists (`:265-267`). Note in `DIFF.md`
   that `data_loader` (which openpi's version only `del`s) is dropped —
   intentional.

5. **`init_train_state`'s resume branch** (`:201-202`): comment only, no
   behavior change — note that the returned sharding is now a restore
   contract, not just a jits'-`in_shardings` convenience.

6. **Optional, low-risk**: extend the existing restore log line
   (`:325-329`) to include the mesh shape, so a cross-topology resume is
   visible in logs.

## What does not change

`openpi/` (stays clean, doubles as the differential-test reference);
`dsrl_env.py`; `advantage_weighted_sft_learner.py` / `best_of_n_learner.py`
(already correct); `_refresh_train_step` / `_refresh_critic_update_function`
(already recompute sharding from the current mesh post-restore — this is
why the fix only needs to correct *where restored arrays land*, not what
downstream jits expect); any `scripts/` flags, `src/training/config.py`,
the replay buffer, checkpoint *layout* (only how it's read back changes).

## Notes for implementation caution

- Exact line numbers above are from the discovery pass (2026-09-21) and were verified then — re-verify against the actual current file before editing, since line numbers drift.
- The discovery/plan agents ran CPU-only probes proving: (a) annotating `jax.eval_shape` leaves with sharding is NOT sufficient — `PyTreeCheckpointHandler` never calls `construct_restore_args` on the item, so `restore_args` must be built and passed explicitly; (b) `dataclasses.replace` inside `_split_params` raises a `TypeCheckError` on the `step` field outside `at.disable_typechecking()` on a faithful minimal replica (not yet confirmed on the real `TrainState` — confirm this yourself while implementing, and if the real `TrainState` behaves differently, note it in `DIFF.md`); (c) `args=ocp.args.Composite(...)` and the legacy `items=`/`restore_kwargs=` form are the same underlying object in the pinned orbax 0.11.13.
- If you hit anything that requires a non-raise/swallow-and-log/sentinel/silently-applied-default code path, STOP and surface it rather than writing it — CLAUDE.md's deviation protocol requires explicit sign-off first, and you cannot get that mid-task (report it in your final summary instead and leave that piece undone).
- Reaching `openpi`'s private `_split_params`/`_merge_params` is deliberate (sanctioned by `best_practices.md:319-320` given a comment explaining why the public API doesn't work) — add that comment.
