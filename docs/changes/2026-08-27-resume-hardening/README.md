# Resume hardening — "make resumes tight"

**Tier 2** (cross-cutting: `ShardedReplayBuffer` public signature, the resume
manifest schema, a new learner hook, two learner packages, four shell recipes).
This directory is the **discovery pass** output — triage + step 1 only. No source
was touched.

## The problem

A requeued/resumed OGPO run does **not** continue with the same training
behavior as an uninterrupted one, and the recipe stack makes an accidental
restart destructive.

Four independent gaps, all verified against source on 2026-08-27:

1. **The success buffer is not persisted.** `save_epoch_state` saves exactly one
   buffer — `agent._online_data_buffer` (`src/training/runtime_state.py:82`).
   OGPO's `_success_data_buffer` (`src/rl/ogpo/ogpo_learner.py:91-106`) has no
   save and no restore path anywhere in the tree. After a resume it is empty, so
   the BC anchor silently falls back to the mostly-failed online batch
   (`ogpo_learner.py:533-536`) and `critic_success_oversample` silently no-ops
   (`ogpo_learner.py:460-464`) until it refills — roughly one collection
   interval (10k steps) of a different algorithm than the one being studied.
2. **`_success_task_ranges` goes with it** (`ogpo_learner.py:94`, appended at
   `:144-147`, consumed at `:696-734`), so `MT_BAL`
   (`rl.balance_success_buffer_tasks`) degrades to uniform sampling. The ranges
   and the buffer are a package: restoring one without the other is *worse* than
   restoring neither (see BLAST-RADIUS §4.2).
3. **`_adv_scale` is not checkpointed** (`ogpo_learner.py:112`, updated at
   `:575-589`, documented at `src/training/config.py:282-283`). Every resume
   resets the advantage normalizer to `min_scale`, i.e. a transient effective-LR
   spike on exactly the knob the stability study introduced to stop scale drift.
4. **The save is not crash-atomic and the restore is not tolerant.**
   `save_epoch_state` commits the orbax checkpoint for step N *first*
   (`runtime_state.py:74`), which garbage-collects step M (`max_to_keep=1`,
   openpi `checkpoints.py:48`), and only then writes the shard and the manifest.
   A crash in that window leaves a manifest pointing at a checkpoint that no
   longer exists — the run is unresumable without hand-editing JSON. Separately,
   `restore_shards` replays **every** `step_*.h5` in the directory with no step
   cutoff (`src/rl/replay_buffer.py:271`), so a rolled-back resume ingests data
   from the future.

On top of that, the recipe stack defaults to `--overwrite`
(`scripts/ogpo_multitask_4task.sh:242`, `scripts/stability_study.sh:128`), which
`rmtree`s the checkpoint directory, and none of the maxlab `*.sbatch` wrappers
handle `exp.py`'s exit-42 requeue contract. A resubmit of an interrupted run is
therefore a wipe, not a resume.

## The change

Make a resume reproduce the uninterrupted run's state, and make an accidental
restart non-destructive:

- persist and restore the success buffer, its task ranges, and `_adv_scale`,
  through a learner-owned hook so the generic manifest writer stays generic;
- give `restore_shards` a step cutoff so the buffer can never run ahead of the
  checkpoint it is paired with;
- reorder `save_epoch_state` (and add per-step manifests) so that every crash
  window leaves a fully consistent step on disk, and make the restore pick it;
- default the recipes to `--resume` behind an explicit `FRESH=1` escape hatch,
  and give the sbatch wrappers the exit-42 requeue trap that `launcher.py`
  already has (`scripts/launcher.py:65-82`, `:281-290`).

Full spec, verified cites, sweep results, open decisions and the verification
plan: [`BLAST-RADIUS.md`](./BLAST-RADIUS.md).

## Status

Discovery pass complete. **Tier 2 stops here** — the plan phase (plan mode) is
the approval gate, and must be built from this record rather than re-derived.
There are seven open decisions marked `D1`–`D7` in BLAST-RADIUS §6 that the plan
must resolve, three of which (D1, D5, D6) are deviation-protocol items: they
propose non-raise error paths and need explicit sign-off before implementation.
