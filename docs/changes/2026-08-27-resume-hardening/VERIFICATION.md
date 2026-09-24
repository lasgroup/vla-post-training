# VERIFICATION — resume hardening

Step 3 of the CLAUDE.md change workflow. Independent verification agent, fresh
context, given only `BLAST-RADIUS.md` + `PLAN.md` + `DIFF.md` + the source — not
the implementing session's reasoning. Findings below are verbatim and unfiltered,
failures included.

Verifier tests live in **`tests/ogpo/test_resume_hardening_verifier.py`** (new,
58 tests). No source file, no recipe, no sbatch and no implementer test file was
modified. Nothing was launched or submitted.

---

## 0. Verdict

The design does what BLAST-RADIUS §4.6 claims for a **fully post-change**
checkpoint tree: all seven crash rows were driven through the real
`save_epoch_state` and every one resolves to the claimed step with a mutually
consistent orbax / rl_state / manifest / shard set. `restore_shards(max_step=None)`
is byte-identical to the pre-change implementation. The rebase, the resolver, the
manifest schema, both clone-family reorders and the recipe precedence are all
correct as specified.

**Three defects found**, one of them in the migration path every in-flight run
takes on its first post-change save:

| # | Severity | Where | What |
|---|---|---|---|
| **V1** | **Medium-high — blocks a recoverable resume, and its message recommends the wipe** | `src/training/runtime_state.py:98-120` | The legacy pointer fallback is gated on `not manifest_steps`, so a **mixed** tree raises instead of resuming a step that is complete on disk. |
| **V3** | Medium — diagnostics loss | `scripts/ogpo_multitask_4task_maxlab.sbatch:10`, `scripts/ogpo_multitask_4task_ref_maxlab.sbatch:10` | `#SBATCH --requeue` added without `#SBATCH --open-mode=append`; the cited precedent (`launcher.py:220`) pairs them. Each requeue truncates the log of the attempt it is recovering from. |
| **V2** | Low — log text only | `src/rl/filtered_sft_agent/filtered_sft_learner.py:404-413` | The "not the manifest pointer's step" WARNING also fires when the resolved step is **ahead** of a stale pointer, where its text states the opposite of what happened. |

Neither V1 nor V2 nor V3 is a data-corruption defect: no path was found that
restores a buffer inconsistent with the restored weights, or the ranges without
the buffer.

---

## 1. Defects

### V1 — mixed legacy/post-change tree refuses a resumable step
**`src/training/runtime_state.py:98-120` (`resolve_resume_step`).**

```python
if not manifest_steps:
    if pointer_step is not None and pointer_step in orbax_steps:
        return pointer_step
    raise FileNotFoundError(...)          # D1 case 3
for step in sorted(orbax_steps & manifest_steps, reverse=True):
    if required_ok(step):
        return step
raise FileNotFoundError(...)              # "no step has all of {...}"
```

The pre-change fallback is reachable **only when `manifest_steps` is empty**.

Failure scenario (verified end to end against the real `save_epoch_state`):

1. A run written before this change is resumed. Its tree has `resume_state.json`
   naming M, orbax M, `rl_state/M`, and **no per-step manifests** — the D1 case-1
   path, which is every in-flight run and both `--resume` probes.
2. It reaches its first post-change boundary N. `save_epoch_state` writes shard N,
   the success shard, and `resume_state_<N>.json` — then the process dies before
   `agent.save_checkpoint(N)` commits orbax (SIGKILL, preemption, node failure,
   the babel-m9-16 NIC fault the wrapper's own header documents).
3. On the next attempt: `orbax_steps = {M}`, `manifest_steps = {N}`, pointer M.
   `orbax_steps & manifest_steps` is empty, so the loop finds nothing and the
   function raises:

```
FileNotFoundError: Resume requested, but no step has all of {orbax checkpoint,
per-step manifest, learner rl_state}: committed orbax steps [10000], per-step
manifests [20000], pointer step 10000. Restore the missing component for one of
those steps, or start over with --overwrite (FRESH=1 in the shell recipes).
```

Step M is complete and resumable — orbax has it, the pointer names it, `rl_state/M`
exists, and `restore_shards(max_step=M)` would drop shard N correctly. The run is
recoverable and the code refuses it, telling the operator to wipe.

This is precisely the crash window §4.6 exists to close, in precisely the tree
shape §4.6's "Backward compatibility" paragraph promises to support. It is a
one-boundary window per run (after the first successful post-change save the tree
is fully post-change and the intersection is never empty), but every in-flight run
passes through it.

Reproduced by `test_mixed_legacy_tree_should_still_resume_the_complete_older_step`
(`xfail(strict=True)`, so it flips to a failure the moment it is fixed), and
directly:

```
$ uv run python -c "from src.training.runtime_state import resolve_resume_step
print(resolve_resume_step({10000}, {20000}, lambda s: True, 10000))"
FileNotFoundError: Resume requested, but no step has all of {...}
```

**Suggested fix** (one branch, no behavior change for any other row): before the
final raise, retry the pointer —

```python
if pointer_step is not None and pointer_step in orbax_steps and required_ok(pointer_step):
    return pointer_step
```

placed after the intersection loop. That keeps D1 case 3 intact (the pointer
naming a GC'd checkpoint still raises) and makes the `not manifest_steps` special
case redundant rather than load-bearing.

### V2 — the resume WARNING states the wrong direction
**`src/rl/filtered_sft_agent/filtered_sft_learner.py:404-413.**

```python
if pointer_step is not None and step != pointer_step:
    logging.warning(
        "Resuming at step %d, not the manifest pointer's step %d: that step "
        "is missing an orbax checkpoint or a learner sidecar (the process "
        "died inside save_epoch_state). Everything collected after step %d "
        "is discarded.", step, pointer_step, step)
```

The guard is `step != pointer_step`, but the text only describes `step < pointer_step`.
BLAST-RADIUS §4.6's last row ("between (3) and (4)", i.e. the orbax commit
succeeded and the process died before the pointer refresh) has **step > pointer_step**:
the resolver correctly picks the newer, fully durable N while the pointer still
names M. In that row nothing is missing and nothing is discarded, and the operator
is told both.

Reproduced by `test_resolve_resume_state_warning_text_matches_the_direction`
(`xfail(strict=True)`). Log-only; no training-state consequence. Fix: branch the
message on `step < pointer_step` vs `step > pointer_step`, or gate the existing
message on `step < pointer_step` and log the ahead-of-pointer case at INFO.

### V3 — `--requeue` without `--open-mode=append`
**`scripts/ogpo_multitask_4task_maxlab.sbatch:10`,
`scripts/ogpo_multitask_4task_ref_maxlab.sbatch:10`.**

Both wrappers now carry `#SBATCH --requeue` and the `launcher.py:65-82` trap
shape, which is correct. But the precedent they cite sets **two** flags together:

```
scripts/launcher.py:219-220
        if requeue:
            bsub_cmd += "--requeue --open-mode=append "
```

Neither wrapper sets `--open-mode=append`, and `sbatch`'s default open mode for
batch jobs is *truncate*. Both wrappers write to
`--output=/home/pchellap/logs/%x_%j.out` and a requeue keeps the same `%j`, so
every requeue **truncates the log of the attempt it is recovering from**. Since
this change is what makes requeue the routine operating mode (mt4 now exits 42
every 23.5 h), the practical effect is that the log of the last 23.5 h is
destroyed at exactly the moment something needs post-mortem. No training state is
lost. Fix: add `#SBATCH --open-mode=append` next to `#SBATCH --requeue` in both.

---

## 2. What was verified, and how

### 2.1 `restore_shards` — differential against the pre-change implementation

The pre-change method body is transcribed **verbatim** into
`tests/ogpo/test_resume_hardening_verifier.py:73-102` as
`_restore_shards_pre_change` (from `git show 5b94510:src/rl/replay_buffer.py:257-`).
It is frozen in the test file rather than fetched with `git show HEAD` on purpose:
a `git show HEAD` differential degenerates into a self-comparison the moment the
change is committed — the exact rot that has already disabled three
`test_verifier_alignment.py` legs (see §4).

Every leg compares the full restorable state — `ptr`, `size`, `total_inserted`,
`obs_ptr`, `obs_total`, `valid_start`, `persisted_total_inserted`, `obs_pos`,
`next_obs_pos`, the whole `obs_storage` and all five transition leaves, and the
serialized RNG state — not just the summary counters.

| Leg | Result |
|---|---|
| `max_step=None`, three unclipped shards | **identical** |
| `max_step=None` with a non-initial `rng_state_json` | **identical** |
| `max_step=None` in the clipped case (`save_shard`'s `min(delta_count, size)` clamp fired) | **identical** |
| `max_step=None` on an empty (existing) shard directory, from a dirty buffer | **identical**, both reset to `(0, 0, 0)` |
| missing shard directory | both raise `FileNotFoundError` with a **string-identical** message |
| a foreign `step_latest.h5` present, `max_step=None` | same exception type from both — DIFF divergence D3 confirmed deliberate, `_shard_step` is not reached |

The cutoff itself is differential too, against a *separate directory that only
ever received the retained shards*, parametrized over `cut ∈ {0, 1000, 2000}`:
`restore_shards(full_dir, max_step=cut)` is state-identical to
`_restore_shards_pre_change(truncated_dir)` in every case. Cutoff inclusivity, a
cutoff below every shard (empty buffer, `persisted_total_inserted == 0`), and the
foreign-name raise under a cutoff are covered separately.

`_shard_step`'s raise is unreachable from the atomic writers: `_write_h5_atomic`
prefixes temp files with `.` (`replay_buffer.py:254`) and `_atomic_write_text`
does the same (`runtime_state.py:128`), so neither matches `step_*.h5` or
`resume_state_*.json`. Checked, not assumed.

### 2.2 The §4.6 crash table — driven through the real `save_epoch_state`

`test_resume_hardening.py` asserts the resolver over hand-written step sets. That
proves the resolver, not the save order. These tests instead run the **real**
`save_epoch_state(agent, config, prepare_for_resume=True)` against a stand-in
agent that emulates AWR's write shape (`rl_state/<step>` then the orbax commit)
and orbax's `max_to_keep=1` GC, kills the process at each write point, and then
resolves the surviving tree with the real `resolve_resume_step` +
`step_manifest_steps`. The manifest and pointer kills are injected by
monkeypatching `_atomic_write_text` to raise on its 1st / 2nd call, so the write
really is torn at that point rather than simulated.

| Crash point | shard N | success shard N | manifest N | resolver picks | verified |
|---|---|---|---|---|---|
| during (1) online shard write | absent | absent | absent | **M** | ✅ |
| between (1) and (2) | present | absent | absent | **M** | ✅ |
| between (2) and (3) | present | present | absent | **M** | ✅ |
| inside (4), before the rl_state write | present | present | present | **M** | ✅ |
| inside (4), after rl_state, before the orbax commit | present | present | present | **M** | ✅ |
| inside (4), after the orbax commit (M GC'd) | present | present | present | **N** | ✅ |
| between (4) and (5) (pointer refresh killed) | present | present | present | **N** | ✅ |
| no crash | present | present | present | **N** | ✅ |

Each row additionally asserts that the chosen step is internally consistent
(present in `all_steps()`, `rl_state/<step>` exists, its manifest loads and its
`step` field matches) and that `restore_shards(max_step=chosen)` ingests exactly
the transitions that existed at that step — never the newer shard.

Also verified against the real `save_epoch_state`:

- the pointer is written **last** and is byte-identical to the per-step manifest;
- the payload has exactly the six expected top-level keys;
- `prepare_for_resume=False` writes the checkpoint and **nothing else** (no
  pointer, no per-step manifest, no shard directory) — the branch BLAST-RADIUS
  §4.6 requires to keep working;
- `resume_state.json` is not matched by the `resume_state_*.json` glob;
- the pre-change fallback (per-step manifest removed) resolves to and loads the
  pointer manifest;
- the D1 case-3 raise (pointer names a GC'd checkpoint, no per-step manifests);
- a `keep_period`-pinned older orbax step is selected when the newest step lost
  its `rl_state` sidecar.

`FilteredSFTLearner._resolve_resume_state` was also exercised unbound (it takes no
`super()`), confirming it returns the per-step manifest when one exists, falls
back to the pointer manifest when none does, and logs
`"Resume resolved to step N (orbax steps=..., per-step manifests=...)"`.

### 2.3 Rebase semantics

`_rebase_task_ranges` is new code, so there is no pre-change copy to diff against.
Instead it is checked against an **independent transcription of the spec**
(BLAST-RADIUS §4.2 prose, written without reading the implementation) over 400
randomized `(saved_total, restored_total, valid_start, ranges)` draws — including
the case the two express differently (the spec drops on `hi + shift <= valid_start`
*before* clamping, the implementation compares *after* clamping). They agree on
every draw; the two orderings are equivalent because `valid_start <= restored_total`
always holds.

A second property sweep (200 draws) asserts the output invariant that actually
matters downstream: every surviving range satisfies `lo < hi <= restored_total`
and `hi > valid_start`, i.e. nothing the rebase emits can reach
`sample(ordinals=...)`'s `"ordinals reference evicted or unwritten transitions"`
guard (`replay_buffer.py:168-169`). Identity, low-drop and whole-task-eviction
cases are pinned separately.

Integration: the happy-path `_restore_extra_resume_state` test restores a real
two-task success buffer from a shard, rebases, then runs
`OGPOAgentLearner._balanced_success_ordinals` unbound and samples — no raise, and
`np.bincount` over the sampled `reward` tag gives exactly 4/4 for `batch_size=8`.

**Package property** (ranges and buffer never restore one without the other) is
enforced as a source invariant as well as behaviorally: the tree contains exactly
one `_success_data_buffer.restore_shards` call, it is inside
`_restore_extra_resume_state`, `self._success_task_ranges =` follows it, and there
is no `return` between the two.

### 2.4 OGPO restore — every branch

Called unbound against a `SimpleNamespace` (the method touches only `self`
attributes and has no `super()`), because `OGPOAgentLearner.__init__` is GPU-only:

| Branch | Verified |
|---|---|
| `extra` empty (D1 case 1) | WARNING logged, `_adv_scale` left at today's default, ranges left empty, buffer untouched |
| manifest has success state, run has `use_success_buffer` off | `ValueError`, message names the flag |
| run has the buffer, manifest has no success state | `ValueError`, message names the flag |
| `adv_scale` only (success buffer off) | restored — confirms PLAN §5 / DIFF divergence D1 |
| shard directory missing | `FileNotFoundError` |
| non-zero `success_total_inserted`, no shard at or below the step | `FileNotFoundError` ("holds no shard at or below") |
| happy path | buffer, ranges, RNG stream and `_adv_scale` all restored; balanced sampling works |
| an orphan success shard at a step **ahead** of the resolved step | excluded by `max_step` |

Success and online shards are confirmed to live in separate directories
(`success_shards` vs `replay_shards`), so the `step_*.h5` glob can never mix them.

BLAST-RADIUS §3.3's per-task-critics claim was checked on the success path
specifically: a shard saved from a buffer carrying `task_index` refuses to restore
into a shared-critic buffer with
`ValueError: Insert transition structure does not match buffer structure`. The
desired fail-fast, inherited for free.

**Exactly one `# best-effort:` comment** exists across all six changed Python
modules, and it is the sanctioned D1 case-1 path in `ogpo_learner.py:230`.
Asserted as a test, not eyeballed.

### 2.5 Manifest schema strictness

- five-key pre-change payload constructs; `extra` defaults to `{}` and is
  **per-instance** (two `ResumeState`s do not share one dict);
- an unknown top-level key still raises `TypeError` (checked with both
  `adv_scale` and `success_shard_dir` at top level);
- a missing required key still raises `TypeError`;
- the dataclass is still frozen (`FrozenInstanceError` on assignment);
- `extra` is the last field, so positional construction is unchanged;
- `replay_shard_dir`'s annotation is now `str` (N6 fixed);
- round-trip through `json.dumps` → `load_resume_state(step=…)` and
  `load_resume_state()` keeps `extra` intact and the two agree.

### 2.6 Inheritance / clone family

Verified by class-dict inspection and by calling the reordered `save_checkpoint`
on `object.__new__` instances with `FilteredSFTLearner.save_checkpoint`
monkeypatched to a recorder:

- `FilteredSFTLearner.save_extra_resume_state` → `{}`,
  `FilteredSFTLearner._resume_required_paths` → `[]`;
- **`OGPOAgentLearner` is the only** class overriding `save_extra_resume_state`
  (checked across AWR / BofN / MPO / FlowGRPO / OGPO);
- `_resume_required_paths` is overridden **only** by AWR; MPO and FlowGRPO
  inherit the identical function object (so they require `rl_state` too); BofN
  deliberately does not override it and keeps its own warn-and-start-fresh
  `_restore_rl_checkpoint` — OQ-6 preserved, asserted against the source of both
  classes;
- AWR `_resume_required_paths` returns `[rl_state/<step>]`, and
  `[rl_state/<step>, rl_state/task_registry_<step>.json]` when a registry is set;
- AWR `save_checkpoint` order is **rl_state → registry → orbax**;
- BofN `save_checkpoint` order is **rl_state → orbax**;
- BofN still normalizes `step=None` to `self.training_steps` (divergence 1);
- BofN's existing-path skip now skips **only** the rl_state write and still runs
  the orbax commit (DIFF divergence D2) — asserted directly, this is the one
  place where a mechanical "move the line" would have silently dropped the
  checkpoint;
- AWR still raises `TypeError` on `save_checkpoint(step=None)` — the documented
  unreachable divergence (BLAST-RADIUS §2.2) was **not** harmonized away.

`MPOWeightedSFTLearner`, `FlowGRPOLearner` and `DSRLLearner` define no
`save_checkpoint` of their own (`grep 'def save_checkpoint'` over `src/rl/`
returns AWR, BofN, base, DSRL, and the `Agent` ABC only), so the two clones are
the complete set that had to move.

`FilteredSFTLearner.save_checkpoint`, `_get_online_replay_buffer`,
`rng_state_json` and `set_rng_state_json` were AST-compared against
`git show HEAD:` and are **character-identical** — BLAST-RADIUS §8's non-goals
(full-EMA recomposition, the manifest RNG round trip) are untouched.

### 2.7 Recipes and sbatch (read-only, nothing submitted)

`bash -n` clean on all seven affected shell files:
`ogpo_multitask_4task.sh`, `stability_study.sh`, `ogpo_multitask_4task_ref.sh`,
`ws_bcbb_pipeline.sh`, `ogpo_multitask_4task_maxlab.sbatch`,
`ogpo_multitask_4task_ref_maxlab.sbatch`, `ogpo_ref_smoke_maxlab.sbatch`.

`DRY=1` precedence, with `CKPT_BASE_DIR`/`STORE_ROOT` pointed at a scratch dir
(the default `/data/group_data/...` is not writable from this node):

| invocation | emitted flag | banner |
|---|---|---|
| `DRY=1` | `--resume` | `checkpoint mode=--resume (continues an existing run; FRESH=1 to wipe)` |
| `DRY=1 FRESH=1` | `--overwrite` | `checkpoint mode=--overwrite (WIPES the checkpoint dir)` |
| `CKPT_MODE_FLAG=--overwrite DRY=1` | `--overwrite` | as above |
| `CKPT_MODE_FLAG=--resume DRY=1 FRESH=1` | `--resume` | as above |

Precedence is exactly **explicit `CKPT_MODE_FLAG` > `FRESH` > `--resume`**, and
the last row's emitted command line is byte-identical to the default row's. The
default↔FRESH diff is **exactly one flag** (`--resume` ↔ `--overwrite`) plus the
banner line. Exactly one mode flag reaches `exp.py` in every case.

`MAX_RUNTIME` plumbing: `MAX_RUNTIME=84600 DRY=1 …` emits `--max_runtime 84600`,
so the mt4 wrapper's `export MAX_RUNTIME="${MAX_RUNTIME:-84600}"` reaches
`exp.py`. 84600 s = 23.5 h against `#SBATCH --time=25:00:00` — N1 fixed.
`ogpo_multitask_4task_ref.sh` (which `exec`s the mt4 recipe, so exit 42
propagates) emits `--resume` and `--max_runtime 169200` = 47 h under its 48 h
wall, as DIFF states.

`stability_study.sh` (no `DRY` support, inspected): the hardcoded `--overwrite`
is replaced by `"$CKPT_MODE_FLAG"` and the file contains no other `--overwrite`
or `--resume` in its flag list. `ws_bcbb_pipeline.sh` sets neither
`CKPT_MODE_FLAG` nor `FRESH` and both stages `bash` into `stability_study.sh`, so
it inherits `--resume` — as §3.1 predicted, no edit needed.

Requeue traps: both are the `launcher.py:65-82` shape verbatim
(`child_status=0`; `cmd || child_status=$?`; `[[ -eq 42 ]]` → `scontrol requeue`
→ `exit 0`; else `exit "$child_status"`), with `bash` rather than `exec` so the
status comes back. All `#SBATCH` directives precede the first non-comment line in
both files, and the `#   #SBATCH --exclude=babel-m9-16` line inside the comment
block is indented behind a `#`, so SLURM will not parse it as a directive.
`ogpo_ref_smoke_maxlab.sbatch` was left untouched by this change (D5) — its
working-tree diff belongs entirely to `2026-08-26-mt2-batch128-recipe-knobs`.

See **V3** for the one thing the trap is missing.

### 2.8 Regression sweep

- **No `getattr(cfg, ..., default)`** introduced anywhere in the change (grep over
  all `+` lines of the working-tree diff; the only hits are false positives —
  `TaskRegistry` contains the substring `try`).
- **No jit signature, sharding annotation, `donate_argnums` or RNG-split-arity
  change** in any file this change touches. The working tree *does* contain one
  `in_shardings` arity change in `ogpo_learner.py` (a `task_index` sidecar), but
  it belongs to the concurrent `2026-08-21-per-task-critics` work, not here — the
  resume-hardening additions to that file are `_rebase_task_ranges`,
  `save_extra_resume_state`, `_restore_extra_resume_state` and one `__init__`
  call, none of which touch a jit.
- **No `try`/`except`, no swallow-and-log, no sentinel return** among the added
  lines. One sanctioned `# best-effort:` block, as approved (D1 case 1).
- `src/training/collect.py` and `scripts/ogpo_ref_smoke_maxlab.sbatch` are dirty
  in the working tree but their diffs are entirely other in-flight changes;
  `scripts/exp.py`'s only edit is the `max_step=resume_state.step` kwarg.
- `ruff check` on all seven changed Python files plus the new test file: **3
  findings, all pre-existing and outside the change** — unused `logging` import
  (AWR), unused `mesh_utils` import and one `E731` (`filtered_sft_learner.py:539`).
  DIFF's claim confirmed.
- Only three `restore_shards` call sites exist in the tree; two pass `max_step`
  (`filtered_sft_learner.py:295-299`, `exp.py:167`) and the third is
  `tests/ogpo/test_per_task_critics_verifier.py:565`, which still works
  positionally. Both existing tests that exercise the changed buffer API pass
  (§4).

---

## 3. Notes and gotchas discovered (not defects)

- **N-a — `step_manifest_steps` matches directories.** `state_dir.glob("resume_state_*.json")`
  returns directories as well as files, so a directory with that name would be
  counted as a manifest step and then fail later in `read_text()`. No code path
  creates one; noted only because the function's job is to be strict about what
  is in that directory.
- **N-b — stale shards ahead of the resume point are never deleted.** After a
  resume at M that drops shard N, `step_N.h5` (online and success) stays on disk
  and is only made harmless by being **overwritten** at the next boundary. That
  works because `start_step = agent.training_steps` (`exp.py:90`) and both
  boundaries are step-multiples (`step % collect_interval`, `step % eval_interval`),
  so the resumed run lands on exactly the same step values. It is a real
  dependency, not an accident, and it would break if a resume changed
  `collect_interval`/`eval_interval` — the leftover shard would then sit between
  two live ones and be replayed on the next restore. Worth a sentence in
  `docs/code/training.md`.
- **N-c — `free_buffer_before_eval=True` loses one collection round's shard per
  resume.** On resume at a boundary M the loop re-enters `step == M`, re-collects,
  and (only under `free_buffer_before_eval`) calls `save_epoch_state` again at
  `training_steps == M`. `save_shard` writes only the delta since restore, so
  `step_M.h5` is overwritten with the *post-resume* delta and the delta it
  previously held is permanently gone from the shard set. Pre-existing and
  strictly out of scope — `free_buffer_before_eval` defaults to `False`
  (`config.py:469`) and no recipe sets it — but this change makes requeue routine,
  so if that flag is ever turned on the loss repeats every 23.5 h. Flagged, not
  fixed.
- **N-d — the legacy fallback does not apply `required_ok`.** `resolve_resume_step`
  returns `pointer_step` without checking the learner sidecars. That is today's
  behavior and consistent with D1 case 1, but it means a *pre-change* tree caught
  in the old AWR sub-window (orbax N committed, `rl_state/N` never written) still
  fails inside orbax's restore with an unhelpful message instead of stepping back.
  Only affects pre-change trees; correct to leave if that is the intent, but it is
  not stated anywhere.
- **N-e — `${SLURM_JOB_ID}` under `set -u`** in the new traps: unbound if a
  wrapper is ever run outside SLURM *and* the child exits 42. Identical to the
  `launcher.py` script it copies; noted only for completeness.
- **N-f — the two probe sbatch comments are now stale.**
  `probe_counterfactual_rollouts.sbatch:68` and `probe_value_next_spread.sbatch:46`
  still say `CKPT_MODE_FLAG=--resume   # replaces the recipe's --overwrite`. The
  override is now redundant-but-correct; BLAST-RADIUS §3.1 assigns the comment fix
  to the step-4 docs pass, which has not run. Still open.
- **N-g — `--resume` with an empty orbax dir does not clean `runtime_state/`.**
  `initialize_checkpoint_dir` downgrades `resume=True` to a fresh start when
  `all_steps()` is `()` or `(0,)` (`openpi/.../checkpoints.py:56-61`) **without**
  wiping the directory, so a fresh-start run can begin against a
  `runtime_state/` left by an earlier attempt. In practice it is self-healing —
  the only way `runtime_state/` is non-empty with no committed orbax step is a
  crash inside the first `save_epoch_state`, whose shard and manifest are both at
  the same step the new run will overwrite. Worth knowing, not worth code.

---

## 4. Test results

### New verifier file
```
$ uv run pytest tests/ogpo/test_resume_hardening_verifier.py -q
56 passed, 2 xfailed in 9.28s
```
The two `xfail`s are the reported defects, both `strict=True` so they become
failures the moment V1/V2 are fixed:
- `test_mixed_legacy_tree_should_still_resume_the_complete_older_step` → V1
- `test_resolve_resume_state_warning_text_matches_the_direction` → V2

### Implementer's file
```
$ uv run pytest tests/ogpo/test_resume_hardening.py -q
16 passed in 8.50s
```

### `pytest tests/ogpo`
Collects **296** tests (244 as of DIFF, plus this file's 52). The suite **cannot
be run whole on this login node**: it dies with `Fatal Python error: Aborted`
inside XLA's CPU compile under the node's hard 16 GiB `ulimit -v` (16777216 KB,
not raisable from the shell). Reproduced exactly as DIFF describes. Per-file:

| file | result |
|---|---|
| `test_ema_utils.py` | 1 passed |
| `test_grad_norm_decomposition.py` | **aborts** (XLA compile) — not executed here |
| `test_group_dedup.py` | **aborts** — not executed here |
| `test_per_task_critics.py` | 12 pass, then **aborts** — remainder not executed here |
| `test_per_task_critics_verifier.py` | **aborts** — but the two legs that exercise the changed buffer API were run by node id and **both pass** (see below) |
| `test_per_task_critics_verifier2.py` | 30 passed |
| `test_resume_hardening.py` | 16 passed |
| `test_resume_hardening_verifier.py` (new) | 56 passed, 2 xfailed |
| `test_reward_and_value_bounds.py` | 10 passed |
| `test_sampling.py` | **aborts** — not executed here |
| `test_split_equivalence.py` | **aborts** — not executed here |
| `test_verifier_alignment.py` | 114 passed, **3 failed**, 3 skipped |

The changed-API legs, run explicitly:
```
$ uv run pytest \
  "tests/ogpo/test_per_task_critics_verifier.py::test_task_index_is_a_transition_field_that_survives_sample_and_shards" \
  "tests/ogpo/test_per_task_critics_verifier.py::test_num_tasks_none_buffer_has_no_task_index_key" -q
2 passed in 7.66s
```
The first of these calls `restore_shards(shard_dir)` positionally — the
signature change is confirmed additive.

The **3 failures are pre-existing and unrelated**, exactly as the brief and DIFF
state:
```
FAILED tests/ogpo/test_verifier_alignment.py::test_head_value_distribution_differential_over_every_registered_config
FAILED tests/ogpo/test_verifier_alignment.py::test_head_value_distribution_differential_is_a_real_change_at_201_bins
FAILED tests/ogpo/test_verifier_alignment.py::test_head_wrapper_differential_over_a_randomized_flag_script
```
They diff the working tree against **git HEAD** for `src/rl/value_distribution.py`
and `src/envs/wrappers.py` and now find them equal because commit `5b94510`
landed that work. Neither file appears in this change's diff (`git status`
confirms neither is even modified). Not fixed; out of scope.

`sbatch scripts/run_ogpo_tests.sbatch` — the documented way to run the aborting
legs — **was not submitted**: that requires explicit permission.

---

## 5. What could NOT be verified here, plainly

- **The end-to-end resume.** Learner construction, the orbax restore, real π0.5
  weights, a simulator, a GPU. `OGPOAgentLearner.__init__` device_puts the EMA to
  `pinned_host` (`ogpo_learner.py:124-127`) and is GPU-only by construction. Every
  claim above about learner methods was established by calling them unbound
  against stand-ins or on `object.__new__` instances — that exercises the logic,
  **not** the wiring inside a real `__init__`, and not orbax at all. In particular:
  - that `self._resume_state` is populated before `OGPOAgentLearner.__init__`
    reaches `_restore_extra_resume_state`,
  - that `self._checkpoint_manager.all_steps()` returns what orbax has committed
    at the moment `_resolve_resume_state` runs,
  - that `_checkpoints.restore_state(..., step=resolved)` accepts the resolved
    step,

  were checked by **reading** the call order (`filtered_sft_learner.py:258` →
  `:292-299` → `:306-320`; `ogpo_learner.py:131-155`; AWR sets
  `_num_critic_tasks`/`_task_registry` at `:88-89` before `super().__init__` at
  `:115`), not by executing it.
- **Orbax's asynchronous commit.** `save_state` runs under
  `AsyncOptions(timeout_secs=7200)` and `max_to_keep=1` GC is orbax's, not ours.
  The crash-table tests emulate "commit then GC" as an atomic step. If orbax can
  ever GC step M before step N is durable, a crash in that sub-window would leave
  `all_steps()` empty and `initialize_checkpoint_dir` would silently downgrade to
  a fresh start (`checkpoints.py:56-61`). That is outside this change and cannot
  be tested without a real orbax write.
- **Anything requiring SLURM.** The requeue traps were read and `bash -n`'d; no
  job was submitted, so the exit-42 → `scontrol requeue` → resume loop is
  unverified in the field. V3 was established by reading the precedent
  (`launcher.py:219-220`) and `sbatch`'s documented default open mode, not by
  observing a truncated log.
- **`stability_study.sh` end to end.** It has no `DRY` support; its checkpoint
  mode was verified by reading the flag list and grepping for competing
  `--overwrite`/`--resume` occurrences.
- **Disk/retention behavior (D4).** No live run's checkpoint tree is visible from
  this node (`run_store/checkpoints/**` is empty), so the "one ~1 KB manifest per
  boundary" cost is arithmetic, not measurement — same limitation BLAST-RADIUS §D4
  already records.

---

## 6. Conclusions

1. The core design is sound and, for a fully post-change tree, verified row by row
   against the implemented save order — not merely against the resolver in
   isolation.
2. `restore_shards(max_step=None)` is proven byte-identical to the pre-change
   implementation on six differential legs, so the signature change is genuinely
   additive.
3. **V1 must be fixed before this ships to a live run.** It fires in the exact
   migration window every in-flight run passes through, and its message points at
   `--overwrite`, which is the destructive action this change exists to prevent.
   The fix is one `if` after the intersection loop.
4. **V3 should be fixed with it** — one `#SBATCH` line per wrapper, and it is the
   difference between having and not having the log of a crash the requeue just
   recovered from.
5. V2 is cosmetic but sits on the same log line an operator reads after a crash;
   fix it in the same pass.
6. The step-4 docs pass still owes the two probe-sbatch comments (N-f), on top of
   BLAST-RADIUS §10's list.

---

# Post-verification fixes (2026-08-27)

*Appended by the implementation session after the verification above. Nothing
above this line was edited: the findings, their severities and the verifier's
conclusions stand as written.* All three defects are fixed; details per file are
in [`DIFF.md`](./DIFF.md) § "Post-verification fixes".

| # | Status | Fix |
|---|---|---|
| **V1** | Fixed | `resolve_resume_step` (`src/training/runtime_state.py`) retries `pointer_step` **after** the intersection loop fails, accepting it when `pointer_step in orbax_steps and required_ok(pointer_step)`. The mixed legacy/post-change tree now resumes step M instead of raising. The `not manifest_steps` early branch is retained so the D1 case-3 message is unchanged; the load path already read the `resume_state.json` pointer manifest for a resolved step with no per-step file, so it needed no edit. |
| **V2** | Fixed | `FilteredSFTLearner._resolve_resume_state` splits the WARNING: `step < pointer_step` keeps the "behind the pointer / everything after is discarded" text; `step > pointer_step` gets its own WARNING stating that the resolved step is fully durable, the pointer refresh is what did not run, and nothing is lost. |
| **V3** | Fixed | `#SBATCH --open-mode=append` added next to `#SBATCH --requeue` in `scripts/ogpo_multitask_4task_maxlab.sbatch` and `scripts/ogpo_multitask_4task_ref_maxlab.sbatch`, with a comment citing `scripts/launcher.py:219-220`. |

The two `xfail(strict=True)` markers in
`tests/ogpo/test_resume_hardening_verifier.py` were removed (only the markers —
no test body, helper or assertion in that file was touched), so both tests now
assert the fixed behavior directly.

Re-run:

```
$ uv run pytest tests/ogpo/test_resume_hardening.py \
                tests/ogpo/test_resume_hardening_verifier.py -q
74 passed, 11 warnings in 35.44s      # 16 + 58, zero xfail/xpass
```
```
$ uv run pytest tests/ogpo/test_ema_utils.py \
    tests/ogpo/test_per_task_critics_verifier2.py \
    tests/ogpo/test_reward_and_value_bounds.py \
    "…verifier.py::test_task_index_is_a_transition_field_that_survives_sample_and_shards" \
    "…verifier.py::test_num_tasks_none_buffer_has_no_task_index_key" -q
43 passed, 30 warnings in 8.93s
```

Also re-checked: `bash -n` clean on both edited wrappers and every `#SBATCH`
directive still precedes the first non-comment line; `ruff check` on the two
edited Python modules and the verifier test file reports only the two
pre-existing findings §2.8 already lists; §1's V1 repro
(`resolve_resume_step({10000}, {20000}, lambda s: True, 10000)`) now returns
`10000`, while the same call with `required_ok(10000)` false and the D1 case-3
call still raise `FileNotFoundError`.

The remaining aborting legs (`ulimit -v` 16 GiB on this node) were **not**
re-run — unchanged from §4 — and no job was submitted, nothing launched, nothing
committed. Verifier notes N-a … N-g were observations rather than defects and are
left as recorded; N-f stays with the step-4 docs pass.
