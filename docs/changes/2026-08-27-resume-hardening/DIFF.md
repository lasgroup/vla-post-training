# DIFF — resume hardening

What actually changed, file by file, plus every divergence from
[`PLAN.md`](./PLAN.md). Implementation pass, 2026-08-27. No run was launched, no
sbatch submitted, nothing committed.

---

## Source

### `src/rl/replay_buffer.py` (plan §1)

- New module-level `_shard_step(path) -> int`: parses the `step_%08d` stem of a
  shard filename and **raises** `ValueError` (with the fix in the message) on a
  `step_*.h5` name that does not parse — the atomic writer's temp files are
  leading-dot and never match the glob, so such a match is a foreign file.
- `restore_shards` gains keyword-only `max_step: int | None = None` (additive;
  both production call sites and `tests/ogpo/test_per_task_critics_verifier.py:565`
  keep working unchanged) plus a docstring saying why the cutoff exists. When
  `max_step` is not `None`, the glob result is filtered to shards `<= max_step`.

### `src/training/runtime_state.py` (plan §2)

- `ResumeState`: `replay_shard_dir` annotation corrected `Path` → `str` (N6, it
  was always a JSON string); new
  `extra: dict[str, Any] = dataclasses.field(default_factory=dict)` — one nested
  field, so five-key pre-change payloads still construct and an unknown
  **top-level** key still raises `TypeError`.
- New helpers: `success_shard_dir` / `success_shard_path`
  (`runtime_state/success_shards/step_%08d.h5`), `step_manifest_path`
  (`runtime_state/resume_state_%08d.json`), and `step_manifest_steps(config)`
  (globs the per-step manifests; raises on a foreign `resume_state_*.json`).
- New pure `resolve_resume_step(orbax_steps, manifest_steps, required_ok,
  pointer_step) -> int`: greatest step in `orbax_steps ∩ manifest_steps` passing
  `required_ok`; with no per-step manifests at all it falls back to
  `pointer_step` when that step is committed (D1 case 1's sibling — the
  pre-change directory path); when the intersection search finds nothing it
  retries `pointer_step` (V1 fix, below); otherwise it raises `FileNotFoundError`
  naming both sets (D1 case 3). No orbax dependency, so it is unit-testable.
- `load_resume_state(config, step=None)`: `step` reads that step's manifest,
  `None` reads the `resume_state.json` pointer. Docstring corrected (N7 — it
  claimed to return `None` where it raises).
- `save_epoch_state` reordered on the `prepare_for_resume=True` path to
  (1) online `save_shard(step N)`, (2) `agent.save_extra_resume_state(N)`,
  (3) atomic `resume_state_<N>.json`, (4) `save_checkpoint(N)` +
  `wait_until_finished`, (5) atomic refresh of the `resume_state.json` pointer.
  The `prepare_for_resume=False` branch keeps today's checkpoint-only behavior
  (it now saves the checkpoint inside that branch rather than before the split).

### `src/rl/filtered_sft_agent/filtered_sft_learner.py` (plan §3)

- New base hooks next to `save_checkpoint`: `save_extra_resume_state(step) -> dict`
  returning `{}` (docstring names OGPO as the only override) and
  `_resume_required_paths(step) -> list[epath.Path]` returning `[]`.
- New `_resolve_resume_state()` (placed after `__init__`): reads the pointer step
  when the pointer exists, gathers `self._checkpoint_manager.all_steps()` and
  `step_manifest_steps`, calls `resolve_resume_step` with
  `required_ok = lambda s: all(p.exists() for p in self._resume_required_paths(s))`,
  logs the chosen step (plus a direction-specific WARNING when it differs from
  the pointer — see the V2 fix below), and loads the per-step manifest — or the
  pointer manifest in the pre-change and mixed-tree fallbacks, where the resolved
  step has no per-step manifest.
- `__init__` resume block now calls `_resolve_resume_state()` and passes
  `max_step=self._resume_state.step` to `restore_shards`.

### `src/rl/advantage_weighted_sft/advantage_weighted_sft_learner.py` (plan §4)

- `save_checkpoint`: `rl_state/<step>` (+ `task_registry_<step>.json`) are now
  written **before** `super().save_checkpoint()` commits orbax, closing the
  newest-orbax-step-without-critics sub-window (§2.1/#6 of BLAST-RADIUS).
- New `_resume_required_paths` override: `[rl_state/<step>]`, plus the registry
  JSON when `self._task_registry is not None` (both are set before
  `super().__init__`, so the dispatch during base init is safe; the class-level
  defaults on `FilteredSFTLearner:224-225` cover the shared-critic case).

### `src/rl/best_of_n/best_of_n_learner.py` (plan §4 — N3 clone family)

- Same reorder, written second on purpose (OQ-2 duplication-by-copy). Both
  documented divergences preserved: the `step is None` normalization stays, and
  the existing-path skip now guards only the rl_state write (see divergence D2
  below). `_resume_required_paths` is deliberately **not** overridden here, so
  BofN keeps warning-and-starting-fresh on a missing rl_state (OQ-6).

### `src/rl/ogpo/ogpo_learner.py` (plan §5)

- New imports: `logging`, `pathlib.Path`, `typing.Any`,
  `success_shard_dir`/`success_shard_path`.
- New module-level pure `_rebase_task_ranges(saved_ranges, saved_total_inserted,
  restored_total_inserted, valid_start)`: uniform `shift = restored - saved`,
  drops ranges that end at or below `valid_start` (and ranges emptied by the
  clamp), clamps `hi` to `restored_total_inserted`, drops tasks with no surviving
  range. Factored out of the learner so it is testable without a GPU.
- `save_extra_resume_state(step)`: always returns `{"adv_scale": ...}`; when the
  success buffer exists it additionally writes
  `runtime_state/success_shards/step_%08d.h5` and returns `success_shard_dir`,
  `success_total_inserted`, `success_task_ranges`, `success_rng_state_json`.
- `_restore_extra_resume_state()`, called from `__init__` under `self._resuming`,
  **after** the temporary config swap is undone (`ogpo_learner.py:105` gotcha) and
  after the `_adv_scale` default it replaces. Behavior:
  - `extra` empty → WARNING + today's behavior, carrying a `# best-effort:`
    comment (D1 case 1, the one sanctioned warn-and-degrade path);
  - `extra` present → `_adv_scale` restored, then the success buffer restored with
    `max_step=<resolved step>` and the ranges rebased as a package; a `shift != 0`
    logs a WARNING naming both counts;
  - every other inconsistency raises with the fix in the message (missing shard
    dir; a declared non-zero success count with no shard at or below the step;
    `rl.use_success_buffer` disagreeing with the manifest in either direction).
- The `_adv_scale` comment lost its now-false "Not checkpointed" sentence. The
  same claim in `src/training/config.py:282-283`, `docs/code/rl-ogpo.md` and
  `CLAUDE.md` is left to the docs pass (plan step 4 / BLAST-RADIUS §10).

### `scripts/exp.py` (plan §6)

- The `free_buffer_before_eval` re-restore passes `max_step=resume_state.step`.

## Recipes and sbatch

### `scripts/ogpo_multitask_4task.sh` (plan §7)

- Checkpoint mode resolved before the banner: an explicit `CKPT_MODE_FLAG` wins,
  else `FRESH=1` → `--overwrite`, else `--resume`. Verified precedence by `DRY=1`
  (see below). New `[mt4] checkpoint mode=…` banner line that says outright when
  the run will wipe. `FRESH` documented in the env-var header.

### `scripts/stability_study.sh` (plan §7)

- Same-shaped block (deliberately a second copy — OQ-2, the two preambles have
  already diverged and are not unified here), replacing the hardcoded
  `--overwrite` in the flag list with `"$CKPT_MODE_FLAG"`. Banner line and header
  documentation to match.

### `scripts/ogpo_multitask_4task_maxlab.sbatch` (plan §8)

- `#SBATCH --requeue` + `#SBATCH --open-mode=append` (V3 fix, below);
  `export MAX_RUNTIME="${MAX_RUNTIME:-84600}"` (23.5 h under
  the 25 h wall — fixes N1, which made the exit-42 path dead code here);
  `exec bash …` replaced with the capture-status + `scontrol requeue` shape from
  `scripts/launcher.py:65-82`.

### `scripts/ogpo_multitask_4task_ref_maxlab.sbatch` (plan §8)

- `#SBATCH --requeue` + `#SBATCH --open-mode=append`; same requeue trap; the
  preempt-header reasoning corrected
  (`save_interval` is never read by `exp.py`; the real exposure is the SIGKILL
  that skips the exit-42 handoff). No `MAX_RUNTIME` override needed — the
  recipe's 47 h default is already under this wrapper's 48 h wall.

### `scripts/ws_bcbb_pipeline.sh` — **not edited, behavior changes anyway**

Both of its stages delegate to `stability_study.sh` (`:36-40`, `:50-52`) and it
never sets a checkpoint mode, so it inherits `--resume`. Its stage-A guard
(`:33`, `if [ ! -d …/20000 ]`) previously meant a stage A interrupted at 15k
re-ran from scratch **and wiped**; it now continues from the last boundary
instead. Net improvement, as BLAST-RADIUS §3.1 predicted; recorded here per
plan §7. `scripts/ogpo_ref_smoke_maxlab.sbatch` untouched (D5).

## Tests

### `tests/ogpo/test_resume_hardening.py` (new, plan §9)

16 pytest-native tests, CPU-only, model-free, modeled on the
`_buffer_dummy`/`_insert_episode` helpers of
`tests/ogpo/test_per_task_critics_verifier.py:520-580` (local copies; transitions
are stamped with their global ordinal so payload identity is directly checkable):

- shard round-trip ordinal/payload exactness (the evidence for the uniform-shift
  claim the rebase rests on);
- the clipped case: `expected_dropped` computed from `save_shard`'s documented
  clamp, then asserted equal to `restored_total - saved_total`, and every live
  transition asserted to carry the payload of `ordinal - shift`;
- `max_step` **differential** against a second shard directory that only ever
  received the early shards (plus: the cutoff is inclusive, and no-cutoff is
  unchanged); foreign shard name raises;
- `_rebase_task_ranges` units (identity, shift, low-drop, `hi` clamp,
  empty-task removal) and an integration through `_balanced_success_ordinals`
  (called unbound against a `SimpleNamespace`, since `OGPOAgentLearner.__init__`
  is GPU-only) asserting no raise, in-range ordinals and balanced per-task counts,
  plus a test showing what an unclamped `hi` does (the raise from
  `replay_buffer.py:155-156`) — i.e. why the clamp is load-bearing;
- `ResumeState` old-payload construction, per-instance `extra`, unknown-key
  `TypeError`, manifest JSON round trip through both `load_resume_state` forms,
  `step_manifest_steps` behavior;
- `resolve_resume_step` over every row of the BLAST-RADIUS §4.6 crash table plus
  both D1 raise cases and the sidecar-missing skip.

---

## Divergences from PLAN.md

**D1 — `adv_scale` is always saved** (already flagged in plan §5 as a deliberate
divergence from BLAST-RADIUS §4.1). `save_extra_resume_state` returns
`{"adv_scale": …}` even when `rl.use_success_buffer` is off; returning `{}` there
would drop `_adv_scale` for NORM-only runs, which is the case the field exists
for. Consequence: an `extra` block is now present in every post-change OGPO
manifest, so the "no `extra`" warn-and-degrade path means exactly "written before
this change", which is what D1 case 1 is scoped to.

**D2 — BofN's existing-path skip is reshaped, not moved.** The original
`save_checkpoint` returned early when `rl_state/<step>` already existed; with the
write moved ahead of `super().save_checkpoint()`, a bare `return` would have
skipped the orbax commit too. The skip is now `if not path.exists():` around the
rl_state write only, so the divergence's meaning (never re-save an existing
rl_state) is preserved and the orbax commit always runs.

**D3 — shard-name parsing happens only under a cutoff.** `_shard_step` is called
only when `max_step is not None`, so `max_step=None` remains byte-identical to
today (a foreign `step_*.h5` still reaches h5py, as before) — the plan's "`None` =
today's behavior" taken literally. The raise is therefore reachable from the
filtered path only.

**D4 — two extra raise paths beyond the plan's enumeration**, both following the
rule that every non-D1-case-1 inconsistency raises: (a) the manifest declares a
non-zero success count but the shard directory holds no shard at or below the
resolved step; (b) `rl.use_success_buffer` disagrees with the manifest in either
direction (state present with no buffer, or a buffer with no state). Each names
the flag to flip or `--overwrite`/`FRESH=1`.

**D5 — the manifest-step glob lives in `runtime_state.py`, not the learner.** The
plan put "manifest steps by globbing `resume_state_*.json`" inside
`FilteredSFTLearner.__init__`; it is `step_manifest_steps(config)` instead, next
to `step_manifest_path`, so the filename contract and its foreign-file raise sit
in one file and are unit-testable without a learner.

**D6 — the OGPO restore is a method, not an inline block.** Plan §5 describes
inline code after `ogpo_learner.py:105`/`:112`; it is
`self._restore_extra_resume_state()` called from exactly that point. Same
ordering constraints, but readable and separately reviewable.

**D7 — the stale `_adv_scale` source comment in `ogpo_learner.py` was corrected
here**, since the change makes it actively false in a file this change edits. The
identical claims in `src/training/config.py:282-283`, `docs/code/rl-ogpo.md` and
`CLAUDE.md` are left for the docs pass, as the plan's step 4 lists them.

No jit signature, sharding annotation, `donate_argnums` or RNG split arity was
touched anywhere in this change.

---

## Post-verification fixes (V1 / V2 / V3)

The step-3 verifier found three defects
([`VERIFICATION.md`](./VERIFICATION.md) §1). All three are fixed, file by file:

**V1 — `src/training/runtime_state.py`, `resolve_resume_step`.** The legacy
pointer fallback was gated on `not manifest_steps`, so a **mixed** tree raised
instead of resuming a step that is complete on disk: a pre-change run (pointer M,
no per-step manifests) whose first post-change save writes `resume_state_<N>.json`
and then dies before the orbax commit leaves `orbax={M}`, `manifests={N}` — an
empty intersection, while M has its checkpoint, its rl_state and the pointer.
Every in-flight run passes through that window once. Fix: after the intersection
loop, retry `pointer_step` and accept it when
`pointer_step in orbax_steps and required_ok(pointer_step)`; only then raise. The
`not manifest_steps` early branch is kept for its more specific message (and so
the D1 case-3 raise text is unchanged). The load path needed no edit —
`_resolve_resume_state` already reads the `resume_state.json` pointer manifest
when the resolved step has no per-step file, which is exactly this case.

**V2 — `src/rl/filtered_sft_agent/filtered_sft_learner.py`, `_resolve_resume_state`.**
The "not the manifest pointer's step" WARNING fired on `step != pointer_step`,
but its text only describes `step < pointer_step`; on the crash row where the
orbax commit succeeded and the pointer refresh did not (`step > pointer_step`) it
claimed a missing component and discarded data, both false. Fix: the existing
message is gated on `step < pointer_step` (and now says "behind"), and
`step > pointer_step` gets its own WARNING saying the resolved step is fully
durable, the pointer refresh is what did not run, and nothing is lost.

**V3 — `scripts/ogpo_multitask_4task_maxlab.sbatch`,
`scripts/ogpo_multitask_4task_ref_maxlab.sbatch`.** `#SBATCH --requeue` was added
without `#SBATCH --open-mode=append`; sbatch's default open mode truncates and a
requeue reuses the same `%j`, so each attempt erased the log of the attempt it was
recovering from. Both wrappers now carry the second directive (the cited
precedent, `scripts/launcher.py:219-220`, pairs them), with a comment saying why.
All `#SBATCH` lines still precede the first non-comment line in both files.

**Verifier tests.** `tests/ogpo/test_resume_hardening_verifier.py` pinned V1 and
V2 as `xfail(strict=True)`; those two markers — and nothing else in that file —
were removed, so
`test_mixed_legacy_tree_should_still_resume_the_complete_older_step` and
`test_resolve_resume_state_warning_text_matches_the_direction` now assert the
fixed behavior directly.

Verifier notes N-a … N-g are observations, not defects, and are left as recorded.
N-f (the two stale probe-sbatch comments) remains the docs pass's item, per
BLAST-RADIUS §3.1.

---

## Verification performed in this pass

(The independent verifier's findings go in `VERIFICATION.md`; this is only what
the implementation pass ran.)

- `tests/ogpo/test_resume_hardening.py`: **16 passed**. `pytest tests/ogpo`
  collects **244** tests (228 pre-existing + 16 new).
- `pytest tests/ogpo` as a whole **cannot complete on this login node**: it dies
  with `Fatal Python error: Aborted` inside XLA's CPU compile (first at
  `test_grad_norm_decomposition.py::test_cosine_is_in_range_and_finite`). The
  node enforces a hard `ulimit -v` of 16 GiB that cannot be raised from the shell,
  and **every** PaliGemma/BroNet-jit leg aborts on it while every pure-Python test
  in the same files passes — the documented condition under which the full suite
  needs `sbatch scripts/run_ogpo_tests.sbatch` (not submitted: that requires
  permission). Re-running each file with the aborting test ids deselected gives:

  | file | passed | failed | skipped | aborts (never executed) |
  |---|---|---|---|---|
  | `test_ema_utils.py` | 1 | 0 | 0 | 0 |
  | `test_grad_norm_decomposition.py` | 0 | 0 | 0 | 8 |
  | `test_group_dedup.py` | 0 | 0 | 0 | 3 |
  | `test_per_task_critics.py` | 18 | 0 | 0 | 6 |
  | `test_per_task_critics_verifier.py` | 16 | 0 | 0 | 5 |
  | `test_per_task_critics_verifier2.py` | 30 | 0 | 0 | 0 |
  | `test_resume_hardening.py` (new) | 16 | 0 | 0 | 0 |
  | `test_reward_and_value_bounds.py` | 10 | 0 | 0 | 0 |
  | `test_sampling.py` | 0 | 0 | 0 | 5 |
  | `test_split_equivalence.py` | 0 | 0 | 0 | 6 |
  | `test_verifier_alignment.py` | 114 | 3 | 3 | 0 |
  | **total** | **205** | **3** | **3** | **33** |
- Three **pre-existing, unrelated failures** in
  `tests/ogpo/test_verifier_alignment.py`
  (`test_head_value_distribution_differential_over_every_registered_config`,
  `…_is_a_real_change_at_201_bins`, `test_head_wrapper_differential_over_a_randomized_flag_script`):
  they diff the working tree against **git HEAD** for
  `src/rl/value_distribution.py` and `src/envs/wrappers.py` and now find them
  equal, because commit `5b94510` landed that work. Neither file is touched by
  this change; not fixed.
- The two existing tests that exercise the changed buffer API
  (`test_task_index_is_a_transition_field_that_survives_sample_and_shards`,
  `test_num_tasks_none_buffer_has_no_task_index_key`) pass.
- `bash -n` clean on `ogpo_multitask_4task.sh`, `stability_study.sh`,
  `ogpo_multitask_4task_maxlab.sbatch`, `ogpo_multitask_4task_ref_maxlab.sbatch`,
  `ogpo_multitask_4task_ref.sh`, `ws_bcbb_pipeline.sh`.
- `DRY=1 bash scripts/ogpo_multitask_4task.sh` vs `DRY=1 FRESH=1 …`: the emitted
  command lines differ in **exactly one flag** (`--resume` ↔ `--overwrite`), plus
  the intended banner line. An explicit `CKPT_MODE_FLAG` overrides both (checked
  with `CKPT_MODE_FLAG=--resume FRESH=1` → `--resume`).
- `ruff check` on every changed Python file: only three **pre-existing** findings,
  all outside the change (unused `logging` import in AWR, unused `mesh_utils`
  import and one `E731` in `filtered_sft_learner.py`).

### Re-run after the V1/V2/V3 fixes

```
$ uv run pytest tests/ogpo/test_resume_hardening.py \
                tests/ogpo/test_resume_hardening_verifier.py -q
74 passed in 35.44s          # 16 + 58; the two former xfails now pass outright
```
```
$ uv run pytest tests/ogpo/test_ema_utils.py \
    tests/ogpo/test_per_task_critics_verifier2.py \
    tests/ogpo/test_reward_and_value_bounds.py \
    "…test_per_task_critics_verifier.py::test_task_index_is_a_transition_field_that_survives_sample_and_shards" \
    "…test_per_task_critics_verifier.py::test_num_tasks_none_buffer_has_no_task_index_key" -q
43 passed in 8.93s
```
`bash -n` clean on both edited wrappers; every `#SBATCH` directive still precedes
the first non-comment line. `ruff check` on the two edited Python files and the
verifier test file: the same two pre-existing findings, none in the change. The
V1 repro from VERIFICATION §1 now returns the step instead of raising
(`resolve_resume_step({10000}, {20000}, lambda s: True, 10000) == 10000`), while
the same call with `required_ok(10000)` false and the D1 case-3 call both still
raise.

**Not verified here, and not verifiable on this node:** the end-to-end resume —
learner construction, orbax restore, real π0.5 weights, a simulator, a GPU.
`OGPOAgentLearner.__init__` device_puts the EMA to `pinned_host` and is GPU-only
by construction, which is why the rebase and the resolver were factored as pure
functions in the first place.
