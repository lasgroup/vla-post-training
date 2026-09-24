# Blast radius & change spec — resume hardening

Discovery pass, 2026-08-27. Every cite below was **re-verified against source in
this pass**; where the seeding audit's cite or claim was wrong, the correction is
stated inline and flagged.

---

## 1. Intent

A requeued / resumed run must continue with the same training behavior as an
uninterrupted one, and no ordinary restart may destroy state.

Concretely, after the change:

- **State parity.** Success buffer contents, `_success_task_ranges` and
  `_adv_scale` survive a resume, alongside what already survives (policy +
  EMA, critics, normalizer, task registry, online buffer, both RNG streams,
  `training_steps`, `total_collected_episodes`).
- **Crash atomicity.** For every point at which the process can die inside
  `save_epoch_state`, the on-disk tree still contains one step whose orbax
  checkpoint, `rl_state`, replay shards and manifest are mutually consistent,
  and the restore selects exactly that step.
- **No wipe footgun.** Re-running a recipe against an existing checkpoint
  directory resumes it; wiping requires an explicit opt-in. A job that exits 42
  gets requeued by its sbatch wrapper instead of ending as a failure.

---

## 2. Verification of the seeded audit

### 2.1 Findings that survived verification (with corrected cites)

| # | Claim | Verdict | Verified cites |
|---|---|---|---|
| 1 | Success buffer never persisted | **Confirmed** | `runtime_state.py:82` is the only non-test `save_shard` call site (grep across `src/`, `scripts/`, excluding submodules). `_success_data_buffer` appears only in `ogpo_learner.py` (`:91`, `:106`, `:125`, `:133`, `:138`, `:141`, `:462-465`, `:534-540`, `:685-687`, `:704`) — no `save_shard`/`restore_shards` among them. Silent fallbacks at `:533-536` (BC) and `:460-464` (`critic_success_oversample`) are exactly as described. |
| 2 | `_success_task_ranges` lost with it | **Confirmed** | Init `ogpo_learner.py:94`; appended `:144-147`; consumed `_balanced_success_ordinals` `:696-734`, called at `:539`. Keyed on `str(task_description)` at `:145`. |
| 3 | `_adv_scale` not checkpointed | **Confirmed** | Init `ogpo_learner.py:112` (audit said `:112` — correct); read/updated `:577-589`; documented at `src/training/config.py:282-283` and `docs/code/rl-ogpo.md` gotchas. |
| 4 | `restore_shards` has no step cutoff | **Confirmed** | `src/rl/replay_buffer.py:271` (`sorted(shard_dir.glob("step_*.h5"))`, replayed unconditionally at `:273-281`). Filenames encode the step, zero-padded to 8 (`runtime_state.py:32`). |
| 5 | `--overwrite` requeue footgun | **Confirmed** | `scripts/ogpo_multitask_4task.sh:242` (`CKPT_MODE_FLAG="${CKPT_MODE_FLAG:---overwrite}"`, consumed `:254`); `scripts/stability_study.sh:128` hardcodes it; `openpi/src/openpi/training/checkpoints.py:26-29` rmtrees on overwrite. No exit-42 handling in any `*.sbatch` (`ogpo_multitask_4task_maxlab.sbatch:39`, `ogpo_multitask_4task_ref_maxlab.sbatch:44` both `exec bash …`; `ogpo_ref_smoke_maxlab.sbatch:85` plain `bash`). `launcher.py:65-82` (trap) + `:281-290` (`apply_requeue_flags`, called `:311-312`) is the working precedent. |
| 6 | Crash window in `save_epoch_state` | **Confirmed, and wider than stated** | `runtime_state.py:74` orbax save (+ GC of the previous step), `:82` shard, `:83-91` manifest. `max_to_keep=1` is hardcoded at `openpi/src/openpi/training/checkpoints.py:48` — not overridable from our config, only `keep_period` is. Additional sub-window found: `AdvantageWeightedSFTLearner.save_checkpoint:266-272` writes `rl_state/<step>` **after** `super().save_checkpoint()` has already committed orbax and GC'd the previous step, so a crash there leaves the newest orbax step without critics. See §4.6. |

### 2.2 Findings that did **not** survive verification

- **"`restore_shards` renumbers ordinals from 0, so persisted raw ranges would
  be wrong."** *Not true in the normal case.* `restore_shards` resets
  `total_inserted = 0` (`replay_buffer.py:267`) and re-inserts every shard in
  filename order, so the restored ordinal of a transition is
  `original_ordinal - K`, where `K` is the number of transitions that were
  **never persisted**. `K` is zero unless the `min(delta_count, self.size)` clip
  at `:200` fired — i.e. unless more than `buffer_capacity` transitions were
  inserted between two saves. At the shipped settings (capacity 500k, ~20
  episodes per 10k-step collection round) `K = 0` always, and persisted raw
  ranges would in fact be exactly correct.

  This matters for the design: the rebase is a **single uniform integer shift**,
  not a rebuild, and it is exact (every surviving transition shifts by the same
  `K`, because the clip always drops a prefix). The spec therefore persists the
  success buffer's `total_inserted` alongside the ranges and derives
  `shift = restored_total_inserted - saved_total_inserted`, asserting/logging when
  it is non-zero. No prompt archaeology and no buffer schema change are needed.
  (`shift` is only inexact for a range that straddles a drop boundary — reachable
  only inside the already-accepted `:200` non-determinism.)

- **"AWR's `_rl_checkpoint_path(step)` would crash on `save_checkpoint(step=None)`."**
  True as written (`advantage_weighted_sft_learner.py:268` → `int(None)`), but
  **unreachable**: the only production call site is `runtime_state.py:74`, which
  always passes `agent.training_steps` (an `int`). `BestofNLearner` defends
  against it (`best_of_n_learner.py:192-193`), AWR does not — a divergence in the
  clone family (§3.1), not a live defect. Not in scope to fix.

- **Ref sbatch preempt reasoning.** `ogpo_multitask_4task_ref_maxlab.sbatch:21-24`
  says preempt is unsafe because *"save_interval=100000, so the first checkpoint
  lands at step 100000"*. **`save_interval` is never read by `scripts/exp.py`** —
  grep across `src/` and `scripts/` returns zero uses; it is an openpi
  `TrainConfig` field (`openpi/src/openpi/training/config.py:518`) consumed only
  by openpi's own `train.py:272`. Our loop checkpoints at every collect/eval
  boundary (`exp.py:185-190`), i.e. every 10k steps. The stated reason is wrong;
  the *conclusion* is still right for a different reason — see §2.3.

### 2.3 New findings this pass

- **N1 — `ogpo_multitask_4task_maxlab.sbatch` can never requeue gracefully.**
  `--time=25:00:00` (`:9`) but `MAX_RUNTIME` defaults to 169200 s = 47 h
  (`ogpo_multitask_4task.sh:218`) and the wrapper does not override it. SLURM
  SIGKILLs at 25 h, ~22 h before `exp.py`'s `runtime_exceeded` check
  (`exp.py:187-188`) can ever fire, so the exit-42 path is dead code for this
  wrapper. The recipe's own header warns about exactly this
  (`ogpo_multitask_4task.sh:59-61`: *"With `--time=25:00:00` use
  `MAX_RUNTIME=86400`"*) — the wrapper just never applied it. Combined with the
  `--overwrite` default, an interrupted 25 h job resubmitted by hand wipes
  everything. `ogpo_multitask_4task_ref_maxlab.sbatch` is consistent (48 h vs
  47 h). **This is the real reason preempt is unsafe** — not `save_interval`.
- **N2 — the correct preempt argument.** Under preemption the job is SIGKILLed
  mid-step with no exit-42, and the resubmit runs the recipe again with
  `--overwrite`. The exposure is not "everything since step 0", it is "everything
  since the last 10k boundary, *plus the whole run if the resubmit is not
  hand-edited to `--resume`*". Item 5's fix removes the second half; the header
  comment should be corrected to say so.
- **N3 — a second, unlisted clone family.** `_rl_checkpoint_state` /
  `_rl_checkpoint_dir` / `_rl_checkpoint_path` / `_restore_rl_checkpoint` /
  `save_checkpoint` are near-duplicated between
  `advantage_weighted_sft_learner.py:234-272` and
  `best_of_n_learner.py:164-202`. Not in CLAUDE.md's known-clones list. It is
  directly in this change's path (§4.6 reorders the rl_state write; §6/D7
  proposes rl_state retention). Recommend adding it to
  `docs/code/rl-learners.md`'s duplication note in step 4 regardless of what the
  plan decides.
- **N4 — `_success_task_ranges` buckets are wrong today for the shipped mt4 task
  set.** The ranges dict keys on the prompt string (`ogpo_learner.py:145`), but
  libero_90 prompts are not unique across task ids — `libero_90_79` and
  `libero_90_82` share one (`src/rl/task_registry.py:5-11`, verifier finding
  F1), and **both are in the mt4 default `TASKS`**
  (`ogpo_multitask_4task.sh:87`). So `MT_BAL` currently balances over 3 buckets,
  not 4, and 79+82 jointly receive one task's share. Pre-existing, independent
  of resume. `save_episode` already receives `task_id`
  (`ogpo_learner.py:117-118`, passed from `collect.py:202`), so the fix is a
  one-line re-key — but it **changes MT_BAL's sampling distribution**, which is a
  science change and must not ride silently on a resume PR. Decision **D6**.
- **N5 — the two "read-only" probes construct the real learner under `--resume`.**
  `probe_counterfactual_rollouts.py:185` and `probe_value_next_spread.py:149`
  build `OGPOAgentLearner(cfg)` with `CKPT_MODE_FLAG=--resume`
  (`probe_counterfactual_rollouts.sbatch:68`,
  `probe_value_next_spread.sbatch:46`), against checkpoint directories written
  **before** this change. Any new restore code in `OGPOAgentLearner.__init__`
  runs in those probes. They never save (verified: no `save_epoch_state` /
  `save_checkpoint` in any `scripts/probe_*.py`), so they cannot corrupt state,
  but a hard-raise on a missing success shard would break them. Decision **D1**.
  The other three probe sbatches (`probe_critic_action_sensitivity`,
  `probe_noise_level_sweep`, `probe_policy_candidate_spread`) bypass the recipe
  entirely and read orbax directly — no `CKPT_MODE` assumption, unaffected.
- **N6 — `ResumeState.replay_shard_dir` is annotated `Path` but is always a
  `str`.** `load_resume_state` does `ResumeState(**json.loads(...))`
  (`runtime_state.py:64-65`); JSON has no `Path`. Harmless (`restore_shards` does
  `Path(shard_dir)` at `:261`) but the annotation lies, and the same trap awaits
  any new path-typed manifest field.
- **N7 — `load_resume_state`'s docstring contradicts its code** (`:54-58` says it
  returns `None` when no manifest exists; `:61-62` raises). Already listed in
  `docs/code/training.md` gotchas. We are editing this function; fix the
  docstring as a rider.

---

## 3. Mandatory sweeps

### 3.1 Duplication sweep

| Clone family | Touched by this change? | Disposition |
|---|---|---|
| `stability_study.sh:41-82` ↔ `ogpo_multitask_4task.sh:100-147` env preamble (already diverged) | **Yes** — item 5 changes the checkpoint-mode default in both | The preamble itself is not edited. The `--overwrite` default lives *outside* the cloned block in both files (`stability_study.sh:128` inline in the flag list; `ogpo_multitask_4task.sh:239-242` a variable). **The fix must be written twice, in different shapes.** Do not attempt to unify the preamble here — out of scope, and `ogpo_multitask_4task_ref.sh` already delegates rather than cloning a third time (`:1-8`, `:79`). |
| AWR `update_critic.py` ↔ BofN `update_critic.py` (11 helpers) | No | Untouched. |
| `_pad_last_dim` / best-of-N scoring block, AWR ↔ BofN learners | No | Untouched. |
| `_get_on_policy_action`, BofN ↔ MPO | No | Untouched. |
| `init_train_state`, `filtered_sft_learner.py` ↔ `dsrl_env.py` | No | Untouched (`src/rl/dsrl/` is quarantined and unreachable). |
| **N3 (new): rl-checkpoint block, `advantage_weighted_sft_learner.py:234-272` ↔ `best_of_n_learner.py:164-202`** | **Yes if §4.6 reorders the rl_state write, or if D7 adds retention** | Both copies must move together, or neither. They have already diverged in two ways: BofN normalizes `step=None` (`:192-193`) and skips a re-save when the path exists (`:195-196`); BofN warns on a missing rl_state while AWR raises (`:176-183` vs `:255`, adjudicated OQ-6). Preserve both divergences. |
| Success-buffer save path | n/a — **nothing to duplicate.** The success buffer reuses `ShardedReplayBuffer.save_shard` verbatim; the only new code is the call site and the manifest fields. | |
| `ws_bcbb_pipeline.sh` CKPT_MODE assumptions | **Yes, indirectly** | It never sets a checkpoint mode; both stages delegate to `stability_study.sh` (`:36-40`, `:50-52`), so it inherits `--overwrite` today and inherits the fix for free. Its stage-A guard (`:33`, `if [ ! -d …/20000 ]`) means a stage A interrupted at 15k re-runs from scratch **and wipes**; with a `--resume` default it continues instead. Net improvement, no edit needed beyond `stability_study.sh`. Verify by inspection. |
| Probe sbatch CKPT_MODE assumptions | **Yes** | Two of five override `CKPT_MODE_FLAG=--resume` explicitly (`probe_counterfactual_rollouts.sbatch:68`, `probe_value_next_spread.sbatch:46`) with comments explaining that it *replaces* the recipe's `--overwrite`. If the recipe default becomes `--resume`, those overrides become redundant but stay correct. **Their comments become stale** and must be updated in step 4 — they assert "replaces the recipe's `--overwrite`". |

### 3.2 Inheritance sweep

Verified class graph (`grep '^class .*Learner'`):

```
Agent
├── FilteredSFTLearner
│   ├── BestofNLearner
│   └── AdvantageWeightedSFTLearner
│       ├── MPOWeightedSFTLearner
│       │   └── FlowGRPOLearner
│       └── OGPOAgentLearner
└── DSRLLearner            (quarantined, unreachable from exp.py:76-85)
```

**Correction to CLAUDE.md:** `AdvantageWeightedSFTLearner` has **three**
subclasses (MPO, FlowGRPO, OGPO), not four; `FilteredSFTLearner` has five
descendants, which matches. Worth fixing in step 4 only if a doc sentence
asserts the number.

Relevant overrides:

- `update()` — overridden by **every** concrete learner (`filtered_sft_learner.py:880`,
  `advantage_weighted_sft_learner.py:748`, `best_of_n_learner.py:599`,
  `flow_grpo_learner.py:22`, `ogpo_learner.py:355`; MPO inherits AWR's). This
  change adds no `update()` behavior, so **OQ-10 (EMA advance ownership) is not
  engaged**. `_adv_scale` is written in `update()` (`ogpo_learner.py:584-586`)
  but only read there too, and only OGPO has it — restoring it in `__init__`
  cannot reach a sibling.
- `save_checkpoint()` — overridden by `FilteredSFTLearner:621` (base),
  `AdvantageWeightedSFTLearner:266`, `BestofNLearner:191`, `DSRLLearner:727`.
  MPO/FlowGRPO/OGPO inherit AWR's. §4.6's reorder, if adopted, edits **two**
  of these (AWR and BofN — the N3 clone family); the base and DSRL are untouched.
- `save_episode()` — overridden by `FilteredSFTLearner:716` (base),
  `BestofNLearner:308`, `AdvantageWeightedSFTLearner:620`, `OGPOAgentLearner:116`
  (which is where the success-buffer write and the range append live). Not
  edited unless D6 is taken.
- `_get_online_replay_buffer()` — not overridden anywhere; called at
  `filtered_sft_learner.py:286`, `ogpo_learner.py:106` (success buffer, via the
  temporary config swap at `:95-105`), and `exp.py:166`.

**Mechanism consequence.** `save_epoch_state` is generic over agents
(`runtime_state.py:68-72`, `agent: Any`) and already reaches into privates
(`agent._checkpoint_manager`, `agent._online_data_buffer`, documented as a gotcha
in `docs/code/training.md`). Only OGPO owns a success buffer and `_adv_scale`.
The **save** side therefore needs a polymorphic hook; the **restore** side does
not, because `OGPOAgentLearner.__init__` runs *after*
`FilteredSFTLearner.__init__` has already set `self._resuming` and
`self._resume_state` (`filtered_sft_learner.py:287-292`) and *after* the success
buffer is constructed (`ogpo_learner.py:95-106`) — so the restore is plain
OGPO-local code with no hook and no `isinstance` anywhere. This asymmetry is the
cheapest correct shape; see D2 for the alternative.

### 3.3 Gotchas checked

- `docs/code/rl-core.md:263-270` — the `save_shard` `delta_count` clip is a
  **documented, accepted** non-determinism. §2.2 shows it is also the *only*
  thing that makes ordinals shift, so the rebase must handle it rather than
  assume it away. `save_shard`/`restore_shards` return `None` despite `-> dict`
  annotations — leave as-is unless the plan adds a return value.
- `docs/code/rl-ogpo.md` — success buffer built by temporarily mutating
  `self._config` (`ogpo_learner.py:95-105`): "anything reading config during
  `_get_online_replay_buffer` sees the swapped value". Any restore code must sit
  **after** `self._config = orig_config` (`:105`).
- `docs/code/rl-ogpo.md` — per-task critics are **new-run only** (decision D8 of
  `docs/changes/2026-08-21-per-task-critics/`): neither a shared-critic
  `rl_state` nor a pre-change shard directory resumes into a per-task run. The
  success-shard schema inherits this for free (same `dummy_data`, same
  `task_index` field), so a per-task success shard cannot be restored into a
  shared-critic buffer and vice versa — `insert` raises "Insert transition
  structure does not match buffer structure" (`replay_buffer.py:92-93`). That is
  the desired fail-fast; state it in the record, do not "fix" it.
- `docs/code/rl-learners.md` — OQ-6: BofN warns on missing rl_state, AWR raises.
  Precedent for D1's compat split, and the reason §4.6 must not "harmonize" them.
- `docs/code/training.md` — `save_epoch_state` reaching into agent privates is a
  known coupling; the hook in §4.1 reduces it but does not remove it (the online
  shard write stays where it is unless D2 is taken).
- `docs/code/scripts.md` — the `stability_study.sh` ↔ `ogpo_multitask_4task.sh`
  ~60-line divergent clone; `--skip_requeue` inverts its help text
  (`launcher.py:303`); `apply_requeue_flags` computes an unused `overrides` dict
  (`:283`, `:289-290`). None of these are fixed here.
- `openpi/` and `molmospaces/` are **quarantined**. `max_to_keep=1`
  (`checkpoints.py:48`) and the `rmtree` on overwrite (`:26-29`) are read-only
  facts the design must work around, never edit. No openpi file appears in any
  touch list below.

---

## 4. Per-item spec

Files to touch, in the order the implementation should take them.

### 4.1 Persist the success buffer, its ranges, and `_adv_scale`

**Files**

| File | Change |
|---|---|
| `src/training/runtime_state.py` | `ResumeState` gains `extra: dict[str, Any] = dataclasses.field(default_factory=dict)` (one field, nested — so old manifests still satisfy `ResumeState(**payload)` at `:65`, and future additions need no further schema edits). `save_epoch_state` calls the new hook and merges its return into `payload["extra"]`. New helper `success_shard_dir(config)` → `runtime_state/success_shards/`, mirroring `:27-32`. Docstring fix (N7). |
| `src/rl/filtered_sft_agent/filtered_sft_learner.py` | New method `save_extra_resume_state(self, step: int) -> dict` returning `{}`. Base-class no-op with a docstring naming OGPO as the only override. Placed next to `save_checkpoint` (`:621`). |
| `src/rl/ogpo/ogpo_learner.py` | Override `save_extra_resume_state`: write `success_shard_dir/step_%08d.h5` via `self._success_data_buffer.save_shard(...)`, return `{"success_shard_dir": …, "success_total_inserted": int, "success_task_ranges": {task: [[lo, hi], …]}, "success_rng_state_json": …, "adv_scale": float}`. Returns `{}` when `self._success_data_buffer is None`. Restore block added after `:106` (after the config swap is undone) and before `:112`. |

**Semantics**

- The success shards live in their **own directory** so the existing
  `restore_shards(dir)` glob works unchanged and can never mix the two buffers.
- `_adv_scale` restore replaces the literal at `ogpo_learner.py:112`; when absent
  from the manifest it keeps today's `min_scale` initialization. Resolves the
  `docs/code/rl-ogpo.md` gotcha and the `config.py:282-283` comment (both must be
  updated in step 4 — the comment is a *documented claim about behavior* that
  this change falsifies).
- No `getattr(cfg, …, default)` anywhere. The hook is a real method with a real
  base implementation; the manifest `extra` block is read with explicit key
  presence checks, and every absent-key path is one of the two D1 cases.

### 4.2 Rebase `_success_task_ranges` on restore

Per §2.2 the correct operation is a uniform shift, not a rebuild:

```
shift  = success_buffer.total_inserted (after restore) - manifest["success_total_inserted"]
ranges = {task: [(lo + shift, hi + shift) for lo, hi in rs] for task, rs in saved.items()}
```

then drop ranges with `hi + shift <= buf.valid_start` and clamp
`hi = min(hi + shift, buf.total_inserted)` before storing. `shift != 0` means the
`replay_buffer.py:200` clip fired; log it at WARNING with both counts.

Two correctness requirements the plan must not lose:

1. **The `hi` clamp is load-bearing.** `_balanced_success_ordinals` clamps only
   the low end (`ogpo_learner.py:709`, `lo2 = max(lo, buf.valid_start)`); a `hi`
   above `total_inserted` reaches `sample(ordinals=…)` and raises "ordinals
   reference evicted or unwritten transitions" (`replay_buffer.py:155-156`).
2. **Ranges and buffer are a package.** Restoring the buffer but *not* the ranges
   is worse than restoring neither: `_balanced_success_ordinals` would then
   sample only from post-resume successes while the buffer holds all of them, a
   silent distribution change. Either both restore or the whole `extra` block is
   ignored (D1).

The prompt-vs-`task_id` keying question is **N4 / decision D6**; the rebase is
key-agnostic and works either way.

### 4.3 Step cutoff for `restore_shards`

**File:** `src/rl/replay_buffer.py`

`restore_shards(self, shard_dir, *, rng_state_json=None, max_step: int | None = None)`.
Filter `shard_paths` (`:271`) by the integer parsed from the `step_%08d` stem;
`None` keeps today's behavior. Keyword-only with a default, so the existing
positional call in `tests/ogpo/test_per_task_critics_verifier.py:565` and the two
production call sites keep working.

Fail-fast on an unparseable `step_*.h5` name rather than skipping it silently —
the atomic writer's temp files are `.step_….tmp` (leading dot,
`replay_buffer.py:239-244`) and never match the glob, so an unparseable match
means someone put a foreign file in the shard directory.

**Call sites to update (both, and there are only two):**

- `src/rl/filtered_sft_agent/filtered_sft_learner.py:289-292` →
  `max_step=self._resume_state.step`
- `scripts/exp.py:167` (the `free_buffer_before_eval` re-restore) →
  `max_step=resume_state.step`. Same-step no-op today; correct under §4.6's
  reorder, where a shard for a step ahead of the manifest can exist.

Per §4.1 the success-buffer restore uses the same cutoff.

**Tier-2 note:** this is a public signature change on a `src/rl/` primitive
shared by all six learners. It is additive and keyword-only, so no sharding,
donation or RNG contract moves.

### 4.4 Recipe: `--resume` by default

**Files:** `scripts/ogpo_multitask_4task.sh` (`:239-242`),
`scripts/stability_study.sh` (`:128`).

Flip the default to `--resume`, keep `--overwrite` behind an explicit `FRESH=1`.
The pattern is safe on a first launch because `initialize_checkpoint_dir`
downgrades `resume=True` to a fresh start when the directory is absent or holds
no checkpoints (`openpi/src/openpi/training/checkpoints.py:22-33`, `:56-61`) —
the same reason `launcher.py:281-290` can set `resume=True` on *first*
submission. `stability_study.sh` has no `CKPT_MODE_FLAG` variable today; add one
mirroring mt4's so the two divergent clones at least share the shape.

**Trade to state explicitly in the plan:** with `--resume` as the default,
re-running the same `ARM`/`SEED` intending a *fresh* run silently continues the
old one instead of wiping. That is the intended direction (a silent continue is
recoverable; a silent `rmtree` is not), but it changes muscle memory. `FRESH=1`
must be loud in the `[mt4]` banner line (`:231-232`).

`ws_bcbb_pipeline.sh` needs no edit (§3.1) but its stage-A guard behavior changes
for the better; note it in `DIFF.md`.

### 4.5 sbatch: exit-42 requeue trap and a `MAX_RUNTIME` consistent with `--time`

**Files:** `scripts/ogpo_multitask_4task_maxlab.sbatch`,
`scripts/ogpo_multitask_4task_ref_maxlab.sbatch`. (`ogpo_ref_smoke_maxlab.sbatch`
is a 2 h smoke — requeue is pointless there; leave it, or add for uniformity, D5.)

- Replace `exec bash …` with the capture-status/`scontrol requeue` shape from
  `launcher.py:65-82`. Add `#SBATCH --requeue` explicitly rather than relying on
  the cluster default.
- `ogpo_multitask_4task_maxlab.sbatch`: `export MAX_RUNTIME="${MAX_RUNTIME:-84600}"`
  (23.5 h under the 25 h wall) — fixes N1.
- `ogpo_multitask_4task_ref_maxlab.sbatch:21-24`: correct the preempt-header
  reasoning per §2.2 / N2. Tier-0 rider, comment only.

### 4.6 Crash-atomic save + tolerant restore

**Files:** `src/training/runtime_state.py`,
`src/rl/advantage_weighted_sft/advantage_weighted_sft_learner.py:266-272`,
`src/rl/best_of_n/best_of_n_learner.py:191-202` (N3 clone family — both or
neither), `src/rl/filtered_sft_agent/filtered_sft_learner.py:287-313`.

`max_to_keep=1` is fixed inside openpi, so the previous checkpoint **is** deleted
the moment the new one commits. The window can only be closed by writing
everything else *before* that commit, and by resolving the resume step from what
is actually on disk.

**Proposed order in `save_epoch_state`** (currently orbax → shard → manifest):

1. online shard for step N, plus the §4.1 hook (success shard) — both delta
   writes, both atomic (`replay_buffer.py:238-253`);
2. per-step manifest `runtime_state/resume_state_%08d.json` (atomic,
   `runtime_state.py:39-50`);
3. `agent.save_checkpoint(step=N)` — inside which `rl_state/<N>` is written
   **before** `super().save_checkpoint()` commits orbax (this is the AWR/BofN
   edit; it also removes the AWR sub-window found in §2.1/#6);
4. atomically refresh `resume_state.json` as the latest pointer (kept for
   backward compatibility and human inspection; the resolver does not depend on
   it).

**Restore resolver** (in `FilteredSFTLearner.__init__`, replacing the bare
`load_resume_state` at `:288`): choose the greatest step `S` such that
`S ∈ checkpoint_manager.all_steps()` **and** `resume_state_<S>.json` exists
**and** `rl_state/<S>` exists (plus `task_registry_<S>.json` when
`_num_critic_tasks is not None`). Then restore orbax, rl_state and both buffers
at `S`, with `max_step=S`.

Crash-window table under this design:

| Crash point | On disk | Resolver picks | Lost |
|---|---|---|---|
| during (1) shard write | pointer M, orbax M, manifest M, rl_state M; partial shard N discarded by the atomic rename | M | since M |
| between (1) and (2) | as above, plus shard N present | M (shard N excluded by `max_step=M`) | since M |
| between (2) and (3) | manifest N present, orbax still M | M (no orbax N) | since M |
| inside (3), before orbax commit | rl_state N present, orbax still M | M | since M |
| inside (3), after orbax commit (M GC'd) | orbax N, rl_state N, manifest N, shard N — all complete | **N** | nothing |
| between (3) and (4) | as above | **N** | nothing |

Every row is recoverable. The current code has two unrecoverable rows.

**Backward compatibility:** a directory written before this change has only
`resume_state.json` and no per-step manifests. The resolver must fall back to
today's behavior (`resume_state.json`'s step, restore at that step) when no
`resume_state_*.json` exists — that path is what the two `--resume` probes (N5)
and every in-flight run take. Decision **D1** covers what happens when the
fallback manifest names a step orbax no longer has.

`prepare_for_resume=False` never occurs in the tree (`exp.py:138`, `:190` both
pass `True`) but the branch at `runtime_state.py:77-79` must keep working.

---

## 5. Expected behavior after

- Resuming an OGPO run reproduces the uninterrupted run's success-buffer
  contents, task ranges, `_adv_scale`, online buffer, RNG streams, critics,
  normalizer, task registry, policy and EMA. The remaining known divergences are
  the two out-of-scope ones in §7.
- `restore_shards` never ingests a shard newer than the checkpoint being resumed.
- Killing the process at any point inside `save_epoch_state` leaves a resumable
  tree; the resume logs which step it selected and, when that is behind the
  latest pointer, what it dropped.
- `bash scripts/ogpo_multitask_4task.sh` against an existing checkpoint dir
  resumes; `FRESH=1 bash …` wipes and starts over.
- An sbatch job that exits 42 requeues itself and continues; the mt4 maxlab
  wrapper actually reaches that exit before its wall clock.
- `pytest tests/ogpo` stays green (228 tests), including
  `test_per_task_critics_verifier.py::test_task_index_is_a_transition_field_that_survives_sample_and_shards`,
  which calls `restore_shards` with no cutoff.

---

## 6. Open decisions for the plan gate

**D1 — missing / stale resume state: raise or degrade?** *(deviation protocol)*
Three sub-cases, and they should not all get the same answer:

| Case | Proposed | Rationale |
|---|---|---|
| Manifest has no `extra` block (pre-change directory) | **Warn, start the success buffer empty, `_adv_scale = min_scale`** — i.e. today's behavior | This is the only way the two `--resume` probes (N5) and every in-flight run keep working. Direct precedent: `best_of_n_learner.py:176-183` in the sanctioned-non-fail-fast table (`best_practices.md` §7). Carries a `# best-effort: <why>` comment. |
| Manifest declares success state but the shard dir / shards are missing | **Raise**, with the fix in the message | Corruption or a hand-edited tree, not a compat case. Mirrors `TaskRegistry.from_json` (`task_registry.py:67-73`). |
| Resolver finds no fully consistent step, and the fallback manifest names a step orbax no longer has | **Raise**, naming the available steps and the manifest step | Anything else silently trains from the wrong weights. |

The first row is a new entry in the sanctioned-non-fail-fast table and needs
explicit sign-off.

**D2 — hook shape: `FilteredSFTLearner` method, or `Agent` ABC?**
Recommend the `FilteredSFTLearner` method (§4.1): `save_epoch_state`'s only
callers are `exp.py:138`/`:190` and the memdiag wrapper, and every dispatchable
learner (`exp.py:76-85`) is a `FilteredSFTLearner`. Adding to the `Agent` ABC
would be the "correct" interface move and would let `save_epoch_state` stop
duck-typing, but it drags `DSRLLearner` (quarantined) into the diff and widens a
Tier 2 change for no reachable benefit. A third option — move the online shard
write into the learner too, so `runtime_state.py` stops touching
`agent._online_data_buffer` (the `docs/code/training.md` gotcha) — is a real
cleanup but doubles the diff and collides with `exp.py:139`/`:166`, which pokes
the same private. **Recommend: hook only, gotcha left standing.**

**D3 — how far to go on §4.6.** Three tiers:
 (a) *step cutoff only* (§4.3) — cheap, fixes the rolled-back-resume half,
 leaves both unrecoverable crash rows;
 (b) *cutoff + tolerant restore against the single latest manifest* — closes the
 "manifest points at a GC'd checkpoint" row, but the restored RNG /
 `total_collected_episodes` then belong to a *later* step than the weights, a
 small silent skew;
 (c) *cutoff + per-step manifests + reorder + resolver* (§4.6 as written) — no
 skew, every crash row recoverable, and it subsumes seed item 7. Costs the
 AWR/BofN clone-family edit and one small JSON file per boundary (~1 KB × 10 per
 run).
**Recommend (c).** (b)'s skew is exactly the class of silent-wrongness this
codebase has been burned by. Note that (c) makes seed item 7 *free*.

**D4 — retention policy (seed item 8).** Today: orbax keeps 1 step plus
`keep_period` multiples (mt4 passes `--keep_period 100000`, i.e. only the final
step is pinned), while `rl_state/<step>/`, `task_registry_<step>.json` and every
replay shard accumulate forever. RL state therefore outlives the policy state it
belongs to, and replay shards are the dominant term — each shard stores the
observations for its delta, including 224×224×3 uint8 images per camera
(~300 KB/observation), so a run's shard directory grows with total collected
experience and is never pruned. **No live run's disk was measurable in this pass**
(`run_store/checkpoints/**` is empty and the group-data mount is not visible from
this node), so the cost is stated as a formula, not a number — the plan should
measure one real mt4 run before choosing.
Options: (i) leave alone, document; (ii) prune `rl_state/<step>` to the steps
orbax still has *plus* every step named by a surviving per-step manifest — must
not race the resolver (§4.6) and must be written into **both** N3 clones;
(iii) additionally prune replay shards below the oldest resumable step — but
those shards are the buffer's *only* copy of that experience, so pruning them
narrows how far back a resume can go, permanently. **Recommend (i) for this
change and a separate measured pass for (ii)/(iii)** — deleting training data to
save disk is not a resume-hardening decision.

**D5 — scope of the sbatch edits.** Requeue trap in the two long-run wrappers
only, or also `ogpo_ref_smoke_maxlab.sbatch` (2 h smoke, its EXIT trap already
prints the memory peak and would need care to compose)? **Recommend: the two
long-run wrappers only.**

**D6 — N4: re-key `_success_task_ranges` from prompt to `task_id`?** It is a
one-line fix in code this change already touches, and it makes MT_BAL correct for
the shipped 4-task set (4 buckets, not 3). But it **changes the training
distribution** of every MT_BAL run, so folding it into a resume PR would make
"resume hardening" a science change and make before/after runs incomparable.
**Recommend: out of this change, recorded here, its own Tier 1 record.** If the
plan disagrees, it must be a separately-flagged commit with its own verification.

**D7 — do the AWR/BofN `save_checkpoint` copies both move?** §3.1/N3 says yes if
§4.6(c) is taken. The alternative — reorder only AWR, since BofN is not in the
active research path — leaves the two copies with different crash semantics and
no comment saying why. **Recommend: move both, and add the family to
`docs/code/rl-learners.md`'s duplication note.**

---

## 7. Out of scope (recorded so it is not re-litigated)

- **Offline data iterator seeding on resume** (`filtered_sft_learner.py:267`
  comment). OGPO pins `online_ratio = 1.0` (`ogpo_learner.py:60-66`), so
  `self._data_iter is None` (`:275-284`) and there is nothing to seed for the
  runs in question.
- **W&B step overlap on resume.** `Logger(config, resuming=agent._resuming, …)`
  (`exp.py:87`); metric duplication across a requeue boundary is a logging
  concern, not a training-state one.
- **`save_shard`'s `delta_count` clip** (`replay_buffer.py:200`). Documented and
  accepted (`docs/code/rl-core.md:263-267`). §4.2 *handles* its effect on
  ordinals rather than removing it.
- **openpi's `max_to_keep=1` and `rmtree`-on-overwrite.** Quarantined submodule;
  §4.6 is designed around them.
- **Unifying the `stability_study.sh` ↔ `ogpo_multitask_4task.sh` preamble.**
  Adjudicated debt (OQ-2); §4.4 writes the same fix twice on purpose.
- **AWR `save_checkpoint(step=None)` crashing** (§2.2). Unreachable.
- **`--skip_requeue`'s inverted help text**, `apply_requeue_flags`'s dead
  `overrides` dict, `save_shard`/`restore_shards` lying return annotations.
  Unrelated pre-existing defects.

## 8. Verified-correct — non-goals, do not "fix"

- Full-EMA recomposition in `save_checkpoint` (`filtered_sft_learner.py:621-637`)
  and the trainable-only re-slice on resume (`:329-332`): a resumed run's EMA is
  bit-identical to the pre-crash one, including frozen SigLIP leaves.
- Critic / value / normalizer / task-registry checkpointing
  (`advantage_weighted_sft_learner.py:234-272`), restored at `:153-154` from
  `self.training_steps`, which the base has already set from the manifest
  (`filtered_sft_learner.py:311`).
- Manifest round-trip of `step`, `total_collected_episodes`, and both RNG streams
  (`runtime_state.py:84-91` ↔ `filtered_sft_learner.py:311-313`, `:289-292`;
  `rng_state_json` / `set_rng_state_json` at `:639-658` and
  `replay_buffer.py:187-194`).
- Step-0-gated initial rollouts (`collect.py:133-134`) — a resume at step > 0
  correctly does **not** re-pay `num_initial_rollouts`.
- Step-derived env seeding (`collect.py:124`, `env.seed(config.seed + step)`) —
  deterministic in the step, so a resume reproduces the same env seeds.
- Stateless per-task advantage normalization (`update_actor.py:311-340`) — pure
  function of the batch's tokenized prompts, nothing to checkpoint.
- Per-task critics being new-run-only (§3.3) — the shard schema mismatch raises
  by design.

---

## 9. Verification approach (for step 3's independent verifier)

**pytest-native, CPU-only, model-free** — all of this is pure data plumbing and
belongs in `tests/ogpo/`, modelled on
`tests/ogpo/test_per_task_critics_verifier.py:520-580` (its `_buffer_dummy` /
`_insert_episode` helpers already build a real `ShardedReplayBuffer` with and
without `task_index`):

1. **Shard round-trip preserves ordinals exactly.** Save from a buffer with
   multiple episodes, restore into a fresh one, assert `total_inserted`,
   `valid_start` and per-ordinal payload equality. This is the *evidence* for the
   §2.2 correction and must be written before the rebase code is trusted.
2. **Clipped case pins the shift.** Force `delta_count > size` (small
   `max_capacity`, one large insert between saves), assert
   `restored_total_inserted - saved_total_inserted` equals the number of dropped
   transitions and that a uniform shift maps every surviving transition to its
   original payload — the formula in §4.2, not a restatement of the code.
3. **`max_step` cutoff.** Shards at steps 0/1000/2000; restore with
   `max_step=1000`; assert the result is byte-identical to a buffer that only
   ever received the first two shards (**differential**, not self-comparison).
   Assert the no-`max_step` call is unchanged.
4. **Range rebase helper** as a pure function: shift, low-drop against
   `valid_start`, `hi` clamp against `total_inserted`, empty-after-clamp task
   removal. Then an integration assert that `_balanced_success_ordinals` over the
   rebased ranges returns only ordinals in `[valid_start, total_inserted)` and
   balanced per-task counts — the failure mode is a raise from
   `replay_buffer.py:155-156`, so make the test assert no raise *and* the counts.
5. **Manifest schema compatibility.** `ResumeState(**old_payload)` (five keys,
   no `extra`) still constructs; a new payload round-trips through
   `json.dumps`/`load_resume_state` with `extra` intact; an unknown top-level key
   still raises `TypeError` (i.e. we did not accidentally make the schema loose).
6. **Resolver as a pure function** over `(orbax_steps, manifest_steps,
   rl_state_steps, registry_steps, pointer_step)` → chosen step, with one case
   per row of the §4.6 crash table plus the two D1 raise cases. Building a real
   orbax directory is not required and should not be attempted — factor the
   resolver so it takes the step sets as arguments.

**Honest fallbacks — state plainly what was not verified:**

- The end-to-end resume (learner construction, orbax restore, real π0.5 weights,
  a GPU, a simulator) **cannot be run here**. Do not imply otherwise.
- Recipe changes: `DRY=1 bash scripts/ogpo_multitask_4task.sh` and
  `DRY=1 FRESH=1 bash scripts/ogpo_multitask_4task.sh` — diff the two emitted
  command lines and confirm exactly one flag changed. `stability_study.sh` has no
  `DRY` support; inspect the flag list by reading it.
- sbatch changes: read-only inspection plus `bash -n` on each wrapper. Do **not**
  submit anything.
- Run `pytest tests/ogpo` (228 tests, CPU, seconds; the PaliGemma-backed legs take
  minutes — the full suite needs `sbatch scripts/run_ogpo_tests.sbatch`, which
  requires permission). `pytest tests/` still fails at collection
  (`tests/nnx_networks/test_nnx_network_implementation.py:6-7`); out of scope
  (OQ-4).
- A CPU smoke that instantiates `OGPOAgentLearner` is **not** available — its
  `__init__` device_puts the EMA to `pinned_host` (`ogpo_learner.py:84-87`) and
  is GPU-only by construction (stated in that comment). The restore code must
  therefore be factored so its logic is testable without the learner.

## 10. Docs to update in step 4 (scoped by this blast radius)

- `docs/code/rl-core.md` — `ShardedReplayBuffer.restore_shards` signature +
  symbol bullet; the consumer list at `:296-301` (a second buffer instance is now
  persisted).
- `docs/code/rl-ogpo.md` — delete the "`_adv_scale` is not checkpointed" gotcha;
  update the success-buffer bullets.
- `docs/code/rl-learners.md` — add the N3 clone family to the duplication note;
  correct "four subclasses" if asserted (§3.2).
- `docs/code/training.md` — `save_epoch_state` / `load_resume_state` /
  `ResumeState` symbol bullets; delete the `load_resume_state` docstring gotcha
  (N7) once fixed; adjust the "reaches into agent privates" gotcha per D2.
- `docs/code/scripts.md` — checkpoint-mode defaults, the sbatch requeue trap.
- `src/training/config.py:282-283` — the in-source comment claiming `_adv_scale`
  is not checkpointed becomes false; `CLAUDE.md`'s sharp-edges list carries the
  same claim and the `--overwrite` / requeue items.
- `best_practices.md` §7 sanctioned-non-fail-fast table — one new row, if D1's
  first case is approved.
