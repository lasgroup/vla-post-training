# Resume hardening — implementation plan

**Tier 2.** Built from `docs/changes/2026-08-27-resume-hardening/` (README + BLAST-RADIUS); scope is that record's, not re-derived. This plan is written verbatim to the record's `PLAN.md` as the first action of the implementation pass.

## Context

A resumed OGPO run silently diverges from an uninterrupted one: the success buffer, `_success_task_ranges`, and `_adv_scale` are not persisted (BC anchor degrades, `critic_success_oversample` no-ops, effective-LR spike); `restore_shards` has no step cutoff; `save_epoch_state` has two crash windows that leave an unresumable tree; and the recipe stack defaults to `--overwrite` (a resubmit wipes the run) while the mt4 maxlab wrapper's exit-42 requeue path is dead code (`MAX_RUNTIME` 47 h vs `--time` 25 h).

**Decisions resolved (user-approved 2026-08-27):** D1 = warn-and-degrade for pre-change manifests (new sanctioned non-fail-fast row), raise for the other two cases. D2 = hook on `FilteredSFTLearner`, not the `Agent` ABC. D3 = (c) full save reorder + per-step manifests + restore resolver. D4 = document retention, no pruning. D5 = requeue trap in the two long-run wrappers only. D6 = MT_BAL prompt-collision re-key is OUT (own future Tier 1 record). D7 = both AWR/BofN `save_checkpoint` clones move together.

## Implementation (ordered)

### 1. `src/rl/replay_buffer.py` — step cutoff
`restore_shards(self, shard_dir, *, rng_state_json=None, max_step: int | None = None)`. Filter `shard_paths` by the int parsed from the `step_%08d` stem; `None` = today's behavior. Raise (fix in message) on a `step_*.h5` name that doesn't parse — the atomic writer's temp files never match the glob, so an unparseable match is a foreign file.

### 2. `src/training/runtime_state.py` — manifest schema, per-step manifests, save reorder
- `ResumeState` gains `extra: dict[str, Any] = field(default_factory=dict)` (single nested field: old five-key payloads still construct; unknown top-level keys still raise `TypeError`). Fix the `replay_shard_dir` annotation to `str` (N6) and the `load_resume_state` docstring (N7).
- New helpers: `success_shard_dir(config)` → `runtime_state/success_shards/`; `success_shard_path(config, step)`; `step_manifest_path(config, step)` → `runtime_state/resume_state_%08d.json`.
- `resolve_resume_step(orbax_steps: set[int], manifest_steps: set[int], required_ok: Callable[[int], bool], pointer_step: int | None) -> int` — **pure function** (testable without orbax): greatest step in `orbax_steps ∩ manifest_steps` passing `required_ok`. No per-step manifests at all → legacy fallback: return `pointer_step` if in `orbax_steps`, else raise naming both (D1 case 3). No consistent step in new format → same raise.
- `load_resume_state(config, step=None)`: `step` given → per-step file; else the `resume_state.json` pointer.
- `save_epoch_state` reorder (`prepare_for_resume=True` path): (1) online `save_shard(step N)`; (2) `extra = agent.save_extra_resume_state(step)` (writes the success shard); (3) atomic write of `resume_state_<N>.json` with `payload | {"extra": extra}`; (4) `agent.save_checkpoint(step)` + `wait_until_finished` (orbax commit + GC of step M happens here, **after** everything else is durable); (5) atomically refresh the `resume_state.json` pointer. `prepare_for_resume=False` keeps today's checkpoint-only behavior.

### 3. `src/rl/filtered_sft_agent/filtered_sft_learner.py` — hook + resolver
- New method next to `save_checkpoint` (~`:621`): `save_extra_resume_state(self, step: int) -> dict` returning `{}`; docstring names OGPO as the only override.
- New overridable `_resume_required_paths(self, step: int) -> list[Path]` returning `[]` (base has no rl_state).
- In `__init__` (`:287-292`), replace the bare `load_resume_state` with the resolver: gather orbax steps from `self._checkpoint_manager.all_steps()`, manifest steps by globbing `resume_state_*.json`, `required_ok = lambda s: all(p.exists() for p in self._resume_required_paths(s))`; then `load_resume_state(config, step=S)` and `restore_shards(..., max_step=S)`. Log the chosen step and, when behind the pointer, what was dropped.

### 4. `src/rl/advantage_weighted_sft/advantage_weighted_sft_learner.py` + `src/rl/best_of_n/best_of_n_learner.py` — N3 clone family, both or neither
- `save_checkpoint`: write `rl_state/<step>` (and `task_registry_<step>.json`) **before** `super().save_checkpoint()` commits orbax — removes the newest-orbax-step-without-critics sub-window. Preserve the two existing divergences (BofN normalizes `step=None` and skips existing paths; BofN warns on missing rl_state where AWR raises — OQ-6).
- AWR overrides `_resume_required_paths`: `[rl_state/<step>]`, plus the registry JSON when `self._task_registry is not None` (set before `super().__init__`, so dispatch during base init is safe).

### 5. `src/rl/ogpo/ogpo_learner.py` — success buffer, ranges, `_adv_scale`
- Override `save_extra_resume_state`: always include `"adv_scale": float(self._adv_scale)` (**divergence from BLAST-RADIUS §4.1, which returned `{}` when the success buffer is off — that would lose `_adv_scale` for NORM-only runs; record in DIFF.md**). When `_success_data_buffer` is not None, also write `success_shards/step_%08d.h5` via `save_shard` and include `success_shard_dir`, `success_total_inserted`, `success_task_ranges` (`{task: [[lo, hi], ...]}`), `success_rng_state_json`.
- Restore block in `__init__`, **after** `self._config = orig_config` (`:105`, the config-swap gotcha) and after the `_adv_scale` default (`:112`), gated on `self._resuming`, reading `self._resume_state.extra` with explicit key checks:
  - `extra` empty/absent → WARNING + today's behavior (`# best-effort:` comment; D1 case 1, new sanctioned row).
  - success keys present but shard dir/shards missing → raise with the fix in the message (D1 case 2, mirrors `TaskRegistry.from_json`).
  - Otherwise: `restore_shards(success_shard_dir, rng_state_json=..., max_step=S)`; rebase ranges via a **pure helper** `_rebase_task_ranges(saved_ranges, saved_total_inserted, restored_total_inserted, valid_start)`: uniform `shift = restored - saved`, drop ranges with `hi + shift <= valid_start`, clamp `hi` to `restored_total_inserted` (load-bearing — `_balanced_success_ordinals` only clamps the low end and an over-high `hi` raises out of `replay_buffer.py:155-156`), WARNING when `shift != 0`. Ranges and buffer restore as a package — never one without the other.
  - `self._adv_scale = extra["adv_scale"]`.

### 6. `scripts/exp.py` — `free_buffer_before_eval` re-restore (`:167`) passes `max_step=resume_state.step`.

### 7. Recipes — `--resume` default behind `FRESH=1`
- `scripts/ogpo_multitask_4task.sh:239-242`: explicit `CKPT_MODE_FLAG` env wins; else `FRESH=1` → `--overwrite`; else `--resume`. Mode shown in the `[mt4]` banner.
- `scripts/stability_study.sh:128`: introduce the same-shaped `CKPT_MODE_FLAG` variable replacing the hardcoded `--overwrite`. (Divergent clones — the fix is written twice on purpose, OQ-2.) `ws_bcbb_pipeline.sh` inherits; its stage-A guard stops being destructive — note in DIFF.md, no edit.
- Safe on first launch: openpi downgrades `resume=True` to fresh when the dir is absent/empty (same reason `launcher.py:281-290` can).

### 8. sbatch — requeue trap + `MAX_RUNTIME`
- `ogpo_multitask_4task_maxlab.sbatch`, `ogpo_multitask_4task_ref_maxlab.sbatch`: `#SBATCH --requeue`; replace `exec bash` with capture-status + `scontrol requeue` on exit 42 (shape from `launcher.py:65-82`).
- mt4 wrapper: `export MAX_RUNTIME="${MAX_RUNTIME:-84600}"` (23.5 h under the 25 h wall — fixes N1, the dead exit-42 path).
- Ref wrapper `:21-24`: correct the preempt-header reasoning (`save_interval` is never read by `exp.py`; the real exposure is N2). `ogpo_ref_smoke_maxlab.sbatch` untouched (D5).

### 9. Tests written during implementation (`tests/ogpo/test_resume_hardening.py`)
Pure-data, CPU, model-free, modeled on `test_per_task_critics_verifier.py:520-580` helpers: shard round-trip ordinal exactness; clipped-case uniform shift; `max_step` cutoff (differential against a buffer that only ever saw the early shards); `_rebase_task_ranges` unit + `_balanced_success_ordinals` integration (no raise + balanced counts); `ResumeState` old/new payload round-trip + unknown-key `TypeError`; `resolve_resume_step` — one case per §4.6 crash-table row plus both D1 raise cases.

## Out of scope
N4/MT_BAL re-key (D6 — own record), retention pruning (D4 — document only), offline-iterator seeding, W&B step overlap, the `save_shard` delta clip, anything in `openpi/`, preamble unification, `--skip_requeue` help text. Non-goals in BLAST-RADIUS §8 stay untouched.

## Pass structure (orchestrated; each step an Opus-xhigh agent)
1. **Implementation agent**: writes `PLAN.md` (this file, verbatim) into the record dir first, then implements per `docs/code/best_practices.md`, runs `pytest tests/ogpo`, writes `DIFF.md`.
2. **Verification agent** (fresh context, spec + diff only, per CLAUDE.md step 3): adversarial pytest-native tests per BLAST-RADIUS §9 — including the differential shard tests — plus `DRY=1` recipe diffs (`DRY=1` vs `DRY=1 FRESH=1`, exactly one flag differs), `bash -n` on the sbatch wrappers, full `pytest tests/ogpo`. States plainly what cannot be verified (end-to-end resume needs GPU + π0.5 weights; `OGPOAgentLearner.__init__` is GPU-only, hence the pure-function factoring). Findings land verbatim in `VERIFICATION.md`.
3. **Docs agent** (step 4, scoped by the blast radius): BLAST-RADIUS §10 list — `rl-core.md`, `rl-ogpo.md` (delete the `_adv_scale` gotcha), `rl-learners.md` (add N3 clone family; fix "four subclasses"), `training.md`, `scripts.md`, `src/training/config.py:282-283` comment, CLAUDE.md sharp-edges (`_adv_scale`, `--overwrite`/requeue items), `best_practices.md` §7 new sanctioned row (D1), stale comments in the two probe sbatches.

## Verification summary
`pytest tests/ogpo` green (228 existing + new); differential tests for every refactored numeric/data path; `DRY=1` inspection for recipes; `bash -n` for sbatch; **no launches, no sbatch submissions**; honest statement of the untestable end-to-end path.
