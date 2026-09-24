# VERIFICATION — `docs/changes/2026-09-07-single-task-ref-recipe`

Independent adversarial pass (fresh-context agent, Opus, xhigh effort). Read-only: no files modified, no `sbatch`, no `uv run scripts/exp.py` training launch. The only executions were `DRY=1` shell renders and a **parsing-only** tyro resolve (a scratchpad script calling `src.training.config.cli()` on the rendered argv — no learner constructed, no device work, `JAX_PLATFORMS=cpu`).

## (a) Diff audit

`scripts/stability_study.sh` is uncommitted in the working tree along with the sibling change `docs/changes/2026-09-07-stability-recipe-task-knob/`, so `git diff` against HEAD shows **both** changes. Attribution checked hunk by hunk: the `TASK` / `CKPT_BASE_DIR` / LIBERO-seeding / `CKPT_MODE_FLAG` / `RUN=(echo …)` / `"$@"` hunks are all claimed verbatim by `2026-09-07-stability-recipe-task-knob/DIFF.md`. The hunks belonging to *this* change are exactly the four `DIFF.md` claims, and no others:

- `:46-60` header docs for `TD_W`, `SUCC_BONUS`, plus the pointer to `stability_study_ref.sh`.
- `:78-79` `TD_W="${TD_W:-1}"`, `SUCC_BONUS="${SUCC_BONUS:-0}"`.
- `:132-135` `EXTRA_FLAGS+=(--collect.success_reward_bonus "$SUCC_BONUS")` as the first `EXTRA_FLAGS` entry.
- `:244-245` `--rl.critic.td_weight_schedule.init_value/end_value` literal `1` → `"$TD_W"`.

`DIFF.md`'s "Nothing else in the file changed" holds. Both new files are untracked-new as claimed. `bash -n` passes on all three files.

## (b) DRY=1 regression check (baseline path)

Reconstructed the pre-change script in a scratchpad by reverting exactly those three functional edits, then token-diffed the two renders across every existing caller's env shape (bare `ARM/GPU`, `ws_bcbb_pipeline.sh` stage A, stage B, `probe_candidate_q_spread_stab.sbatch`). In every case the only diff is `+ --collect.success_reward_bonus 0`. `td_weight_schedule.init_value`/`end_value` render as `1`/`1`. Resolved-config check (not just command text): the baseline render resolves to `collect.success_reward_bonus 0.0`, `td_weight_schedule {init 1.0, end 1.0, switch 999999}` — identical to pre-change. **Regression check passes.**

## (c) DRY=1 ref-recipe check (both tasks)

`ARM=verify_ref_31 TASK=libero_90_31 GPU=0` and `ARM=verify_ref_38 TASK=libero_90_38 GPU=1` render identically except the task/exp-name tokens. Resolving the task-31 argv through the real `_config.cli()`:

```
config_name pi05_libero_online_ogpo_ref | collect.tasks/eval_tasks ['libero_90_31']
collect.success_reward_bonus 90.0 | td_weight_schedule {0.95, 0.95, 999999}
critic.num_qs/num_vs 10 10 | critic.reduction mean | rl.n_samples 8
rl.critic_success_oversample True | rl.advantage_combination grpo_conservative
rl.normalize_group_advantage False | rl.normalize_advantage_per_task False | rl.adv_clip_sym None
rl.clip_epsilon 0.1 | rl.discount 0.995 | rl.group_num_samples 8 | rl.use_success_buffer True
```

**Differential check against the established path:** resolved `GPU=0 DRY=1 bash scripts/ogpo_multitask_4task_ref.sh` the same way. Every reference-alignment field above is byte-for-byte equal between the new single-task path and the already-in-use multitask ref path (they differ only in `collect.tasks`, `buffer_capacity`, `project/group/exp` names). The single-task recipe reaches the reference stack by config default where the multitask recipe reaches it by explicit flag — same resolved config.

Also verified empirically that tyro takes the **last** occurrence of a repeated flag (appending `--collect.success_reward_bonus 5` to the ref argv resolves to `5.0`), so the trailing-args escape hatch still works over the new unconditional flag.

## (d) Adversarial check of the BLAST-RADIUS claim — CONFIRMED

Enumerated every field `pi05_libero_online_ogpo_ref` overrides (`config.py:842-872`) against every **unconditionally emitted** flag in `stability_study.sh`. Of those, only `td_weight_schedule` (needs `TD_W`) and `success_reward_bonus` (needs `SUCC_BONUS`, not even a ref-config field) are clobbered/missing; every other field (`num_qs/num_vs`, `reduction`, `n_samples`, `critic_success_oversample`, `advantage_combination`, both normalizer flags, `adv_clip_sym`) is untouched by any unconditional flag and flows through from the config's own default. **Verdict: the BLAST-RADIUS.md claim is correct as stated.**

One field is unconditionally clobbered but *not* on the change record's list: `policy.update_interval`/`training_start_step` are forced to `10`/`900` (ref config's own default is `20`/`100`). **Not a defect** — `ogpo_multitask_4task.sh:315-316` clobbers those identically, so the completed multitask ref arms already ran at 10/900; the new single-task path stays comparable to them and to the pre-existing single-task campaign. Worth knowing, previously undocumented anywhere.

**One asymmetry found by test, not by reading:** because most reference-alignment fields now reach the run as *config defaults* rather than flags, env vars cannot turn them back off through `stability_study_ref.sh`. `CONS=0 QS=2 bash scripts/stability_study_ref.sh` resolves `num_qs 2` correctly (the `QS` gate works) but `advantage_combination` stays `grpo_conservative` (`CONS=0` is a silent no-op under this config), and `reduction`/`n_samples`/`critic_success_oversample` have no knob at all. `ogpo_multitask_4task.sh:205-214` calls out exactly this hazard as the reason its own knobs are unconditional; that contract does not hold on this path. Deliberate per the record ("adding no-op knobs would be unrequested scope") and harmless for the intended forward run, but a foot-gun for anyone trying to reproduce a *baseline* arm through `stability_study_ref.sh` via env vars — trailing CLI args (`--rl.advantage_combination reduced`) remain the working escape hatch. Documented in the header now (see below).

## (e) Other-callers collision check

Grepped every reference to `stability_study.sh`. `ws_bcbb_pipeline.sh`, `stability_wave1_bc0.sh`, `probe_candidate_q_spread_stab.sbatch` set neither `TD_W`, `SUCC_BONUS`, nor `--collect.success_reward_bonus` — all render identically to before. No `tests/` file references `stability_study.sh`.

**Export-inheritance exposure (minor, now documented):** `ws_bcbb_pipeline.sh` launches via `env GPU=… ARM=… bash stability_study.sh`, inheriting the caller shell's exports. `TD_W`/`SUCC_BONUS` join the `TASK`/`LORA` trap family the header already warns about for `TASK` — a shell that exported `SUCC_BONUS=90` for a ref arm would silently change the reward function of a `ws_bcbb_pipeline.sh` arm launched from the same shell.

## (f) sbatch sanity check

`bash -n scripts/stability_study_ref_maxlab.sbatch` → OK. **`sbatch` itself was not invoked** — not submission, not `--test-only` (excluded by ground rules) — so SLURM's own acceptance of the directives is unverified.

Directives identical to `ogpo_multitask_4task_ref_maxlab.sbatch` apart from `--job-name`. Exit-42 requeue block is a faithful copy, tested by running the file as a plain script with `ARM`/`TASK` unset (fails fast at the right lines, before the launch). Calls `stability_study_ref.sh`, not `stability_study.sh` — correct. `stability_study_ref.sh` confirmed cwd-independent.

**Finding acted on:** the sbatch did not set `N_STEPS`, so it inherited `stability_study.sh`'s `N_STEPS=100000` → `range(start_step, 100000)` → last step 99999 → eval fires at 10k…90k only, **no eval at 100k** (`exp.py` eval condition is `step % eval_interval == 0`). `ogpo_multitask_4task_ref_maxlab.sbatch` sets `NUM_STEPS=100001` for exactly this reason. **Fixed**: added `export N_STEPS="${N_STEPS:-100001}"` to `stability_study_ref_maxlab.sbatch`.

## (g) Deviation-protocol check

No new non-fail-fast path. Two shell env-var defaults (`${TD_W:-1}`, `${SUCC_BONUS:-0}`), both documented with their exact prior-behavior equivalence — the established recipe pattern (`ogpo_multitask_4task.sh:215-228`), not a Python silent config default; a malformed value fails loudly at tyro parse. Nothing to escalate.

## (h) Not verified

- Any runtime/training behavior, GPU memory envelope, or whether the reference alignment actually helps on libero_90_31/38 — that's the experiment, not this verification. The sbatch's memory estimate is extrapolated from the mt4 ref smoke (job 10178533), not remeasured single-task.
- SLURM acceptance of the sbatch directives (no `sbatch` invoked, per ground rules).
- `pytest tests/ogpo` not run — no `src/` change; out of scope, stated as skipped rather than silently omitted.
- `ws_bcbb_pipeline.sh` stage A end-to-end (`CONFIG_NAME=…_unfrozen_backbone`, a different `ENTRY`) — pre-existing arrangement, unaffected by this change, compared textually only.

## (i) Verdict

**No bugs. The change is correct and the change record's load-bearing claim is confirmed.** Baseline behavior is bit-identical in resolved-config terms for all existing callers. The ref path resolves to exactly the reference-aligned stack, field-for-field identical to the already-validated multitask ref path.

Three items handed back; two fixed post-verification, one is a documentation-only residual now closed by step 4:

1. **`N_STEPS=100001`** — fixed in `stability_study_ref_maxlab.sbatch`.
2. **`CONS=0`/`reduction`/`n_samples`/`critic_success_oversample` have no working env-var override under the ref config** — documented in `stability_study_ref.sh`'s header (trailing CLI args are the escape hatch); accepted as out of scope for new knobs.
3. **`docs/code/scripts.md` update** — done in step 4 (env-var list, caller tree, Gotchas export-inheritance entry, stale line-count/cites).
