# Diff

## `scripts/stability_study.sh` (edited)

- Header comment: documents `TD_W` and `SUCC_BONUS`, and points at
  `stability_study_ref.sh` for the reference-aligned wrapper.
- `TD_W="${TD_W:-1}"` and `SUCC_BONUS="${SUCC_BONUS:-0}"` added next to the
  existing `TASK=` default.
- `EXTRA_FLAGS+=(--collect.success_reward_bonus "$SUCC_BONUS")` added
  unconditionally, first line of the `EXTRA_FLAGS` assembly (mirrors
  `ogpo_multitask_4task.sh:221-222`).
- `--rl.critic.td_weight_schedule.init_value 1` / `...end_value 1` (fixed
  literals) → `"$TD_W"` / `"$TD_W"`.

Nothing else in the file changed. `DRY=1` output for every existing caller
(no `TD_W`/`SUCC_BONUS` set) is unchanged except for the new, always-present
`--collect.success_reward_bonus 0` — same non-goal as the reference-alignment
commit's own multitask change: "the emitted command line gains these flags...
The RESOLVED config is unchanged at the defaults" (`ogpo_multitask_4task.sh:212-214`).
Confirmed by `DRY=1` run, see `VERIFICATION.md`.

## `scripts/stability_study_ref.sh` (new)

Thin wrapper: exports `CONFIG_NAME=pi05_libero_online_ogpo_ref` (default),
`TD_W=0.95`, `SUCC_BONUS=90`, `exec bash stability_study.sh "$@"`. No other
env var forced — see `BLAST-RADIUS.md` for why the rest of the
reference-alignment fields need no wrapper-level override here.

## `scripts/stability_study_ref_maxlab.sbatch` (new)

maxlab, 1 GPU, VRAM_96GB, 16 cpus, 150G, 48h, `--requeue`/`--open-mode=append`,
exit-42 requeue handling — copied from
`ogpo_multitask_4task_ref_maxlab.sbatch`. Requires `ARM` and `TASK` via
`--export` (fails fast if unset, same contract as `stability_study.sh`'s own
`ARM`/`GPU` requirements). Calls `scripts/stability_study_ref.sh`.

## Divergence from PLAN

No `PLAN.md` — Tier 1, one pass. Matches `BLAST-RADIUS.md`.

## Self-check (DRY=1, before handing to the verifier)

```
ARM=smoketest GPU=0 DRY=1 bash scripts/stability_study.sh
  -> pi05_libero_online_ogpo_sft, td_weight_schedule 1/1, success_reward_bonus 0

ARM=REF_libero_90_31 TASK=libero_90_31 GPU=0 DRY=1 bash scripts/stability_study_ref.sh
  -> pi05_libero_online_ogpo_ref, collect.tasks/eval_tasks libero_90_31,
     td_weight_schedule 0.95/0.95, success_reward_bonus 90,
     exp_name stab_REF_libero_90_31

ARM=REF_libero_90_38 TASK=libero_90_38 GPU=1 DRY=1 bash scripts/stability_study_ref.sh
  -> same, task/exp_name swapped to libero_90_38 / stab_REF_libero_90_38
```
