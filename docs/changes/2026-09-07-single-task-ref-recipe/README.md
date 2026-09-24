# Single-task reference-aligned OGPO recipe

**Tier 1.** Shell-recipe change (`stability_study.sh`) plus two new files
(`stability_study_ref.sh`, `stability_study_ref_maxlab.sbatch`). No `src/`,
config, jit, or checkpoint change.

## Why

Maintainer request (2026-09-07): single-task OGPO runs on libero_90_31 and
libero_90_38 using the reference-aligned recipe added by commit `5b94510`
(`pi05_libero_online_ogpo_ref` — 10-head mean critic, 95%TD/5%MC target,
success oversampling, best-of-8 collection, terminal success bonus).

That config is only wired up for multi-task today
(`ogpo_multitask_4task_ref.sh`, a thin wrapper over `ogpo_multitask_4task.sh`
that sets the reference-alignment env knobs `ogpo_multitask_4task.sh` exposes
unconditionally). `scripts/stability_study.sh`, the single-task launcher, has
no such knobs. Pointing its existing `CONFIG_NAME` override at
`pi05_libero_online_ogpo_ref` alone is **not sufficient** — two of its fixed
CLI flags silently clobber the new config's tuned defaults regardless of
`CONFIG_NAME`:

- `--rl.critic.td_weight_schedule.init_value/end_value` is hardcoded to `1`
  (100% TD). The ref config's own default is `0.95`.
- `--collect.success_reward_bonus` is never emitted at all, so it stays at
  the global `CollectionConfig` default of `0.0` for *any* `CONFIG_NAME` —
  it is not baked into the ref config's own dataclass default either
  (`src/training/config.py:843-872` has no `collect=` override), exactly
  mirroring why `ogpo_multitask_4task_ref.sh` sets `SUCC_BONUS` explicitly.

Everything else the reference alignment touches — `num_qs`/`num_vs`,
`critic.reduction`, `critic_success_oversample`, `n_samples` (best-of-N),
`advantage_combination`, the three normalizer flags — **is** baked into
`pi05_libero_online_ogpo_ref`'s own dataclass default, and `stability_study.sh`
never touches any of those fields unconditionally (its `QS`/`CONS`/`NORM`/
`CLIP_SYM` knobs are all optional gates, default off/unset), so the config's
defaults flow through untouched. Verified directly against
`src/training/config.py:843-872` and `CriticTrainingConfig` (`:123-143`), not
assumed from the multitask recipe's shape.

## What changes

- `scripts/stability_study.sh`: add `TD_W` (default `1`, reproduces current
  behavior exactly) and `SUCC_BONUS` (default `0`, ditto) env knobs,
  unconditionally emitted — same contract as `ogpo_multitask_4task.sh:204-229`
  for these two fields only. No other knob is touched (see BLAST-RADIUS.md for
  why the rest don't need it).
- `scripts/stability_study_ref.sh` (new): thin wrapper setting
  `CONFIG_NAME=pi05_libero_online_ogpo_ref`, `TD_W=0.95`, `SUCC_BONUS=90`,
  delegating to `stability_study.sh` — the single-task analogue of
  `ogpo_multitask_4task_ref.sh`.
- `scripts/stability_study_ref_maxlab.sbatch` (new): maxlab wrapper for the
  above, mirroring `ogpo_multitask_4task_ref_maxlab.sbatch`'s resource request
  and exit-42 requeue handling, sized single-task (1 GPU).

See `BLAST-RADIUS.md`, `DIFF.md`, `VERIFICATION.md`.
