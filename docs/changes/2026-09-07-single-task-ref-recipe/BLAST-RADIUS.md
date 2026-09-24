# Blast radius

## Files to touch

- `scripts/stability_study.sh` (edit) — add `TD_W`/`SUCC_BONUS` knobs.
- `scripts/stability_study_ref.sh` (new) — single-task ref wrapper.
- `scripts/stability_study_ref_maxlab.sbatch` (new) — maxlab launcher.

## Verified against source, not the docs

`src/training/config.py:843-872` (current `pi05_libero_online_ogpo_ref`
registration) and `:123-143` (`CriticTrainingConfig` field defaults) read
directly, not taken from `docs/code/` or the commit message. Confirms:

- `reduction` field default is `"min"` at the dataclass level; the ref config
  overrides to `"mean"` as *its own* default. `stability_study.sh` never emits
  `--rl.critic.reduction` (no such flag anywhere in the file), so this flows
  through untouched for both `pi05_libero_online_ogpo_sft` and
  `pi05_libero_online_ogpo_ref`. No knob needed.
- `num_qs`/`num_vs`, `n_samples`, `critic_success_oversample`,
  `advantage_combination`, `normalize_group_advantage`,
  `normalize_advantage_per_task`, `adv_clip_sym` are likewise only touched by
  `stability_study.sh` through optional, default-off gates (`QS`, `CONS`,
  `NORM`, `CLIP_SYM`) or not at all (`n_samples`,
  `critic_success_oversample`) — none of them are unconditionally-emitted
  fixed flags, so the `CONFIG_NAME`-selected config's own defaults reach the
  run in every case. Confirmed by reading the full flag block
  (`stability_study.sh:193-244` pre-change), not inferred.
- There is no `collect=CollectionConfig(success_reward_bonus=...)` override in
  the ref config's registration — `success_reward_bonus` stays `0.0` under
  `pi05_libero_online_ogpo_ref` exactly as under the baseline config unless a
  caller emits `--collect.success_reward_bonus` itself. This is also true of
  `ogpo_multitask_4task_ref.sh` (`SUCC_BONUS=90` is a recipe-level default,
  not a config-level one) — same mechanism, confirmed independently for the
  single-task side rather than assumed to match.
- `td_weight_schedule` **is** baked into the ref config's own default
  (`init_value=0.95, end_value=0.95`, `config.py:866-868`) — the only reason it
  needs a knob here is that `stability_study.sh`'s fixed flag block
  (pre-change) hardcodes `1`/`1` unconditionally, overriding it regardless of
  `CONFIG_NAME`.

## Duplication sweep (OQ-2: `stability_study.sh` ↔ `ogpo_multitask_4task.sh`)

This is the named clone family. This change does not touch
`ogpo_multitask_4task.sh` — it already has `TD_W`/`SUCC_BONUS` (and every
other reference-alignment knob) as unconditionally-emitted fields
(`:204-229`). This change brings `stability_study.sh` to parity for exactly
the two fields it was missing; the other reference-alignment fields are
deliberately *not* mirrored as new env knobs here (per the point above, they
need none — adding no-op knobs for them would be unrequested scope). The two
scripts remain structurally divergent (documented, accepted state).

## Other consumers of `stability_study.sh`, checked for collision

- `scripts/ws_bcbb_pipeline.sh` (`:50,:63`) — calls `stability_study.sh`
  without setting `CONFIG_NAME`/`TD_W`/`SUCC_BONUS`; new defaults (`TD_W=1`,
  `SUCC_BONUS=0`) reproduce current resolved behavior exactly, so this
  pipeline is unaffected. Its own `CONFIG_NAME=...unfrozen_backbone` override
  (`:47`) is for a different `ENTRY` script entirely, not `stability_study.sh`.
- `scripts/stability_wave1_bc0.sh` — same: no `TD_W`/`SUCC_BONUS`/`CONFIG_NAME`
  set, defaults hold.
- `scripts/probe_candidate_q_spread_stab.sbatch` (`:98`) — appends
  `--rl.n_samples "$BON_N"` as a trailing arg; does not touch
  `td_weight_schedule` or `success_reward_bonus`. Unaffected.
- No test under `tests/` references `stability_study.sh` (grepped, zero hits),
  so no existing pytest exercises its rendered command line.

## Inheritance sweep

N/A — no `src/` change. `OGPOSFTLearnerConfig`/`AdvantageWeightedSFTLearnerConfig`
are read-only here (verifying their current defaults), not edited.

## Gotchas checked

- `scripts.md` Gotchas section (existing) doesn't list anything about
  `stability_study.sh`'s `td_weight_schedule` hardcode or the
  `success_reward_bonus` no-op — this was previously undocumented, not a known
  trap being rediscovered.
- The `EXP_NAME="stab_${ARM}"` collision risk is avoided by construction:
  `ARM=REF_libero_90_31` / `ARM=REF_libero_90_38` land at
  `stab_REF_libero_90_31` / `stab_REF_libero_90_38` under this repo's own
  `run_store`, distinct from the sibling `vla_single_task` checkout's
  `stab_NCB_libero_90_38` (different config, different store root).

## Verification plan

- `DRY=1` rendered-command inspection for both the baseline path (no env vars
  set — confirm byte-identical `td_weight_schedule`/`success_reward_bonus`
  flags to the pre-change script) and the new `stability_study_ref.sh` path
  (confirm `CONFIG_NAME`, `TD_W=0.95`, `SUCC_BONUS=90` land correctly, for both
  `TASK=libero_90_31` and `TASK=libero_90_38`).
- `sbatch --test-only` (or equivalent dry submission check) against the new
  `.sbatch` file if a real submission isn't exercised.
- No `pytest tests/ogpo` re-run needed — no `src/` change; noted as
  out-of-scope in `VERIFICATION.md` rather than silently skipped.
