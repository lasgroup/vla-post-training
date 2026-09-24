# `stability_study.sh`: TASK knob, DRY mode, trailing-argument passthrough

**Tier 1.** Shell-recipe change plus one new sbatch wrapper. No `src/`,
config, jit, or checkpoint change.

## Why

Maintainer request (2026-09-07): run the candidate Q-spread probe
(`docs/changes/2026-09-07-candidate-q-spread-rollouts/`) on the single-task
checkpoint trained on task 38 alone. Of the two such checkpoints the
maintainer chose **`stab_NCB_libero_90_38`** — the frozen-backbone
stability-study arm (normalizer + symmetric clip 4.0 + 1000-step critic
burst, no conservative gating), job 10210760, completed 2026-08-24 at step
100000, under the sibling store root
`/data/group_data/maxlab/common_datasets/pchellap/vla_single_task/checkpoints/stability_study/pi05_libero_online_ogpo_sft/`.

That run was launched from the sibling `vla_single_task` checkout (commit
`c3dbcac`), whose `stability_study.sh` has a `TASK` env var
(`--collect.tasks "${TASK:-libero_90_44}"`, `:171-172`). This repo's copy
hardcodes `libero_90_44` and has no trailing-argument passthrough, so a probe
routed through it would roll out task 44 and could not set `rl.n_samples`
(`experiments/language_grounding/stage3_rollouts.py:24` already notes the
missing passthrough). Three additions, each defaulting to the old behaviour:

- `TASK` (default `libero_90_44`), mirroring the sibling recipe;
- `DRY=1` (print the command), mirroring `ogpo_multitask_4task.sh`;
- `"$@"` appended after the flag block, mirroring `ogpo_multitask_4task.sh`.

Plus `scripts/probe_candidate_q_spread_stab.sbatch`, the wrapper that reaches
these checkpoints the way `stage3_rollouts.sbatch` does (`STORE_ROOT`
override, `ARM=NCB_libero_90_38`, `ENTRY` swap, `--resume`) and appends
`--rl.n_samples 8`.

See `BLAST-RADIUS.md`, `DIFF.md`, `VERIFICATION.md`.
