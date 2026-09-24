# Blast radius — verified against source, not the docs

## Files touched

| File | Change |
|---|---|
| `scripts/ogpo_multitask_4task.sh` | `TASKS` array env-overridable (`:75`); `BATCH` env knob replacing the `--batch_size 32` literal (`:279`); header env-var docs |
| `scripts/ogpo_multitask_4task_ref_maxlab.sbatch` | `GPU` default → `CUDA_VISIBLE_DEVICES` |
| `scripts/ogpo_ref_smoke_maxlab.sbatch` | same `GPU` fix; memlog samples max-used GPU |

## Consumers of `ogpo_multitask_4task.sh` (grepped, all 10)

`ogpo_multitask_4task_ref.sh` · `ogpo_multitask_4task_maxlab.sbatch` ·
`ogpo_ref_smoke_maxlab.sbatch` (via ref.sh) · `submit_mt4_data_arm.sh` ·
`submit_mt4_stability_arms.sh` · `probe_counterfactual_rollouts.sbatch` ·
`probe_noise_level_sweep.sbatch` · `probe_value_next_spread.sbatch` ·
`probe_policy_candidate_spread.sbatch` · `tests/ogpo/test_verifier_alignment.py`,
`tests/ogpo/test_per_task_critics_verifier.py`

All unaffected: both knobs default to the previous literal.

## Name-collision sweep

`grep` across `scripts/*.sh` and `scripts/*.sbatch` for `TASKS=` / `BATCH=`
assignments: the only hit is `ogpo_multitask_4task.sh:75` itself
(`HELDOUT_TASKS`/`EVAL_TASKS` excluded — distinct names, both still assigned
internally and not env-overridable). No caller exports either name, so nothing
leaks in through `test_verifier_alignment.py::_dry`, which inherits `os.environ`
(`:504`) — the same exposure every other knob in this script already has.

## Duplication sweep (CLAUDE.md clone families)

The relevant family is **`stability_study.sh` ↔ `ogpo_multitask_4task.sh`** (the
~60-line env preamble, already diverged). `stability_study.sh` is single-task:
its task is a literal `--collect.tasks libero_90_44` (`:136-137`) with no array,
and its `--batch_size 32` (`:169`) has no env knob. **Deliberately not
propagated** — it has no `TASKS` array to override, adding one would be a second
divergence, and no run needs it. Recorded here so the asymmetry is intentional
rather than an oversight.

`ogpo_multitask_4task_ref.sh` needed no change: it already forwards `"$@"` (`:80`)
and sets only env vars, so both new knobs pass through it untouched.

## Inheritance sweep

N/A — no Python touched. No learner, no `Agent` ABC, no jit signature, no RNG
split, no `donate_argnums`, no config dataclass. Nothing in this change can alter
a resolved `OnlineTrainConfig` at the defaults; that equivalence is what the
verification below pins.

## Gotchas checked

- `filtered_sft_learner.py:232` — `batch_size % jax.device_count() != 0` raises.
  128 % 2 = 0 ✓. Documented in the new `BATCH` comment so the next person raising
  it on an odd device count sees the constraint.
- `ogpo_multitask_4task.sh` per-task semantics (header `:17-24`) — `N_ROLLOUTS`,
  `INIT_ROLLOUTS`, `EVAL_ROLLOUTS` are **per task** and multiply by `len(TASKS)`.
  A `TASKS` override silently rescales every episode total. Called out in the new
  `TASKS` comment and in the header docs, since the script cannot rescale them
  itself without overriding a deliberate user setting.
