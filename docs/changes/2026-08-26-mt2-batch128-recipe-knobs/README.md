# 2026-08-26 — `TASKS` / `BATCH` recipe knobs + 2-GPU plumbing

**Tier 1.** Three shell recipes. No source, no config dataclass, no numerics.

## What

Enable a 2-task, batch-128, 2-GPU OGPO arm (`libero_90_44` + `libero_90_79`) to be
launched from the existing reference recipe without forking it:

1. `scripts/ogpo_multitask_4task.sh` — `TASKS` and `BATCH` become env-overridable,
   matching the script's existing one-env-var-per-knob design.
2. `scripts/ogpo_multitask_4task_ref_maxlab.sbatch` — `GPU` now defaults to the whole
   `CUDA_VISIBLE_DEVICES` list, not `0`.
3. `scripts/ogpo_ref_smoke_maxlab.sbatch` — same `GPU` fix, plus the memory sampler
   now reports the max-used GPU instead of only GPU 0.

## Why

**`TASKS`.** The train set was hardcoded at `:75`. A trailing `--collect.tasks` on the
CLI *would* work (tyro takes the last occurrence of a repeated flag — verified against
tyro directly), but `EVAL_TASKS` (`:84`), the `[mt4]` banner (`:214`) and
`PER_TASK_CRITIC`'s `${#TASKS[@]}` slot count all derive from the array, so a CLI
override leaves three downstream consumers reporting the 4-task default. The banner
lie is the dangerous one: it is the only task-set record in a run's log.

**`BATCH`.** `--batch_size 32` was a literal at `:279`. It is the dominant memory knob
— the actor jit expands it to `BATCH × rl.group_num_samples` SDE chains
(`update_actor.py:214`), so 32→128 takes 256→1024 chains through a 3B expert.

**`GPU` in the two sbatch wrappers.** Both hardcoded `GPU="${GPU:-0}"`, so a
`--gres=gpu:2` allocation would still have been pinned to one card and `FSDP=2` would
have failed the `jax.device_count() % num_fsdp_devices` check
(`openpi/src/openpi/training/sharding.py:18-21`). The non-ref wrapper
(`ogpo_multitask_4task_maxlab.sbatch:36`) already read `CUDA_VISIBLE_DEVICES`; these
two had drifted from it.

**Smoke memory sampler.** `head -1` reads GPU 0 only. Under FSDP both cards carry the
same order of memory, so a 2-GPU peak would have been under-reported — and the peak
is the entire output of that harness.

## Not in scope

No change to any default. Every knob added is `${VAR:-<the previous literal>}`.
