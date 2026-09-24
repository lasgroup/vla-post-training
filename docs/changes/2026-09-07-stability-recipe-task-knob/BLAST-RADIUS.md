# Change spec

## Files

- `scripts/stability_study.sh` — edited: `TASK` variable and its two flag
  uses; `task=$TASK` on the banner; `RUN=(uv run "$ENTRY")` with the `DRY=1`
  echo swap; `"$@"` after `"${EXTRA_FLAGS[@]}"`; header comment for the three.
- `scripts/probe_candidate_q_spread_stab.sbatch` — new.

## Callers of the recipe (verified by grep)

| Caller | How it invokes | Affected? |
|---|---|---|
| `scripts/ws_bcbb_pipeline.sh:50, :63` | `bash .../stability_study.sh`, no args, no `TASK` | No: `TASK` unset → `libero_90_44` as before; `"$@"` empty |
| `scripts/stability_wave1_bc0.sh:25` | `env GPU=… ARM=… "$@" nohup bash scripts/stability_study.sh` — the `"$@"` there are `env` assignments, not script args | No |
| `experiments/language_grounding/stage3_rollouts.sbatch:57`, `stage4_trained_instructions.sbatch:39` | `exec bash scripts/stability_study.sh`, no args, no `TASK` | No (they build their own env task lists) |
| `scripts/*.sbatch` stability wrappers | none reference the recipe | — |

No caller sets `TASK`, `DRY`, or passes positional arguments, so every
existing invocation renders the identical command. Verified two ways: the
implementer's render (pre-edit copy with `ENTRY=/bin/echo`, post-edit file
with `DRY=1`, `uv run scripts/exp.py` normalised to `/bin/echo`, three arm
sets, token-identical) and the verifier's sounder one (a bash function
shadowing `uv` that dumps argv, applied identically to both files, ten arm
sets including `ws_bcbb_pipeline` stages A/B, stage 3/4, `LORA=1 FSFT=1
WEIGHT_LOADER=… FC_INT=… QS=10`, `CKPT_MODE_FLAG`, `FRESH=1`; all identical,
95–120 tokens). See `VERIFICATION.md`.

## Divergence from the sibling recipe found the hard way

The sibling checkout's `stability_study.sh` (which produced the stab_* runs)
seeds LIBERO's config file when `$STORE_ROOT/libero/config.yaml` is absent
(`:140-141`); this repo's copy did not, and `ogpo_multitask_4task.sh:253-254`
does. Pointing this recipe at the sibling store therefore hit LIBERO's
interactive prompt (`libero/libero/__init__.py:101-104`, `input(...)` when
the config file is missing) and died on `EOFError` — job 10352129. The block
is now copied in. The stage-3 probe that used the same store root on
2026-08-31 (job 10274728) did not prompt, so the file has since gone missing
from that root or was never there and stage 3 imported LIBERO through a path
that found `~/.libero`; not resolved — the seeding makes it moot.

## Duplication sweep

The `DRY` swap and the `"$@"` line are copies of
`ogpo_multitask_4task.sh:255-258, :311`; `TASK` is a copy of the sibling
checkout's recipe. The two preambles are already a documented divergent pair
(OQ-2, `docs/code/scripts.md` gotchas); this adds three lines to that
divergence rather than unifying them.

## Inheritance sweep

None — shell only.

## Probe wrapper mechanism

Same as `probe_candidate_q_spread.sbatch` except: 1 GPU / `FSDP=1` (the
recipe hardcodes `--fsdp_devices 1`, which is what the checkpoint was saved
with, so the id-mapped restore works); `STORE_ROOT` pointed at the sibling
store (checkpoints and the openpi asset cache both live under it — the same
thing `stage3_rollouts.sbatch` does); `ARM`/`TASK`; the NCB arm's three
flags re-exported for CLI fidelity (inert at probe time — they only enter
`update()`); `--rl.n_samples 8` via the new passthrough, and `QSPREAD_PASSES=4`, so
**M = 32 candidates per state** (the probe's multi-pass knob, see the
candidate-q-spread record). `--mem=200G` against the checkpoint's own
training run at 138.6 GiB MaxRSS (job 10210760); the stage-3 probe on this
root sat exactly at its 128G cap (censored), and the "10274386" the first
draft cited was a stage-2 job at 43 GiB — verifier finding 1. Config resolves to `pi05_libero_online_ogpo_sft`: 2 Q heads,
`reduction=min`, `episode_steps_multiplier=1` → 400 env steps, 82 chunks.

Resume-manifest note: this checkpoint predates the 2026-08-27 resume
hardening, so OGPO's restore warns and degrades to an empty success buffer
and `adv_scale=min_scale` (sanctioned path, `best_practices.md` §7). Neither
is read by the probe.

## Verification

`bash -n` both files; before/after render equivalence for two existing arm
invocations; `DRY=1` render with the wrapper's env block confirming
`--exp_name stab_NCB_libero_90_38`, `--checkpoint_base_dir` under the sibling
root, `--collect.tasks libero_90_38`, `--collect.eval_tasks libero_90_38`,
`--rl.n_samples 8`, `--fsdp_devices 1`, `--resume`; then the GPU job itself
(the maintainer asked for this run explicitly).
