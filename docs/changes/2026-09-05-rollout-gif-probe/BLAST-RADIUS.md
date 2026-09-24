# Change spec

## Files added

- `scripts/probe_rollout_gifs.py` — restores the checkpoint via the same tyro
  CLI as `scripts/exp.py` (ENTRY-swap contract: the recipe owns the env
  preamble and the config flag block, so it is not cloned a second time to
  drift; the knobs that matter for restore — `CONFIG_NAME`, `ARM`, `SEED`,
  `TASKS`, `EP_MULT`, `FSDP` — render identically to the training run's, a
  few inert-at-eval ones don't, per `probe_counterfactual_rollouts.sbatch`'s
  own convention). Requires `--resume` (hard guard — a missing `--resume`
  would construct the learner with `overwrite=True` and wipe the checkpoint;
  same reasoning as Tier C and stage 4) **and** requires `agent._resuming`
  (second guard, added post-verification — `--resume` against a missing/empty
  checkpoint dir silently downgrades to a fresh start in openpi, which would
  otherwise roll out untrained base weights mislabeled as the trained
  checkpoint). Rolls out `ROLLOUT_EPISODES` episodes per `cfg.collect.eval_tasks`
  task, each episode in a wave from its own seed, writes one GIF per episode
  plus a JSON summary (success, step count, GIF path) per task.
- `scripts/probe_rollout_gifs.sbatch` — preempt, single GPU (`FSDP=1`),
  `ENTRY=scripts/probe_rollout_gifs.py` through `scripts/ogpo_multitask_4task_ref.sh`
  with `ARM=mt2_b128_ep2 TASKS="libero_90_38 libero_90_82" EP_MULT=2`, mirroring
  the `SRC` mount-hang guard from `probe_counterfactual_rollouts.sbatch`.

## Expected behavior after

`sbatch scripts/probe_rollout_gifs.sbatch` restores `mt4_mt2_b128_ep2_s0`'s
latest checkpoint, prints the restored step (so "final checkpoint" is
verified against what's actually on disk, not assumed), and writes
`<out>/rollouts/<task>/ep<NN>_seed<seed>_<succ|fail>.gif` for each of 8
(default) episodes per task, plus `<out>/rollout_results.json`.

## Deliberate divergence from the training config

The run trained on `FSDP=2` / 2 GPUs (`--gres=gpu:2`); this probe restores
with `FSDP=1` / 1 GPU (`--gres=gpu:1`) for scheduling convenience on a preempt
job. Orbax restores by the checkpoint's global array shape and the CURRENT
process's target sharding, not by the saving-time mesh, so this does not
change which weights are loaded — only how they're sharded across devices.
`BATCH=128` is still passed (harmless at eval — `batch_size` only sizes
policy-update batches, never taken here) for CLI-line fidelity with the
original launch; `TASKS`/`EP_MULT` are the two knobs that actually matter and
both are set to match the training run exactly.

`--mem=300G` matches the training run's own allocation rather than this
probe's actual footprint — `OGPOAgentLearner(cfg)` unconditionally restores
the full online replay buffer and success buffer on `--resume` regardless of
whether a read-only probe ever touches them, and by the run's final step
those hold substantially more than the 83k/48k rows measured at step 40k
(verification finding 7, not confirmable without running it — the training
run used 300G at this batch size so this reuses that number rather than
guessing a smaller one).

## Duplication sweep

`rollout`/`policy_chunk`/`reset_wave` in the new script duplicate the
same-named helpers in `probe_counterfactual_rollouts.py` (return/candidate
tracking stripped, frame capture added); `write_gif` duplicates
`stage4_trained_instructions.write_gif` verbatim. Neither source script
exposes these as an importable utility module, and each has already diverged
from the others (this repo's documented duplication-by-copy convention,
CLAUDE.md Decisions log OQ-2) — this adds a fourth divergent copy rather than
extracting a shared module now, consistent with the existing probe family.

## Inheritance sweep

None. The script drives `OGPOAgentLearner` through its existing public
`sample_actions` / `start_data_collection` / `end_data_collection` surface —
the same surface Tier C and stage 4 already use. No learner method is
overridden or touched, and nothing calls `agent.update()` or any save path.

## Blast radius

Two new files under `scripts/`. Nothing in `src/` changes. No config
dataclass field added — all probe-specific knobs are env vars
(`ROLLOUT_OUT_DIR`, `ROLLOUT_EPISODES`, `ROLLOUT_SEED`, `ROLLOUT_GIF_STRIDE`),
matching the `experiments/language_grounding/stage4_trained_instructions.py`
convention rather than adding a `--probe.*` tyro field.

## How it will be verified

GPU/simulator execution cannot run from the login node (no `/data` mount, no
GPU — see `VERIFICATION.md`). Verified instead: `py_compile` syntax check, and
`DRY=1 bash scripts/ogpo_multitask_4task_ref.sh` with the sbatch's exact
env-var block, confirming the rendered command carries `ENTRY` swapped to the
new script, `--collect.tasks libero_90_38 libero_90_82`,
`--collect.episode_steps_multiplier 2`, and `--resume`.
