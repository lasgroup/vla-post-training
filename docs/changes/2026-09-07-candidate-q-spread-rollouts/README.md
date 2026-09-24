# Candidate Q-spread along rollouts (BoN on / off)

**Tier 1.** New read-only analysis script in the `scripts/probe_*` family plus
its sbatch wrapper. No algorithm, jit signature, checkpoint layout, or
config-dataclass change. Nothing in `src/` changes.

## Why

Maintainer request (2026-09-07): from a trained checkpoint, roll out N episodes
of a task. At every policy query sample M candidate action chunks, score each
with the critic, and record the **variance of the M reduced Q-scores** — how
diverse the policy's candidates look to the critic at that state. Two arms:

- `BoN=1` — execute the argmax-Q candidate (production best-of-N collection);
- `BoN=0` — execute candidate 0, an iid policy draw, i.e. the plain-policy
  control. The M candidates are still sampled and scored for the trace.

Output: one plot per rollout (chunk index vs. Q-variance, success/failure in
the title), one plot of the mean trace over the N rollouts with the alive
count underneath, and a JSON with every raw score so anything can be
re-plotted. All under a caller-specified directory.

## What already existed

- `AdvantageWeightedSFTLearner.sample_actions` with `rl.n_samples = M` IS
  "sample M candidates, score with the critic, pick argmax"
  (`src/rl/advantage_weighted_sft/advantage_weighted_sft_learner.py:412-654`).
- The opt-in `_bon_record` hook (added in
  `docs/changes/2026-08-22-critic-action-probes/`) already captures the M
  candidates (absolute robot-space, directly steppable) and their M
  *reduced* Q-scores per env per query (`:621-646`). The reduction honours
  `rl.critic.reduction`, so the spread is over the same scalar the run
  selected on.
- `scripts/probe_rollout_gifs.py` already restores a checkpoint read-only via
  the ENTRY-swap contract and drives N per-seed rollouts of
  `cfg.collect.eval_tasks` in waves of `collect.env_num`.

So the probe is the gif probe's rollout loop with the hook armed at every
query, a variance over axis 1 of the recorded `scores`, and matplotlib.

## Settled design choices (from the discussion)

1. y-axis = variance across the M **reduced** Q-scores (not per-head
   disagreement, not action-space dispersion). Raw scores are saved so the
   others can be derived later without a rerun.
2. Checkpoint and task come through the recipe's existing env vars
   (`ARM`/`SEED`/`CONFIG_NAME` → `checkpoint_dir`; `TASKS` → `collect.tasks`
   and `collect.eval_tasks`), or any `--flag` override on the CLI. The step is
   whatever the directory holds: the restore path picks the newest complete
   step (`src/training/runtime_state.py:79`) and openpi keeps one
   (`max_to_keep=1`).
3. Candidate sampling regime is production's: deterministic ODE from M
   different initial-noise draws, `noise_level=0`. M is `BON_N` → `--rl.n_samples`.
4. `BoN` is a probe env var (`QSPREAD_BON`), not a config-dataclass field —
   that would be Tier 2 and reach three learners.
5. x-axis = chunk index = the k-th policy query within that episode, from 0.
6. No overlay of all traces; a mean trace instead. Rollouts end at different
   chunk counts, so the mean at index k is over the rollouts still alive at k;
   the alive count is drawn in a panel beneath it (one axis per panel — never a
   second y-axis).

See `BLAST-RADIUS.md` for the change spec, `DIFF.md` for what was written,
`VERIFICATION.md` for what could and could not be checked.
