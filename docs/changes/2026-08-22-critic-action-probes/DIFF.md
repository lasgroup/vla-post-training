# DIFF

## Source changes (2 files, both additive and default-off)

### `src/rl/advantage_weighted_sft/advantage_weighted_sft_learner.py`

Opt-in observability hook for the best-of-N selection.

- `__init__`: `self._bon_record: list[dict] | None = None`.
- `sample_actions`, immediately after `best = group_actions[...]`: when
  `_bon_record` is not None, append `{indices, candidates, scores, best_idx}`
  for that task group.

Rationale: Tier C needs the candidate set and the scores the argmax was taken
over. The alternative was a ~130-line clone of the scoring block (normalisation
order, prefix mean-pooling, task slots, value distribution) inside the probe --
precisely this repo's signature bug class (Decisions log OQ-2). Default `None`
in every production path, so collection is bit-identical; no jit, no sharding,
no signature change.

**Inheritance sweep**: `sample_actions` is overridden by `BestofNLearner` (its
own copy, unaffected), `FilteredSFTLearner` (the base single-sample path,
unaffected) and `DSRLLearner` (unwired). `OGPOAgentLearner` and
`MPOWeightedSFTLearner` inherit the AWR implementation and therefore gain the
hook; both keep `_bon_record = None` unless a probe sets it.

### `scripts/ogpo_multitask_4task.sh`

`ENTRY="${ENTRY:-scripts/exp.py}"`, used in `RUN=(uv run "$ENTRY")` and its
`DRY=1` twin. Lets a probe reuse this recipe's env preamble **and** its config
flag block verbatim, so there is no second copy of the config to drift. Default
unchanged; consistent with the recipe's "every knob is an env var" convention.

## New files (all read-only analysis; nothing launches)

| file | what |
|---|---|
| `scripts/probe_noise_level_sweep.py` | Tier B battery across a `noise_level` ladder; prints the reference-equivalence table at startup |
| `scripts/probe_noise_level_sweep.sbatch` | preempt, 80GB+, 4h, arms `mt4_ref_s0` + `mt4_ref_nopg_s0`, ladder `0 .01 .02 .05 .07 .1 .2 .3 .5` |
| `scripts/probe_counterfactual_rollouts.py` | Tier C: determinism check / calibrate / full phases |
| `scripts/probe_counterfactual_rollouts.sbatch` | preempt, 80GB+, 6h, `PHASE` env var, drives the ref recipe via `ENTRY` |

### Tier C design decisions

- **Replay, not sim-state restore.** `set_init_state(get_sim_state())` *is* an
  exact MuJoCo round trip (`set_state_from_flattened` + `sim.forward()`, no
  settle steps) but both methods live on the raw LIBERO env, so through the
  vector env they bypass `Pi0ObservationWrapper` / `QueryFrequencyWrapper` and
  return an obs the policy cannot consume, with `TimeLimit` unrestored. Instead
  every env in a wave is driven to the probe state by identical seeds (a
  per-worker identical seed list; `LiberoWrapper.reset` draws its init state
  from a seeded rng) plus the trunk's own recorded chunks, through the real step
  path. Phase `determinism` asserts this rather than assuming it.
- **`calibrate` before `full`.** Delta = G(argmax) - mean_k G_k has per-state
  noise ~1.06*sigma_cont, so the required N is set by the continuation noise
  floor. `calibrate` executes the SAME candidate in all 8 envs and prints the N
  needed for a target SE. Running `full` without it is a guess at the runtime.
- **Probes grouped by task**: `LiberoWrapper.reset` rebuilds the env whenever
  the bddl file changes.
- **Continuation runs the plain policy** (the wave's `sample_actions` is the
  deployed best-of-N path; the candidate under test is forced only on the first
  chunk). Stated as a caveat, not silently.

## Divergence from plan

None -- there was no prior `PLAN.md` for this record; the work was scoped
directly from the Tier B findings.
