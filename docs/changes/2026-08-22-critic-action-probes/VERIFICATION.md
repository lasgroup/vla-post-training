# VERIFICATION

## What was verified

| check | result |
|---|---|
| `python -c compile` on all three new/changed python files | **pass** |
| import `AdvantageWeightedSFTLearner` + `OGPOAgentLearner` after the hook | **pass** |
| `_bon_record` present in both `__init__` and `sample_actions` | **pass** |
| import `probe_noise_level_sweep.py` (loads Tier A + Tier B by path) | **pass** |
| import `probe_counterfactual_rollouts.py` | **pass** |
| `bash -n` on the changed recipe and both new sbatch files | **pass** |
| `--probe.*` / tyro argv split (`parse_known_args`) | **pass** — probe args stripped, `['pi05_...','--resume','--collect.tasks','libero_90_79']` reaches tyro |
| reference-noise equivalence recomputed independently of the script | **pass** — 0.06972 both ways |
| `EXP_NAME` the recipe builds for `ARM=ref SEED=0` == the checkpoint dir name | **pass** — `mt4_ref_s0` |
| `SubprocVectorEnv.step(action, id=[...])` accepts a subset and returns aligned arrays | **pass** — read-verified at `src/envs/venv.py:773-840` |
| `env.seed([s]*G)` sets a per-worker identical seed | **pass** — `venv.py:870-892` takes a list verbatim |

## What could NOT be verified, and why

- **`pytest tests/ogpo` did not run.** It aborts on the login node inside
  `jax/_src/compiler.py:324 backend_compile` with `Fatal Python error: Aborted`,
  on `tests/ogpo/test_group_dedup.py` alone as well as the whole directory. This
  is the documented login-node kill (CLAUDE.md: "the login node kills it"); the
  abort is in XLA compilation, not in any changed code. **The suite needs
  `sbatch scripts/run_ogpo_tests.sbatch` and has not been run against this diff.**
- **No end-to-end execution of either probe.** Both need `/data/group_data/maxlab`
  (not mounted on the login node), real pi0.5 weights, and -- for Tier C -- a GPU
  and the LIBERO simulator. Nothing here has been run.
- **Whether orbax `CheckpointManager.__init__` garbage-collects under
  `max_to_keep=1`.** Not confirmed from source. Two mitigations instead of an
  assumption: the probe records `all_steps()` into its JSON, and the sbatch runs
  against a `cp -al` hardlink shadow so nothing it does can reach the real run.
- **Whether the wave really is bit-deterministic.** This is the load-bearing
  assumption of Tier C's replay-to-probe-state. It is not assumed: phase
  `determinism` measures it and the result is recorded, with a loud warning if
  the spread is non-zero.

## Hazards found and closed during implementation

1. **`overwrite=not resume` wipes the run.** `initialize_checkpoint_dir` calls
   `checkpoint_dir.rmtree()` when `overwrite=True`
   (`openpi/src/openpi/training/checkpoints.py:26-29`), and the learner passes
   `overwrite=not self._config.resume`. A Tier C invocation missing `--resume`
   would delete a finished 100k-step run before reading anything.
   *Closed twice*: the probe raises before constructing the learner if
   `cfg.resume` is False, and the sbatch points `CKPT_BASE_DIR` at a hardlink
   shadow.
2. **`set_init_state`/`get_sim_state` look like the obvious way to return to a
   probe state and are a trap.** They live on the raw LIBERO env, so through the
   vector env they bypass `Pi0ObservationWrapper` and `QueryFrequencyWrapper`,
   return an obs the policy cannot consume, and leave `TimeLimit` unrestored.
   Replay is used instead. (The MuJoCo round trip itself is exact --
   `set_state_from_flattened` + `sim.forward()`, no settle steps -- so this is a
   plumbing trap, not a physics one. Worth a `docs/code/` gotcha.)
3. **`min_noise_std`/`max_noise_std` in the reference recipes are inert.**
   Reading them as the reference's noise level (0.01) is wrong by 5x; the tapered
   path uses `constant_noise_std` (0.05).

## Known limitation of the noise sweep

It changes only the sampler, on states the *current* policy visited. Actually
raising collection noise would also shift the state distribution, which a static
probe cannot see. The sweep bounds the dispersion question; it does not predict
the run.

## Run log and current status (2026-08-23)

**Noise sweep: COMPLETE.** Job 10194496, 11m06s, babel-z5-28. Result in
`/home/pchellap/logs/noise_sweep_10194496.json`. Verdict: `noise_level` is not a
diversity knob. Over a 25x sweep (0 -> 0.5) on `mt4_ref_s0`, candidate spread
rises only 1.31x (eps 0.0761 -> 0.1000) and rho_noise goes 0.194 -> 0.217, then
saturates (sigma_within is *lower* at 0.5 than at 0.3). Mean Q is flat to four
digits throughout. Meanwhile `dist_from_ode_action` reaches 1.06, so samples do
move -- they just move together, because the SDE is marginal-preserving. This
refutes the post-Tier-B recommendation that raising collection noise was the
cheap lever; a real temperature knob would have to scale the initial x_1 draw,
which has no flag today.

**Tier C: NOT YET RUN.** Five submissions, five infrastructure/plumbing
failures, no measurement. All five causes are now closed in the scripts:

| job | elapsed | cause | fix |
|---|---|---|---|
| 10193108 | 75 min | hung `/data` mount; guard `ls` blocked in D state, GPU idle | `timeout 120 ls` in all probe sbatch files |
| 10193109 | 2 s | `cp -al` across `/data/user_data` -> `/data/group_data` (different devices) | shadow moved, then removed entirely |
| 10193599 | 110 min | `cp -al` on the right device but still hardlinking thousands of ocdbt files over NFS | shadow removed; verified unnecessary (orbax GC runs only in `save()`) |
| 10194497 | 13 s | recipe hardcodes `--overwrite`, probe adds `--resume`; mutually exclusive | `CKPT_MODE_FLAG` env var, one flag not two |
| 10194608 | 5m54s | `sample_actions` outside a collection window: `_train_state.ema_params` is None except between `start_data_collection` and `end_data_collection` | probe opens/closes the window; `collect.env_num == G` asserted |

10194658 carried all five fixes and was **cancelled by the maintainer before it
produced output** -- an NFS mount problem on nas6, unrelated to the probe.

**To resume once nas6 is healthy:**

```bash
PHASE=calibrate sbatch scripts/probe_counterfactual_rollouts.sbatch
```

Read `sigma_cont_mean` and `states_needed_for_target_se` out of the resulting
JSON before running `PHASE=full` -- that number sets the runtime.
