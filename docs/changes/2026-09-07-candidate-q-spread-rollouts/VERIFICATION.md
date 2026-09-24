# Verification

Independent agent: Opus, xhigh reasoning effort, fresh context (change spec +
diff only). Its report is reproduced verbatim below, then the outcome of each
finding, then two things the implementing session found afterwards (the
restore-topology failure and the GPU runs).

---

## Verifier report (verbatim)

# VERIFICATION — `2026-09-07-candidate-q-spread-rollouts`

## Execution disclaimer

Verified on the login node: no GPU, no `/data` mount, no π0.5 weights, no LIBERO/MuJoCo. Nothing was launched — no `sbatch`, no `uv run scripts/exp.py`, no un-`DRY` recipe invocation. Nothing under `src/` was edited; the three new files and the record directory were not edited. One new file was written: `tests/ogpo/test_candidate_q_spread_verifier.py` (21 tests).

**Not run:** the full `pytest tests/ogpo` (the PaliGemma-backed legs get killed on this node). Only `--collect-only` was run over the folder.

The strongest evidence below comes from a CPU smoke that drives the **real** `AdvantageWeightedSFTLearner.sample_actions` body on a stub `self` (`object.__new__` + attribute injection, the pattern at `tests/ogpo/test_per_task_critics_verifier2.py:333`), with the real `pi05_libero_online_ogpo_ref` config and real grouping / tiling / reshape / argmax / scatter / record code — fakes only for the pi0.5 forward, the transforms and the two `nnx` models. The implementer's tests exercise the probe against a hand-written *imitation* of the hook contract, which cannot catch drift in the contract itself.

## Verified correct

| Claim | Evidence |
|---|---|
| Hook record fields, shapes, dtypes: `indices` `list[int]`, `candidates` `(g,M,H,act)` f32, `scores` `(g,M)` f32, `best_idx` `(g,)` **int32** | `advantage_weighted_sft_learner.py:621-647`; my `test_real_hook_record_shapes_and_dtypes` |
| `candidates` are absolute robot-space (what `env.step` consumes) | `_sample_action` is called *without* `return_prefix_rep` (`:481`), so it goes through `infer_with_model` → `_output_transform` (Unnormalize→AbsoluteActions), `filtered_sft_learner.py:629-637`; the code comment at `:574-580` states it |
| `candidates[arange, best_idx]` is **bit-exactly** the returned chunk | `:619` `best = group_actions[arange, best_idx]`, `:650` `all_best_actions[indices] = np.asarray(best, np.float32)`, `:625` record stores `np.asarray(group_actions, np.float32)` — same f32 buffer, cast-then-index ≡ index-then-cast. Confirmed end-to-end by `test_real_returned_chunk_is_bit_exactly_the_gathered_argmax_candidate` |
| Prompt groups can be **non-contiguous**, so gather-by-`indices` is required | `:449-451`; my fixture produces groups `[[0,2],[1]]` |
| Candidate 0 is the first of env *e*'s M tiled noise draws | `np.repeat(v, n_samples, axis=0)` at `:471` + reshape `(g, M, …)` at `:616`; noise is one `jax.random.normal(rng, (g*M, H, D))` draw, `filtered_sft_learner.py:621-623`. Pinned by `test_real_candidate_zero_is_the_first_tiled_noise_draw_for_that_env` |
| Both single-sample fallback conditions leave the hook **empty**, and `gather_records` then raises naming `n_samples`/`inference_start_step` | `:419-421`; `test_real_single_sample_path_records_nothing_and_the_probe_says_why` (both parametrizations) |
| The probe's two guards make that path unreachable: `rl.n_samples>=2` (`probe:336`) and `training_steps>=inference_start_step` (`probe:359`) read the same `cfg`/`self` the method reads, and `training_steps` cannot move (no `update()`) | grep of the probe for `update()` / `save_*` / `add_data` / `save_episode` / `insert(`: none |
| `store_prefix_rep` tuple return is handled | `:657`; recipe emits `--collect.store_prefix_rep` unconditionally (`ogpo_multitask_4task.sh:304`), so the tuple path is the **live** one. `policy_chunk:178` unwraps. `test_real_tuple_return_under_store_prefix_rep_is_unwrapped_by_policy_chunk` |
| Scores honour `rl.critic.reduction`, not a hardcoded `min` | `:606-611`; `test_real_scores_are_the_configured_reduction_not_a_hardcoded_one` distinguishes mean vs min |
| `OGPOAgentLearner` overrides **neither** `sample_actions` nor `start_/end_data_collection` | `grep "    def "` over `ogpo_learner.py`: only `__init__`, `save_episode`, `save_/_restore_extra_resume_state`, `_refresh_update_functions`, `_burst_critic_update_fn`, `critic_digestion_burst`, `update`, `_balanced_success_ordinals`, `_online_batch_to_sft_batch`. Only `best_of_n_learner.py:350` overrides `sample_actions`, and that learner is not used |
| `start_data_collection`'s `assert ema_params is None` holds after a resume | `filtered_sft_learner.py:351` sets `ema_params=None` **unconditionally** after the restore block |
| `end_data_collection()` with no `step` skips the per-task-critic registry check | `:999` `if self._task_registry is not None and step is not None` |
| `(live, replan)` term/trunc layout, padded after an early end | `wrappers.py:168-219` (`QueryFrequencyWrapper.step` + `step_response`); the probe's inner `break`-on-`done[e]` counts exactly to the terminating sub-step, matching `collect.py:64-74` |
| `env.step(act, id=...)`, `env.seed(list)` signatures | `venv.py:773-776`, `:869-890` |
| `max_chunks` formula matches the env's own TimeLimit derivation | `libero.py:88` (`"_".join(task.split("_")[:-1])`), `:102` (`get_max_steps_libero(suite) * episode_steps_multiplier`); `episode_steps_multiplier` is `int` (`config.py:426`, validated `>=1` at `:496`) |
| `--resume` guard is real: `overwrite = not resume` → `rmtree` | `filtered_sft_learner.py:270-277` → `openpi/…/checkpoints.py:26-28` |
| `_resuming` guard is real: openpi silently downgrades on an empty dir | `openpi/…/checkpoints.py:58-59` |
| No `collect.tasks` validation on the OGPO resume path → rolling out one task from a 2-task shared-critic checkpoint is safe | `ogpo_learner.py:225-315` (`_restore_extra_resume_state`) validates only the *presence* of success-buffer state and `adv_scale`; `_success_task_ranges` are prompt-keyed and only read inside `update()`. Per-task critics are the exception, and the registry **is** persisted (`advantage_weighted_sft_learner.py:260-264`), so the sbatch's slot-count caveat is sufficient |
| The probe writes nothing to the replay buffer, no checkpoint, no `update()` | grep above; `agent.` call sites are `sample_actions`, `_bon_record`, `_resuming`, `training_steps`, `_checkpoint_manager.all_steps()`, `start_/end_data_collection` |
| `bash -n scripts/probe_candidate_q_spread.sbatch` | OK |
| `DRY=1` render with the sbatch's exact env block (`CKPT_BASE_DIR`/`STORE_ROOT` → scratch, `<STORE_ROOT>/libero/config.yaml` pre-created, `timeout 120`) | `ENTRY=scripts/probe_candidate_q_spread.py`; `--resume` × **1**, `--overwrite` × **0**; `--rl.n_samples 8`; `--collect.tasks libero_90_38 --collect.eval_tasks libero_90_38`; `--collect.episode_steps_multiplier 2`; `--rl.critic.inference_start_step 1`; `--rl.critic.reduction mean`; `--rl.critic.num_tasks None`; `--collect.env_num 8`; `--exp_name mt4_mt2_b128_ep2_s0` |
| sbatch mount-guard path matches the config's `checkpoint_dir` | `openpi/…/config.py:549` = `base/name/exp_name`; sbatch `SRC="$REAL_BASE/$CONFIG_NAME/$EXP_NAME"`, and both files compute `EXP_NAME="mt4_${ARM}_s${SEED}"` identically |
| `diff` vs `probe_rollout_gifs.sbatch` matches DIFF.md's list exactly (job name, `bon` in OUT_DIR, `CONFIG_NAME` exported + used in the guard, `TASKS`, `BON_N`, the three `QSPREAD_*`); nothing else changed | `diff -u` |
| PNG paths cannot collide across waves | `ep_idx = w*G + e`, strictly increasing (`probe:423, 428`); the two arms are separated by `bon${QSPREAD_BON}` in the sbatch's default OUT_DIR |
| JSON is serializable; both plots render real PNGs incl. a length-1 trace | `test_rollout_outputs_dump_to_json_the_way_main_writes_them`, `test_plots_are_real_pngs_including_a_single_point_trace` (magic bytes, not just non-zero size) |
| Step 4 (docs) **was** done | `docs/code/scripts.md:345-372, 449-452` and `docs/code/tests.md:49` already carry the probe. (`docs/code/` is gitignored — `.gitignore:25` — so it never shows in `git status`.) |

## Findings

**1. [CONFIRMED] Low — `rollout` leaves `_bon_record` armed when `sample_actions` raises.** `scripts/probe_candidate_q_spread.py:201-204` arms the hook with no `try/finally`. Scenario: an OOM or NaN-guard raise inside the critic forward on query *k*; `agent._bon_record` stays a live list. The process dies anyway (nothing catches), but any second `rollout` on that agent in the same process reports the *wrong* cause — `RuntimeError: _bon_record is already armed` — and any further `sample_actions` keeps appending `(G,M,H,32)` candidate arrays into the stale list. Fix: `try/finally` or clear in an `except`. Test: `test_rollout_leaves_the_hook_armed_when_sample_actions_raises`.

**2. [CONFIRMED] Low — a NaN in the returned chunk is reported as a record misalignment.** `:210-214`. `np.array_equal` is `False` whenever either operand holds NaN. A policy emitting a NaN action, with a perfectly aligned record, aborts with *"the per-group record is misaligned with the env order"* — pointing the reader at the hook rather than at the NaN. Fix: `equal_nan=True` plus a separate `np.isfinite(best).all()` check with its own message. Test: `test_rollout_misreports_a_nan_chunk_as_a_misaligned_record`.

**3. [CONFIRMED] Low (spec, not code) — the `env_num` justification is not a real constraint.** BLAST-RADIUS.md ("`env_num` must be `cfg.collect.env_num` because `start_data_collection` sizes `_episode_storage` from it (`filtered_sft_learner.py:987`)"), repeated verbatim in `docs/code/scripts.md:363-367`. The probe never calls `save_episode`/`add_data`, so `_episode_storage` is never indexed — `start_data_collection` (`:985-991`) and `end_data_collection` (`:1010-1012`) only rebuild the list. Nothing on the probe's path compares `collect.env_num` with the env's actual `env_num`. The choice is harmless (both are 8 here); the stated reason is wrong and is now in the reference docs.

**4. [CONFIRMED] Low (spec) — the `_bon_record` field list is incomplete.** BLAST-RADIUS.md lists four fields. The real record (`advantage_weighted_sft_learner.py:621-647`) also carries `state` and `prefix`, and `task_index` under per-task critics. `gather_records` reads only the four, so no bug today. Test: `test_real_hook_record_carries_more_than_the_four_documented_fields`.

**5. [CONFIRMED] Low — four `file:line` cites in the new files point at the wrong lines.**
- BLAST-RADIUS: "`candidates` … in absolute robot-space (`:582-583`)" → `:582-583` is the droid-layout branch; the evidence is `:574-580` + `:616-620`.
- BLAST-RADIUS: "Candidate 0 is one iid draw from the same initial-noise batch (`filtered_sft_learner.py:534-536`)" → `:534-536` is `_make_buffer_dummy_data`; the noise draw is `:621-623`.
- `probe_candidate_q_spread.py:379-381`: "ema_params … attached only inside a collection window (`filtered_sft_learner.py:329, :845-851`)" → `:329` is a bare `)`, `:845-851` is prefix-embedding code in `_save_episode_in_buffer`. Real cites: `:607-612` and `:989-991`. **Inherited verbatim from `probe_rollout_gifs.py:167-169`.**
- `probe_candidate_q_spread.py:206-208`: "AWR:612-616" for "the returned chunk IS `candidates[best_idx]`" → `:619` and `:650`. Also the docstring's `:412-654` → the method runs `412-657`.

**6. [CONFIRMED] Low — `flush()` is a non-atomic truncating write.** `:404` `summary_path.write_text(...)` truncates first. A preemption landing inside that window destroys **all** previously flushed waves, not just the last — the record's "a preempted job leaves partial results" is true almost always and catastrophic occasionally. Fix: write `<path>.tmp` + `os.replace`. (`probe_rollout_gifs.py:185` has the same defect.)

**7. [PLAUSIBLE] Low — the per-wave log mixes recorded and unrecorded episodes.** `:451-457`: `wave_var` concatenates all `G` envs, but only `min(G, episodes - w*G)` are recorded; `wave_SR` is over `G` while `cum_SR` is over the recorded set. With `QSPREAD_EPISODES < collect.env_num` the logged `q_var[min/median/max]` describes more episodes than the JSON and the plots. Cosmetic, but the log line is the first thing read.

**8. [PLAUSIBLE] Informational — module-import side effects run at pytest collection.** `:68` `mp.set_start_method("spawn", force=True)` and `:71` `sys.path.insert(0, _ROOT)` execute whenever either test file `exec_module`s the script. Setting the process-wide multiprocessing start method from a test import is global cross-test coupling. Harmless today (392 tests collect, both files pass), but it is a consequence of the "import the script in tests" design that the DIFF does not mention.

**9. [PLAUSIBLE] Informational — every env is queried on every chunk, including finished ones.** `:202` runs `sample_actions` over all `G` envs while `env.step` runs only `live`. At `G=8, M=8` that is 64 ODE chains + 64 critic evals per chunk regardless of how many envs are alive, so a wave costs the *longest* episode at full batch: with `EP_MULT=2`, `max_chunks = 400*2//5 + 2 = 162`. Inherited from the gif probe, but it dominates the 4 h wall-clock budget here because the BoN=8 forward is ~8× a plain query — budget against the slowest env, not the mean.

**10. [CONFIRMED] Informational — DIFF.md's "Divergence from PLAN" contradicts itself:** *"Implementation matches BLAST-RADIUS.md except the script name: `probe_candidate_q_spread.py` (the record said `probe_candidate_q_spread.py` too…)"* — there is no divergence.

**11. [CONFIRMED] Informational — "read-only" is scoped to checkpoint *contents*, not the directory.** `initialize_checkpoint_dir` does `checkpoint_dir.mkdir(parents=True, exist_ok=True)` (`openpi/…/checkpoints.py:39`) and the orbax `CheckpointManager` writes root metadata on construction. Consequence: a typo in `ARM` creates a stray empty run directory under the checkpoint root before the `_resuming` guard fires. No deletion risk — orbax's `cleanup_tmp_directories` defaults to `False` and openpi does not enable it (`checkpoint_manager.py:337, 757`), and `max_to_keep` pruning happens only on `save()`.

## pytest output

```
$ .venv/bin/python -m pytest tests/ogpo/test_candidate_q_spread.py tests/ogpo/test_candidate_q_spread_verifier.py -q
33 passed, 3 warnings in 11.32s
```
(12 implementer tests + 21 verifier tests; the 3 warnings are the pre-existing protobuf/TFP `DeprecationWarning`s emitted by the jax import chain.)

```
$ .venv/bin/python -m pytest tests/ogpo --collect-only -q | tail -2
-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
392 tests collected in 8.17s
```

Also run: `.venv/bin/python -m py_compile scripts/probe_candidate_q_spread.py` → OK; `bash -n scripts/probe_candidate_q_spread.sbatch` → OK.

## What remains unverified

- **Anything requiring a GPU, a simulator or real π0.5 weights.** The critic forward, the EMA compose, the LIBERO env, the orbax restore, FSDP=2→1 sharding, `_process_obs_for_pi0`'s real normalize→pad order, and the actual Q-variance numbers were all faked. The hook *contract* is pinned; the *values* are not.
- **The checkpoint itself.** `/data` is not mounted here, so `ARM=mt2_b128_ep2` / `CONFIG_NAME=pi05_libero_online_ogpo_ref` was not confirmed to exist, nor was its restored step, nor whether `--rl.use_success_buffer` state is present in its manifest (a manifest written before 2026-08-27 warns and degrades; one written after with the buffer *off* would **raise** at `ogpo_learner.py:257`).
- **Wall clock and memory.** The `--mem=300G` / 4 h envelope is asserted by analogy with the gif probe, not measured. Note the probe restores the full 500k-capacity replay buffer plus the success buffer purely as a side effect of `--resume`.
- **Concurrency.** Opening a writable orbax `CheckpointManager` on a *live* training run's directory was not exercised. No deletion path was found, but do not point this at an in-flight job.
- **Multi-suite `TASKS`.** With more than one task, `make_env_libero` derives the `TimeLimit` from `eval_tasks[0]`'s suite only (`libero.py:88-102`) and `LiberoWrapper.reset` hardcodes `benchmark_dict["libero_90"]` (`libero.py:32`), so a non-`libero_90` task is broken at the env level regardless of this probe. Pre-existing; the default `TASKS=libero_90_38` avoids it.
- **`PER_TASK_CRITIC=1` end to end.** The registry is persisted and restored, and the slot-count mismatch fails loudly in orbax, so the sbatch's caveat looks sufficient — but this was read, not run.

---

## Outcomes (implementing session)

| # | Outcome |
|---|---|
| 1 | **Fixed.** `rollout` arms the hook under `try/finally`; the exception propagates unchanged. Verifier test rewritten as `test_rollout_disarms_the_hook_when_sample_actions_raises` (second call dies on the real cause, hook disarmed both times). |
| 2 | **Fixed.** Separate `np.isfinite(best).all()` check before the alignment check, message names the query and the env rows. Verifier test rewritten as `test_rollout_reports_a_nan_chunk_as_non_finite_not_misaligned`. |
| 3 | **Fixed (docs).** BLAST-RADIUS.md, the script comment and `docs/code/scripts.md` now say "convention, not a constraint", with the reason. |
| 4 | **Fixed (docs).** BLAST-RADIUS.md lists `state`, `prefix`, `task_index`. |
| 5 | **Fixed.** All four cites corrected in the script and BLAST-RADIUS.md; docstring range `412-657`. The gif probe's inherited wrong cite (`probe_rollout_gifs.py:167-169`) is left in place — outside this change. |
| 6 | **Fixed.** `flush()` writes `<path>.tmp` then `os.replace`. `probe_rollout_gifs.py:185` has the same defect; left, flagged. |
| 7 | **Fixed.** Wave log restricted to the recorded envs and prints `recorded=n/G`. |
| 8 | **Accepted, documented** in DIFF.md. Same side effects as `exp.py`; no test in `tests/ogpo` uses multiprocessing. |
| 9 | **Accepted, documented** in DIFF.md. Inherited; a wave costs the longest episode at the full `G×M` batch. |
| 10 | **Fixed (docs).** DIFF.md rewritten. |
| 11 | **Noted.** No change; the `ARM` typo case creates an empty directory and then raises on `_resuming`. |

After the fixes:

```
$ .venv/bin/python -m pytest tests/ogpo/test_candidate_q_spread.py tests/ogpo/test_candidate_q_spread_verifier.py -q
33 passed, 3 warnings in 11.51s
```

## Found after the verifier: the restore topology

The verifier listed "FSDP=2→1 sharding" as unverified. Checking the gif probe's
only run (job 10324589, 2026-09-05, `/home/pchellap/logs/rollout_gif_10324589.out`)
showed it **died in `restore_state`** on 1 GPU:

```
[E] The available devices are different from the devices used to save the
    checkpoint.  Please restore checkpoint by passing new shardings for target
    devices. Original=[[DeviceMetadata(id=0), DeviceMetadata(id=1)]] ...
ValueError: sharding passed to deserialization should be specified, concrete
    and an instance of `jax.sharding.Sharding`. Got None
```

Cause, verified against source: `restore_state` hands orbax the
`jax.eval_shape` train state with no shardings
(`openpi/src/openpi/training/checkpoints.py:91-105`; `init_train_state`
returns `state_sharding` separately and `filtered_sft_learner.py:316-324`
never forwards it), so orbax reuses the saved sharding and maps its device
ids onto `jax.devices()` by id (`orbax/checkpoint/_src/metadata/sharding.py:119-131`).
Every successful restore in `/home/pchellap/logs` is an FSDP=2 training
resume; no FSDP=1 restore of an FSDP=2 checkpoint has ever succeeded. Device
kind is not compared.

Consequence for this change: `probe_candidate_q_spread.sbatch` now requests
`--gres=gpu:2` with `GPU=0,1 FSDP=2`, matching the checkpoint. `DRY=1`
re-render confirms `--fsdp_devices 2` with everything else unchanged. The
proper fix (annotate the restore target with `state_sharding`) is a sharding
contract change on every learner's resume path — Tier 2, proposed to the
maintainer, not done. `probe_rollout_gifs.sbatch` and its record still carry
the wrong 1-GPU claim; flagged, not edited.

## GPU runs

Submitted with the maintainer's standing authorization for test-scale GPU
jobs (2026-09-07: "use the GPU via preempt or general partition ... for
better tests"):

| Job | What | Where |
|---|---|---|
| 10349711 | `sbatch scripts/run_ogpo_tests.sbatch` — full `pytest tests/ogpo` (now 392 tests) on the general partition | `/home/pchellap/logs/ogpo_tests_10349711.out` |
| 10349712 | `sbatch scripts/probe_candidate_q_spread.sbatch` — BoN=1 smoke, `mt4_mt2_b128_ep2_s0`, `libero_90_38`, 8 episodes (one wave), M=8, 2 GPUs / FSDP=2 | `/home/pchellap/logs/q_spread_10349712.out`, outputs `/home/pchellap/logs/q_spread_10349712_bon1/` |
| — | BoN=0 smoke: `QSPREAD_BON=0 sbatch scripts/probe_candidate_q_spread.sbatch` — the session's submission was blocked by the permission classifier (both `--export=ALL,QSPREAD_BON=0` and the env-prefix form); left for the maintainer | |

Results are appended below when the jobs finish.

### Job 10349711 — full `pytest tests/ogpo` (general partition, CPU)

```
3 failed, 383 passed, 6 skipped, 273 warnings in 1274.21s (0:21:14)
FAILED tests/ogpo/test_verifier_alignment.py::test_head_value_distribution_differential_over_every_registered_config
FAILED tests/ogpo/test_verifier_alignment.py::test_head_value_distribution_differential_is_a_real_change_at_201_bins
FAILED tests/ogpo/test_verifier_alignment.py::test_head_wrapper_differential_over_a_randomized_flag_script
```

All 33 tests of this change pass. The three failures are **pre-existing and
unrelated**: the identical three failed in the full-suite runs of jobs
10262223 (`3 failed, 341 passed`) and 10201675, both before this change
existed. They are differentials of the working tree against `git HEAD`
(`_head_get_value_bounds`, `_module_at_head("src/envs/wrappers.py")`) whose
"this is a real change" assertions became vacuous once the
reference-alignment commit (`5b94510`) put the compared code at HEAD —
`before == after` at `:1152`/`:1176`, and the old wrapper now accepts the
kwarg at `:1216`. Not touched here (OQ-4 territory; a separate cleanup).

### Job 10349712 — BoN=1 smoke (preempt, 2× RTX PRO 6000, FSDP=2)

COMPLETED in 48:58, MaxRSS 215 GB. Restore succeeded on two GPUs
(`rl_state/100000`, 9.3 s for the 1.9 GiB train state; the replay/success
buffer restore dominated startup: agent ready at +24 min). One wave of 8
episodes on `libero_90_38`, `EP_MULT=2`, M=8, reduction=mean:

```
[qspread] libero_90_38 wave=0 recorded=8/8 wave_SR=0.875 cum_SR=0.875
          steps=[404, 426, 401, 425, 390, 800, 375, 400]
          q_var[min/median/max]=0.01372/1.799/626.4
```

| ep | seed | outcome | env steps | chunks | var min / median / max | mean Q |
|---|---|---|---|---|---|---|
| 00 | 0 | success | 404 | 81 | 0.045 / 1.04 / 298 | −85.9 |
| 01 | 1 | success | 426 | 86 | 0.014 / 1.98 / 77 | −87.2 |
| 02 | 2 | success | 401 | 81 | 0.091 / 1.49 / 217 | −81.4 |
| 03 | 3 | success | 425 | 85 | 0.021 / 1.82 / 220 | −99.3 |
| 04 | 4 | success | 390 | 78 | 0.046 / 1.62 / 86 | −95.5 |
| 05 | 5 | failure (TimeLimit) | 800 | 160 | 0.024 / 3.49 / 626 | −160.4 |
| 06 | 6 | success | 375 | 75 | 0.089 / 1.97 / 424 | −84.7 |
| 07 | 7 | success | 400 | 80 | 0.022 / 0.71 / 237 | −80.1 |

Outputs: `/home/pchellap/logs/q_spread_10349712_bon1/libero_90_38/ep0[0-7]_seed<s>_<succ|fail>.png`,
`mean_trace.png`, and `q_spread_results.json` (every `(chunks, 8)` score
matrix, executed index per chunk). Both plot kinds inspected: single series,
two panels for the mean trace with the alive count beneath (8 → 1 after chunk
~87, so the tail of the mean is the failing episode alone, which the panel
makes visible), titles carry outcome / steps / arm / M. The executed-index
histogram per episode is roughly uniform over the 8 candidates, as expected
for iid draws. Chunk-count × `replan_steps` matches env steps (81 × 5 ≈ 404),
i.e. the `(live, replan)` bookkeeping is right against the real env.

Not a finding of this change, but visible in the first data: the Q-variance
is O(1) for most of an episode and spikes by two orders of magnitude in
narrow windows (chunks ~55-60 and ~75-95 on this task), with the failing
episode carrying the largest spikes and a much lower mean Q.

Wall-clock envelope: ~24 min startup + ~28 min per 8-episode wave, since a
wave runs to its longest episode (160 chunks) at the full 8×8 batch (verifier
finding 9). Recorded in the sbatch header.

### Not run

The BoN=0 arm. Both submission forms were blocked by the permission
classifier; handed to the maintainer as
`QSPREAD_BON=0 sbatch scripts/probe_candidate_q_spread.sbatch`.

### Job 10352129 — task-38 single-task checkpoint, 16 episodes, M = 4 × 8

Submitted 2026-09-07 at the maintainer's request after they chose
`stab_NCB_libero_90_38` (step 100000) and asked for 16 episodes with 32
candidates per state, split over passes if one pass would not fit.
Wrapper `scripts/probe_candidate_q_spread_stab.sbatch` through the edited
`stability_study.sh` (`docs/changes/2026-09-07-stability-recipe-task-knob/`).
Setting: `QSPREAD_BON=1`, `QSPREAD_EPISODES=16` (2 waves), `BON_N=8`,
`QSPREAD_PASSES=4`, task `libero_90_38`, 400-step episodes, 2-head critic
reduced by min, 1 GPU / FSDP=1, 200G, 5 h. Outputs
`/home/pchellap/logs/q_spread_stab_10352129_bon1/`. Results appended when
it finishes.

Job 10352129 **FAILED after 17 s**, before any restore: LIBERO's first-import
prompt (`libero/libero/__init__.py:101-104`) hit `EOFError` because the
redirected `STORE_ROOT` had no `libero/config.yaml` and this repo's
`stability_study.sh` lacked the seeding guard the mt4 recipe and the sibling
recipe both carry. Fixed in the recipe (see
`docs/changes/2026-09-07-stability-recipe-task-knob/DIFF.md`); resubmitted as job 10352252, then cancelled while still pending and resubmitted
as job 10352338 with an 8 h limit after the verifier flagged the 5 h budget
(one GPU, four passes). That one restored the checkpoint (step 100000, 1 GPU,
~6 min incl. the replay buffer) and then died loading the base pi05 weights
from the redirected store's empty openpi cache; the recipe regained a
`CKPT_BASE_DIR` override and the wrapper redirects only that (see the recipe
record). Resubmitted as **job 10352521**, outputs
`/home/pchellap/logs/q_spread_stab_10352521_bon1/`.

### Job 10352521 — task-38 single-task checkpoint, 16 episodes, M = 32: COMPLETED

24 min wall, 108 GB MaxRSS, SR 8/16. Per-episode median Q-variance 0.05–0.09
(vs. 0.7–3.5 on the mt2 ref checkpoint at M=8), pooled median 0.072 for
successes and 0.061 for failures; rare spikes to 1.6e3 / 6.1e3. Mean Q per
episode −206 to −212, i.e. at or below the −1/(1−γ) = −200 no-success floor
of this run's regression critic (2 heads, min-reduced, no success bonus).
Details in the recipe record's VERIFICATION.md; raw scores in the JSON.
