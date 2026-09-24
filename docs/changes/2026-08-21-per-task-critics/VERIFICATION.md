# VERIFICATION — per-task critics for multi-task OGPO

Step 3 of the Tier 2 workflow. Section 1 is the independent verifier's report
**verbatim** (Opus, xhigh, fresh context, given the spec + diff only). Section 2 is the
implementing session's disposition of each finding. Section 3 is the post-fix test
record. Section 4 is what remains unverified.

---

## 1. Verifier report (verbatim)

All verification runs are complete. Here is the report.

---

# VERIFICATION — per-task critics (independent step-3 verifier)

Inputs: `README.md` / `BLAST-RADIUS.md` / `PLAN.md` / `DIFF.md` + `git diff` + the three new files. Baseline `git show HEAD` (`5b94510`). No source, script, doc, or existing test was modified. Added file: `tests/ogpo/test_per_task_critics_verifier.py` (21 tests).

## Findings, by severity

### F1 — HIGH · CONFIRMED · the mt4 arm gets 3 critics for 4 tasks, silently
`src/rl/task_registry.py:38` keys slots on `str(task_description)`; the only two producers feed it the LIBERO **language string** (`src/rl/filtered_sft_agent/filtered_sft_learner.py:796`, `src/rl/advantage_weighted_sft/advantage_weighted_sft_learner.py:461-463` ← `src/training/collect.py:193` ← `src/envs/libero.py:51` = `task.language`). The R3 guard (`advantage_weighted_sft_learner.py:57-58`) counts distinct **task ids**.

`scripts/ogpo_multitask_4task.sh:70` = `libero_90_79 libero_90_31 libero_90_82 libero_90_38`. Resolved against the LIBERO benchmark table (run, not inferred):

```
79 'pick up the book and place it in the left compartment of the caddy'
31 'put the black bowl on top of the cabinet'
82 'pick up the book and place it in the left compartment of the caddy'   <-- == 79
38 'put the right moka pot on the stove'
duplicate language strings across libero_90: 12
```

With `PER_TASK_CRITIC=1`: `num_tasks == len(set(tasks)) == 4` passes, but the registry only ever holds **3** entries. libero_90_79 and libero_90_82 **share critic slot 0** — the exact cross-task contamination the change exists to remove — and slot 3's Q and V are allocated (25% of the T× memory and FLOPs), never receive a gradient, and are never read. Nothing raises. The only signal is `critic/q_n_task_3` and `critic/value_n_task_3` pinned at `0.0` forever. `README.md`'s "each task's Q/V trained only on its own transitions" is false for the arm being targeted.

Not specific to these four: 12 libero_90 language strings are duplicated.

Suggested fixes (pick one): key the registry on the LIBERO task id (available at `collect.py:190` as `current_task_ids[env_index]`, but would also need threading into `sample_actions`); or assert `len(self._task_registry) == self._num_critic_tasks` at the end of the first collection round; or at minimum log the registry contents at each checkpoint.

Tests added: `test_registry_collapses_task_ids_that_share_a_description_KNOWN_GAP`, `test_libero_90_mt4_task_descriptions_actually_collide_KNOWN_GAP` (both pass — they pin the defect).

### F2 — HIGH · CONFIRMED · existing test regression: `test_g_stored_prefix_threading`
`src/rl/ogpo/ogpo_learner.py:751` reads `self._task_registry` unconditionally. `tests/ogpo/test_split_equivalence.py:610,614` calls the override **unbound with `self=None`** (its comment: "The override never touches self").

```
tests/ogpo/test_split_equivalence.py::test_g_stored_prefix_threading  -> 1 failed
E  AttributeError: 'NoneType' object has no attribute '_task_registry'
   src/rl/ogpo/ogpo_learner.py:751
```

It dies before the (also-stale) 3-tuple unpack. Contradicts `PLAN.md` §10.7 ("`test_split_equivalence.py` all 6 legs unchanged") and `DIFF.md`. Self-free fix that keeps the old call working and matches how the composed `train_step` already does it (`update_actor.py:691`):
`task_index = online_batch[TASK_INDEX_NAME] if TASK_INDEX_NAME in online_batch else None`
(the test's 3-value unpack still needs updating either way).
Test added: `test_unbound_online_batch_to_sft_batch_still_works_with_stub_self` (FAILS, intentionally red).

### F3 — HIGH · CONFIRMED · the change's only direct jit-1 coverage never runs
`tests/ogpo/test_per_task_critics.py:628` does `from tests.ogpo.test_split_equivalence import ...`. A **site-packages package named `tests`** (`.venv/lib64/python3.12/site-packages/tests/__init__.py`) shadows the repo's `tests/` (no `tests/__init__.py`), so the import raises `ModuleNotFoundError: No module named 'tests.ogpo'`.

```
ERROR tests/ogpo/test_per_task_critics.py::test_jit1_task_index_none_is_identity
ERROR tests/ogpo/test_per_task_critics.py::test_jit1_routes_each_state_to_its_own_task_critics
```

Both ERROR at fixture setup — i.e. the Tier-2 signature edit had **zero** executed test coverage. (Also visible as `ERROR [19%]` at lines 41-42 of the implementing session's own `-v` log.) `docs/code/tests.md:36,195` claims this coverage. Working import: `importlib.import_module("ogpo.test_split_equivalence")` (pytest imports siblings under `ogpo`, since `tests/ogpo/__init__.py` exists and `tests/__init__.py` does not). Both claims re-made and now passing in the verifier file.

### F4 — HIGH · CONFIRMED · second existing-test regression: registered-config dispatch pin
```
FAILED tests/ogpo/test_verifier_alignment.py::test_dispatch_for_every_registered_config_is_unchanged_by_the_new_entry
E  AssertionError: Left contains 1 more item:
E    {'pi05_libero_online_ogpo_sft_pertask': 'OGPOAgentLearner'}
   tests/ogpo/test_verifier_alignment.py:454
```
The new registered config (`src/training/config.py:693`) breaks a test that pins the exact config→learner map. Not mentioned in `DIFF.md`. Either update the pin or drop the registered config (see F7).

### F5 — MEDIUM · CONFIRMED · "routing survives the G-expansion" was untested
`tests/ogpo/test_per_task_critics.py` docstring (§"routing survives the G-expansion") and `:684`. Its fixture inherits `group_num_samples=1` from `test_split_equivalence._build_config` (`:87`), where `_expand` is a no-op. Production mt4 runs `--rl.group_num_samples 8`.
Now covered at G=2 by `test_jit1_g_expansion_routes_each_state` — **PASSES**: perturbing slot 1's Q moves both of state 1's samples and neither of state 0's, i.e. `jnp.repeat` (not tile) ordering is correct and matches the sampler's.

### F6 — MEDIUM · CONFIRMED · memory/compute figure is for a path the recipe does not take
`src/training/config.py:167` and `BLAST-RADIUS.md` §"Memory & compute" say the mt4 default is the MLP path (~0.26 GB at T=4). `DRY=1 bash scripts/ogpo_multitask_4task.sh` emits **`--rl.critic.use_bronet --rl.critic.bronet_hidden_dim 1024`** (depth default 2), `num_qs=num_vs=2`, `--rl.critic.batch_size 1024`. Analytic (prefix 2048 + state + flat action ≈ 2.4k input): ~26M critic params ≈ 0.42 GB for params+Adam m/v+EMA today → **~1.7 GB at T=4 (+~1.25 GB)**, plus 4× the burst's 1000 critic steps/round. Not measured on GPU. The "negligible" justification does not hold for this arm as written.

### F7 — MEDIUM · CONFIRMED · `pi05_libero_online_ogpo_sft_pertask` cannot be constructed as registered
`config.py:693-698` pins `num_tasks=4` while inheriting `collect.tasks=["libero_90_59"]` (`config.py:373-375`), so R3 (`advantage_weighted_sft_learner.py:57-58`) raises at learner construction. The mt4 script never uses it (it drives `pi05_libero_online_ogpo_sft` + `PER_TASK_CRITIC`). It parses but cannot run; it is also the sole cause of F4. Test added: `test_registered_pertask_config_is_not_runnable_as_registered` (passes, pins the fact).

### F8 — MEDIUM · PLAUSIBLE (mechanism CONFIRMED) · `_task_subtree_mask` fails open
`src/rl/advantage_weighted_sft/update_critic.py:144-155`: `_is_task` returns `False` for any unrecognised keypath, and `per_task_clip_chain` never asserts coverage. If the wrapper's storage layout ever changes (rename `tasks`, `nnx.List`, a flax nnx keypath change), **every `optax.masked` becomes a no-op and the critic trains with no gradient clipping at all** — silently. Today it is correct: I confirmed the keypaths are `(DictKey('tasks'), DictKey(<int>), …, GetAttrKey('value'))` and that the T masks partition the tree exactly (84/84 leaves at T=3). Test added: `test_task_masks_partition_the_whole_param_tree` (passes) so a future change fails here. This is the one new **silently-applied default** in the diff under the deviation protocol.

### F9 — LOW · CONFIRMED · out-of-range slot → NaN, no message
`src/rl/networks/per_task_critic.py:57` `_gather_task` does not bounds-check; `jnp.take_along_axis`'s default OOB mode NaN-fills. That NaN then propagates through `per_task_mean`'s einsum (NaN·0 = NaN) into the loss, so it is loud at runtime but names nothing. Unreachable via the registry; reachable from a foreign/corrupt shard. Pinned by `test_gather_out_of_range_slot_yields_nan_not_a_raise_KNOWN_GAP`.

### F10 — LOW · CONFIRMED · `TaskRegistry.from_json` does not bound slots by `num_tasks`
`src/rl/task_registry.py:63-88` validates `num_tasks` equality and slot contiguity, but not `len(slots) <= num_tasks`. A file with 3 tasks and `num_tasks=2` loads and hands out slot 2 (→ F9). Unreachable from `to_json`; reachable from corruption. Pinned by `test_from_json_does_not_bound_slots_by_num_tasks_KNOWN_GAP`.

### F11 — INFO
- `update_critic.py:429-432` / `:501-504` recompute `dist.log_prob(td_target)` and `dist.log_prob(mc_return)` a second time for `_per_task_aux`, inside the differentiated `loss_fn`. Gradient-neutral (XLA will likely CSE), but it is 2× the log_prob graph by construction in the burst's hot loop.
- `value_mean`, `mc_corr`, `grad_norm`, `param_norm` stay **batch-global / tree-global** under per-task critics, i.e. mixed across tasks. Only `loss_task_{t}` / `n_task_{t}` are per-task. `DIFF.md` notes `q_loss` incomparability but not this.
- R4 confirmed as documented: at step 1 the absent task is bitwise unchanged; with F1 slot 3 is permanently absent and drifts only by `lr·wd·p ≈ 1e-14` relative — harmless, nothing reads it.

## What I verified as correct (no finding)

- **`num_tasks=None` bit-identity.** Read every diff hunk for a None-path delta; the only ones are (a) `train_value_step` hoisting `td_target` (same graph — pinned bit-for-bit by their reference test), (b) jit-1 receiving an 8th positional `None` (new input treedef → one extra compile vs HEAD, numerically identical; leg d + `test_jit1_task_index_none_is_bit_identical_to_the_7_arg_call` both pass), (c) the `_online_batch_to_sft_batch` arity change (F2). `_critic_optimizer(None)` is bit-for-bit `_optimizer.create_optimizer` (their test). Buffer schema, info key sets, `sample_actions`, checkpoint I/O all gated. `BestofNLearner` subclasses `FilteredSFTLearner` (not AWR) so it never enters the new `__init__` block; the config guard blocks the flag on AWR/MPO/FlowGRPO/BofN anyway.
- **Routing (D2).** Verified independently at three levels: the wrapper (their test), through `train_q_step` including the **TD target's `V(next_obs)` under the same task** — untested upstream, added `test_q_td_target_bootstraps_next_obs_under_the_same_task` (perturbing the absent task's V leaves the Q update bit-identical; perturbing the present task's V changes it) — and through jit-1 at G=2.
- **Disjointness.** `nnx.state(wrapper)` has exactly one root, `tasks`; every leaf is a `VariableState` array; `num_tasks` is static (graphdef), so no non-array leaf reaches orbax.
- **Per-task clip (D5).** Added an end-to-end test at the `tx` the `TrainState` actually holds (`_critic_optimizer(config)`), not just the exposed helper: task 1's Adam update is invariant to task 0's gradient magnitude, with the `num_tasks=None` global-clip optimizer as a non-vacuous positive control. Passes.
- **`per_task_mean` math.** Independent numpy reference over random data with two leading head axes and two absent tasks; plus balanced-batch == `jnp.mean`. Both pass. Their hand computation checks out.
- **Buffer plumbing (D3/D8).** Exercised on the real `ShardedReplayBuffer` with synthetic data (the plan listed this as unverifiable): `task_index` lands **top-level** (not under `observation`), survives `drop_obs_keys`, keeps `int32`, and round-trips through `save_shard`/`restore_shards`. Inserting a transition without it into a per-task buffer raises `ValueError("Insert transition structure does not match buffer structure")`, and vice versa — which is exactly the D8 fail-fast, since `restore_shards` restores via `insert()` (`replay_buffer.py:283`).
- **Checkpoint layout.** Orbax `StandardCheckpointer` round-trips the per-task `TrainState`, and restoring a shared-critic checkpoint into a per-task template raises. Added.
- **Resume ordering.** `FilteredSFTLearner.__init__` restores buffer shards (`:286-291`) → AWR builds critics → `_restore_rl_checkpoint` (`:147-148`) replaces the registry from JSON. No `index_for` call occurs between shard restore and registry restore, so no path permutes slots. `save_epoch_state` calls `agent.save_checkpoint(step)` **before** writing the resume manifest (`runtime_state.py:74,83-91`), so the JSON and the manifest always name the same step. `exp.py:166`'s `free_buffer_before_eval` rebuild reads the still-set `_num_critic_tasks`, so the rebuilt buffer keeps the schema.
- **Every critic call site carries `task_index`** when `num_tasks` is set: the update-loop critic step, `critic_utd` (all fresh batches go through `_online_batch_to_critic_batch`), `critic_success_oversample` (success buffer built from the same dummy → same schema; the `_config` swap at `ogpo_learner.py:96-107` touches only `rl.buffer_capacity`), the digestion burst including `burst_use_mc_targets`' second jit (`mc_cfg` is a `dataclasses.replace` that preserves `critic.num_tasks`), and the best-of-N `sample_actions` path. A missing key raises `KeyError` at trace time rather than defaulting to slot 0 — I added `test_missing_task_index_in_critic_batch_raises` to prove that through `train_q_step`, not just the wrapper. Registry keys agree between `save_episode` and `sample_actions` (both `str(info["task_description"])`).
- **Tier-2 contracts.** jit-1 has 8 args and 8 `in_shardings` slots; `donate_argnums=()` unchanged; the arity-3 RNG split (`update_actor.py:160-164`) untouched — `test_leg_c_rng_arity_three_and_indexing` **passes**. Nothing new is donated or captured across `del ema_dev` (the sidecar is a `(B,)` int32 already in the device batch). Both backbones ignore `task_index` (`bronet_critic.py:79-85` and the MLP `state_vector_keys`), so the extra obs key is inert. `_maybe_reset_critic_optimizers` reuses `state.tx`, so the per-task clip survives a critic-optimizer reset.
- **Fail-fast audit.** No new `getattr(..., default)`, no `.get(k, default)`, no `try/except`, no swallowed exception in the diff or the two new source files. The only new fail-open path is F8; the `max(count,1)` / `max(n_present,1)` guards are the documented D6 semantics, not error suppression.
- **Script.** `PER_TASK_CRITIC=1 → --rl.critic.num_tasks 4`; `=0 → None`; unset `→ None`. Emitted unconditionally as claimed.
- **Their tests are not tautological.** The `_reference_train_q_step` / `_reference_train_value_step` copies are verbatim against `git show HEAD:src/rl/advantage_weighted_sft/update_critic.py` (only the helpers they call are imported, and those are untouched by the diff). Tolerances (`0.0` exact / `1e-5` reassociation) are stated and appropriate.

## Test output (verbatim summary lines)

Environment for every run: `export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2; ulimit -v 33554432; ulimit -u 2000; taskset -c <cores> uv run pytest <target> -q -p no:cacheprovider`.

```
# per-task critics, fast legs (cores 8-11)
tests/ogpo/test_per_task_critics.py -k "not jit1"
  21 passed, 2 deselected, 70 warnings in 44.42s

# both per-task-critic files (cores 16-19)
tests/ogpo/test_per_task_critics.py tests/ogpo/test_per_task_critics_verifier.py -rEf
  ERROR  tests/ogpo/test_per_task_critics.py::test_jit1_task_index_none_is_identity
  ERROR  tests/ogpo/test_per_task_critics.py::test_jit1_routes_each_state_to_its_own_task_critics
  FAILED tests/ogpo/test_per_task_critics_verifier.py::test_unbound_online_batch_to_sft_batch_still_works_with_stub_self
  1 failed, 41 passed, 102 warnings, 2 errors in 183.80s (0:03:03)

# everything except test_split_equivalence.py (cores 12-15)
test_verifier_alignment + test_reward_and_value_bounds + test_ema_utils
  + test_grad_norm_decomposition + test_group_dedup + test_sampling  -rEf
  FAILED tests/ogpo/test_verifier_alignment.py::test_dispatch_for_every_registered_config_is_unchanged_by_the_new_entry
  FAILED tests/ogpo/test_verifier_alignment.py::test_head_value_distribution_differential_over_every_registered_config
  FAILED tests/ogpo/test_verifier_alignment.py::test_head_value_distribution_differential_is_a_real_change_at_201_bins
  FAILED tests/ogpo/test_verifier_alignment.py::test_head_wrapper_differential_over_a_randomized_flag_script
  FAILED tests/ogpo/test_group_dedup.py::test_rescorer_cached_matches_uncached_forward_and_grad
  FAILED tests/ogpo/test_sampling.py::test_scanned_rescorer_grad_matches_unrolled
  6 failed, 138 passed, 3 skipped, 35 warnings in 740.98s (0:12:20)

# test_split_equivalence.py, one leg per process (cores 8-11)
  test_leg_a_split_and_composer_match_reference_ema_equals_params      COULD NOT RUN (exit 137, SIGKILL)
  test_leg_a_prime_split_and_composer_match_reference_perturbed_ema    COULD NOT RUN (exit 137, SIGKILL)
  test_leg_b_splice_structural_identity            1 passed, 16 warnings in 90.46s
  test_leg_c_rng_arity_three_and_indexing          1 passed, 13 warnings in 121.79s
  test_leg_d_none_branch_recompute_matches_original 1 passed, 13 warnings in 59.80s
  test_g_stored_prefix_threading                   1 failed, 13 warnings in 40.35s
```

Failure detail for the four "rest" failures, and their attribution:

```
tests/ogpo/test_verifier_alignment.py:454  (CAUSED BY THIS CHANGE — F4)
E  AssertionError: Left contains 1 more item:
E    {'pi05_libero_online_ogpo_sft_pertask': 'OGPOAgentLearner'}

tests/ogpo/test_verifier_alignment.py:1143 (PRE-EXISTING, not this change)
E  assert -99.99999999999991 < -99.99999999999991
tests/ogpo/test_verifier_alignment.py:1167 (PRE-EXISTING)
E  assert (-200.49999999999983, 0.49999999999999956) != (-200.49999999999983, 0.49999999999999956)
tests/ogpo/test_verifier_alignment.py:1207 (PRE-EXISTING)
E  Failed: DID NOT RAISE <class 'TypeError'>
```
Those three are head-vs-worktree differentials (`_head_get_value_bounds()`, `_module_at_head("src/envs/wrappers.py")`). `src/rl/value_distribution.py` and `src/envs/wrappers.py` are **unmodified in this working tree**, so HEAD == worktree and the differentials are now vacuous — they went red when the 2026-08-20 reference-alignment change was committed at `5b94510`, not because of this change.

```
tests/ogpo/test_group_dedup.py::test_rescorer_cached_matches_uncached_forward_and_grad (ENVIRONMENT)
tests/ogpo/test_sampling.py::test_scanned_rescorer_grad_matches_unrolled              (ENVIRONMENT)
E  jaxlib.xla_extension.XlaRuntimeError: INTERNAL: ... Out of memory allocating 535486464 bytes.
```

Script inspection:
```
PER_TASK_CRITIC=1     -> --rl.critic.num_tasks 4
PER_TASK_CRITIC=0     -> --rl.critic.num_tasks None
PER_TASK_CRITIC unset -> --rl.critic.num_tasks None
```

## Could NOT be verified — and why

- **`pytest tests/ogpo` as a whole never completes on this login node.** Two attempts died with SIGKILL (exit 137) partway into `test_split_equivalence.py`; a third aborted inside XLA `backend_compile`. Legs a / a' of `test_split_equivalence.py` are individually unrunnable here (SIGKILL at 32 GB vmem; XLA CPU "Out of memory allocating 535 MB" at the default 16 GB). This is exactly what `scripts/run_ogpo_tests.sbatch` exists for and is **pre-existing** — the implementing session's own full run also died (at 27%). **The full suite must be re-run via `sbatch scripts/run_ogpo_tests.sbatch` before this change is called green.** Note also that the suite is **197 tests**, not the 23 CLAUDE.md still claims.
- Real collection: `save_episode` → registry → buffer insert with a live LIBERO env (needs EGL + GPU). Read-verified and substituted with a synthetic `ShardedReplayBuffer` exercise.
- The digestion burst, `critic_utd`, `critic_success_oversample` end-to-end (read-verified only; all funnel through `_online_batch_to_critic_batch`).
- `sample_actions` best-of-N Q scoring with `task_index` (needs π0.5 weights + a policy). Read-verified. Note mt4 emits `--rl.n_samples 1`, so it is off in the target arm; `pi05_libero_online_ogpo_ref` sets `n_samples=8`.
- Actual GPU memory at T=4 (F6 is analytic).
- Multi-device FSDP sharding of the T× critic tree (mt4 uses `--fsdp_devices 1`).
- A real resume (registry sidecar + orbax + buffer shards together). Verified the orbax layer, the registry, and the buffer independently, plus the ordering by reading.
- Nothing was launched: no `sbatch`, no `exp.py`, no `nohup` training.

## Gotchas for a future session

1. **`from tests.ogpo.X import ...` does not work in this repo.** A site-packages package named `tests` shadows `tests/`. Use `importlib.import_module("ogpo.X")` (pytest's own module name, since `tests/ogpo/__init__.py` exists and `tests/__init__.py` does not). This silently ERRORs a fixture rather than failing loudly at collection.
2. **The mt4 recipe forces BroNet-1024**, overriding the config's MLP default. Any per-critic cost estimate must use the script's flags, not `get_config(...)`.
3. **LIBERO task ids are not LIBERO task descriptions.** 12 language strings are duplicated across libero_90. Anything keyed on `task_description` (this registry, `_success_task_ranges`, and `normalize_advantage_per_task`'s prompt hash) silently merges those tasks.
4. `jnp.take_along_axis` NaN-fills out-of-bounds — it does not clamp and does not raise.
5. `git status` hides `docs/code/` (gitignored, `.gitignore:25`). Step 4 has already been applied there; it needs corrections for F1, F3 (`tests.md:36,195` claims jit-1 coverage that does not execute), F6, and `tests.md:112`.
6. Running two pytest sessions concurrently on this node causes SIGKILLs that look like test failures. Run one at a time, or use the sbatch runner.
7. Side effect of my run: importing LIBERO prompts interactively on first use and I answered `N`, creating `~/.libero/config.yaml` with the default paths. Benign (any first LIBERO run creates it); mentioned for completeness.

---

## 2. Disposition (implementing session)

| # | Sev | Status | What was done |
|---|---|---|---|
| F1 | HIGH | **OPEN — user decision required** | Real. Two of the four mt4 tasks (`libero_90_79`, `libero_90_82`) share the LIBERO language string, and the registry is keyed on that string. Not fixed unilaterally: the fix options change the `Agent` ABC (thread the task *id* through `save_episode` / `sample_actions`, a Tier-2 expansion) or change what "task" means for the critic (identical-prompt tasks share a critic, `num_tasks` counted by distinct *prompt*), and the same collision already affects `balance_success_buffer_tasks` and `normalize_advantage_per_task`. Presented to the maintainer as a decision. Until resolved, **`PER_TASK_CRITIC=1` on the mt4 set silently trains 3 critics for 4 tasks** — do not launch it. |
| F2 | HIGH | Fixed | `_online_batch_to_sft_batch` now keys on the batch (`TASK_INDEX_NAME in online_batch`), self-free, matching the composed `train_step`. `test_g_stored_prefix_threading` unpack updated to the 4-tuple (asserts the fourth is `None`). |
| F3 | HIGH | Fixed | `importlib.import_module("ogpo.test_split_equivalence")`; comment explains the site-packages `tests` shadowing. |
| F4 | HIGH | Fixed | Dispatch pin extended with the new config (it is the same dataclass → `OGPOAgentLearner`, which is what the pin exists to check). |
| F5 | MED | Covered by verifier | `test_jit1_g_expansion_routes_each_state` (verifier file) retained. |
| F6 | MED | Fixed (docs) | `config.py` comment, `BLAST-RADIUS.md`, `rl-networks.md` corrected: the mt4 recipe forces BroNet-1024; ~1.7 GB at T=4 (analytic, not measured). |
| F7 | MED | Fixed | `pi05_libero_online_ogpo_sft_pertask` now bakes in the 4-task collect/eval set, so the registered name is self-consistent under R3. Verifier's pin inverted to `test_registered_pertask_config_is_self_consistent`. |
| F8 | MED | Fixed | `_task_subtree_mask` now fails **closed**: any leaf not under `tasks/<0..T-1>` raises at `tx.init`/trace time naming the leaf. |
| F9 | LOW | Documented, not fixed | `_gather_task` cannot raise on a traced value; an out-of-range slot NaNs loudly. Reachable only via a corrupt sidecar, which F10 now rejects at load. Gotcha recorded in `rl-networks.md`. |
| F10 | LOW | Fixed | `from_json` rejects `len(slots) > num_tasks`. Verifier's pin inverted to `test_from_json_bounds_slots_by_num_tasks`. |
| F11 | INFO | Fixed (1st) / documented (2nd) | `log_prob` computed once per target and reused by the loss and `_per_task_aux`. Mixed-task `value_mean`/`mc_corr`/`grad_norm`/`param_norm` noted in `DIFF.md`. |
| — | — | Fixed (docs) | Test count claims (23 / 46) corrected to the true collected count, 197. |

The three `test_verifier_alignment.py` head-vs-worktree failures and the two XLA host-OOM
failures are **pre-existing / environmental** per the verifier's attribution and were
not touched.

## 3. Post-fix test record

See §3a below (fast legs, run by the implementing session after the fixes) and §3b
(targeted PaliGemma-backed legs). The full suite **has not been run to completion** on
this node — it cannot be (see §4). Proposed: `sbatch scripts/run_ogpo_tests.sbatch`
(not launched; needs permission).

### 3a. Fast legs, both per-task-critic files, after fixes
```
tests/ogpo/test_per_task_critics.py tests/ogpo/test_per_task_critics_verifier.py -k "not jit1"
  40 passed, 4 deselected, 92 warnings in 61.73s (0:01:01)
```

### 3b. Targeted PaliGemma-backed legs, after fixes
(appended below when the run completes)
```
tests/ogpo/test_per_task_critics.py::test_jit1_task_index_none_is_identity PASSED
tests/ogpo/test_per_task_critics.py::test_jit1_routes_each_state_to_its_own_task_critics PASSED
tests/ogpo/test_per_task_critics_verifier.py::test_jit1_task_index_none_is_bit_identical_to_the_7_arg_call PASSED
tests/ogpo/test_per_task_critics_verifier.py::test_jit1_g_expansion_routes_each_state PASSED
tests/ogpo/test_split_equivalence.py::test_leg_d_none_branch_recompute_matches_original PASSED
tests/ogpo/test_split_equivalence.py::test_g_stored_prefix_threading PASSED
tests/ogpo/test_verifier_alignment.py::test_dispatch_for_every_registered_config_is_unchanged_by_the_new_entry PASSED
7 passed, 33 warnings in 206.35s (0:03:26)
```
(environment: `JAX_PLATFORMS=cpu OMP_NUM_THREADS=2; ulimit -v 33554432; ulimit -u 2000; taskset -c 0-5`)

## 4. Not verified / still open

- **Full `pytest tests/ogpo` (197 tests) end-to-end.** Cannot complete on the login node
  (SIGKILL / XLA host OOM in `test_split_equivalence` legs a/a′, `test_group_dedup`,
  `test_sampling` — pre-existing, reproduced with the tracked edits stashed). Needs
  `sbatch scripts/run_ogpo_tests.sbatch`; **not launched** (permission required).
  Verified piecewise instead: 40 fast legs + the 7 targeted heavy legs above, and the
  verifier's own 138-passed partial run.
- Three `test_verifier_alignment.py` head-vs-worktree differentials are red
  independently of this change (went red at commit `5b94510`).
- Real collection / burst / best-of-N scoring / GPU memory / multi-device FSDP / a real
  resume: not locally verifiable; see the verifier's list.
- **F1 is open** and blocks a meaningful `PER_TASK_CRITIC=1` run on the current mt4 set.

### 3c. Full suite on a compute node (Slurm job 10187013, `scripts/run_ogpo_tests.sbatch`, approved)
```
[tests] node=babel-s9-24 cpus=8 target=tests/ogpo
[tests] git: 5b94510 (+ uncommitted working tree)
FAILED tests/ogpo/test_verifier_alignment.py::test_head_value_distribution_differential_over_every_registered_config
FAILED tests/ogpo/test_verifier_alignment.py::test_head_value_distribution_differential_is_a_real_change_at_201_bins
FAILED tests/ogpo/test_verifier_alignment.py::test_head_wrapper_differential_over_a_randomized_flag_script
3 failed, 191 passed, 3 skipped, 143 warnings in 929.58s (0:15:29)
```
The three failures are the pre-existing head-vs-worktree differentials (`-99.99999999999991 < -99.99999999999991`,
`DID NOT RAISE TypeError`) — identical assertions to the verifier's attribution to commit `5b94510`;
`src/rl/value_distribution.py` / `src/envs/wrappers.py` are untouched by this change. Everything
else green, including `test_split_equivalence.py` legs a / a′ (unrunnable on the login node) and
all 44 per-task-critic tests. Working tree at this run = slice 1 + post-verification fixes, before
slice 2 (F1).

---

## 5. Slice 2 — critic slots keyed on the task id (F1)

### 5a. Verifier report (verbatim; second independent pass, Opus, xhigh, fresh context)

## VERIFICATION §3 — slice 2 (key critic slots on the task *id*, F1)

Independent step-3 pass. Fresh context, spec + diff only. Nothing under `src/`, `scripts/`, `docs/`, or existing tests was modified. One file added: `/home/pchellap/Projects/OGPO-VLA/vla-post-training/tests/ogpo/test_per_task_critics_verifier2.py` (29 tests).

### Test runs (verbatim summary lines)

```
tests/ogpo/test_per_task_critics_verifier2.py -q -p no:cacheprovider -rfE
29 passed, 19 warnings in 9.28s
```
```
tests/ogpo/test_per_task_critics.py tests/ogpo/test_per_task_critics_verifier.py
tests/ogpo/test_per_task_critics_verifier2.py -q -p no:cacheprovider -k "not jit1" -rfE
70 passed, 4 deselected, 108 warnings in 62.20s (0:01:02)
```
No failures, no errors. Env: `JAX_PLATFORMS=cpu OMP_NUM_THREADS=2; ulimit -v 33554432; ulimit -u 2000; taskset -c 8-11`. Full `tests/ogpo` not run (per instruction).

**Mutation check** (three mutants of `collect.py` built in the scratchpad and imported in place of the real module — no repo file touched):

| mutant | new tests | upstream `test_collect_data_threads_slot_aligned_task_ids_to_the_agent` |
|---|---|---|
| A: `save_episode(task_id=…)` moved *after* the slot reassignment | **FAIL** (2 tests) | FAIL |
| B: `task_id=list(reversed(current_task_ids))` on `sample_actions` | **FAIL** (2 tests) | **PASS** ← |
| C: `task_id=current_task_ids` (alias, not snapshot) | **FAIL** (1 test) | **PASS** |
| baseline (unmutated) | PASS | PASS |

---

### Findings, by severity

**F12 — MEDIUM (test quality). The upstream slot-alignment assertion cannot fail.** `tests/ogpo/test_per_task_critics.py:584-655`. Its stub gives both task ids the *same* description (`desc = {"libero_90_79": "pick up the book", "libero_90_82": "pick up the book"}`), so `assert all(desc[i] == d for d, i in zip(descs, ids))` is satisfied by **any** permutation of the id list. **CONFIRMED** by mutant B: reversing the id list handed to `sample_actions` leaves that test green. The test does kill mutant A (the reassignment-ordering claim it also makes), so only the *slot-alignment* half is unenforced. Covered now by `test_collect_data_ids_stay_slot_aligned_with_distinct_descriptions` and `test_collect_data_filler_slots_carry_a_real_train_task_id`, which use distinct equal-length descriptions and kill B.

**F13 — MEDIUM, pre-existing, live outside mt4. `collect.py` silently truncates `task_description` on a per-env reset.** `src/envs/venv.py:768` builds info leaves with `np.array([str,…])` → fixed-width `<U N` dtype sized by the *first* full reset. `src/training/collect.py:99-104` and `:213-218` then write a freshly-reset env's description **into** that array (`prev_state[env_index] = new_val_leaf[0]`), so a later, longer description is truncated with no error. **CONFIRMED** by `test_collect_py_truncates_task_description_on_a_per_env_reset_KNOWN_GAP` (a 39-char instruction arrives as its first 9 characters).
- mt4 is **accidentally** safe: `libero_90_79` (the longest string) is `TASKS[0]`, and with `env_num=8`, `N_ROLLOUTS=5`, `INIT_ROLLOUTS=10` it fills every initial slot, so the dtype is already at max width.
- **molmo is not safe**: `src/envs/molmo.py:252` re-samples the prompt on every reset, so lengths vary per episode.
- Not introduced by this slice, and not a per-task-critic bug — but it is exactly the corruption `task_id` is immune to (a Python `list[str]`), so it is a further argument for the F1 decision and a reason not to re-key anything else on the description.

**F14 — LOW/MEDIUM. `rl.store_success_episodes_only=True` + per-task critics can abort the run at the first `end_data_collection`.** `advantage_weighted_sft_learner.py:592` returns before `_save_episode_in_buffer`, so a task with zero successes in the first collection round never reaches `_task_slot` and never registers; the new guard (`filtered_sft_learner.py:853-869`) then raises with a message that blames collect.py's plumbing. Default is `False` and no shipped recipe sets it (only `config.py:193` defines it), so latent. **CONFIRMED** by `test_failed_episodes_still_register_their_task_unless_the_flag_is_set`.

**F15 — LOW, accepted-in-spec but now internally inconsistent.** `_success_task_ranges` (`ogpo_learner.py:146`) still keys on `str(task_description)`. An arm running `MT_BAL=1 PER_TASK_CRITIC=1` treats `libero_90_79`/`_82` as **two** critic slots but **one** BC-balancing bucket. Recorded as out of scope in BLAST-RADIUS §Addendum; pinned by `test_success_task_ranges_still_key_on_the_description_KNOWN_GAP` so it fails loudly if it is ever changed.

**F16 — COSMETIC.** `src/rl/task_registry.py:37` (`tasks` property docstring still says "task_description -> slot") and `:40-42` (`index_for(self, task_description: str)` parameter name + docstring). Module docstring was updated; these two were not. Pinned by a KNOWN_GAP test so the rename is noticed.

**F17 — PROCESS.** `docs/code/**` is untouched (`git status`): step 4 of the four-step flow is outstanding for both slices. `DIFF.md` *does* carry a slice-2 section (`:148-175`) — that part is done.

**Nit (not a finding).** `filtered_sft_learner.py:817-822` defines `_task_index` inside `if self._task_registry is not None:` and consumes it at `:839-840` under a second, identical `if`. Correct, but a static checker will read it as possibly-unbound.

---

### Items verified clean

1. **`collect.py` ordering & alignment — CORRECT.** `task_id=current_task_ids[env_index]` at `:202` precedes the reassignment at `:210` (comment at `:196-197` states it). Both `sample_actions` calls (`:60`, `:160`) pass `list(current_task_ids)`; `current_task_ids[env_index]` and the `info` slot are updated in the *same* loop iteration (`:210` / `:218`, `:96` / `:104`), so the two are aligned at the top of every iteration. First alignment comes from `venv.py:717-722` `_filter_kwargs`, which indexes the list by worker index. Filler (`valid_envs=False`) slots are parked on `config.collect.tasks[-1]` / `eval_tasks[-1]` — a **real** train id, so the registry can never overflow on one. `evaluate_policy` verified identically. All CONFIRMED by mutation + the new tests.
2. **Call contract — no `TypeError` anywhere.** Base `sample_actions` → `_generate_actions(observations, task_description, task_id=None)` (`filtered_sft_learner.py:579-588`, `del task_id`). AWR (`:374`) and BofN (`:331`) take `**kwargs` and forward `**kwargs` to `super()` on the early-return legs. DSRL `_generate_actions(observations, **kwargs)` and `save_episode(…, **kwargs)` absorb it (quarantined and unreachable anyway). MPO/FlowGRPO define neither override and inherit AWR's. All four `save_episode` overrides default `task_id=None` and forward it (OGPO forwards to **both** the success-buffer pass and `super()`). Pinned by `test_save_episode_overrides_forward_task_id` (parametrized over all four) and `test_sample_actions_accepts_the_task_id_kwarg_on_every_implementation`. **A molmo run does not break** — `task_id` is domain-agnostic (`config.collect.tasks` strings in both domains); in fact description-keying would have *overflowed* on molmo (per-reset prompt resampling, `molmo.py:252`), so this slice fixes molmo rather than risking it.
3. **Fail-fast audit — clean.** `git diff HEAD -- src/ | grep '^+'` yields no new `getattr(…, default)`, no `dict.get(k, fallback)`, no `except`. The two new `X if cond else None` forms (`ogpo_learner.py:758`, `update_actor.py`) are presence tests on a batch key, not silent config defaults. No path keys a slot on the description: `_task_slot` (`filtered_sft_learner.py:732-747`) takes only the id and raises on `None`; the BofN path reads `kwargs["task_id"]` explicitly and raises with the fix (`awr:397-408`). The pre-existing `kwargs.get("task_description")` is benign under per-task critics — even if it is `None` and everything collapses into one `str(None)` group, the slot vector is still built per env from `task_ids`.
4. **`end_data_collection(step)` discriminator — correct, and not spuriously firable in practice.** `collect_data:225` passes `step=step` (`0 is not None` → guard active at step 0); `evaluate_policy:111` passes nothing. No other caller (`scripts/exp_ogpo_unfrozen_backbone_memdiag.py:366` wraps it with a transparent `*args, **kwargs`). The round provably visits every `collect.tasks` id: the loop exits only when `total_episodes == num_rollouts`, i.e. every queued entry has completed, and each task contributes `num_rollouts_per_task ≥ 1` entries. `num_initial_rollouts` only inflates that (pinned). The registry is never cleared between rounds, so a short/empty later round cannot fire it (pinned) — it effectively only guards round 0 and the first round after a resume, where the registry is restored full by `_restore_rl_checkpoint` (`awr:247-258`, run at `awr:147-148` after `self._task_registry` is set at `:76`). `free_buffer_before_eval` does not touch it. Only known spurious-fire path is F14. Four guard states pinned by tests.
5. **AWR best-of-N slot tiling — CORRECT.** `slots = [self._task_slot(task_ids[i]) for i in indices]` (`awr:480-482`) is in the group's `indices` order, which is definitionally the row order of `group_obs = tree.map(lambda x: x[indices], observations)` and hence of `state`, `processed_obs`, `tiled_obs` and `group_actions`. All three tilings use the identical `repeat(…, n_samples, axis=0)` spelling (env-major, `row = i*n + s`), matching `scores.reshape(group_env_num, n_samples)` and `group_actions.reshape(group_env_num, n_samples, …)`. **CONFIRMED end-to-end** by `test_bofn_slot_list_follows_the_group_index_order_across_two_ids`, which drives the *real* `AdvantageWeightedSFTLearner.sample_actions` with a description group spanning two ids and asserts the recorded `_task_slot` sequence is `["libero_90_79", "libero_90_82"]`; plus `test_repeat_tiling_is_env_major_…` pinning the layout against `np.tile`. No off-by-one, no interleave. The `len(task_ids) != len(task_description)` guard and the missing-`task_id` raise are both pinned.
6. **Resume with a slice-1 (description-keyed) sidecar — cannot silently misroute.** `from_json` has no key semantics, so a 3-entry description-keyed file with `num_tasks=4` *loads*. But the very next id lookup takes the single free slot and the second one overflows (`ValueError`), and even if it did not, the end-of-round guard sees `3 of 4 slots` and raises. Both legs CONFIRMED by `test_slice1_description_keyed_sidecar_cannot_silently_misroute`. Consistent with D8; no such checkpoint exists.

### What could not be verified locally

- A real LIBERO collection round (needs EGL + GPU + π0.5 weights): the `end_data_collection` registry guard against a live simulator, the buffer write of `task_index` under real episodes, and the best-of-N scoring path past `self._policy._input_transform` (the stub stops there — the slot *ordering* is verified, the downstream `q_model(critic_obs, flat_actions)` call is not executed).
- Molmo (CSCS asset path hardcoded at `src/envs/molmo.py:81-83`) — the truncation exposure (F13) there is read-derived, PLAUSIBLE not CONFIRMED.
- GPU/sharding: jit-1's new `task_index` slot under a real mesh; per-task critic memory at T=4.
- `pytest tests/ogpo` in full (login-node limits; must be re-run on a compute node as for slice 1). `pytest tests/` still red at collection (OQ-4), unrelated.

### Gotchas for the next reader

- A stub vector env must return `info` leaves as `np.array([...])` (not `dtype=object`) to reproduce the real `<U`-dtype truncation behaviour of `venv.py:768`; the upstream stub uses `dtype=object` and therefore cannot see F13.
- `task_to_indices` iteration order is dict-insertion order, so "the first description group" is deterministic — the BofN ordering test relies on that.
- `_task_slot` is reachable from two places with different first-seen orders (best-of-N `sample_actions` during collection vs `_save_episode_in_buffer`). Slot *numbers* therefore depend on `n_samples` and `inference_start_step`; only the mapping's stability within a run matters, and the JSON sidecar preserves it.

### 5b. Disposition (implementing session)

| # | Sev | Status | What was done |
|---|---|---|---|
| F12 | MED (test) | Fixed | My stub test now uses distinct equal-length descriptions so a permuted id list fails it; the verifier's two alignment tests are kept. |
| F13 | MED, pre-existing | Documented, not fixed | `collect.py`'s per-env `info` slot update truncates a longer description into the `<U N` array sized by the first reset. Not this change's code; mt4 is accidentally safe, molmo is not. Recorded as a gotcha in `docs/code/training.md`; the verifier's `KNOWN_GAP` test pins it. A fix is a separate change (it touches the collection loop for every learner). |
| F14 | LOW/MED | Fixed | `AdvantageWeightedSFTLearner.save_episode` registers the task **before** the `store_success_episodes_only` early return, so a task with no success in round 0 still owns its slot and the guard cannot misfire. New test `test_store_success_only_still_registers_the_task`. |
| F15 | LOW | Accepted (out of scope, recorded) | `_success_task_ranges` / `normalize_advantage_per_task` stay prompt-keyed; the verifier's pin keeps it visible. |
| F16 | cosmetic | Fixed | `index_for(task_id)`, `tasks` docstring. Verifier's pin inverted. |
| F17 | process | Not an issue | `docs/code/` is gitignored, so `git status` cannot show it; step 4 was applied for both slices (see `DIFF.md`). |
| nit | — | Left | `_task_index` is defined and consumed under two identical gates; correct as written. |

### 5c. Post-fix test record (fast legs, all three per-task files)
```
tests/ogpo/test_per_task_critics.py tests/ogpo/test_per_task_critics_verifier.py tests/ogpo/test_per_task_critics_verifier2.py -k "not jit1"
  71 passed, 4 deselected, 108 warnings in 63.61s (0:01:03)
```
Full `tests/ogpo` on a compute node after slice 2: **not yet run** — proposed as the final
gate (`sbatch scripts/run_ogpo_tests.sbatch`), pending permission.

### 5d. Full suite on a compute node after slice 2 (Slurm job 10187244, approved) — FINAL GATE
```
[tests] node=babel-u9-24 cpus=8 target=tests/ogpo
[tests] git: 5b94510 (+ uncommitted working tree)
FAILED tests/ogpo/test_verifier_alignment.py::test_head_value_distribution_differential_over_every_registered_config
FAILED tests/ogpo/test_verifier_alignment.py::test_head_value_distribution_differential_is_a_real_change_at_201_bins
FAILED tests/ogpo/test_verifier_alignment.py::test_head_wrapper_differential_over_a_randomized_flag_script
3 failed, 222 passed, 3 skipped, 159 warnings in 1004.16s (0:16:44)
```
The same three pre-existing head-vs-worktree differentials as before (identical assertions;
red since commit `5b94510`, untouched files). All 228 − 3 = 225 other tests green, including
all 73 per-task-critic tests across the three files and every `test_split_equivalence.py`
leg. Working tree at this run = slice 1 + slice 2 + all verifier fixes = the final state.
