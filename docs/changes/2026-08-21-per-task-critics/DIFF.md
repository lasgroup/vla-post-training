# DIFF — per-task critics for multi-task OGPO

What actually changed, file by file, against `PLAN.md`. Divergences from the plan are
called out inline as **Δ**.

## New files

### `src/rl/task_registry.py` (new, pure)
`TaskRegistry(num_tasks)`: `index_for()` first-seen assignment, overflow raises with the
task name, the registered set, `num_tasks` and the fix; `to_json` / `from_json`
(missing file ⇒ `FileNotFoundError` citing D8; `num_tasks` mismatch ⇒ `ValueError`;
non-contiguous slots ⇒ `ValueError`); `tasks` read-only view. As planned.

### `src/rl/networks/per_task_critic.py` (new)
`PerTaskStateActionCritic` / `PerTaskStateValue`: T copies of the injected `critic_def`
under `self.tasks`, each seeded from `jax.random.split(rngs(), T)[t]`; `__call__` stacks
all T outputs and `take_along_axis`-gathers by `observation["task_index"]` → today's
`(n_heads, b[, bins])`. Missing key ⇒ explicit `KeyError` (D4). `TASK_INDEX_NAME =
"task_index"` is the single source of the key string.
**Δ** The plan said construction "strips `task_index` only to build the dummy"; it does
not strip anything — both backbones read only their declared vector keys, so the
observation is passed through as-is. Simpler, and the routing test covers both backbones.

### `tests/ogpo/test_per_task_critics.py` (new, 23 tests)
See `VERIFICATION.md` for results. Contains VERBATIM copies of the pre-change
`train_q_step` / `train_value_step` (HEAD `5b94510`) as the differential reference.

## Modified files

### `src/training/config.py` (+52)
- `CriticTrainingConfig.num_tasks: int | None = None` with a mechanism/cost comment.
- `OnlineTrainConfig.__post_init__`: **Δ (addition)** rejects `num_tasks` on any
  `rl` that is not `OGPOSFTLearnerConfig`, and `num_tasks < 1`. Not in the plan; added
  because `BestofNLearnerConfig` carries the same `CriticTrainingConfig` and BofN's
  critic copy is deliberately untouched — the flag would otherwise be silently ignored
  there. This is the mirror image of the existing `use_distributional_critic` guard
  directly above it, which exists for exactly that footgun. Keeps the check in config
  rather than touching `best_of_n_learner.py`.
- Registered `pi05_libero_online_ogpo_sft_pertask`: `_sft` + `critic=CriticTrainingConfig(num_tasks=4)`.
  Pinned by `test_config_num_tasks_cli_and_registered_config` to differ from `_sft` in
  the critic only.

### `src/rl/filtered_sft_agent/filtered_sft_learner.py` (+54 −?)
- Class attributes `_num_critic_tasks = None`, `_task_registry = None` (instance-attribute
  gates; a plain `FilteredSFTLearnerConfig` has no `rl.critic`).
- `_make_buffer_dummy_data`: `return {…}` → `dummy = {…}`; adds `task_index int32 (1,)`
  when gated on.
- `_save_episode_in_buffer`: stamps `task_index` from `self._task_registry.index_for(str(task_description))`
  into the insert dict when gated on. The insert dict is now built as `insert_data`
  then `buf.insert(insert_data)` — same keys/order as before when off.
- **Reaches all five subclasses**; off by default for every one of them.

### `src/rl/advantage_weighted_sft/update_critic.py` (+193 −?)
- Imports `TASK_INDEX_NAME`, the two wrappers.
- `_num_critic_tasks(config)`, `per_task_mean(...)` (typechecked, `*heads b`),
  `_task_subtree_mask(num_tasks, t)` (keypath predicate on `DictKey('tasks'), DictKey(t)`),
  `_critic_optimizer(config)` (None ⇒ exactly `_optimizer.create_optimizer`; else
  `chain(per_task_clip_chain, adamw-from-dataclass-fields)`, non-AdamW ⇒ raise),
  `per_task_clip_chain(num_tasks, max_norm)` (**Δ** split out of `_critic_optimizer` so
  the clip is testable without Adam's scale invariance hiding it), `_per_task_aux(...)`
  (R5 metrics `loss_task_{t}`, `n_task_{t}`).
- `init_state_action_critic_train_state` / `init_state_value_train_state`: `tx =
  _critic_optimizer(config)`; wrap `critic_def` in the per-task wrapper when `num_tasks`
  is set.
- `train_q_step` / `train_value_step`: `task_index = observation[TASK_INDEX_NAME]` when
  set; the `-jnp.mean(...)` lines are kept **verbatim** in the `None` branch and replaced
  by `-per_task_mean(...)` otherwise; per-task aux merged into the returned aux dict.
  **Note:** `train_value_step` now binds `td_target = stop_gradient(bootstrap_target)`
  once and uses it in both branches — same graph as before in the `None` path
  (pinned bit-for-bit by `test_num_tasks_none_train_steps_bit_identical_to_reference`).
- `best_of_n/update_critic.py` **deliberately not changed** (disjoint consumers; BofN is
  critic-only with a frozen policy; the config guard above makes the flag unreachable
  there). This is a recorded divergence in the AWR↔BofN clone family (OQ-2).

### `src/rl/advantage_weighted_sft/advantage_weighted_sft_learner.py` (+70)
- Class attr `_supports_per_task_critics = False`.
- `__init__`, before anything else: when `num_tasks` is set — raise unless
  `_supports_per_task_critics`; raise unless `num_tasks == len(set(collect.tasks))` (R3,
  `==`); raise if `n_samples > 1` and `eval_tasks ⊄ tasks`; then set the two base-class
  gates **before `super().__init__`**. `dummy_obs` gains `task_index`.
- `_task_registry_path(step)` → `rl_state/task_registry_<step>.json` (sidecar next to the
  orbax step dir, R2); `_restore_rl_checkpoint` loads it (missing ⇒ raise, D8);
  `save_checkpoint` writes it.
- `_online_batch_to_critic_batch`: `task_index` into both obs dicts when gated on.
- `sample_actions` best-of-N: `critic_obs[task_index] = full(group_env_num*n_samples, index_for(task))`.

### `src/rl/ogpo/update_actor.py` (+17 −?)
- `sample_and_advantage(..., ema, task_index=None)`: trailing, defaulted; joins
  `critic_observation` before the G-expansion. Arity-3 rng split untouched.
- Composed `train_step`: reads `critic_observation[TASK_INDEX_NAME]` if present, passes it
  through. **Δ** the plan wrote this as a `"task_index" in critic_observation` check; that
  is what was implemented (no `.get` default).

### `src/rl/ogpo/ogpo_learner.py` (+28 −?)
- `_supports_per_task_critics = True`.
- jit-1 `in_shardings`: 8th slot `self._data_sharding` for `task_index` (**the Tier-2
  signature edit**). `donate_argnums=()` unchanged.
- `_online_batch_to_sft_batch`: 3-tuple → 4-tuple; both unpack sites and the jit-1 call
  updated. Nothing new is captured across `del ema_dev`.

### `scripts/ogpo_multitask_4task.sh` (+16)
`PER_TASK_CRITIC` (default 0), emitted **unconditionally**: `1` ⇒ `--rl.critic.num_tasks ${#TASKS[@]}`,
else `--rl.critic.num_tasks None`. Header comment documents it and the `BON_N=1`-with-`HELDOUT=1`
constraint. `DRY=1` verified: `4` / `None` / `None` for `1` / `0` / unset.

## Things to know that are not in the plan

- **Init-rng keystream shift, new config only.** The per-task wrapper consumes one key
  from the critic init stream and splits it T ways; the sub-critics therefore do not
  reproduce a single-critic run's initialization even at T=1. `num_tasks=None` consumes
  the stream exactly as before (bit-identical test).
- **R4, restated:** with one shared `tx`, a task absent from a batch still passes through
  Adam. At step 1 that is a bitwise no-op (m = v = 0; `lr·wd = 1e-14` relative is below
  fp32 resolution — pinned by `test_two_tasks_absent_task_is_untouched_and_reported`);
  after the task has momentum it is not. `n_task_{t}` makes this observable.
- **`critic/q_loss` series are not comparable** across per-task and shared-critic arms
  (per-task-mean-over-present-tasks vs plain batch mean).
- **Test environment.** `pytest tests/ogpo` on this login node needs
  `ulimit -v 33554432; ulimit -u 2000; taskset -c 0-3` (the 16 GB vmem / 1000-thread soft
  limits abort XLA's CPU compiler — pre-existing, reproduced with all tracked edits
  stashed). Recorded in `VERIFICATION.md`.

## Post-verification changes (step 3 findings, see `VERIFICATION.md` §2)

- `src/rl/ogpo/ogpo_learner.py` `_online_batch_to_sft_batch`: gate on
  `TASK_INDEX_NAME in online_batch` instead of `self._task_registry` (F2) — self-free,
  same rule as the composed `train_step`.
- `tests/ogpo/test_split_equivalence.py::test_g_stored_prefix_threading`: 4-tuple unpack,
  asserts the fourth element is `None` (F2).
- `tests/ogpo/test_per_task_critics.py`: fixture imports via
  `importlib.import_module("ogpo.test_split_equivalence")` (F3 — a site-packages package
  named `tests` shadows the repo's).
- `tests/ogpo/test_verifier_alignment.py`: dispatch pin gains the new config (F4).
- `src/training/config.py`: `pi05_libero_online_ogpo_sft_pertask` bakes in the 4-task
  collect/eval set (F7); `num_tasks` comment's memory figure corrected to the BroNet-1024
  recipe path (F6).
- `src/rl/advantage_weighted_sft/update_critic.py`: `_task_subtree_mask` fails closed —
  raises at `tx.init` on any leaf not under `tasks/<t>` (F8); `log_prob` computed once per
  target and shared by the loss and `_per_task_aux` (F11).
- `src/rl/task_registry.py`: `from_json` rejects more tasks than slots (F10).
- `tests/ogpo/test_per_task_critics_verifier.py` (verifier-authored, 21 tests) is kept;
  two of its pins were inverted after the fixes (F7, F10).
- Not fixed: **F1** (task-description collisions in libero_90 — a design decision, see
  `VERIFICATION.md` §2) and F9 (documented gotcha).
- Known-mixed metrics under per-task critics: `value_mean`, `mc_corr`, `grad_norm`,
  `param_norm` stay batch/tree-global; only `loss_task_{t}` / `n_task_{t}` are per-task.

## Slice 2 — critic slots keyed on the task *id* (verifier F1; approved plan in `PLAN.md` §2)

- `src/training/collect.py`: both `sample_actions` calls pass `task_id=list(current_task_ids)`
  (slot-aligned with `info`); `save_episode` passes `task_id=current_task_ids[env_index]`,
  which is still the finished episode's id at that point (the slot is reassigned after).
- `src/rl/filtered_sft_agent/filtered_sft_learner.py`: `save_episode(..., task_id=None)`;
  new `_task_slot(task_id)` (`:732`) — `None` ⇒ raise naming the collect.py fix, else
  `registry.index_for(str(task_id))`; `_save_episode_in_buffer(..., task_id=None)` stamps
  via `_task_slot`; `_generate_actions(..., task_id=None)` accepts and `del`s it (the base
  path does not need it); `end_data_collection(step)` raises when the registry is on,
  `step is not None` (collection, not eval), and `len(registry) != num_tasks`.
- `src/rl/advantage_weighted_sft/advantage_weighted_sft_learner.py`: `save_episode`
  pass-through; best-of-N `sample_actions` requires `kwargs["task_id"]` when the registry is
  on (length-checked against `task_description`) and builds a **per-env** slot array
  `repeat([_task_slot(task_ids[i]) for i in indices], n_samples)` — a description group can
  span two ids. R3 message says "distinct task ids".
- `src/rl/ogpo/ogpo_learner.py`, `src/rl/best_of_n/best_of_n_learner.py`: `save_episode`
  pass-through (BofN: pass-through only).
- `src/rl/task_registry.py`: docstring — key is the id; JSON format unchanged.
- `src/training/config.py`: `num_tasks` comment — keyed on the id; registry must fill after
  the first collection round.
- Tests: `test_collect_data_threads_slot_aligned_task_ids_to_the_agent` (stub env/agent
  through the real `collect_data`), registry tests re-keyed on ids; verifier pins
  `test_registry_keyed_on_task_id_separates_ids_that_share_a_description` and
  `test_libero_90_mt4_task_descriptions_actually_collide` (inverted from `KNOWN_GAP`).
- **Δ** none vs the slice-2 plan.
- Unchanged, recorded: `_success_task_ranges` (`balance_success_buffer_tasks`) and the
  `normalize_advantage_per_task` prompt hash still key on the prompt.
