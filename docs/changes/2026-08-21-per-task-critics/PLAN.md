# Plan — per-task critics for multi-task OGPO (Tier 2, implementation phase)

Built from `docs/changes/2026-08-21-per-task-critics/{README,BLAST-RADIUS}.md`. Decisions
D1–D8 there are settled inputs; this plan resolves the five "residual decisions" and
sequences the work. **First action of implementation: copy this file verbatim to
`docs/changes/2026-08-21-per-task-critics/PLAN.md`**, then load
`docs/code/best_practices.md` before any source edit.

## Context

The 4-task OGPO arm (`scripts/ogpo_multitask_4task.sh`) trains one shared Q ensemble and
one shared V ensemble on a buffer that interleaves four tasks with different return
scales. The stability study's dominant failure is the critic *mis-ranking* fresh actions;
a shared critic adds a permanent cross-task source of that mis-ranking, and the existing
multi-task knobs (`normalize_advantage_per_task`, `balance_success_buffer_tasks`) act
downstream of the critic, not on it. This change gives each task its own Q and V — no
shared parameters, and a task's advantage computed only from its own critics — driven by
one config field, `rl.critic.num_tasks`. `None` (default) is today's path, bit-identical.

## Resolved residual decisions (from BLAST-RADIUS §"Residual decisions")

| # | Decision |
|---|---|
| R1 | **AdamW without openpi's baked-in clip:** build `optax.adamw(lr, b1, b2, eps, weight_decay, mask=None)` locally in `update_critic.py` from the `_optimizer.AdamW` dataclass fields (`openpi/src/openpi/training/optimizer.py:66-85`), chained after T `optax.masked(clip_by_global_norm(c), mask_t)`. Assert `isinstance(optimizer, AdamW)` — SGD has no clip and the per-task path mirrors `AdamW.create` only. openpi is not edited. |
| R2 | **Registry persistence:** JSON sidecar `task_registry_<step>.json` in `_rl_checkpoint_dir()` (next to, not inside, the orbax step dir). Written in `save_checkpoint`, read in `_restore_rl_checkpoint`; missing on a `num_tasks` resume ⇒ raise (D8). |
| R3 | **`num_tasks` vs `collect.tasks`:** assert at construction `num_tasks == len(set(config.collect.tasks))` (`==`, not `>=`: under D8 there is no legitimate spare slot, and a spare slot would let a held-out eval task silently claim an untrained critic). If `rl.n_samples > 1`, additionally assert `set(eval_tasks) ⊆ set(tasks)` — best-of-N scoring on a held-out task can never have a critic. Both messages carry the two numbers/sets and the fix. |
| R4 | **Absent-task optimizer drift:** accepted; made visible by per-task sample counts in the critic info (below). Stated in `DIFF.md`. |
| R5 | **Per-task metrics:** in scope, minimal — `loss_task_{t}` and `n_task_{t}` for both Q and V steps. Keys are static for a given T; `exp.py` NaN-fills missing keys so cadence is safe. |

Two routine calls, flagged for the approval read: new registered config name
`pi05_libero_online_ogpo_sft_pertask` (same values as `_sft` + `critic=CriticTrainingConfig(num_tasks=4)`);
script env var `PER_TASK_CRITIC` (default 0).

## Design in one paragraph

A generic wrapper instantiates T copies of whatever `critic_def` closure the learner
already selects (BroNet or pi0-MLP, `advantage_weighted_sft_learner.py:52-63`), runs all
T on the batch and gathers per sample by `observation["task_index"]`, returning today's
`(num_heads, B[, bins])` shape — so `create_critic`, `summarize_critic_values`,
`critic_values_per_head`, all three advantage branches, the burst, `critic_utd` and
success-oversample are untouched. `task_index` is a new **transition** field in the replay
buffer (survives `drop_obs_keys`), assigned by a host-side first-seen registry keyed on the
same `str(task_description)` that `_success_task_ranges` uses. It reaches the critic via
the critic-batch obs dicts and reaches jit-1 as a second sidecar next to `critic_prefix`.

## Implementation steps (in order)

### 1. `src/rl/task_registry.py` (new, pure)
`TaskRegistry(num_tasks)`: `index_for(description) -> int` (first-seen assignment; full ⇒
`ValueError` naming the task, the registered set and `num_tasks`, and the fix), `__len__`,
`to_json(path)`, `from_json(path, num_tasks)` (mismatched `num_tasks` ⇒ raise), `tasks`
view. No learner dependency — directly unit-testable.

### 2. `src/rl/networks/per_task_critic.py` (new)
- `PerTaskStateActionCritic(critic_def, num_tasks, observation, action, rngs)` and
  `PerTaskStateValue(critic_def, num_tasks, observation, rngs)`.
- `self.tasks = [critic_def(obs_wo_task_index, ..., nnx.Rngs(k_t)) for t]` with
  `k_t = jax.random.split(rngs_key, num_tasks)[t]` — disjoint params, own seed each.
- `__call__`: `idx = observation["task_index"]` (explicit raise with message if the key is
  missing — never default to slot 0); `outs = stack([net(obs, ...) for net in self.tasks])`
  → `(T, n, B[, bins])`; `take_along_axis` on the T axis with `idx` → `(n, B[, bins])`.
- Both backbones ignore the extra key (`bronet_critic.py:79-86`,
  `best_of_n/update_critic.py:80-84` via `state_vector_keys`), so `obs` is passed through
  as-is; construction strips `task_index` only to build the dummy.

### 3. `src/training/config.py`
- `CriticTrainingConfig.num_tasks: int | None = None` (`:122-158`) with a comment block in
  the existing style (what it does, D2 shape, memory note, "None = today").
- Register `pi05_libero_online_ogpo_sft_pertask` next to `_sft` (`:641-648`) — identical
  except `critic=CriticTrainingConfig(num_tasks=4)`; comment points at the change record.
  `isinstance` dispatch (`scripts/exp.py:74-85`) unchanged.

### 4. `src/rl/filtered_sft_agent/filtered_sft_learner.py` (base — gated)
- Class attributes `_num_critic_tasks: int | None = None`, `_task_registry: TaskRegistry | None = None`
  (instance-attribute gate pattern, as `_prefix_embed_dim` / `_buffer_obs_drop_keys`; a
  plain `FilteredSFTLearnerConfig` has no `rl.critic`, and `getattr` defaults are banned).
- `_make_buffer_dummy_data` (`:432-450`): add `"task_index": np.zeros((1,), np.int32)` when
  `_num_critic_tasks is not None`.
- `_save_episode_in_buffer` (`:706-790`): when gated on, `idx = self._task_registry.index_for(str(task_description))`
  and insert `"task_index": np.full((n_windows,), idx, np.int32)` beside `is_success` (`:787`).
  Idempotent for OGPO's double call (success buffer then online).

### 5. `src/rl/advantage_weighted_sft/update_critic.py`
- `_per_task_mean(per_sample, task_index, num_tasks) -> scalar`: one-hot `(B,T)`; per-task
  sum ÷ `max(count,1)`; mean over heads; mean over **present** tasks (`max(n_present,1)`).
  Pure, module-level, tested directly.
- `_critic_optimizer(config)`: `num_tasks is None` ⇒ exactly today's
  `_optimizer.create_optimizer(...)`; else R1 chain. Task mask `t` = callable over the
  param tree (`jax.tree_util.tree_map_with_path`; the leading path entries `tasks/<t>`
  identify the subtree) — pinned by a test that the mask is True exactly on task t's leaves.
- `init_state_action_critic_train_state` / `init_state_value_train_state` (`:110-184`):
  use `_critic_optimizer`; wrap `critic_def` in the per-task wrapper when `num_tasks` set.
- `train_q_step` (`:266-269`) / `train_value_step` (`:337-340`): keep `-jnp.mean(...)`
  verbatim when `num_tasks is None`; else `-_per_task_mean(dist.log_prob(...), obs["task_index"], T)`
  for both td and mc terms, plus R5 metrics. Existing scalar metrics unchanged.
- `best_of_n/update_critic.py` is **deliberately not changed** (disjoint consumers;
  critic-only learner) — record in `DIFF.md`.

### 6. `src/rl/advantage_weighted_sft/advantage_weighted_sft_learner.py`
- `__init__` before `super().__init__`: set `_num_critic_tasks`, `_task_registry`; add
  `task_index` to `dummy_obs` (`:46-50`); R3 asserts; class attr
  `_supports_per_task_critics = False` (OGPO sets `True`) — assert when `num_tasks` set,
  since the AWR/MPO/FlowGRPO actor paths build critic obs without `task_index`.
- `_rl_checkpoint_state`/`save_checkpoint`/`_restore_rl_checkpoint` (`:188-207`): R2 sidecar.
- `_online_batch_to_critic_batch` (`:242-281`): copy `online_batch["task_index"]` into both
  obs dicts when gated on.
- `sample_actions` best-of-N (`:313-471`): `critic_obs["task_index"] = full(group_env_num*n_samples, index_for(task))`
  at `:395` when gated on. Note first-seen assignment here is safe: step-0 collection
  registers every train task before any eval; a held-out task overflows ⇒ raise (D4/R3).
- Confirm the success-buffer config swap (`ogpo_learner.py:88-90`) leaves `rl.critic`
  untouched — assert in a test, not in prose.

### 7. `src/rl/ogpo/update_actor.py`
- `sample_and_advantage` (`:132`): trailing parameter `task_index: at.Int[at.Array, " b"] | None = None`
  (trailing + default keeps the 7-positional test calls and the arity-3 RNG split untouched);
  when not None, add to `critic_observation` (`:186-190`) before `_expand` (`:203-207`).
- Composed `train_step` (`:659-697`): read `critic_observation.get`-free — `task_index = critic_observation["task_index"] if "task_index" in critic_observation else None`, pass through.

### 8. `src/rl/ogpo/ogpo_learner.py`
- `_supports_per_task_critics = True`.
- jit-1 `in_shardings` (`:155-178`): append `self._data_sharding` slot for `task_index`
  (None-passthrough like `critic_prefix`). **This is the Tier-2 signature edit.**
- `_online_batch_to_sft_batch` (`:672`): 3-tuple → 4-tuple, fourth = `online_batch["task_index"]`
  when gated on else `None`; update the two unpack sites (`:504`, `:533`) and the jit-1 call
  (`:530-546`) to pass it. Nothing new is captured across the `del ema_dev` boundary (`:505`).

### 9. `scripts/ogpo_multitask_4task.sh` (`:121-146`)
`PER_TASK_CRITIC="${PER_TASK_CRITIC:-0}"`; emit **unconditionally** (authoritative pattern
of the NUM_QS block): `=1` ⇒ `--rl.critic.num_tasks ${#TASKS[@]}`, else `--rl.critic.num_tasks None`.
Document in the header comment.

### 10. Tests — `tests/ogpo/test_per_task_critics.py` (new) + touch-ups
Dummy variants / module-scoped fixtures per `tests/ogpo/` convention, stated tolerances.
1. Routing: mixed-task batch output == standalone sub-critic output row-by-row (D2).
2. Gradient disjointness: loss on task-t-only batch ⇒ other tasks' grads exactly 0, task t's == standalone (no shared params).
3. **Differential vs verbatim copy** of current `train_q_step`/`train_value_step`
   (`test_split_equivalence.py:142` pattern): `num_tasks=None` bit-exact; T=1 all-slot-0 within tolerance.
4. `_per_task_mean`: hand-built losses; absent task contributes exactly 0, not NaN (D6).
5. Per-task clip: huge-gradient task A leaves task B's update identical to standalone (D5); mask pinned.
6. `TaskRegistry`: first-seen, overflow message contents, JSON round-trip, `num_tasks` mismatch raise.
7. jit-1: `task_index=None` ⇒ identical outputs to the 7-arg call; `test_split_equivalence.py` all 6 legs unchanged.
8. Config: tyro parses `--rl.critic.num_tasks None` and `4`; `_pertask` config registered and dispatches to OGPO.
9. Success-buffer config swap preserves `rl.critic.num_tasks`.

### 11. Change record + docs (step 4)
`DIFF.md` (incl. the BofN non-change, the init-rng keystream shift for the new config only,
R4), `VERIFICATION.md` (verbatim agent findings). Docs sections only: `rl-ogpo.md`
(jit-1 signature, sidecar, gotchas), `rl-learners.md` (critic init, buffer field, BofN
path), `rl-core.md` (buffer schema), `rl-networks.md` (wrapper), `training.md` (field +
config), `scripts.md` (env var). Repair only the cites those sections carry.

## Verification (step 3, independent Opus agent, xhigh, spec + diff only)
- `pytest tests/ogpo` stays green (23 existing + new file). `pytest tests/` still fails at
  collection (OQ-4) — out of scope.
- `DRY=1 GPU=0 PER_TASK_CRITIC=1 bash scripts/ogpo_multitask_4task.sh` shows
  `--rl.critic.num_tasks 4`; `PER_TASK_CRITIC=0` shows `None`.
- **Cannot be verified locally and must be reported as such:** buffer write path under
  real collection, the burst, per-task memory (analytic: ~0.26 GB at T=4 on the MLP path,
  ~9 GB on the BroNet `_ref` path), anything needing π0.5 weights or a simulator.
- No run is launched without explicit permission.

---

# Plan — per-task critics, slice 2: key critic slots on the task *id* (F1)

Continuation of the approved per-task-critics change (`docs/changes/2026-08-21-per-task-critics/`;
the original plan is preserved there as `PLAN.md`). This slice resolves verifier finding F1.
Discovery is recorded in `BLAST-RADIUS.md` §"Addendum — F1". Tier 2: it sits on the
`Agent.save_episode` / `sample_actions` call contract.

## Context

The `TaskRegistry` keys critic slots on the LIBERO *language string* — the only task
identity the learner receives. `libero_90_79` and `libero_90_82` in the mt4 set share
that string (12 libero_90 prompts collide), so `PER_TASK_CRITIC=1` silently trained 3
critics for 4 tasks. Maintainer decision: key slots on the **task id** (`libero_90_79`),
which `collect.tasks`, the R3 count, and the per-task eval metrics already use. After
this slice, a per-task run has exactly one critic per `collect.tasks` entry, and a
registry that does not fill after the first collection round raises.

## Design

`collect.py` already holds the slot-aligned `current_task_ids`; it just never hands it
to the agent. Thread it as a new `task_id` keyword through the existing `**kwargs`
contract, make the registry key on it, and fail fast wherever the registry is on but
the id is absent. `task_description` keeps every existing use (prompting, prompt
embedding grouping, `_success_task_ranges`).

## Steps

1. **`src/training/collect.py`** — `collect_data` and `evaluate_policy`: pass
   `task_id=current_task_ids` (the list; slot-aligned with `info`) to both
   `agent.sample_actions(...)` calls, and `task_id=current_task_ids[env_index]` to
   `agent.save_episode(...)` (`:190`, which runs *before* the slot is reassigned at `:201`).
2. **`src/rl/filtered_sft_agent/filtered_sft_learner.py`**
   - `save_episode(self, is_success, env_index, task_description, task_id=None)` →
     `_save_episode_in_buffer(..., task_id=task_id)`.
   - `_save_episode_in_buffer(..., task_id=None)`: when `self._task_registry is not None`,
     `task_id is None` ⇒ `ValueError` naming the fix (collect.py must pass it); slot =
     `self._task_registry.index_for(str(task_id))`. Description is no longer a key.
   - `_generate_actions(observations, task_description, task_id=None)`: accepted and
     unused here (consumed by the best-of-N scoring override); comment says so, so the
     kwarg from `collect.py` does not `TypeError` on the base path.
   - `end_data_collection(step)`: when the registry is on and `step is not None`
     (collection — `evaluate_policy` calls it without a step), raise if
     `len(registry) != num_tasks`, listing registered vs. `collect.tasks`. This is the
     "registry fills after step-0 collection" guard; it also catches a future id↔slot drift.
3. **`src/rl/advantage_weighted_sft/advantage_weighted_sft_learner.py`**
   - `save_episode(..., task_id=None)` passes through.
   - best-of-N `sample_actions`: `task_ids = kwargs.get("task_id")` is NOT used as a
     silent default — read `kwargs["task_id"]` only when the registry is on (raise with
     the fix if missing), and build a **per-env** index
     `jnp.repeat(jnp.asarray([index_for(task_ids[i]) for i in indices]), n_samples)` —
     a description group may span two ids.
4. **`src/rl/ogpo/ogpo_learner.py`** `save_episode(..., task_id=None)`: pass through to
   both `_save_episode_in_buffer` (success buffer) and `super().save_episode`.
5. **`src/rl/best_of_n/best_of_n_learner.py`** `save_episode(..., task_id=None)`:
   pass-through only (per-task critics are config-blocked there) — keeps the AWR↔BofN
   clone pair in step.
6. **`src/rl/task_registry.py`**: docstring — key is the task id; JSON format unchanged.
7. **`src/rl/advantage_weighted_sft/advantage_weighted_sft_learner.py` R3 message** and
   the `config.py` `num_tasks` comment: say "distinct task ids".
8. **Tests**
   - `tests/ogpo/test_per_task_critics.py`: a stub-agent `collect_data` test (fake
     `BaseVectorEnv` with 2 envs, 2 tasks sharing one description) asserting
     `save_episode` receives each slot's id and `sample_actions` receives the aligned list;
     `TaskRegistry` keyed on two ids with equal descriptions → distinct slots.
   - `tests/ogpo/test_per_task_critics_verifier.py`: invert the two `KNOWN_GAP` collision
     pins into "ids with a shared description get distinct slots".
   - Re-run the fast legs of both files; `test_g_stored_prefix_threading` unaffected.
9. **Record + docs**: `DIFF.md` (slice 2 section), `VERIFICATION.md` (F1 → fixed, with
   the new test output), `rl-learners.md` (`save_episode`/`sample_actions` kwarg,
   `end_data_collection` guard), `rl-core.md` (`task_registry.py` key), `training.md`
   (`collect.py` passes `task_id`), drop the F1 gotcha from `rl-networks.md`.

## Sequencing constraint

Slurm job `10187013` (`scripts/run_ogpo_tests.sbatch`) is running the full suite on the
current working tree. **No source edit until it finishes** (it imports modules lazily;
editing mid-run would test a mixed tree). Implement after its log shows the summary line.

## Verification

- Fast legs of both per-task test files (`-k "not jit1"`), plus the new collect stub test.
- `DRY=1` recipe inspection unchanged (no script change in this slice).
- Not locally verifiable: a real collection round (LIBERO env) exercising
  `end_data_collection`'s registry guard — read-verified; the stub test covers the
  `collect.py` → `save_episode` plumbing.
- Step-3 verifier: a short follow-up pass on this slice's diff (same agent type), findings
  appended to `VERIFICATION.md`.
