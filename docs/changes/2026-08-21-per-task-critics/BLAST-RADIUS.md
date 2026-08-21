# Blast radius & change spec — per-task critics for multi-task OGPO

Seeded from `docs/code/rl-ogpo.md`, `rl-learners.md`, `rl-core.md`, `rl-networks.md`,
then **verified against source** on 2026-08-21. Every cite below was read, not
inherited from the docs.

## Intent

`rl.critic.num_tasks = T` replaces the single shared Q/V pair with T disjoint Q/V
pairs, routed per sample by a `task_index` carried in the replay buffer. Q for a
sample of task *t* is produced by task *t*'s parameters only, in the critic update
and in the OGPO advantage alike. `num_tasks = None` reproduces today exactly.

---

## The design, mechanically

### The one structural insight that shrinks this change

The critic backbone is selected as a **pair of closures** at
`advantage_weighted_sft_learner.py:52-63` — BroNet (`:52-62`) or the pi0-backbone
MLP (`:63`, `_build_pi0_backbone_critic_defs`, `best_of_n/update_critic.py:61`) —
and both are handed to `init_state_action_critic_train_state` /
`init_state_value_train_state` as `critic_def`.

So per-task support belongs in a **generic wrapper module that instantiates T copies
of whatever `critic_def` it is given**, not in either backbone. That covers both
backbones with one implementation and requires **zero edits** to
`bronet_critic.py`, `rl_networks.py`, or `networks/decoders/`.

This matters more than it looks: the mt4 recipe's default config
`pi05_libero_online_ogpo_sft` (`config.py:642`) does **not** set `use_bronet`, so it
runs the MLP path — only `pi05_libero_online_ogpo_ref` (`config.py:657`, `:676`)
uses BroNet. A BroNet-only implementation would have missed the arm being targeted.

### Wrapper contract

```
PerTaskStateActionCritic(critic_def, num_tasks, observation, action, rngs)
  __call__(observation, action, training=False) -> (num_qs, B[, bins])
PerTaskStateValue(critic_def, num_tasks, observation, rngs)
  __call__(observation, training=False)         -> (num_vs, B[, bins])
```

- Holds `num_tasks` independent sub-critics built from the same `critic_def`, each
  seeded from its own rng split. **No shared parameters** (D2).
- Reads `observation["task_index"]` — `int32 [B]`.
- Computes all T sub-critics on the full batch → `(T, num_qs, B[, bins])`, then
  gathers along the task axis with the per-sample index → `(num_qs, B[, bins])`.
- **Output shape and semantics are identical to today's**, which is what keeps
  `summarize_critic_values` (`update_critic.py:69`), `critic_values_per_head`
  (`:89`), `create_critic` (`:48`), and all three `advantage_combination` branches
  (`ogpo/update_actor.py:263-300`) untouched.
- Both existing backbones ignore unknown observation keys — BroNet via
  `_obs_vector_keys` (`bronet_critic.py:79-86`), the MLP path via
  `state_vector_keys` (`best_of_n/update_critic.py:80-84`) — so adding `task_index`
  to the critic observation dict is inert for the sub-critics themselves.

**Why the gather is a hard route, not a soft mix:** the gather selects exactly one
task's output per sample, so (a) no other task's parameters appear in the value used
for policy extraction, and (b) the cotangent of every non-selected sub-critic is
exactly zero — gradients are disjoint by construction, with no explicit loss mask.

### Task index plumbing

1. Learner keeps a registry `dict[str, int]`, keyed by `str(task_description)` — the
   **same key** `_success_task_ranges` already uses (`ogpo_learner.py:136`), so BC
   balancing and critic routing agree by construction. Slots assigned in first-seen
   order; overflow past `num_tasks` raises (D4).
2. `_save_episode_in_buffer` (`filtered_sft_learner.py:706`) writes
   `task_index: int32 (n_windows,)` into the insert dict alongside `is_success`
   (`:787`); `_make_buffer_dummy_data` (`:432-450`) declares it. **Gated on
   `num_tasks is not None`** so existing runs' schema is unchanged.
3. It is a **transition** field, not an observation field, so it survives
   `drop_obs_keys=("image","image_mask")` (`replay_buffer.py:170-172`) and is
   present as a top-level key in every sampled batch (`replay_buffer.py:175`).
4. Critic path: `_online_batch_to_critic_batch` (`awr_learner.py:242`) copies it into
   both `observation_dict` and `next_observation_dict` (same episode → same task).
5. Actor path: `Observation.from_dict` drops unknown keys, so OGPO threads it as a
   **sidecar**, exactly as it already does for `critic_prefix` — `_online_batch_to_sft_batch`
   (`ogpo_learner.py:672`) goes from a 3-tuple to a 4-tuple, and
   `sample_and_advantage` gains a `task_index` argument. Inside jit-1 it joins
   `critic_observation` (`ogpo/update_actor.py:186-190`) and is expanded by G through
   the existing `_expand` (`:203-207`).

### Loss reduction (D6)

`train_q_step` (`update_critic.py:266-269`) and `train_value_step` (`:337-340`)
currently reduce with `-jnp.mean(dist.log_prob(target))` over the batch. Under
`num_tasks`, the reduction becomes: per-sample log-probs → per-task sum ÷
`max(count_t, 1)` → mean over tasks **present** in the batch. Gated on
`num_tasks is not None` so AWR/MPO/FlowGRPO/BofN keep the exact current scalar.

Consequence to state in the run ledger: `critic/q_loss` under this reduction is not
numerically comparable to the single-critic arms' series.

### Per-task gradient clipping (D5)

`CriticTrainingConfig.optimizer` is `_optimizer.AdamW(clip_gradient_norm=1.0)`
(`config.py:146`), and `AdamW.create` bakes the clip in as
`optax.chain(optax.clip_by_global_norm(c), adamw)`
(`openpi/src/openpi/training/optimizer.py:81-85`). Under `num_tasks`, the critic
`tx` is built as `optax.chain(*[optax.masked(clip_by_global_norm(c), mask_t) for t],
adamw_core)` — one clip per task subtree.

**Fork boundary:** `openpi/` is a quarantined submodule and must not be edited. The
per-task `tx` is therefore constructed in `update_critic.py`'s init functions
(`:110-147`, `:150-184`) from the same `OptimizerConfig` fields, not by changing
`create_optimizer`. The plan must decide how to obtain `adamw` without its baked-in
clip without duplicating openpi's body — flagged as a plan-phase item below.

---

## Files to touch

| File | Change | Why it is in radius |
|---|---|---|
| `src/rl/networks/per_task_critic.py` *(new)* | `PerTaskStateActionCritic`, `PerTaskStateValue` | The wrapper. New file, no existing consumers. |
| `src/training/config.py:122-158` | `CriticTrainingConfig.num_tasks: int \| None = None` | D7. Reached by all 4 critic learners; default `None` keeps them identical. |
| `src/training/config.py` (~`:636-686`) | New registered config for the per-task OGPO arm | D7. Sits next to `_ogpo_sft` / `_ogpo_ref`; same dataclass, so the `isinstance` dispatch order at `scripts/exp.py:74-85` is untouched. |
| `src/rl/advantage_weighted_sft/advantage_weighted_sft_learner.py:44-99` | Wrap the two `critic_def` closures when `num_tasks` is set; add `task_index` to `dummy_obs` | Critic construction site. |
| ` ” :188-207` | Persist/restore the task registry beside the RL checkpoint | D3. |
| ` ” :242-281` | `task_index` into critic obs + next-obs dicts | Critic batch assembly. |
| ` ” :313-471` | Task index for the best-of-N Q-scoring path; fail fast on unknown task | `sample_actions` ranks with Q when `n_samples > 1`; mt4's `BON_N` env var can turn this on, and `pi05_libero_online_ogpo_ref` sets `n_samples=8` (`config.py:670`). |
| `src/rl/advantage_weighted_sft/update_critic.py:110-184` | Per-task `tx`; wrapper-aware init | D5. |
| ` ” :233-296`, `:303-358` | Per-task loss reduction; per-task metrics | D6. |
| `src/rl/filtered_sft_agent/filtered_sft_learner.py:432-450`, `:706-790` | Buffer schema field + write | D3. **Base class** — see inheritance sweep. |
| `src/rl/ogpo/update_actor.py:101-201` | `task_index` argument on jit-1; into `critic_observation` | Advantage path. |
| ` ” :594` (`train_step` monolith) | Same argument, kept in sync | Subject of the equivalence test. |
| `src/rl/ogpo/ogpo_learner.py:110-142` | Registry assignment at `save_episode` | Where `task_description` is available. |
| ` ” :135-241` | jit-1 `in_shardings` gains the `task_index` slot | **Tier-2 trigger.** |
| ` ” :530-546`, `:672-680` | 4-tuple sidecar; pass through to jit-1 | Actor call site. |
| `scripts/ogpo_multitask_4task.sh:121-146` | `PER_TASK_CRITIC` / `NUM_TASKS` env-var knob | The recipe stack is how OGPO is actually driven. |
| `tests/ogpo/*` | New tests; repair the jit-1 signature coupling | See verification plan. |
| `docs/code/rl-ogpo.md`, `rl-learners.md`, `rl-core.md`, `training.md`, `scripts.md` | Scoped section updates (step 4) | Same radius. |

---

## Duplication sweep (CLAUDE.md rule 1) — result: TWO FAMILIES REACHED, ONE DELIBERATELY NOT

Grepped every clone family named in CLAUDE.md.

1. **`advantage_weighted_sft/update_critic.py` ↔ `best_of_n/update_critic.py`
   (eleven near-identical helpers).** Verified with
   `grep -rn "init_state_action_critic_train_state\|train_q_step\|create_critic\|summarize_critic_values" src/ tests/`:
   the two copies have **disjoint consumers**. `best_of_n_learner.py:21-25` imports
   only from its own copy; AWR/MPO/FlowGRPO/OGPO import only from the AWR copy.
   **Decision: change the AWR copy only.** BofN is critic-only with a frozen policy —
   per-task critics have no meaning there, and `num_tasks=None` leaves it inert.
   *This is a deliberate divergence and must be recorded in `DIFF.md`.*
2. **`_build_pi0_backbone_critic_defs` is genuinely shared** — AWR imports it *from*
   `best_of_n/update_critic.py` (`awr_learner.py:30`). It is **not modified**; the
   wrapper composes it from outside. Confirmed no edit needed.
3. **`_pad_last_dim` + the ~130-line best-of-N scoring block**
   (`awr_learner.py:304-471` ↔ `best_of_n_learner.py`). The AWR copy **is** in radius
   (it calls the Q model directly at `awr_learner.py:466`). The BofN copy is not,
   per (1). Divergence recorded.
4. **`_get_on_policy_action`** (BofN ↔ MPO), **`init_train_state`** (filtered_sft ↔
   `dsrl_env.py`): not touched — neither reads the critic obs or the buffer schema.
5. **The ~60-line env preamble `stability_study.sh` ↔ `ogpo_multitask_4task.sh`
   (already diverged).** Only the mt4 script gets the new knob; the preamble itself
   is untouched, so the existing divergence neither widens nor is "helpfully" fixed.

## Inheritance sweep (CLAUDE.md rule 2) — result: BASE-CLASS EDIT, 5 SUBCLASSES REACHED

Verified chain (`grep -rn "^class .*Learner" src/rl/*/*.py`):

```
FilteredSFTLearner
├── AdvantageWeightedSFTLearner
│   ├── MPOWeightedSFTLearner → FlowGRPOLearner
│   └── OGPOAgentLearner
└── BestofNLearner
```

- The buffer-schema edit is in **`FilteredSFTLearner`** (`:432-450`, `:706-790`) and
  therefore reaches **all five subclasses**. It must be gated so every other config
  keeps today's schema.
  **Verified constraint on how that gate is written:** `critic: CriticTrainingConfig`
  hangs off `AdvantageWeightedSFTLearnerConfig` (`config.py:166`) and, separately,
  off `BestofNLearnerConfig` (`config.py:189`). A plain `FilteredSFTLearnerConfig`
  has **no `rl.critic` at all** (`config.py:160-165`). So the base class **cannot**
  gate on `self._config.rl.critic.num_tasks` — and the adjudicated rule forbids
  reaching for `getattr(cfg, "x", default)` to paper over it.
  **Recommendation:** the base reads an *instance* attribute the subclass sets —
  the pattern already used for `_prefix_embed_dim` (`awr_learner.py:64-67`) and
  `_buffer_obs_drop_keys` (`best_of_n_learner.py:80-82`). `FilteredSFTLearner`
  declares it off by default; `AdvantageWeightedSFTLearner.__init__` sets it from
  `config.rl.critic.num_tasks` before calling `super().__init__(config)`. Config
  access stays direct and stays where the field actually exists.
- `BestofNLearner` overrides `_make_buffer_dummy_data` (`:204-210`) and declares
  `_buffer_obs_drop_keys` (`:80-82`) — both operate on the *observations* subtree, so
  a transition-level `task_index` passes through untouched. Verified.
- **`update()` overrides** (OQ-10 — those own their own EMA advance):
  `filtered_sft:812`, `awr:613`, `bofn:590`, `flow_grpo:22`, `ogpo:344`.
  Only `ogpo:344` is edited. `MPOWeightedSFTLearner` does **not** override `update()`
  — it inherits AWR's, so the AWR critic-batch edit (`:242-281`) reaches MPO and
  FlowGRPO. Contained by the same gate.
- `flow_grpo/update_actor.py:39-105` and `advantage_weighted_sft/update_actor.py:51-72`
  call `create_critic` + `summarize_critic_values` on a critic obs **without** a
  `task_index`. Under `num_tasks=None` they are unchanged. If one of them were ever
  run with `num_tasks` set, the wrapper must **raise on a missing `task_index` key**
  rather than defaulting to slot 0 (D4).

## Tier-2 triggers, itemized

| Trigger | Present? | Detail |
|---|---|---|
| Jit signature / sharding annotation | **YES** | `sample_and_advantage` gains a `task_index` parameter; `ogpo_learner.py:155-178` gains a matching `in_shardings` slot (`self._data_sharding`, mirroring `critic_prefix`). |
| `donate_argnums` | No | jit-1 is `donate_argnums=()`. Nothing donated changes. Invariant G2 (`ogpo_learner.py:485-488`) untouched. |
| RNG split arity | **No — explicitly** | The arity-3 split with two dead slots (`update_actor.py:128-132`) is **not** touched. The critic init adds `num_tasks` sub-critic seeds, which is a **construction-time** split (`awr_learner.py:80`) — it consumes `self._rng` differently at init and therefore shifts the training keystream for the new config only. Must be stated, not hidden. |
| Replay-buffer schema | **YES** | New transition field. `insert()` raises `ValueError("Insert transition structure does not match buffer structure")` on a treedef mismatch (`replay_buffer.py:91-93`), so `restore_shards` (`:255`) of a pre-change shard directory fails loudly against a `num_tasks` config — the intended fail-fast per D8. |
| `TrainState` / checkpoint layout | **YES** | Critic param tree gains a task level. Old `rl_state` will not structure-match (`awr_learner.py:201-207`) — intended (D8). Plus the registry sidecar (D3). |
| Config dataclass hierarchy | **YES** | New field on `CriticTrainingConfig`; new registered config. `isinstance` dispatch order (`exp.py:74-85`) unchanged — the new config is an `OGPOSFTLearnerConfig`. |
| ≥2 learner packages | **YES** | `filtered_sft_agent/`, `advantage_weighted_sft/`, `ogpo/`. |

## Gotchas checked (CLAUDE.md rule 3)

Read the Gotchas sections of `rl-ogpo.md`, `rl-learners.md`, `rl-core.md`,
`rl-networks.md`, plus `tests/ogpo/OGPO_DEBUG_LOG.md`:

- **`isinstance` dispatch order** — new config subclasses `OGPOSFTLearnerConfig`, so
  the OGPO-first branch still wins. No change.
- **Overriding `update()` owns the EMA advance** — `ogpo:344` already advances it
  (`:593-601`); this change adds no early return and no new branch that could skip it.
- **`del ema_dev` at `ogpo_learner.py:505` is load-bearing** — the `task_index`
  sidecar is a small int array threaded *alongside* `critic_prefix`; it must not be
  captured in any closure that outlives jit-1. No new device-resident handle.
- **`_adv_scale` is not checkpointed** — the direct precedent for D3. The registry
  **is** checkpointed for exactly this reason.
- **Success buffer is built by temporarily swapping `self._config`**
  (`ogpo_learner.py:88-90`) — anything reading config during
  `_get_online_replay_buffer` sees the swapped value. `num_tasks` is on
  `rl.critic`, which the swap (`buffer_capacity` on `rl`) does not alter, so the
  success buffer gets the same schema. Verified, but the plan should assert it.
- **`sample_actions` is inherited from AWR** — with `n_samples > 1` an OGPO run
  silently uses best-of-N collection. That path scores with Q and therefore needs a
  task index; `pi05_libero_online_ogpo_ref` sets `n_samples=8`. In radius.
- **Observation preprocessing order is domain-specific and replicated in three
  places** — untouched; `task_index` is not an observation the transforms see.
- **Molmo transform ordering is positional** (`filtered_sft_learner.py:373-383`) —
  untouched.
- **`replay_buffer.save_shard` non-determinism** (`:200`) — unchanged; noted only
  because the new field rides in the same shard.

---

## Residual decisions for the plan phase

Not blocking discovery; each needs a call before implementation.

1. **How to build AdamW without openpi's baked-in clip** (D5). Options: read the
   `AdamW` dataclass fields and construct `optax.adamw` locally in
   `update_critic.py`; or chain the per-task clips *before* `create_optimizer`'s
   output and accept a redundant global clip at a raised threshold. The first is
   correct, the second is smaller. **Recommendation: the first**, with a comment
   citing `openpi/.../optimizer.py:81-85` as the source it mirrors.
2. **Registry persistence format** (D3). `_rl_checkpoint_state()` goes through
   `ocp.StandardCheckpointer` (`awr_learner.py:104`, `:188-199`), which expects
   array-shaped leaves — a `dict[str, int]` is not one. **Recommendation:** a JSON
   sidecar `task_registry.json` written into `_rl_checkpoint_dir()` next to the step
   directories, loaded before the critic is built so slot assignment is stable
   across a resume. Restoring a registry whose task set does not match
   `collect.tasks` must raise.
3. **`num_tasks` vs `collect.tasks`.** `num_tasks` is an explicit int (D7), but
   `collect.tasks` is already resolved and de-duplicated at config time
   (`config.py:395-431`; note `expand_tasks` can emit duplicates via the `xN`
   multiplier, so compare against `len(set(...))`). **Recommendation:** assert at
   construction that `num_tasks >= len(set(config.collect.tasks))` and fail with the
   two numbers in the message — catches the common misconfiguration before an hour
   of collection rather than at the first `save_episode`.
4. **Optimizer drift on tasks absent from a batch.** With one shared `tx`, every
   task's params take an AdamW step every update — including a task with zero
   gradient that step, whose momentum tail and weight decay still move it. This is
   the one residual coupling the chosen packaging cannot remove. At `batch_size=256`
   (`config.py:497`) and T=4 with uniform sampling every task is present in
   essentially every batch, so **recommendation: accept, and log per-task sample
   counts** so an actually-absent task is visible rather than silent. The plan should
   state this explicitly rather than leave it implied.
5. **Per-task metrics.** `critic/q_loss` etc. are currently scalars. Emitting
   per-task `loss` / `mc_corr` / `value_mean` is cheap and is the only way the
   above is observable. Scope it in or out deliberately — `exp.py` NaN-fills missing
   keys (`EXP:159-163`), so sparsely-emitted keys are safe.

## Memory & compute

Analytic estimates — **not measured**, no GPU run was made.

- **Params.** T× the critic parameter count, plus T× Adam `m`/`v` and T× the critic
  EMA (`use_ema=True`, `update_critic.py:203-214`) ⇒ ~4× the param bytes per task.
  - mt4 default (`pi05_libero_online_ogpo_sft`, MLP path, `NUM_QS=2`, encoder
    `(512,512)`, decoder `(256,256)`): order 4M params total today ⇒ ~64 MB with
    optimizer+EMA; **T=4 ⇒ ~0.26 GB. Negligible.**
  - `pi05_libero_online_ogpo_ref` (BroNet, hidden 1024, depth 2, `num_qs=num_vs=10`):
    order 140M params ⇒ ~2.3 GB today; **T=4 ⇒ ~9 GB.** Against a ~34.7–46 GiB
    documented floor (`rl-ogpo.md`, `ogpo_learner.py:61-71`) this is **not**
    negligible and must be measured before an unfrozen-backbone arm.
- **Compute.** T× critic forward+backward per critic step. The digestion burst runs
  `post_collection_critic_steps` (mt4 default **1000**) critic steps per collection
  round, and `critic_utd` multiplies the per-trainer-step count — so the T× lands
  squarely on the burst's wall-clock. Cheap on the MLP path, material on the
  BroNet/1024-batch ref path.

## Expected behavior after

- `num_tasks=None`: byte-identical behavior for all six learners. This is the
  primary claim the verification must establish.
- `num_tasks=T` on the OGPO multitask arm: each task's Q/V trained only on its own
  transitions with its own clip and its own loss normalization; each sample's
  advantage computed from its own task's Q/V; unknown task ⇒ raise; resume from a
  pre-change checkpoint or buffer ⇒ raise.

## Verification plan (step 3, independent agent)

Pure-math / pure-plumbing, so **pytest-native tests are required**, not a fallback:

1. **Routing test.** A `PerTaskStateActionCritic` over T sub-critics with
   deliberately different initializations: assert the output for a batch of mixed
   `task_index` equals, row by row, the output of the corresponding standalone
   sub-critic on that row. This is the "never use another task's critic" claim (D2).
2. **Gradient-disjointness test.** Backprop a loss on a batch containing only task
   *t*: assert every other sub-critic's gradient tree is exactly zero, and task *t*'s
   is exactly the standalone critic's gradient. This is the "no shared params" claim.
3. **Differential test against the pre-change reduction.** `train_q_step` /
   `train_value_step` are numeric code being refactored, so per CLAUDE.md this needs
   a differential test against a **verbatim copy** of the current implementation —
   the `tests/ogpo/test_split_equivalence.py:142` pattern. With `num_tasks=None` the
   new reduction must reproduce the old scalar bit-for-bit; with T=1 and all samples
   in slot 0 it must reproduce it up to a stated tolerance.
4. **Per-task loss reduction test.** Hand-built per-sample losses and task indices:
   assert per-task mean → mean over present tasks, and that an absent task
   contributes exactly zero (not NaN) — the `max(count,1)` guard (D6).
5. **Per-task clip test.** Two tasks, one with a deliberately huge gradient: assert
   the small-gradient task's update is unchanged from a standalone run (D5). This is
   the whole point of the decision and is the leg most likely to be quietly wrong.
6. **Registry tests.** First-seen assignment; overflow raises with the task name and
   `num_tasks` in the message; save→load round-trip preserves the mapping (D3, D4).
7. **jit-1 signature.** `tests/ogpo/test_split_equivalence.py` compares the split
   against `_reference_mono_train_step` (`:148`), a frozen verbatim monolith, and
   asserts its info is exactly 33 keys (`:62-64`). The new `task_index` parameter
   must be added such that all 6 legs still pass — the plan should default it to
   `None` and keep the `None` branch identical, mirroring how `critic_prefix` is
   handled (`update_actor.py:150-159`, pinned by the `:573` leg).
8. **Regression gate.** `pytest tests/ogpo` (23 tests, CPU, seconds) must stay green.
   `pytest tests/` still fails at collection (`tests/nnx_networks/…:6-7`, OQ-4) — out
   of scope, not touched by this change.
9. **Not verifiable locally, and must be reported as such:** the buffer write path,
   collection, the burst, per-task memory, and anything needing real π0.5 weights or
   a simulator. Honest fallback: a CPU smoke on dummy variants plus
   `DRY=1 bash scripts/ogpo_multitask_4task.sh` to inspect the emitted command line.
