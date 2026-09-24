# Change spec — cross-topology (`fsdp_devices`) resume

Tier 2. Discovery pass, 2026-09-21. Every claim below was verified against source or
by a CPU-only probe; line numbers were re-read, not taken from the docs.

## Intent

On resume, restore the main policy `train_state` onto the **current run's** mesh
instead of the checkpoint's saved device layout, so a run checkpointed at
`fsdp_devices=N` resumes at `fsdp_devices=M` for any `M` the allocation supports.
Bit-identical behavior when `M == N`.

## Verified mechanism

1. `self._mesh = sharding.make_mesh(self._config.fsdp_devices)`
   (`src/rl/filtered_sft_agent/filtered_sft_learner.py:261`). `make_mesh` builds
   `(jax.device_count() // n, n)` over `(BATCH_AXIS, FSDP_AXIS)` and raises on
   non-divisibility (`openpi/src/openpi/training/sharding.py:17-23`).
2. `init_train_state` computes the correct target and discards it on resume:
   ```
   198    train_state_shape = jax.eval_shape(init, init_rng)
   199    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)
   200
   201    if resume:
   202        return train_state_shape, state_sharding
   ```
   `jax.eval_shape` leaves carry `.sharding is None` unless the traced fn declared
   `out_shardings` (`jax/_src/pjit.py:400-407`), so `train_state_shape` is
   sharding-free. `:199` is where the "Sharding X of shape Y along axis N" log lines
   come from (`sharding.py:88-93`).
3. `filtered_sft_learner.py:315-317` binds both returns, then `:319-324` passes only
   `self._train_state` to `_checkpoints.restore_state` — `self._train_state_sharding`
   is never forwarded.
4. `restore_state` calls `checkpoint_manager.restore(step, items={"train_state": …,
   "params": {"params": …}})` with no `restore_kwargs`
   (`openpi/src/openpi/training/checkpoints.py:97-107`). Handlers are plain
   `ocp.PyTreeCheckpointHandler()` (`checkpoints.py:40-46`).
5. **The item's shardings are never read on this path.** With `restore_args=None`,
   `_fill_missing_save_or_restore_args` fills a bare `RestoreArgs()` for every leaf
   (`orbax/checkpoint/_src/handlers/base_pytree_checkpoint_handler.py:246-252`,
   `:746-748`). Only `StandardCheckpointHandler` derives restore args from its item
   (`standard_checkpoint_handler.py:218-224`, via `construct_restore_args`).
6. Every leaf therefore falls into the `_sharding`-file fallback
   (`orbax/checkpoint/_src/serialization/type_handlers.py:1234-1279`), which
   reconstructs the saved `NamedSharding` by **device id**
   (`orbax/checkpoint/_src/metadata/sharding.py:124-133`). Device ids are serialized
   for all device kinds in 0.11.13, not just TPU (`sharding.py:88-110`).

### Corrections to the briefing / prior records

- **Line numbers.** The relevant `filtered_sft_learner.py` lines are **198 / 199 /
  201-202 / 261 / 315-317 / 319-324 / 369 / 515-518**, not the 202/203/205-206/371
  that one sweep reported. Verified by direct read.
- **The device-mismatch ValueError is swallowed.** `get_sharding_or_none`
  (`orbax/checkpoint/_src/metadata/sharding.py:354-358`) catches
  `"The available devices are different from the devices used to save the
  checkpoint…"` into a bare `logging.error` and returns `None`. The *raised* error is
  the generic one at `orbax/checkpoint/_src/serialization/serialization.py:526-530`
  (`"sharding passed to deserialization should be specified, concrete and an instance
  of jax.sharding.Sharding. Got None"`) — which is exactly what job 10324589 printed.
  The informative message is log-only.
- **The briefing's "1→2 GPU resumes silently onto a wrong sharding" is wrong.**
  Measured (2 forced CPU devices, `jax/_src/pjit.py:1733-1753`):

  | current-mesh target sharding | stale arg on device {0} | outcome |
  |---|---|---|
  | replicated on the 2-device mesh | committed, devices {0} | **no error** — broadcast to {0,1}, values correct |
  | genuine 2-way FSDP (`P(None,'fsdp')`) | committed, devices {0} | **raises** `ValueError: Sharding passed to pjit does not match the sharding on the respective arg` |

  So nothing silently *mistrains*. The three real cases are:

  | change | what happens today |
  |---|---|
  | FSDP 2 → 1 (fewer GPUs) | hard error inside orbax restore. Recorded: job 10324589 |
  | FSDP 1 → 2 (more GPUs) | restore succeeds (saved id `{0}` exists); first jit touching a ≥2-D ≥4 MiB leaf raises the pjit mismatch above |
  | GPU count changes, `fsdp_devices=1` throughout | no error; arrays sit on the saved device subset and are broadcast at each jit boundary. Correct, wasteful |

  This is good news and should be stated plainly in the plan: **the current bug is
  loud in both directions that matter.** It is a usability defect, not a silent-
  correctness one, which lowers the risk tier of the *symptom* (not of the fix).

### The fix shape, proven on CPU

`construct_restore_args(target, sharding_tree)` is public, and its second argument is
**not** deprecated in 0.11.13 — the docstring recommends exactly this usage
(`orbax/checkpoint/checkpoint_utils.py:377-382`, `:397-402`). The legacy manager API
accepts a parallel `restore_kwargs` that becomes `PyTreeRestoreArgs(item,
restore_args=…)` (`orbax/checkpoint/checkpoint_manager.py:1501-1510`, `:1576-1599`);
it is docstring-deprecated but **not** runtime-warned, and `item_handlers=` (what
openpi uses) does not warn either.

Probe (`JAX_PLATFORMS=cpu`, `--xla_force_host_platform_device_count=2`, save under a
1-device mesh → restore under a 2-device mesh through a real `CheckpointManager` with
`PyTreeCheckpointHandler`):

```
A bare SDS, no restore_args   -> NOT resharded (stale/saved)   <- today's path
B SDS annotated with sharding -> NOT resharded (stale/saved)
C explicit restore_args       -> RESHARDED (correct)
```

**Option B is dead**: annotating the `jax.eval_shape` leaves with
`jax.ShapeDtypeStruct(..., sharding=…)` is *not* enough, because
`PyTreeCheckpointHandler` never calls `construct_restore_args` on the item. The fix
**must** reach `restore_args`. The reverse direction was also measured (save 2-way
FSDP → restore onto a 1-device mesh): reshards, values bit-intact.

## Files to touch

**One code site, two plausible placements.** The plan phase picks; both are recorded
here because the choice has a real cost either way.

| file | change |
|---|---|
| `src/rl/filtered_sft_agent/filtered_sft_learner.py` | `init_train_state` `:198-202` (resume return) and the restore call `:319-324` — forward `state_sharding` into the restore |
| *either* `openpi/src/openpi/training/checkpoints.py:89-107` | give `restore_state` an optional sharding-tree parameter and build `restore_kwargs` from it |
| *or* `src/rl/filtered_sft_agent/filtered_sft_learner.py` (only) | stop calling `_checkpoints.restore_state`; inline a sharding-aware restore in this repo, reusing `_checkpoints._split_params` / `_merge_params` |

**Placement tradeoff.** `openpi/` is a **git submodule, currently clean** (`git
submodule status` → `11f08089…`, no `+`; `git -C openpi status` empty). A change there
is not captured by a parent-repo commit — only the submodule pointer is — so a fresh
`git submodule update` loses it unless the commit is pushed to `lasgroup/openpi` and
the pointer bumped. `docs/code/best_practices.md:316-318` is directly on point:

> Helpers that would naturally be methods on a submodule class **stay in this repo**
> so the submodule stays clean

That argues for the `src/`-only placement. Its cost: reaching two openpi privates
(`_split_params`, `_merge_params`), which `best_practices.md:319-320` permits with a
comment stating why the public API doesn't work — here, because the public
`restore_state` has no seam for restore args.

Not touched: `scripts/` (no flag changes — `fsdp_devices` already exists and is
already explicit), `src/training/config.py`, the replay buffer, the resume manifest.

## Duplication sweep

- **`init_train_state` clone family** (`filtered_sft_learner.py:144` ↔
  `src/rl/dsrl/dsrl_env.py:70` ↔ `openpi/scripts/train.py:85`).
  - `dsrl_env.py:70-74` takes `(config, init_rng, mesh)` — **no `resume` parameter**,
    so it has no short-circuit and always returns a concrete, current-mesh-sharded
    state. **The clone diverged in exactly the way that spares it. Excluded**, and it
    must *not* be "synced" to the fixed version — adding a `resume` branch would
    import the bug. Independently moot: `src/rl/dsrl/` is unwired and its config
    matches no dispatch branch (CLAUDE.md sharp edges), and `DSRLLearner` restores via
    a concrete target (`dsrl_learner.py:760`) with `use_sharding=False` states.
  - `openpi/scripts/train.py:85` / `:115-117` / `:236` / `:241` is the upstream
    offline trainer carrying the identical bug. **Excluded** — not on this repo's
    entry path (`scripts/exp.py`). Worth an upstream issue, not this change.
- **RL-checkpoint block** (`advantage_weighted_sft_learner.py:235-287` ↔
  `best_of_n_learner.py:165-209`). Both already pass a **concrete** target into
  `StandardCheckpointer.restore` (`:256` and `:185-188`), so both already reshard
  correctly. **Not touched.** Their two documented divergences (BofN warns instead of
  raising on a missing rl_state, OQ-6; the `if not path.exists():` guard) are
  unaffected.
- **`_refresh_train_step` / `_refresh_critic_update_function`** — a fourth,
  previously unlisted clone pair: `filtered_sft_learner.py:515-518` and
  `best_of_n_learner.py:133-135` both recompute
  `self._train_state_sharding = sharding.fsdp_sharding(self._train_state, self._mesh)`.
  Relevant because it means the *target* sharding tree is already recomputed from the
  current mesh after the restore — **the fix only has to correct where the restored
  arrays land, not what the downstream jits expect.** Suggest adding this pair to
  CLAUDE.md's clone-family list in step 4.
- Other listed clone families (`update_critic.py` pair, `_pad_last_dim`, the best-of-N
  scoring block, `_get_on_policy_action`, the recipe preambles): not touched.
- **One additional affected site outside the clone families:**
  `scripts/probe_critic_action_sensitivity.py:130` calls
  `ocp.StandardCheckpointer().restore(ckpt_path)` with **no target**, which falls into
  the same metadata-sharding fallback. Throwaway analysis probe, not on the training
  path — flag, decide in the plan whether to include.

## Inheritance sweep

Chain (verified): `Agent` (`src/rl/agent.py:22`) → `FilteredSFTLearner`
(`filtered_sft_learner.py:234`) → `AdvantageWeightedSFTLearner`
(`advantage_weighted_sft_learner.py:41`) → `{MPOWeightedSFTLearner`
(`mpo_weighted_sft_learner.py:14`) → `FlowGRPOLearner` (`flow_grpo_learner.py:15`)`,
`OGPOAgentLearner` (`ogpo_learner.py:89`)`}`. `BestofNLearner`
(`best_of_n_learner.py:36`) branches off `FilteredSFTLearner` directly — it is a
**sibling of AWR, not a child**, which is why its checkpoint block is a copy.

- **No subclass overrides `init_train_state` or the restore call.** The call at
  `filtered_sft_learner.py:319` is the only `restore_state` call site in `src/`; all
  six learners inherit it. So one edit reaches every learner — which is precisely what
  makes this Tier 2.
- **`update()` overrides are not in play.** OGPO (`ogpo_learner.py:520`), AWR (`:787`),
  BofN (`:621`), FlowGRPO (`:22`) override `update()` and therefore own their EMA
  advance (OQ-10) — but this change touches only `__init__`-time restore, not the
  step loop. **The EMA-ownership trap does not apply.** Stated explicitly so the plan
  phase does not spend effort on it.
- **`__init__` ordering that the fix sits inside** (`FilteredSFTLearner.__init__`):
  mesh `:261` → shardings `:262-267` → checkpoint manager `:270-277` → buffer restore
  `:305-311` → `init_train_state` `:315-317` → **restore `:319-324`** → EMA slice +
  `device_put` `:348-350` → `_refresh_train_step()` `:369`. Subclass work happens
  after `super().__init__`: AWR builds critics on `self._mesh` (`:127-145`) and
  restores rl_state (`:155`); OGPO moves the EMA to `pinned_host` on `self._mesh`
  (`:123-126`) then `_restore_extra_resume_state()` (`:152`). All of it already keys
  off the current mesh.
- **Downstream consumers of the restored state are already current-mesh**, so they
  need no change: `_batch_axis_sharding` + `jax.device_put` for the EMA
  (`filtered_sft_learner.py:348-350`), the five OGPO jits
  (`ogpo_learner.py:328-416`), the two AWR jits
  (`advantage_weighted_sft_learner.py:191-226`).

## Host-side resume state — confirmed topology-independent

No change needed in any of these; verified no `jax.Array` storage, no `device_put`, no
device/sharding recorded:

- **OGPO extra resume state** (`ogpo_learner.py:198-243`): `save_extra_resume_state`
  emits JSON scalars only (`float(_adv_scale)`, shard dir string, ints, task-range
  int pairs, numpy RNG state); `_restore_extra_resume_state` reads them back through
  `float()`/`int()`/`Path()`. `_rebase_task_ranges` is pure Python ints.
- **`src/rl/replay_buffer.py`**: numpy storage, HDF5 shards. Its only device contact is
  `jax.device_put(batch, self.data_sharding)` inside `sample()` (`:192`), using a
  sharding passed in fresh each run from the current mesh — nothing persisted.
- **`src/training/runtime_state.py`**: dataclasses/json/pathlib only.
- **AWR critic/value/normalizer**: built from `self._mesh` every run
  (`update_critic.py:265-273`, `:304-310`; `_normalizer_state` via
  `jax.device_put(..., self._replicated_sharding)`,
  `advantage_weighted_sft_learner.py:149-151`) and restored through a concrete target.

**`fsdp_devices` is recorded nowhere in any manifest and nothing validates it on
resume** (greps across `src/`, `scripts/`, `docs/changes/`). The topology lives only
in orbax's array metadata. Nothing to relax — and equally, no existing mechanism would
warn about a topology change. Note for the plan: adding `fsdp_devices` to the manifest
would have to go in the per-learner `extra` block, because `ResumeState(**payload)` is
strict about unknown top-level keys (`runtime_state.py:22`, pinned by
`tests/ogpo/test_resume_hardening.py:327`).

## Expected behavior after

- `sbatch --export=ALL,FSDP=2 --gres=gpu:2 scripts/ogpo_multitask_4task_maxlab.sbatch`
  against a run checkpointed at FSDP=1 resumes and trains, resharding the policy
  params/opt_state/EMA to 2-way FSDP. The reverse (FSDP=2 checkpoint, 1-GPU job)
  resumes replicated.
- `M == N` resumes are numerically identical to today (same sharding tree, now stated
  explicitly rather than read from `_sharding`).
- The per-resume orbax `UserWarning` about restoring "on a different topology than the
  checkpoint was saved with" disappears.
- The four `--resume` probes stop being pinned to the checkpoint's GPU count;
  `probe_candidate_q_spread.sbatch:83`'s `FSDP=2` workaround and
  `probe_rollout_gifs.sbatch`'s stale header/`--gres=gpu:1` mismatch become fixable.
- `batch_size % jax.device_count() == 0` (`filtered_sft_learner.py:250-253`) still
  binds — changing the GPU count can still be rejected there, by design.

## Open design questions for the plan phase

1. **Placement** — `src/`-only vs. a submodule edit (see the tradeoff table above).
2. **Splitting the sharding tree.** `restore_state` routes `ema_params` into the
   `params` item and blanks the other via `_split_params`
   (`openpi/.../checkpoints.py:145-152`); the sharding tree must be split the same
   way. With `ema_decay` set (every reachable config) the resume-path eval_shape state
   has non-None `ema_params`, so `params := state.ema_params`. Note
   `dataclasses.replace` on a `TrainState` of shardings re-invokes its beartype-checked
   `__init__` (`step: at.Int[...]` would reject a `NamedSharding`), so the split must
   run inside `at.disable_typechecking()` — which `restore_state` already does at
   `checkpoints.py:97`. Confirm at implementation time.
3. **Which sharding to request for the `params`/EMA item.** Save writes it *mixed*:
   `save_checkpoint` composes trainable leaves from a **replicated** `device_put` of
   `self._ema` with frozen leaves from the **FSDP-sharded** `train_state.params`
   (`filtered_sft_learner.py:757-767`, `src/rl/ema_utils.py:16-28`). Requesting a
   uniform `fsdp_sharding` for that item would change the same-topology path too: the
   ~11.3 GiB trainable EMA would land FSDP-sharded and then be all-gathered by the
   `device_put` at `:350` (`_batch_axis_sharding` returns replicated whenever
   `mesh.shape[BATCH_AXIS] == 1`, which is the case for a 1-node FSDP=2 run). Given
   the ~34.7 GiB memory floor and that `del ema_dev` is load-bearing, **peak init
   memory is a real risk here** and the plan should either mirror the save-side
   composition or justify the uniform choice.
4. **Whether to include `scripts/probe_critic_action_sensitivity.py:130`.**

## Blast radius

One or two source files; no config field, no new flag, no jit signature, no
`donate_argnums`, no RNG split arity, no replay-buffer schema, no checkpoint *layout*
change (what is written is unchanged — only how it is read back). Reaches all six
learners through the single inherited restore call, and the four `--resume` probes.

Docs to update in step 4: `docs/code/rl-learners.md` (the `init_train_state` bullet at
`:75` says resume "returns only the shape and sharding, leaving restoration to orbax" —
that becomes wrong), `docs/code/scripts.md:429-442` (the "Restore topology" note
becomes obsolete — the gotcha is resolved and should be deleted per CLAUDE.md step 4),
and CLAUDE.md's clone-family list (add the `_refresh_train_step` pair). If the plan
adds a `# best-effort:` comment anywhere,
`tests/ogpo/test_resume_hardening_verifier.py:1135` asserts on the count of those and
will need updating.

## How it will be verified

CPU-only; a real cross-topology resume needs π0.5 weights, two GPUs and a cluster
allocation and **will not be run from the login node** — the verifier states that
plainly rather than implying validation.

1. **New pytest, multi-device CPU.** There is **no precedent in this repo**: greps of
   `tests/` for `XLA_FLAGS` / `xla_force_host_platform_device_count` / `jax.device_count`
   return zero hits, and every mesh in the suite is `sharding.make_mesh(1)`
   (`test_split_equivalence.py:400`, `test_per_task_critics.py:153`, …). The flag must
   be set before the first `import jax`, so this goes in a **standalone module that
   sets `os.environ` at the top and is run in its own pytest process** — *not* in
   `tests/ogpo/conftest.py`, where it would turn every existing `make_mesh(1)` into a
   `(2,1)` mesh and perturb the 302-test suite.
2. **Round-trip assertions**, modeled on the one existing orbax round-trip test
   (`tests/ogpo/test_per_task_critics_verifier.py:591-617`, whose `_states(config, rng)`
   at `:120-135` already takes `mesh = sharding.make_mesh(1)` — parameterizing that
   mesh is most of the work): save under a 1-device mesh, restore under a 2-device
   mesh and assert (a) the restored leaves carry the requested sharding, (b) values are
   bit-identical, (c) the reverse direction likewise. The scratch probes written during
   this discovery pass (`probe_restore_sharding.py`, `probe_restore_sharding2.py`)
   already demonstrate all three on a toy tree and can seed the test.
3. **Same-topology no-op test**: assert the `M == N` restore produces the identical
   sharding tree it does today — the differential-test requirement, since this is a
   change to numeric-data plumbing.
4. **`jax.eval_shape` tree assertion** if the plan annotates the target leaves, on the
   precedent of `tests/ogpo/test_backbone_lora_verifier.py:80-104`.
5. **Dummy-variant full-`TrainState` smoke** if a realistic tree is needed:
   `tests/ogpo/test_split_equivalence.py:80-103` (`paligemma_variant="dummy"`,
   `action_expert_variant="dummy"`, BroNet `hidden_dim=32, depth=1`) plus the
   module-scoped fixture at `:395-419`, which builds a `TrainState` *without*
   `init_train_state` and so sidesteps the GPU-only learner `__init__`.
6. **`pytest tests/ogpo`** (302 tests) for regressions. Watch
   `test_resume_hardening_verifier.py:860` and `:872`, which assert on `__dict__`
   membership across all six learner classes and will trip if a method is added or
   moved on the chain. `pytest tests/` still fails at collection (OQ-4) — out of scope.
7. **`DRY=1 bash scripts/ogpo_multitask_4task.sh`** to confirm the rendered command is
   token-identical (no flag changes expected).
8. **Proposed, not submitted:** a 2-GPU `--export=ALL,FSDP=2 --gres=gpu:2` resume of an
   existing FSDP=1 checkpoint, and the mirror. These are the only true end-to-end
   checks and need the maintainer's explicit go-ahead.

## Gotchas checked

- `rl-learners.md` Gotchas: the RL-checkpoint clone pair (both already correct), the
  AWR-vs-BofN warn/raise split (OQ-6, untouched), `update()`/EMA ownership (OQ-10, not
  in play — `__init__`-time change only).
- `rl-ogpo.md` Gotchas: RNG split arity, donated-argument reuse, `del ema_dev`,
  per-task-critic new-run-only — none reached by an `__init__`-time restore change.
  `del ema_dev` is adjacent to open question 3 above (peak memory), not violated by it.
- `training.md` Gotchas: stale shards ahead of the resume point, manifest strictness —
  unaffected; the shard path is topology-independent (§C).
- CLAUDE.md sharp edges: `isinstance` dispatch order (untouched), resume manifests
  written before 2026-08-27 (unchanged — this fix is orthogonal to the `extra` block),
  `save_shard` non-determinism (unchanged).
- `best_practices.md:304-320` (fork/submodule boundary) — the governing rule for the
  placement decision; `:319-320` requires a comment if openpi privates are reached.
