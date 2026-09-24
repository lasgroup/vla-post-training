# DIFF — cross-topology (`fsdp_devices`) checkpoint resume

Tier 2, implementation pass (phase 3 of 3), 2026-09-21. Plan: `PLAN.md` in this
directory. Verification was a separate agent's pass; its report is recorded
verbatim in `VERIFICATION.md`. Changes made after that report are listed under
"Post-verification amendments" at the end of this file.

## Files changed

### `src/rl/filtered_sft_agent/filtered_sft_learner.py` (only source file touched)

**1. Imports (`:13-18`)** — two additions to the third-party group:

```python
import orbax.checkpoint as ocp
from orbax.checkpoint.checkpoint_utils import construct_restore_args
```

`import orbax.checkpoint as ocp` matches the house spelling already used by
`advantage_weighted_sft_learner.py:13`.

**2. `init_train_state` resume branch (`:201-207`)** — comment only, no behavior
change. Records that the returned `state_sharding` is now a *restore contract*
(it becomes orbax restore args), not just the jits' `in_shardings`.

**3. New module-level helper `_params_item_sharding` (`:227-248`)** — builds the
restore sharding for the on-disk `params` (EMA) item by reusing
`compose_full_params` (`src/rl/ema_utils.py:16-28`), the same function
`save_checkpoint` uses to compose that item:

```python
replicated = jax.tree.map(lambda _: replicated_sharding, state_sharding.params)
return compose_full_params(state_sharding.params, replicated, trainable_filter)
```

Result: frozen + non-Param leaves keep the current mesh's FSDP spec, trainable
(EMA) leaves are replicated — the save-side layout exactly
(`filtered_sft_learner.py:854-870`). Docstring records why the simpler uniform
`fsdp_sharding` target was rejected (it would change even the working
single-GPU path, adding an unmeasured FSDP→all-gather round trip on the
~11.3 GiB EMA immediately before the replicated `device_put` at `:447-449`).

**4. New module-level helper `_restore_state_sharded` (`:251-313`)** — a local
reimplementation of `openpi.training.checkpoints.restore_state`
(`openpi/src/openpi/training/checkpoints.py:89-107`) that adds explicit
`restore_args`. It:

- runs inside `at.disable_typechecking()`;
- calls `_checkpoints._split_params` twice — once on the shape tree, once on the
  sharding tree — so the two-item split is identical on both;
- builds the params-item sharding via `_params_item_sharding`;
- restores with
  `checkpoint_manager.restore(step, args=ocp.args.Composite(train_state=ocp.args.PyTreeRestore(item=…, restore_args=construct_restore_args(…)), params=ocp.args.PyTreeRestore(item={"params": …}, restore_args=construct_restore_args(…))))`;
- recombines with `_checkpoints._merge_params`.

Both openpi privates carry the comment `best_practices.md:319-320` requires,
stating why the public API does not work (no seam for restore args) and why the
typechecking guard is needed.

**5. Call site (`:414-428`)** — `_checkpoints.restore_state(...)` replaced with
`_restore_state_sharded(self._checkpoint_manager, self._train_state,
self._train_state_sharding, trainable_filter=self._config.trainable_filter,
replicated_sharding=self._replicated_sharding, step=self._resume_state.step)`.
The existing `logging.info` gained one field, the mesh shape
(`dict(self._mesh.shape)`), so a cross-topology resume is visible in the log.

`grep` confirms no `_checkpoints.restore_state` call remains anywhere in `src/`,
`scripts/` or `tests/`: the single inherited call site now routes all six
learners through the sharded path.

### Files deliberately NOT changed

| File | Why |
|---|---|
| `openpi/` (submodule) | Stays clean per `best_practices.md:316-318`; also serves as the verbatim pre-change reference for the mandatory differential test |
| `src/rl/dsrl/dsrl_env.py` | Its `init_train_state` clone has no `resume` parameter and therefore not this bug — "syncing" it would import the bug |
| `advantage_weighted_sft_learner.py`, `best_of_n_learner.py` | Their RL-state restore already passes a concrete current-mesh target and already reshards |
| `scripts/*`, `src/training/config.py`, replay buffer, checkpoint layout | No flag, schema, or on-disk format change — only how the checkpoint is read back |
| `scripts/probe_critic_action_sensitivity.py:130` | **Knowingly not fixed.** Same bug class (`StandardCheckpointer().restore(path)` with no target), but a throwaway analysis probe off the training path with no mesh at all. Per PLAN.md decision, a separate small change |

## Divergence from PLAN.md

**None in substance.** Three mechanical notes:

1. **Line numbers.** The plan's numbers were from the discovery pass and had
   drifted. Actual current locations: imports `:13-18`, resume branch `:201-207`,
   `_params_item_sharding` `:227`, `_restore_state_sharded` `:251`, call site
   `:414`, save composition `:854-870`, EMA replicated `device_put` `:447-449`.
   The plan's `:217` / `:318-324` / `:265-267` / `:757-767` / `:348-350` are all
   the same code at different offsets.
2. **`data_loader` dropped**, as the plan anticipated: openpi's `restore_state`
   takes it only to `del` it on the first line.
3. **Plan note (b) confirmed on the real `TrainState`, not just the replica.**
   `_split_params` on a sharding-valued `TrainState` raises
   `jaxtyping.TypeCheckError` outside `at.disable_typechecking()` — the error
   names `step`'s annotation `Union[Int[Array, ''], Int[ndarray, ''], np.number,
   int]` against a `NamedSharding`. Worth recording *why* this does not also fire
   at `fsdp_sharding` (`:199`), which builds the very same tree: openpi patches
   `jaxtyping._decorator._check_dataclass_annotations`
   (`openpi/src/openpi/shared/array_typing.py:26-41`) to skip checking when the
   call stack contains `jax._src.tree_util`, i.e. during tree *unflattening*.
   `dataclasses.replace` is a direct call, so it is not whitelisted and the guard
   is genuinely required. The comment in `_restore_state_sharded` says this.

## Nothing left undone pending sign-off

No non-raise error path, swallow-and-log, sentinel return, or silently-applied
default was written, so the deviation protocol was not triggered. The change adds
no `# best-effort:` comment, so
`tests/ogpo/test_resume_hardening_verifier.py:1135` (which asserts on their
count) is unaffected.

## Self-checks run by the implementing session

Run on a compute node (`maxlab-cpu`, jobs 10522776 and 10522784) via
`scripts/_test_fsdp_topology_resume.sbatch`, with the probe at
`temp/probe_fsdp_topology_resume.py`. Both are one-off scratch artifacts, not
part of the change. The login node cannot run any of this: `RLIMIT_NPROC` 1000 /
`RLIMIT_AS` 16 GiB there abort XLA's thread-pool creation and the 162 MB
embedding allocation, and `/data` (the default `CKPT_BASE_DIR`) is not mounted.

**Probe (dummy-variant π0.5, two forced CPU devices, job 10522784) — all legs
pass:**

```
PROBE A: raised TypeCheckError outside guard <-- guard IS required
PROBE B: params target/sharding treedef match: True
PROBE B: frozen-leaf specs   : [PartitionSpec('fsdp',None), PartitionSpec(), ... ]
PROBE B: trainable-leaf specs: ['PartitionSpec()']
PROBE C(i):  same-topology restore values identical: True; ema present: True; step = 0
PROBE C(ii): train_state params leaves = 51, genuinely sharded = 41, sharding mismatches = 0
PROBE C(ii): ema leaves = 51, genuinely sharded = 19 (frozen only, by design), mismatches = 0
PROBE C(ii): cross-topology values bit-identical (params/ema): True True
PROBE C(ii): device sets -> [0, 1]
PROBE C(iii): openpi restore_state specs (unchanged path): ['PartitionSpec()']
```

C(iii) is the differential leg: in the *same process*, on the *same*
checkpoint, openpi's untouched `restore_state` returns only the saved
(1-device, replicated) layout while `_restore_state_sharded` returns the
requested 2-way FSDP layout with identical values.

**`pytest tests/ogpo` (job 10522776): 3 failed, 392 passed, 6 skipped in
711 s.** The suite is 401 tests on this branch, not the 302 CLAUDE.md quotes.
All three failures are in `tests/ogpo/test_verifier_alignment.py`
(`test_head_value_distribution_differential_over_every_registered_config`,
`..._is_a_real_change_at_201_bins`, `test_head_wrapper_differential_over_a_randomized_flag_script`)
— the verifier suite for the **2026-08-20 reference-alignment** change, whose
embedded verbatim pre-change copy of `get_value_bounds` now equals the current
implementation (`assert -99.99999999999991 < -99.99999999999991`). Pre-existing
branch state, not a regression from this change: that module imports only
`src.envs.wrappers`, `src.rl.replay_buffer`, `src.rl.value_distribution`,
`src.training.config`, `src.rl.advantage_weighted_sft.update_critic`,
`src.rl.ogpo.update_actor`, `src.rl.networks.bronet_critic` — none of which
imports `filtered_sft_learner`. Both `src/rl/value_distribution.py` and that
test file already carry uncommitted working-tree edits.

**`DRY=1 bash scripts/ogpo_multitask_4task.sh`:** renders, 133 tokens, and is
byte-identical to the render taken before the compute-node move once the
`--checkpoint_base_dir` path is normalized. `--fsdp_devices 1` unchanged; no
flag added or removed. Note this could not be a literal `git stash` diff — the
working tree is shared with five live training jobs, so reverting a source file
in place was not safe. It does not need to be: `ogpo_multitask_4task.sh:283`
sets `RUN=(echo uv run "$ENTRY")` under `DRY=1`, so no `src/` module is imported
or executed on that path and the render cannot depend on a `.py` edit.

Results are relayed in the implementing session's report; the adversarial
verification that produced `VERIFICATION.md` was a separate agent's pass. It
independently reproduced 392 / 3 / 6 and **proved** (not merely argued) that the
3 `test_verifier_alignment.py` failures are identical with and without this change
(VERIFICATION.md §4). The two scratch artifacts named above, and the earlier
throwaway `scripts/_test_topology_producer_fsdp1.sbatch`, were deleted after
verification.

## Post-verification amendments

Made by the implementing session after `VERIFICATION.md`; all comment/docstring
only, `py_compile` clean on a compute node (job 10523677).

1. **Both new helpers' docstrings now name the certifying test**
   (`best_practices.md` §8): `_params_item_sharding` cites
   `tests/ogpo/test_cross_topology_resume_verifier.py` and the measured real-scale
   memory/placement parity; `_restore_state_sharded` cites the same-topology
   differential test and states the strictness delta below.
2. **Call-site comment at `init_train_state`'s use in `__init__`** recording that
   both returns must come from one call (static `tx`/`ema_decay` fields compare by
   identity; two calls raise `Mismatch custom dataclass node data`) —
   VERIFICATION.md B.1, judged moderately fragile and cryptic.
3. **Not adopted:** the optional mesh-consistency `assert` in
   `_restore_state_sharded`. It would be new behaviour beyond the approved plan,
   and the single call site derives both arguments from `self._mesh`.

### Behaviour change recorded (VERIFICATION.md §3 B.7a)

The change is a no-op at an unchanged `fsdp_devices` — identical values, identical
per-leaf sharding, no peak-memory regression on real π0.5 weights at 1 and 2 GPUs.
It is **not** a no-op in one respect: restore is now **stricter on a target-vs-
stored leaf *shape* mismatch**. `construct_restore_args` builds an
`ArrayRestoreArgs` carrying `global_shape` with `strict=True`, so a target whose
leaf shape differs from the stored one raises
`ValueError: Requested shape … is not compatible with the stored shape …`.
The old path used a bare `RestoreArgs` and silently returned the *stored* shape
into a tree whose jits were compiled for the target's. Leaf-set and structure
mismatches already raised identically on both paths. Judged an improvement in line
with the repo's fail-fast rule: the restore target is `jax.eval_shape(init)` under
the current config, so a shape difference means the config builds a different
network than the checkpoint holds; and the real π0.5 `TrainState` restored at
three topologies (1→2, 2→2, 2→1) without a strict-shape rejection. A target *dtype*
difference is likewise now cast rather than ignored (unreachable from any config;
pinned by a test that skips in the current tree because it has no bf16 leaf).
