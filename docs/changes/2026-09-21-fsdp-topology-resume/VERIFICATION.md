# VERIFICATION — cross-topology (`fsdp_devices`) resume of the policy `train_state`

> **Provenance.** Everything from the next heading down to "Coordinator notes" is the final
> report of the independent verifier (fresh-context Opus agent, CLAUDE.md step 3), recorded
> verbatim by the implementing session. An earlier interim handback from the same verifier
> was incomplete (regression run and GPU legs still in flight) and is superseded by this
> report; nothing from it is lost — its results are folded in below.

Independent adversarial verification of `docs/changes/2026-09-21-fsdp-topology-resume/`.
**This report supersedes the earlier incomplete handback.** All owed items are now closed: the test-23
behaviour delta is resolved, the full regression and attribution legs are collected, and all four GPU
legs (real π0.5 weights) ran. Nothing was left running; all scratch is deleted.

---

## 1. Verdict

**Correct and safe to keep. Confidence: high, now including real-weights GPU evidence at three
topologies. No blocking issues.** One deliberate behaviour change must be surfaced to the maintainer
(§3, B.7a) — it is a strictness *improvement*, not a regression, and I recommend a one-line note in the
change record rather than any code change.

Verified end-to-end on real π0.5 weights:

- **M == N is a no-op.** At FSDP=1→1 and FSDP=2→2 the new path and openpi's untouched `restore_state`
  produce an **identical per-leaf sharding histogram** (at 2 GPUs: 28 replicated + 23 sharded across 9
  distinct `PartitionSpec`s), identical EMA placement, identical values, and **no peak-memory
  regression** (2-GPU peak 16.03/15.75 GiB new vs 16.03/15.89 GiB old).
- **The fix works.** An FSDP=1 checkpoint (step 200) resumed on 2 GPUs at `fsdp_devices=2`, logged
  `onto mesh {'batch': 1, 'fsdp': 2}`, emitted **zero** "Sharding info not provided" warnings, and
  trained on from step 200 to 399 with buffers and metrics continuing, then saved an FSDP=2 checkpoint.
- **The original failure is reproduced and fixed on GPU.** That FSDP=2 checkpoint on 1 GPU: old code
  dies with verbatim job 10324589's `ValueError: sharding passed to deserialization should be
  specified, concrete and an instance of 'jax.sharding.Sharding'. Got None`; new code restores cleanly.

Non-blocking findings, descending interest:

1. **Behaviour change (deliberate, must be surfaced):** the new path is **stricter** than the old on a
   target-vs-stored *shape* mismatch. Old: silently returns the **stored** shape into a tree whose jits
   were compiled for the target's. New: raises. Judged an improvement — see §3 B.7a for the full
   characterization and the argument that it cannot reject a legitimate resume.
2. **`_params_item_sharding` is only correct for the `ema_params is not None` branch** (§3 B.2).
   Unreachable in production; pinned by two tests.
3. **`_restore_state_sharded` does not assert that `state_sharding` and `replicated_sharding` share a
   mesh.** One call site, both from `self._mesh`, so it cannot currently diverge. Cheap hardening.
4. **`at.disable_typechecking()` is wider than strictly needed** but is *exactly* openpi's own scope,
   and `_merge_params` is correctly outside the guard on both paths. Parity, not a regression.
5. **B.1's "same `init_train_state` call" requirement is real** — two calls give non-identical, non-`==`
   `tx` objects → `ValueError: Mismatch custom dataclass node data`. Production satisfies it; the
   failure would be loud but cryptic. One comment at the call site is worth it.
6. **Out-of-scope side finding, but significant:** openpi's `nnx_utils.state_map` is a **complete no-op**
   under the pinned flax 0.10.6, so `init_train_state`'s "convert frozen params to bfloat16"
   (`filtered_sft_learner.py:184-188`) never happens. Confirmed on real weights (§6).

---

## 2. Per-test results

Permanent module: **`tests/ogpo/test_cross_topology_resume_verifier.py`** — 27 test items.

**Final clean result (job 10523467, leg A, maxlab-cpu): `26 passed, 1 skipped, 30 warnings in 68.82s`.**
(Reproduced at job 10523407: `26 passed, 1 skipped in 59.86s`.) The one skip is intentional and carries
its reason: `no bf16 (frozen) leaf in this tree; dtypes present = ['float32']` — see §6.

**Skip behaviour (job 10523467, leg B):** collected after another module imported jax —
`9 passed, 27 skipped`; all 27 of mine skip with the reason *"needs >= 2 JAX devices, have 1; jax was
already imported when this module loaded (full-suite run), so XLA_FLAGS could not be set. Run
standalone: uv run pytest tests/ogpo/test_cross_topology_resume_verifier.py"*. `tests/ogpo/conftest.py`
was not touched.

Design: two forced CPU devices set **only if jax is not already imported**; meshes built directly over
device subsets (`Mesh(np.array(jax.devices()[:n]).reshape(1,n), (BATCH_AXIS, FSDP_AXIS))`), never
`sharding.make_mesh`; checkpoints written through the **real** path
(`_checkpoints.initialize_checkpoint_dir` → `PyTreeCheckpointHandler` item handlers →
`_checkpoints.save_state`), with the `params` item composed by copying `save_checkpoint`'s body
verbatim; the TrainState is real (`nnx.State`/`VariableState`/optax `opt_state`/`nnx.GraphDef`,
non-`None` `ema_params`).

| # | Test | Asserts | Result |
|---|---|---|---|
| 1 | `test_two_devices_are_available_and_meshes_are_distinct` | 2 devices; mesh1 fsdp=1, mesh2 fsdp=2 | PASS |
| 2 | `test_the_fixture_has_BOTH_frozen_and_trainable_leaves` | Anti-vacuity: 19 trainable of 51 — the **same split the real π0.5 tree shows on GPU** | PASS |
| 3 | `test_the_fixture_checkpoint_really_shards_a_big_leaf` | A ≥4 MiB ≥2-D leaf exists and gets a non-replicated spec (below `min_size_mbytes=4` everything replicates) | PASS |
| 4 | `test_no_leaf_falls_back_to_a_bare_RestoreArgs` | Spy captures the real `ocp.args.Composite`; every leaf of **both** items is an `ArrayRestoreArgs` with `sharding is not None` on the current mesh; >50 leaves | PASS |
| 5 | `test_restore_args_treedef_matches_the_item_treedef` | `tree.structure(restore_args) == tree.structure(item)` for both items on the REAL TrainState | PASS |
| 6 | `test_new_path_emits_no_sharding_fallback_warning` | No "Sharding info not provided" `UserWarning` | PASS |
| 7 | `test_old_path_does_emit_the_sharding_fallback_warning` | Non-vacuity for #6 | PASS |
| 8 | `test_same_topology_restore_is_identical_to_openpi_restore_state[1]` | **THE differential, 1 device**: equal treedef, bit-identical values, per-leaf sharding equal both semantically *and* strictly (`==`) | PASS |
| 9 | `…[2]` | Same at M == N = 2 | PASS |
| 10 | `test_same_topology_two_device_restore_is_genuinely_sharded` | Anti-vacuity for #9 | PASS |
| 11 | `test_forward_reshard_one_device_checkpoint_onto_two_device_mesh` | 1→2: values bit-identical to what was saved; **full** expected placement (train_state item *and* mixed EMA item); params device set == {0,1} | PASS |
| 12 | `test_reverse_reshard_two_device_checkpoint_onto_one_device_mesh` | 2→1 mirror; device set {0}; fully replicated | PASS |
| 13 | `test_old_path_leaves_a_one_device_checkpoint_unusable_on_the_two_device_mesh` | The 1→2 failure mode: old restore succeeds but pins to device {0}; a jit with the current mesh's `in_shardings` **raises**; the new path's leaf passes the same jit | PASS |
| 14 | `test_fsdp_devices_one_on_a_two_device_allocation` | (2,1) mesh replicates everything; new path spans {0,1}; old pins to {0} | PASS |
| 15 | `test_split_params_takes_the_same_branch_on_shape_and_sharding_trees` | Both take the EMA branch; structures equal for both halves | PASS |
| 16 | `test_split_params_on_a_sharding_tree_raises_without_the_typecheck_guard` | Raises outside `at.disable_typechecking()`, error names `step`; no raise inside | PASS |
| 17-18 | `test_params_item_sharding_equals_what_save_checkpoint_actually_writes[1]/[2]` | Leaf-for-leaf equality with the actual `.sharding` of the tree `save_checkpoint` composes | PASS |
| 19 | `test_params_item_sharding_is_genuinely_mixed_and_differs_from_uniform_fsdp` | At `min_size_mbytes=0`: trainable replicated, frozen FSDP-sharded, and genuinely ≠ uniform fsdp | PASS |
| 20 | `test_ema_params_none_branch_round_trips` | `ema_decay=None` restores, `ema_params is None`, values bit-identical | PASS |
| 21 | `test_ema_none_params_item_layout_is_replicated_for_trainable_leaves` | Documents finding §1.2 | PASS |
| 22 | `test_step_none_restores_the_latest_step` | `step=None` == `step=7` | PASS |
| 23 | `test_a_different_trainable_filter_at_restore_is_placement_only` | Two different filters (incl. `nnx.Param`) give bit-identical values | PASS |
| 24-25 | `test_new_path_is_STRICTER_than_old_on_a_target_vs_stored_shape_mismatch[5]/[3]` | **Rewritten test 23.** Old: returns, leaf carries the **stored** shape. New: raises `ValueError` matching "not compatible with the stored shape" | PASS |
| 26 | `test_leaf_set_mismatches_raise_identically_on_both_paths` | Bounds the change: extra/missing leaf → both paths raise "tree structures do not match" | PASS |
| 27 | `test_a_target_dtype_difference_is_CAST_by_the_new_path_and_ignored_by_the_old` | Isolated dtype pin | **SKIP** — `no bf16 (frozen) leaf in this tree; dtypes present = ['float32']` (see §6) |

### Failures encountered along the way (all resolved; verbatim)

- **Run 1 (job 10523063), 3 failed / 20 passed.** Two were my bug —
  `dataclasses.replace` on a tree of shardings without the guard:
  `beartype.roar.BeartypeCallHintParamViolation: Method openpi.training.utils.TrainState() parameter
  step="NamedSharding(mesh=Mesh('batch': 1, 'fsdp': 2, …))" violates type hint
  typing.Union[jaxtyping.Int[Array, ''], …]`. An independent third confirmation that the guard in
  `_restore_state_sharded` is required. The third was a vacuous leg (`action_horizon` changes no param
  shape).
- **Run 2 (job 10523143), 1 failed / 22 passed** — `AssertionError: fail-fast parity broken: old=None
  new=ValueError('Requested shape: (5,) is not compatible with the stored shape: (4,). Truncating/padding
  is disabled by setting of strict=True …')`. **Not a test bug** — a genuine behaviour delta; resolved
  in §3 B.7a and rewritten as tests 24-26.
- **Runs 3-4 (jobs 10523333, 10523349), 1 failed each** — my dtype pin:
  `AttributeError: 'VariableState' object has no attribute 'dtype'` (nnx flat state yields
  `VariableState`, not the `ShapeDtypeStruct`), then `AssertionError: no frozen bf16 leaf; dtypes present
  = ['float32']`, which uncovered the `state_map` no-op (§6).

---

## 3. Adversarial checks B.1–B.7

**B.1 — "same `init_train_state` call" requirement. VERIFIED and quantified** (job 10523063):

```
[B1] one call -> shape/sharding treedefs equal: True
[B1] two calls -> tx identical=False tx==tx:False graphdef identical=False graphdef==graphdef:True
[B1] two calls -> treedefs equal: False
[B1] CROSS-CALL construct_restore_args(shape_A, sharding_B): ValueError: Mismatch custom dataclass node
     data: (GradientTransformationExtraArgs(init=<function chain.<locals>.init_fn at 0x14ef68413ba0>, …
[B1]   perturbing static field 'tx':        ValueError: Mismatch custom dataclass node data
[B1]   perturbing static field 'model_def': trees still zip
[B1]   perturbing static field 'ema_decay': ValueError: Mismatch custom dataclass node data
```

`tx` (optax closures) and `ema_decay` bind; `model_def` does not (GraphDef has a structural `__eq__`).
Production satisfies it (`filtered_sft_learner.py:411-422`, one call), and `_refresh_train_step`
(`:614-617`) recomputes the sharding from `self._train_state` itself, preserving identity.
**Fragility: moderate — loud but cryptic.** Recommend one comment at the call site.

**B.2 — edge cases.**
- `ema_params is None`/`ema_decay=None`: **VERIFIED restores correctly** (values bit-identical). But
  `_split_params` then writes the *live params* to the `params` item (fully FSDP-sharded at save), while
  `_params_item_sharding` still asks trainable leaves **replicated** → placement shifts. **Unreachable**
  (every registered config sets `ema_decay`; `filtered_sft_learner.py:447` would crash on a `None` EMA
  first). Pinned by tests 20-21.
- `step=None`: **VERIFIED** identical to the pinned step.
- `trainable_filter` differing at restore: **VERIFIED placement-only, never numeric.** A
  `backbone_lora` toggle additionally changes the leaf set, which both paths already reject.
- `fsdp_devices=1` on a >1-device allocation: **VERIFIED**; new path replicates across both devices,
  old pinned to the saved subset {0}. A small improvement (removes a per-jit broadcast); note it means
  the new path is *not* placement-identical to the old in this one case — it **is** identical whenever
  the mesh is unchanged, which is the property that matters.

**B.3 — original failure reproduced and fixed. VERIFIED twice.**

*CPU, cross-process* (job 10523063; 2 devices save → 1-device process restore):

```
ERROR:root:The available devices are different from the devices used to save the checkpoint.
  Original=[[DeviceMetadata(id=0), DeviceMetadata(id=1)]], current available=[CpuDevice(id=0)]
[consumer] (a) OLD PATH raised ValueError: sharding passed to deserialization should be specified,
           concrete and an instance of `jax.sharding.Sharding`. Got None
[consumer] (b) NEW PATH: restored OK
[consumer] (b) leaves=143 bit-identical=True mismatched=[]
[consumer] (b) devsets={(0,)} specs=['PartitionSpec()']
```

This confirms BLAST-RADIUS.md's correction that the informative "available devices are different"
message is **log-only** and the raised error is the generic one.

*GPU, real π0.5 weights* (job 10523311, FSDP=2 checkpoint on 1 GPU) — §4.

The 1→2 direction was reproduced in-process (test 13): old restore succeeds, first jit raises.

**B.4 — remaining callers. VERIFIED: none.** `grep -rn "restore_state" --include=*.py src scripts tests
temp` returns only the new helper, its single call site (`filtered_sft_learner.py:415`),
docstring/comment mentions, my test module, and the implementer's scratch probe. Other restore sites
confirmed untouched and unaffected: `advantage_weighted_sft_learner.py:256` and
`best_of_n_learner.py:185` pass a **concrete** current-mesh target into `ocp.StandardCheckpointer`,
whose handler derives `ArrayRestoreArgs` from the target — they already reshard;
`dsrl_learner.py:760` is in the quarantined unwired tree.
`scripts/probe_critic_action_sensitivity.py:130` (`StandardCheckpointer().restore(path)`, no target)
still has the same bug class and is correctly recorded in `DIFF.md` as knowingly-not-fixed.

**B.5 — memory. MEASURED on real GPUs (supersedes the earlier argument-only answer).**
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.75` makes `nvidia-smi` useless (it shows the 71 GiB preallocation),
so I used JAX's `device.memory_stats()` (`peak_bytes_in_use`), which is the same instrument the repo's
Phase-F device-bytes gate used.

| Leg | Job | peak, OLD | peak, NEW | in_use, OLD | in_use, NEW |
|---|---|---|---|---|---|
| M==N=1 (FSDP=1→1) | 10523309 | **29.81 GiB** | **28.24 GiB** | 16.15 GiB | 16.13 GiB |
| M==N=2 (FSDP=2→2), dev0 | 10523380 | 16.03 GiB | 16.03 GiB | 8.10 GiB | 8.13 GiB |
| M==N=2, dev1 | 10523380 | 15.89 GiB | 15.75 GiB | 8.05 GiB | 8.05 GiB |
| 2→1 (mirror) | 10523311 | *(raises)* | 28.27 GiB | — | 16.16 GiB |

**No memory regression anywhere; the new path is equal or marginally lower.** This is the measurement
the mixed-layout (mirror-the-save) decision was made for, and it holds. Consistent with the placement
argument: at M == N the requested shardings are strictly equal to the metadata-derived ones, so orbax
allocates into the identical layout with no intermediate; the new path's only extra allocations are
host-side (one extra `_split_params` over a tree of `NamedSharding` references plus two
`construct_restore_args` trees of small dataclasses).

**Restore wall-time: no attributable difference.** G2 showed old 323.0 s vs new 40.5 s, but old ran
first on the first-ever read of a 28 GB checkpoint — a page-cache confound. I reversed the order in G5:
new 50.1 s (first) vs old 48.9 s (second). **The apparent 8× speedup is not supported once order is
controlled**; I am reporting the G2 number only as confounded.

**B.6 — conformance with CLAUDE.md non-negotiables and `docs/code/best_practices.md`.** Reviewed against
`filtered_sft_learner.py:13-18, 201-209, 227-313, 414-428`:
- **Deviation protocol: clean.** No `try/except`, no swallow-and-log, no sentinel/`None` return, no
  `getattr(cfg, "x", default)`, no silently-applied default. The one default (`step: int | None = None`)
  mirrors openpi's signature and is passed explicitly at the call site. No `# best-effort:` comment
  added — **confirmed empirically**: `test_resume_hardening_verifier.py` (which counts them, ~line 1135,
  and asserts `__dict__` membership across the six learner classes, ~lines 860/872) is **58 passed, 0
  failed** (job 10523064 leg C2). The two new helpers are module-level functions, not methods, so the
  `__dict__` assertions are untouched.
- **`best_practices.md:319-320` (openpi-private reach needs a comment saying why the public API fails):
  satisfied** at `:282-290` ("the public `restore_state` has no seam for restore args"), plus the
  typecheck-guard rationale.
- **`:316-318` (helpers stay in this repo): satisfied.** `git -C openpi status` is empty.
- **§6 "derive sharding trees from state, don't hand-write them": satisfied** —
  `_params_item_sharding` goes through the same `compose_full_params`/`filter_state`/`merge_state` the
  save uses.
- **§8 comments/docstrings: good.** Operational rationale with real numbers, the `FSL:`/`OGL:` legend,
  `path:LINE` for openpi. **Gap:** §8 says "name the test that certifies an equivalence claim"; both
  docstrings assert equivalences and cite the change directory. They should now cite
  `tests/ogpo/test_cross_topology_resume_verifier.py`.
- **`at.disable_typechecking()` scope:** identical to openpi's, `_merge_params` outside on both paths.
  Aside: it is a `@contextmanager` with **no `try/finally`** (`openpi/src/openpi/shared/array_typing.py:57-61`),
  so a raise inside leaves typechecking globally off — pre-existing openpi behaviour, identical exposure.
- **Log line:** `dict(self._mesh.shape)` added; lazy `%s` args. Verified in the real GPU log.
- **Dropped `data_loader`:** safe (openpi `del`s it immediately); `save_state` still uses
  `self._data_loader` for norm stats, unaffected.
- **No training run launched without permission; no source, docs or `VERIFICATION.md` written by me; no
  commits.**

**B.7 — other ways it could break.**

**(a) THE BEHAVIOUR CHANGE — target-vs-stored shape strictness. Characterized exactly (job 10523216),
five mismatch classes, both paths:**

| Class | OLD | NEW |
|---|---|---|
| matched target (control) | OK, correct shapes/dtypes | OK, identical |
| target leaf **larger** (`action_dim` 4→5) | **returns OK carrying the STORED shape `(4,)`** — the target's `(5,)` is silently ignored | **raises** `ValueError: Requested shape: (5,) is not compatible with the stored shape: (4,). Truncating/padding is disabled by setting of strict=True …` |
| target leaf **smaller** (4→3) | **returns OK carrying the STORED shape** | **raises**, same class |
| target has an **extra** leaf | raises `ValueError: User-provided restore item and on-disk value metadata tree structures do not match` | **identical** message |
| target **missing** a leaf | raises, same message | **identical** message |
| `freeze_filter` change (dtype + opt_state tree) | raises structure mismatch | **identical** |

So **only per-leaf shape strictness moved**; leaf-set and structure mismatches already raised on both.
The old behaviour is precisely this codebase's signature failure class: a stored-shape array silently
restored into a tree whose jits were compiled for a different shape — a run that trains and means
nothing.

**(iii) Can the stricter check reject a LEGITIMATE resume? No — asserted, and backed by the GPU legs.**
The restore target is `jax.eval_shape(init)` under the *current* config, so a shape difference means the
current config would build a different network than the checkpoint holds; restoring the stored shapes
there is never correct. Specifically: `action_horizon` and `max_token_len` change **no** parameter shape
(that is why my first attempt was vacuous), so changing them across a resume is unaffected;
`backbone_lora` changes the leaf *set*, already rejected by both; a dtype-only difference is unreachable
without an accompanying `opt_state` structure change, which both reject; `step` is a `()` scalar on both
sides; padding-style migrations live in the data-transform layer, not the checkpoint restore. **The
decisive evidence is empirical:** a full real π0.5 `TrainState` (51 params leaves + opt_state + step +
the mixed EMA item) restored through the new path **three times at three different topologies**
(1→2, 2→2, 2→1) without a single strict-shape rejection. **Verdict: improvement, consistent with the
repo's fail-fast non-negotiable — but it is a behaviour change and should be recorded in the change
record, not left implicit.**

**(b) Second, much smaller delta:** `construct_restore_args` also sets `dtype=<target dtype>`, so a
target dtype difference would be **cast** on the new path where the old returned the stored dtype. Pinned
by test 27, which **skips** here because no leaf in the tree is bf16 (§6). Unreachable from any config.

**(c) Old-code checkpoint restored by new code:** the save path is byte-identical (unchanged), so every
checkpoint is an old-code checkpoint — every test and every GPU leg restores one. **(d) New-code
checkpoint restored by old code:** also unchanged — G5's old leg restored the checkpoint written by
G3 (new code) with an identical sharding histogram. A rollback is safe. **(e) LoRA:** covered in B.2;
additionally the real runs report `actor/grad_norm_lora=0.0000` with `--no-backbone_lora`, and the
restore path is filter-driven, not lora-aware. **(f) `del ema_dev` / EMA pinned-host:** confirmed
untouched on real hardware — the EMA histogram is `{'PartitionSpec()|pinned_host': 19}` on **both**
paths at both topologies. **(g) Leaf-count difference in the saved metadata:** the new path never reads
the metadata's leaf set; it builds args from the target, so a mismatch raises exactly as before
(test 26).

---

## 4. Regression (C), recipe render (D), and the GPU legs

### C — regression, job 10523064 (maxlab-cpu), all legs collected

| Leg | Result |
|---|---|
| **C1** `pytest tests/ogpo` (excluding my new module — like-for-like with the author's claim) | **3 failed, 392 passed, 6 skipped, 273 warnings in 760.11s (12m40s)** → **401 tests**, not the 302 CLAUDE.md quotes. **Exactly reproduces the author's 392/3/6.** |
| **C1b** same, **including** my module | **3 failed, 392 passed, 29 skipped in 761.24s** — +23 skips (the module's size at that time), same pass and same fail counts ⇒ the module skips cleanly and does not perturb the suite |
| **C2** `test_resume_hardening_verifier.py` alone | **58 passed, 0 failed in 5.14s** — the `__dict__`-membership and `# best-effort:`-count assertions still pass |
| **C3** `test_verifier_alignment.py`, live tree | **3 failed, 114 passed, 3 skipped in 52.39s** |
| **C5** import graph | `filtered_sft_learner in sys.modules after importing test_verifier_alignment: []` |

The 3 failures (verbatim): `test_head_value_distribution_differential_over_every_registered_config`,
`test_head_value_distribution_differential_is_a_real_change_at_201_bins`,
`test_head_wrapper_differential_over_a_randomized_flag_script`, asserting
`assert -99.99999999999991 < -99.99999999999991` and
`assert (-200.49999999999983, 0.49999999999999956) != (-200.49999999999983, 0.49999999999999956)`.

### Determination on the 3 failures: **PROVEN pre-existing and unrelated to this change.**

Three independent legs, and one earlier attempt that was invalid — stated plainly:

1. **Mechanism (decisive).** Those tests build their pre-change reference by running
   `git show HEAD:src/rl/value_distribution.py` with `cwd=_ROOT` and diffing it against the
   **working-tree** implementation (`tests/ogpo/test_verifier_alignment.py:1096-1135`). Their outcome
   depends only on the HEAD blob and on the working-tree `src/rl/value_distribution.py` /
   `src/envs/wrappers.py` — files this change does not touch. The change's diff touches exactly one
   file, `src/rl/filtered_sft_agent/filtered_sft_learner.py` (hunk-by-hunk verified against `git diff`:
   imports `:16,:18`; resume-branch comment `:204-208`; helpers `:227-313`; call site + log `:414-428`).
2. **Import graph (C5):** `filtered_sft_learner` never enters `sys.modules` when that test module is
   imported — measured, empty list.
3. **Empirical A/B (job 10523443):** the same command run from the live tree and from a mirror whose
   only difference is this change reversed —
   **live: `3 failed, 114 passed, 3 skipped in 56.81s`; reversed: `3 failed, 114 passed, 3 skipped in
   59.65s`. Identical.**

**What was wrong the first time (reported, not hidden):** my first attribution job (10523064 leg C4) and
its retry (10523411) both **failed to load the reversed tree** — `uv run` and `python -c` from the live
cwd each re-add the live project root ahead of `PYTHONPATH`, and the printed
`filtered_sft_learner.__file__` proved it. A third attempt (10523423) did load the reversed tree but
reported `114 passed, 6 skipped, 0 failed`, which looked like the change *causing* the failures — it was
not: the scratch mirror had no `.git`, so `git show HEAD:…` failed and the 3 tests **skipped** (`pytest.skip(f"cannot read HEAD blob: …")`).
Giving the mirror a read-only `.git` symlink produced leg 3 above. **What is proven:** the 3 failures are
identical with and without this change, and are mechanically unreachable from it. **What is not proven:**
what actually causes them (they are a differential against the uncommitted edits to
`src/rl/value_distribution.py` from the earlier 2026-08-20 reference-alignment work — outside my scope).

### D — recipe render (job 10523063)

`DRY=1 GPU=0 bash scripts/ogpo_multitask_4task.sh` → exit 0, single-line
`uv run scripts/exp.py pi05_libero_online_ogpo_sft … --fsdp_devices 1 --resume …`.
`DRY=1 GPU=0 ARM=N bash scripts/stability_study.sh` → exit 0, same shape, `--fsdp_devices 1` present.
No flag added or removed in either.

**How "unchanged" was established, and its worth: I agree with the author, and confirmed it in source.**
`ogpo_multitask_4task.sh:283` and `stability_study.sh:217` both set `RUN=(echo uv run "$ENTRY")` under
`DRY=1` — the render is a pure `echo`, importing no `src/` module. The change touches only
`filtered_sft_learner.py`, so the render **cannot** depend on it; and the reversed mirror's `scripts/`
is byte-identical, so a diff would be trivially empty. **This check confirms the recipes still render;
it carries no evidential weight for this change.**

### GPU legs — real π0.5 weights, scratch `CKPT_BASE_DIR`, ≤2 GPUs at any time (chained by `--dependency`)

**G1 — producer (job 10523141, 1 GPU, 5m14s).** 1 LIBERO task, 8 rollouts (SR 0.625),
`collect_interval=200`, `num_train_steps=200`. Trained 0→199 and wrote `Saved resumable epoch state at
step 200`. `recipe exit=0`. 26 GB.

**G2 — M==N=1, old vs new (job 10523309, 1 GPU, 6m27s).** Same harness, only `PYTHONPATH` differs;
`[probe:old] learner module = …/prechange/…` and `[probe:new] learner module = …/vla-post-training/…`
confirm which tree each leg loaded.

```
[probe:old] CODE = OLD (openpi restore_state)
[mem:old] post-init dev0 in_use=16.15GiB peak=29.81GiB     init seconds = 323.0
[probe:old] resuming=True training_steps=200 mesh={'batch': 1, 'fsdp': 1}
[probe:old] train_state.params sharding histogram = {'PartitionSpec()': 51}
[probe:old] ema sharding histogram = {'PartitionSpec()|pinned_host': 19}
[probe:old] 'Sharding info not provided' warnings = 143
[probe:new] CODE = NEW (sharded restore)
[mem:new] post-init dev0 in_use=16.13GiB peak=28.24GiB     init seconds = 40.5
[probe:new] train_state.params sharding histogram = {'PartitionSpec()': 51}
[probe:new] ema sharding histogram = {'PartitionSpec()|pinned_host': 19}
[probe:new] 'Sharding info not provided' warnings = 0
```

Six sampled leaves' sha1 digests are **identical** between the two legs. Placement identical, values
identical, warnings 143 → 0, no memory regression.

**G3 — THE HEADLINE: FSDP=1 checkpoint resumed on 2 GPUs at `fsdp_devices=2` (job 10523310, 2 GPUs, 7m09s).**

```
16:05:33.407 [I] Restored training checkpoint from …/mt4_verifytopo_s0 at committed step 200
                 onto mesh {'batch': 1, 'fsdp': 2}   (filtered_sft_learner.py:423)
16:11:56.764 [I] Saved resumable epoch state at step 400
[g3] recipe exit=0
```

- **`'Sharding info not provided'` count in the whole log: 0.**
- Genuine 2-way FSDP on real leaves, e.g. `Sharding .params['PaliGemma']['llm']['embedder']['input_embedding'].value of shape (257152, 2048) (2009.00 MiB) along axis 0`, and the 510.68 MiB SigLIP Dense kernels along axis 2/1.
- **Training continued rather than restarting:** metrics logged at steps 200, 225, …, 375, with
  `online_buffer_size` 2560 → 5655 and `success_buffer_size` 1387 → 2136 (i.e. the replay and success
  buffers resumed and were added to, not reset).
- Wrote the FSDP=2 checkpoint at step 400 used by G4/G5.

**G4 — mirror, FSDP=2 checkpoint on 1 GPU (job 10523311, 1 GPU, 2m14s). This is job 10324589 reproduced on GPU.**

```
[probe:old] CODE = OLD (openpi restore_state)  ← …/prechange/…
Traceback (most recent call last):
    raise ValueError(
ValueError: sharding passed to deserialization should be specified, concrete and an instance of
`jax.sharding.Sharding`. Got None
[g4_restore_fsdp2to1] CODE=old exit=1

[probe:new] CODE = NEW (sharded restore)
[mem:new] post-init dev0 in_use=16.16GiB peak=28.27GiB     init seconds = 76.3
[probe:new] resuming=True training_steps=400 mesh={'batch': 1, 'fsdp': 1}
[probe:new] train_state.params sharding histogram = {'PartitionSpec()': 51}
[probe:new] 'Sharding info not provided' warnings = 0
[g4_restore_fsdp2to1] CODE=new exit=0
```

**G5 — M==N=2, NEW leg run FIRST to deconfound page cache (job 10523380, 2 GPUs, 2m07s).**

```
[probe:new] init seconds = 50.1   peak dev0=16.03GiB dev1=15.75GiB   warnings = 0
[probe:old] init seconds = 48.9   peak dev0=16.03GiB dev1=15.89GiB   warnings = 143
both: train_state.params sharding histogram =
  {'PartitionSpec()': 28, "PartitionSpec(None, None, 'fsdp')": 3, "PartitionSpec(None, 'fsdp', None)": 3,
   "PartitionSpec(None, 'fsdp', None, None)": 3, "PartitionSpec(None, None, None, 'fsdp')": 5,
   "PartitionSpec(None, 'fsdp')": 4, "PartitionSpec('fsdp', None)": 1,
   "PartitionSpec(None, None, None, 'fsdp', None)": 2, "PartitionSpec(None, None, 'fsdp', None)": 2}
both: ema sharding histogram = {'PartitionSpec()|pinned_host': 19}
```

**Byte-identical placement across 9 distinct `PartitionSpec`s at 2-way FSDP on real weights** — the
strongest form of the M == N no-op claim.

---

## 5. What could NOT be verified

- **A multi-day / long-horizon resume.** My checkpoints are at steps 200 and 400 with a 1-task,
  8-rollout configuration. The mechanism is step-independent, but I did not resume a real 40k-step
  campaign checkpoint (and deliberately never opened a `CheckpointManager` on any live run's directory).
- **`fsdp_devices` > 2, and multi-node / `jax.process_index() > 0`.** Capped at 2 GPUs; this repo is
  single-process.
- **The claimed root cause of the 3 `test_verifier_alignment.py` failures.** Proven independent of this
  change (§4); what actually causes them is out of scope.
- **A pure dtype-difference restore** (test 27): skipped, because no leaf in the tree is bf16 (§6).
  The asymmetry is reasoned, not measured.
- **Restore wall-time attribution**: the one cold-cache measurement is confounded (§3 B.5). No speed
  regression was observed; no speedup is claimed.
- **Real-GPU behaviour of the `ema_decay=None` branch** — unreachable in production, tested on CPU only.

---

## 6. Gotchas, surprises, and suggested follow-ups

1. **Out-of-scope but significant: `nnx_utils.state_map` is a complete no-op under the pinned flax
   0.10.6**, so `init_train_state`'s "Convert frozen params to bfloat16"
   (`filtered_sft_learner.py:184-188`) never happens. Mechanism, measured:
   `state_map` computes `filtered_keys = set(state.filter(filter).flat_state())`, which yields 32
   **`(path, VariableState)` tuples**, while `state.map(lambda k, v: …)` passes `k` as a bare path tuple
   — so `k in filtered_keys` is **always False**. Probe output:
   `len(filtered_keys) = 32`, `sample filtered_keys = [(('PaliGemma','llm','layers','mlp','linear'), VariableState(…)`,
   `sample k passed to state.map = [('PaliGemma','img','Transformer','encoder_norm','bias'), …]`,
   `membership of first map-k in filtered_keys: False`, and
   `after state_map cast, dtype histogram: {'float32': 51}`.
   **Confirmed at real π0.5 scale on GPU on both code paths**: `train_state.params dtype histogram =
   {'float32': 51}`. This predates the change, is identical on both paths, and affects the repo's memory
   model (the "frozen SigLIP bf16 duplicate" several docstrings assume) — **worth its own Tier-1
   investigation.** Note `tests/ogpo/test_split_equivalence.py:108-111` carries a comment claiming the
   same cast.
2. **The change's own trap bites test authors:** `dataclasses.replace` on a *tree of shardings* raises
   `BeartypeCallHintParamViolation` on `step` unless wrapped in `at.disable_typechecking()`.
3. **`action_horizon` changes no parameter shape in pi0** — use `action_dim` for a structurally
   different tree.
4. **`XLA_PYTHON_CLIENT_MEM_FRACTION=0.75` makes `nvidia-smi` useless for this repo's memory work.** Use
   `jax.local_devices()[i].memory_stats()['peak_bytes_in_use']`.
5. **`uv run` and `tests/ogpo/conftest.py` both re-add the live project root ahead of `PYTHONPATH`** —
   any future A/B against a scratch copy must either run a script whose own directory has no `src/`, or
   set `cwd` to the copy. This silently invalidated two of my attribution attempts.
6. **Doc follow-ups** (I touched no docs):
   - **Record the strictness behaviour change** (§3 B.7a) in the change record — it is the one non-no-op
     at equal topology.
   - `docs/code/rl-learners.md:75` — the `init_train_state` bullet saying resume "leaves restoration to
     orbax" is now wrong.
   - `docs/code/scripts.md:429-442` — the "Restore topology" gotcha is resolved; per CLAUDE.md step 4
     it should be deleted. `probe_candidate_q_spread.sbatch:83`'s `FSDP=2` workaround can go.
   - CLAUDE.md clone-family list — add the `_refresh_train_step` / `_refresh_critic_update_function`
     pair (`filtered_sft_learner.py:614-617` ↔ `best_of_n_learner.py:133-135`).
   - Both new docstrings should name `tests/ogpo/test_cross_topology_resume_verifier.py`
     (`best_practices.md` §8).
   - CLAUDE.md still says 302 tests; the suite is **401** (+27 from my module, all of which skip in a
     normal run).
7. **Optional hardening:** a mesh-consistency assert in `_restore_state_sharded`, and a comment at
   `filtered_sft_learner.py:411` noting both trees must come from the same `init_train_state` call.

---

## 7. Files created, jobs, cleanup, and tree state

**Permanent — the only file I added to the repo:**
`tests/ogpo/test_cross_topology_resume_verifier.py` (27 tests; final clean result
`26 passed, 1 skipped in 68.82s` standalone, all 27 skip inside a normal `pytest tests/ogpo`).

**Scratch — all DELETED.** `/home/pchellap/claude_verify_fsdp_topo/` (sbatch files, `probe_b1_static_identity.py`,
`probe_b3_cross_process.py`, `probe_strictness.py`, `probe_paths.py`, `probe_gpu_restore.py`,
`reverse_change.py`, `mk_probe_sbatch.sh`, the `prechange/` reversed mirror, `isolated/`, DRY renders)
— removed with `rm -rf` after first unlinking its symlinks (`.git`, `.venv`, `openpi`, `molmospaces`,
`run_store`, `scripts`, `uv.lock`) so nothing followed into the live tree; verified absent.
Job logs remain under `/home/pchellap/logs/` (`vfsdp*`, `vtopo_g*`, `g2_restore_fsdp1_*`,
`g4_restore_fsdp2to1_*`, `g5_restore_fsdp2_*`, `vattrib*`, `vpaths_*`, `vskip_*`, `vcleanup_*`).

**GPU scratch checkpoints — DELETED** (job 10523551):
```
[cleanup] BEFORE: 28G  .../checkpoints/verify_fsdp_topology_resume
[cleanup] AFTER:  (removed)
[cleanup] sibling dirs still present: ogpo_multitask_4task, ogpo_sweep_babel_unfrozen_backbone,
          probe_keep, stability_study
```
28 GB reclaimed; every pre-existing experiment directory untouched.

**Jobs — all 22 that I submitted are terminated** (`sacct -X`): 10523063, 10523064, 10523141, 10523143,
10523216, 10523309, 10523310, 10523311, 10523315, 10523333, 10523349, 10523380, 10523388, 10523400,
10523405, 10523407, 10523411, 10523423, 10523443, 10523467, 10523551 = COMPLETED; **10523312 = CANCELLED
by me** (my own G5, cancelled and resubmitted as 10523380 to reverse the old/new order). **Nothing else
was cancelled.** The live runs 10518424, 10518880, 10514865/66/67 and other users' jobs were never
touched. GPU usage never exceeded 2 concurrent GPUs of my own (enforced by `--dependency=afterany`
chaining). **Note:** four jobs named `vcap_transport_p/m` and `vcap_ph40_both/rec` (10523412, 10523416,
10523455, 10523456) appeared under this username during my session; **I did not submit them and did not
touch them.**

**Shared tree unmodified.** `git status --porcelain -- src scripts tests openpi` at the start and again
at the end lists the **same 25 modified tracked files**; `git -C openpi status --porcelain` is empty
(submodule clean); the only new untracked entry attributable to me is
`tests/ogpo/test_cross_topology_resume_verifier.py`. I ran no `git stash`/`checkout`/`restore`/`reset`,
edited no existing file under `src/`, `openpi/`, `scripts/` or `tests/`, and made no commits.

---

## Coordinator notes (implementing session; not part of the verifier's report)

- **Acted on** (recorded in `DIFF.md` under "Post-verification amendments"): the behaviour-change note;
  both new docstrings now cite the test module (`best_practices.md` §8); a comment at the
  `init_train_state` call site recording the same-call requirement (B.1).
- **Not acted on, deliberately:** the optional mesh-consistency `assert` in `_restore_state_sharded`
  (new behaviour beyond the approved plan; one call site, both arguments derive from `self._mesh`);
  the `state_map` bf16 no-op (§6.1, pre-existing, out of scope — needs its own change). Note it is
  **already a documented standing defect** (`docs/changes/2026-08-29-backbone-lora/BLAST-RADIUS.md`
  §Standing defects, and the `init_train_state` bullet in `docs/code/rl-learners.md`); the verifier's
  GPU measurement (`{'float32': 51}` at real π0.5 scale) is the first confirmation at that scale, not a
  new discovery;
  the stale "302 tests" count in CLAUDE.md and the stale bf16 comment in
  `tests/ogpo/test_split_equivalence.py:108-111` (pre-existing, outside the blast radius).
- The verifier's B.4 mention of "the implementer's scratch probe" refers to
  `temp/probe_fsdp_topology_resume.py` and `scripts/_test_fsdp_topology_resume.sbatch`; both, plus the
  earlier throwaway `scripts/_test_topology_producer_fsdp1.sbatch`, were removed after verification.
