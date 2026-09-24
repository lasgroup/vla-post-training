# VERIFICATION — backbone LoRA for OGPO

Step-3 independent verification. A fresh-context agent (Opus, xhigh reasoning
effort) received the change record (README, BLAST-RADIUS, PLAN, DIFF) and the
scoped diff — not the implementing session's reasoning — with an adversarial
mandate. Its report is recorded **verbatim** below, followed by the
implementing session's dispositions of its findings.

Full-suite runs were executed on the preempt partition
(`sbatch --partition=preempt --qos=preempt_qos scripts/run_ogpo_tests.sbatch`;
GPU test jobs on preempt authorized by the maintainer 2026-08-29). One
environmental fix was required en route: the first full-suite run
(job 10262117) segfaulted at ~61% — the three new model-backed test modules
run alphabetically before the heavy split-equivalence compiles and pushed RSS
past the runner's 32G cap; `jax.clear_caches()` teardowns were added to the
new module fixtures and the resubmitted run (job 10262223) completed.

---

## Verifier report (verbatim)

Scope authority: `docs/changes/2026-08-29-backbone-lora/DIFF.md`. Everything below was re-derived from source and run on this login node under `taskset -c 0-7`; no training run, no `sbatch`, no source or existing-test edits. One new file created: `tests/ogpo/test_backbone_lora_verifier.py` (9 probes, all passing).

**Verdict: no confirmed defects.** Five items are recorded as PLAUSIBLE-RISK (one of them the coverage hole the brief predicted), everything the record claims as inert or correct held up under attack, and several BLAST-RADIUS numbers were reproduced independently on the real config.

### Findings, most severe first

#### F1 — `--backbone_lora` silently re-freezes the backbone on the unfrozen-backbone config, and `ws_bcbb_pipeline.sh` can reach it — PLAUSIBLE-RISK

`OnlineTrainConfig.__post_init__` (`src/training/config.py:545`, `:585`) overwrites `freeze_filter` with `_make_ogpo_freeze_filter(allow_lora=True)` for **any** `OGPOSFTLearnerConfig`, including `pi05_libero_online_ogpo_sft_unfrozen_backbone` (registered at `scripts/exp_ogpo_unfrozen_backbone.py:54-69`, an `OGPOSFTLearnerConfig` whose whole point is `freeze_filter=PathRegex(".*PaliGemma/img.*")`). The guard does not fire, because the adapters are trainable — only the 2B base silently flips from trainable to frozen.

```
unfrozen config: llm base frozen? False
  + --backbone_lora -> llm base frozen? True  adapters frozen? False  (raised? no)
```

Reachability is not hypothetical: `scripts/stability_study.sh:123-127` now emits the flag **unconditionally**, and `scripts/ws_bcbb_pipeline.sh:40-46` invokes it via `env GPU=… bash stability_study.sh` — `env` without `-i` inherits the caller's environment, so an exported `LORA=1` reaches stage A with `CONFIG_NAME=pi05_libero_online_ogpo_sft_unfrozen_backbone`. Stage A then runs frozen-backbone-plus-LoRA while its own header still says "unfrozen BC warmstart", and the new LoRA-trap note in that header (`ws_bcbb_pipeline.sh:20-24`) describes a *different* failure (LoRA A → non-LoRA B) that would not be the one that occurred.

Suggested fix, cheapest first: (a) one sentence in the `ws_bcbb_pipeline.sh` header saying `LORA=1` converts stage A into a frozen+LoRA run; or (b) raise in `__post_init__` when `backbone_lora` is set on a config whose `freeze_filter` is not the plain `_make_ogpo_freeze_filter()` output.

#### F2 — the guard's one-path probe does miss a partial adapter freeze — PLAUSIBLE-RISK (the brief's claim B, confirmed)

`config.py:591-594` probes exactly one leaf, `("PaliGemma","llm","layers","attn","q_einsum","lora_a")`. A filter that freezes only the FFN adapters passes the probe and the config constructs silently, leaving 4 of the 10 adapter leaves randomly initialized (`normal(0.01)` for `lora_a`), never loaded from the checkpoint, and never trained — the exact trap the guard exists for, on a subset. Demonstrated in `tests/ogpo/test_backbone_lora_verifier.py::test_v3_guard_probe_misses_a_partial_adapter_freeze` (passes, i.e. the hole exists).

Mitigating: `freeze_filter` is `tyro.conf.Suppress`ed, so such a filter cannot come from the CLI — it takes an edit to `config.py` or a wrapper like `exp_ogpo_unfrozen_backbone.py`. So this is a **coverage limit, not a live defect**.

Recommendation, per the brief's own framing: either widen the probe to one path per adapter family (5 tuples — `attn/q_einsum`, `attn/kv_einsum`, `attn/attn_vec_einsum`, `mlp/gating_einsum_lora_a`, `mlp/linear_lora_a`, which also pins the two *different* naming shapes), or state in the record that the guard is a tripwire for the known CLI trap and not a proof of filter correctness. The 5-tuple version is ~4 extra lines and closes it completely for the tree that exists.

#### F3 — the extra prefix forward also fires at **evaluation**, where the prefix is discarded — PLAUSIBLE-RISK (throughput/VRAM, GPU-only to confirm)

`_generate_actions` passes `return_prefix_rep=self._config.collect.store_prefix_rep` (`filtered_sft_learner.py:707`), and `--collect.store_prefix_rep` is set in both live recipes (`stability_study.sh`, `ogpo_multitask_4task.sh`). `evaluate_policy` reaches the same `agent.sample_actions` path (`src/training/collect.py:55`, `:155`). So under `backbone_lora` the second, adapter-zeroed 968-token forward at `filtered_sft_learner.py:668-670` runs on **every eval query too**, and its result is thrown away by the caller. The in-code comment (`:655-658`) says "per policy query at collection … roughly doubles it at collection" — accurate for collection, silent about eval. At `EVAL_ROLLOUTS=32 × 29 tasks` that is not a rounding error.

Second-order, same site: `zero_lora_params` runs **eagerly** here (and in `_attach_prefix_embeddings_to_episode_data:819` and the two best-of-N scoring blocks), so each call allocates 10 fresh device arrays totalling **111.5 MB f32** (27,869,184 params, measured below) that are then dropped. Per policy query. Correct, but a caching or hoisting opportunity, and it adds to the collection-time peak.

Neither is a correctness problem. Both are GPU-only to quantify.

#### F4 — the guard and the `create_trained_policy` twin are blind to `action_expert_variant` LoRA — PLAUSIBLE-RISK (low; outcome today is loud)

The guard tests `"lora" in self.model.paligemma_variant` (`config.py:589-590`) and the twin strips `_lora` only from `paligemma_variant` (`filtered_sft_learner.py:398-408`). `--model.action_expert_variant gemma_300m_lora` is a live tyro flag:

```
action_expert_variant=gemma_300m_lora: constructed OK, paligemma='gemma_2b' expert='gemma_300m_lora'
   guard fired? no.  expert-adapter frozen by the OGPO filter? False
   create_trained_policy twin still carries expert lora: gemma_300m_lora
```

Today this is **not** a silent trap: the expert adapters land under `.*llm.*_1.*`, which the OGPO filter leaves trainable, and the strict `BaseModelConfig.load` inside `create_trained_policy` then raises on the checkpoint's missing expert-lora keys — a loud crash at learner construction. Worth recording only so the DIFF's "the twin is the lora-less config" is read as "…for the PaliGemma variant".

#### F5 — half-fixed clone: the DSRL copy of `init_train_state` got no `zero_lora_b_params` — PLAUSIBLE-RISK (inert)

`src/rl/dsrl/dsrl_env.py:93-94` is the second copy of the `params = nnx.state(model)` → `state_map` block and was not given the R3 zeroing. DSRL is unwired and its config matches no dispatch branch, so this is inert — but the PLAN's duplication sweep said "one copy at most" and the pair is now asymmetric. A one-line note in `DIFF.md` §Duplication sweep would close it.

#### Two behavioral notes for the run ledger (not defects)

- **At step 0 of a real LoRA run, `lora_a` receives exactly zero gradient.** With `lora_b = 0`, `∂L/∂a ∝ b = 0`; only `lora_b` moves on the first update, and `lora_a` comes alive from step 1. Pinned in `test_v4_zeroing_both_factors_is_permanently_dead_zeroing_b_is_not`. `grad_norm_lora` is still `> 0` at step 0 (it is `lora_b`'s). Worth saying so nobody reads "lora_a isn't moving" at step 0 as a plumbing bug. (AdamW `weight_decay=1e-10`, `openpi/.../optimizer.py:73`, so decay-on-zero-grad is negligible.)
- **Fixture reach:** on the dummy-LoRA fixture, `PaliGemma/img/head/kernel` is zero-initialized, so the image tokens are identically zero regardless of image content — the prefix is bit-identical across completely different images. The adapter-gradient and dedup tests therefore exercise the LLM prefix over 776 constant-valued token positions, not image-conditioned features. The structural claims (grads flow, dedup cotangent equivalence, R1 invariance) are unaffected; the tests are just narrower than "the model sees the images".

### Claims attacked and not broken

#### A — inertness for every non-LoRA config — CONFIRMED-OK

| # | Claim | Evidence |
|---|---|---|
| A1 | `_zero_matching` returns the input `is`-identical when nothing matches | `nnx.filter_state` returns `State({})`, `len 0`, falsy — on flat **and** nested trees. `test_v1_…`; also `test_backbone_lora_init.py::test_zero_lora_params_is_identity_without_adapters` on a real dummy Pi0 state. Verified on the real `gemma_2b` tree: `LORA_FILTER` matches 0 of 51 leaves. |
| A2 | `_sample_action`'s else branch is byte-identical to the pre-change fused call | `git diff -w -U0 -- src/rl/filtered_sft_agent/filtered_sft_learner.py` shows **no** `-`/`+` on the fused-call lines — the change is leading indentation only. Also re-verified from `openpi/.../pi0.py:450-455`: `return_prefix_rep=False, return_info_dict=False` ⇒ `results` has length 1 ⇒ `sample_actions` returns `x_0` bare, so the LoRA branch's `raw_actions = …` unpacking is right. |
| A3 | jit-1's `critic_prefix is None` recompute traces to the same graph on a lora-less tree | Two independent legs. (i) `zero_lora_params(params) is params` on a lora-less tree, so `update_actor.py:194-196` is literally the pre-change expression. (ii) `test_split_equivalence.py::test_leg_d_none_branch_recompute_matches_original` is a genuine differential — `fx.prefix` comes from `_original_recompute` (`:146-152`, the transcribed pre-change body on the un-zeroed model) and the None branch must reproduce it at `atol=1e-6`. It passed in the cluster full-suite run. |
| A4 | The `create_trained_policy` twin is a valued no-op for every registered config | For all 11 registered `OnlineTrainConfig`s: `removesuffix("_lora")` is a no-op, `twin.model == cfg.model`, **zero** differing fields, and `twin.freeze_filter is cfg.freeze_filter`. Two openpi-side configs do carry `gemma_2b_lora` — `pi0_libero_low_mem_finetune`, `pi0_fast_libero_low_mem_finetune` — but neither is an `OnlineTrainConfig`, so neither reaches the guard, the twin, or `init_train_state`. |
| A5 | Every registered config still parses with `--no-backbone_lora` and without any flag | 33 real `_config.cli()` parses (11 configs × {no flag, `--no-backbone_lora`, `--backbone_lora`}). All 11 resolve to `backbone_lora=False`, `gemma_2b`, unchanged filter type in both no-flag forms. `--backbone_lora` raises the intended `ValueError` on all 8 non-OGPO configs and resolves correctly on the 3 OGPO ones. |
| A6 | `_get_policy_model(zero_lora=…)` default leaves all existing callers untouched | Exhaustive grep of `_get_policy_model` call sites: `advantage_weighted_sft_learner.py:301` (`zero_lora=True`, prefix), `best_of_n_learner.py:228` (`zero_lora=True`, prefix), `best_of_n_learner.py:556` (`_get_on_policy_action`, action path, default), `mpo_weighted_sft_learner.py:25` (`_get_on_policy_action`, action path, default — resolves to AWR's `@staticmethod` through the MPO→AWR chain). FlowGRPO defines none and inherits AWR's `_sft_batch_to_actor_batch`, which routes through `_recompute_prefix_embedding(zero_lora=True)`. No caller loses its old behavior. |

Independent reproduction of the BLAST-RADIUS tree numbers, on the **real** `Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False, paligemma_variant=…, action_expert_variant="gemma_300m")` via `nnx.eval_shape` (no weights, no GPU):

```
=== gemma_2b_lora: 61 leaves, 3,381,303,056 params
   LORA_FILTER (.*lora.*)             leaves= 10 params=27,869,184
   LORA_B_FILTER (.*lora_b.*)         leaves=  5 params=15,482,880
   paths containing 'lora': 10 == LORA_FILTER matches: True
   FROZEN allow_lora=False            leaves= 42 params=2,951,204,592
   TRAINABLE allow_lora=False         leaves= 19 params=430,098,464     <- the trap: adapters frozen
   FROZEN allow_lora=True             leaves= 32 params=2,923,335,408
   TRAINABLE allow_lora=True          leaves= 29 params=457,967,648     <- +6.48%
=== gemma_2b: 51 leaves, 3,353,433,872 params
   FROZEN allow_lora=False            leaves= 32 params=2,923,335,408
   FROZEN allow_lora=True             leaves= 32 params=2,923,335,408
   two filter modes identical on lora-less tree: True
```

All ten adapter paths match the record's enumeration exactly; no base leaf's path contains the substring `lora`, so `PathRegex(".*lora.*")` has no false positives on the real tree. The "inert-but-safe under a variant flip" claim (the reason `Pi0Config.get_freeze_filter()` was rejected) holds on the real tree, not just on the test's literal path list.

#### B — the guard's coverage hole — see F2

#### C — R1 completeness — CONFIRMED-OK

Independent grep of every producer/consumer of `PREFIX_EMBEDDING_NAME` and `get_prefix_rep` across `src/` and `scripts/`. Producers, and their status:

| Producer | Status |
|---|---|
| `filtered_sft_learner.py:649-670` `_sample_action` | gated on `backbone_lora`; adapter-zeroed model on the flag-on branch |
| `filtered_sft_learner.py:817-825` `_attach_prefix_embeddings_to_episode_data` | `zero_lora_params(compose_full_params(...))` |
| `advantage_weighted_sft_learner.py:290-308` `_recompute_prefix_embedding` | `zero_lora=True`; serves the critic-batch recomputes (`:365`, `:368`) **and** `_sft_batch_to_actor_batch` (`:393`), which is the path MPO and FlowGRPO inherit |
| `advantage_weighted_sft_learner.py:474-483`, `:566` best-of-N scoring | `prefix_model` = zeroed compose |
| `update_actor.py:194-199` jit-1 `critic_prefix is None` | `zero_lora_params(policy_state.params)` |
| `best_of_n_learner.py:228`, `:392`, `:474` | same two edits, clone-family complete |
| `advantage_weighted_sft_learner.py:88`, `best_of_n_learner.py:45` | `model.get_prefix_rep(fake_obs)[0]` at learner `__init__`, **shape only** — provably harmless |

Consumers need no change, verified: the critic TD update reads `observation_dict[PREFIX_EMBEDDING_NAME]` (stored or passthrough), the OGPO advantage sidecar is a passthrough (`ogpo_learner.py:922` → `update_actor.py:198-208`), `bronet_critic.py:82-83` and `best_of_n/update_critic.py:81-82` only key on presence, and the four `scripts/probe_*.py` read prefixes **out of the buffer** — which is base-backbone by construction once the two writers above are zeroed. FlowGRPO has no prefix producer of its own. Eval reaches the same `_sample_action`.

The zeroed prefix reads only frozen leaves: `Pi0.get_prefix_rep` (`pi0.py:460-475`) touches `PaliGemma.img` (frozen), `llm.embedder` (frozen) and LLM stack 0 (frozen). So with the adapters zeroed the critic prefix is a pure function of the observation and invariant to params-vs-EMA and to the training step, exactly as `lora_utils.py:9-16` claims. Certified end-to-end by `test_backbone_lora_init.py::test_zeroed_lora_prefix_equals_the_base_model_prefix` (bit-exact) and at the jit-1 level by `test_backbone_lora_grads.py::test_critic_prefix_recompute_is_invariant_to_adapter_values`.

#### D — R3 correctness — CONFIRMED-OK

- **Cannot fire on the resume path.** `init_train_state` runs the inner `init` twice: `jax.eval_shape(init, init_rng)` (unconditional, shape-only) and the `jax.jit(init, …)(init_rng, partial_params)`, which is reached only **after** the `if resume: return` short-circuit. On resume the train state comes from `_checkpoints.restore_state`, which overwrites the whole tree.
- **`jax.eval_shape` output unaffected.** `jnp.zeros_like` preserves shape and dtype; verified structurally in `test_v2_…` (both zeroing helpers, mixed f32/bf16 leaves). This matters because that same tree drives `sharding.fsdp_sharding` and the jitted init's `out_shardings` — derived from the same function, so they cannot disagree.
- **`ema_params` at step 0 equals the zeroed params.** Same object, not a copy.
- **The weight loader's `.*lora.*` back-fill cannot overwrite the zeros.** The back-filled lora keys are `ShapeDtypeStruct`s, stripped by `_load_weights_and_validate`, so no lora key is in `partial_params` and `nnx.replace_by_pure_dict` never touches them. The zeroing runs afterwards regardless. The regex is `fullmatch` against **slash-joined** paths, so both naming shapes match.
- **The dead-adapter claim itself.** `test_v4_…`: with both factors zero, `∂L/∂a` and `∂L/∂b` are exactly `0.0` — dead forever. With only `lora_b` zero, `∂L/∂b ≠ 0` and `∂L/∂a == 0`, and after one step on `b`, `∂L/∂a ≠ 0`.

#### E — the zero-init adaRMS gate discovery — CONFIRMED-OK, and the discovery is structural, not fixture-specific

Confirmed independently, three ways:

1. **Source.** `gemma.RMSNorm`'s adaptive branch builds its modulation head as `nn.Dense(3·width, kernel_init=nn.initializers.zeros)` (`openpi/src/openpi/models/gemma.py:128`), so `scale = shift = gate = 0` at init; `_gated_residual` then drops both the attention and the FFN branch of every suffix block. Variant-independent.
2. **Tree.** `.*norm.*Dense_0.*` matches exactly the 6 adaRMS heads, all on the action-expert (`_1`) branch, all all-zero (`test_v6_…`).
3. **Behavior.** With gates at init, the sampled chain is **bit-identical** under two very different adapter settings (`test_v7_…`, `maxdiff = 0.0`); randomizing the gates restores the dependence (`maxdiff = 0.139`).

**Is the fixture load-bearing rather than cosmetic?** Yes, and it does not launder a failure: `test_v8_…` runs the real jit-2a under the real production `allow_lora` filter on un-unblocked params and gets `global_norm(lora grads) == 0.0` exactly. So without `_unblock_adarms_gates`, the grads tests' three adapter assertions would **fail**, not silently pass. The fixture makes the tests possible; it does not change what they claim about production, where the gates come from the trained π0.5 checkpoint (a π0.5 with zero gates would be a policy that ignores its own observations — though I cannot check the real checkpoint here). The DIFF's flag that the pre-existing suite's backbone-grad coverage at random init is weaker than it looks is correct and worth carrying into `docs/code/tests.md`.

**Is the dedup tolerance masking a scaling bug?** No. `test_v9_…`, measured:

```
[V9] adapter grads: n_off=0.262512 n_on=0.262347 |Δnorm|/norm=6.302e-04 max|Δ|=2.441e-04 cos=0.999997438
```

The aggregate norms agree to **6.3e-4** relative — 16× tighter than the shipped 1% aggregate bound — and the two gradient vectors are cosine-aligned to `1 − 2.6e-6`. Max elementwise `|Δ| = 2.44e-4`, matching the DIFF's stated measurement and 4× inside the shipped `atol=1e-3`. Direction agreement two orders tighter than the elementwise bound is the signature of bf16 reassociation on small elements, not a systematic scale error. The shipped tolerances **reject** the two failure modes the test exists for: a 2× cotangent scaling fails both checks, and a dropped cotangent (which would zero the adapters outright) fails too. The relaxation is honest.

#### F — metrics plumbing — CONFIRMED-OK

`grad_norm_lora`/`grad_norm_rest` are added to `loss_aux` at `update_actor.py:637-639`. Every consumer is key-generic: jit-2b's aux out-sharding is a single replicated broadcast; micro-batch averaging is `jax.tree.map`; `exp.py:174-182` unions keys and `nanmean`s. Exhaustive grep for key-set/count assertions found only `test_split_equivalence.py` (`_N_INFO_KEYS = 33` on the frozen reference, correctly unchanged; `_SPLIT_ONLY_INFO_KEYS` correctly +2). `optax.global_norm` on an empty `State` returns an exact `float32` `0.0` (`test_v5_…`), and the complement carries the whole norm bit-exactly. 33 + 5 split-only = 38 info keys, consistent with the updated schema comment.

#### G — recipes — CONFIRMED-OK, including stability_study.sh (better than inspection)

- `bash -n` passes on all three scripts.
- **mt4**, via the shipped `DRY=1` harness: LORA=unset/0 → `--no-backbone_lora` (once); LORA=1 → `--backbone_lora` (once); re-parsed through the real tyro CLI by the shipped test.
- **stability_study.sh has no `DRY` mode**, but honors `ENTRY` and `STORE_ROOT`: pointing `ENTRY` at an argv-dumping script gave a real dry inspection (nothing written into the repo). LORA=unset/0/1 emit exactly one correct flag, and feeding the captured argv back through `_config.cli()` resolves to the right variant + filter polarity in both directions. So the flag-pair spelling is proven to parse from **both** recipes. Residual limitation: this exercised the flag block and full flag stack, not the recipe's GPU/EGL preamble.
- The dedup-comment divergence (⚠2 in DIFF.md) is real: bash cannot carry a comment on a `\` continuation line. The note lives in the knob blocks. Acceptable.

### Test tally

New/edited by this change, run locally under `taskset -c 0-7`:

| File | Result |
|---|---|
| `tests/ogpo/test_backbone_lora_config.py` | **10 passed**, 30.3 s |
| `tests/ogpo/test_backbone_lora_init.py` | **8 passed**, 115.6 s |
| `tests/ogpo/test_backbone_lora_grads.py` | **6 passed**, 185.3 s |
| `tests/ogpo/test_backbone_lora_verifier.py` (verifier-authored, 9 probes) | **9 passed**, 142.4 s |

Full suite on the cluster (`/home/pchellap/logs/ogpo_tests_10262223.out`):

```
3 failed, 341 passed, 6 skipped, 255 warnings in 1078.44s (0:17:58)
```

350 collected = the 326 baseline + the 24 new tests. The three failures are **pre-existing and unrelated**: self-invalidating HEAD-differential tests in `test_verifier_alignment.py` (`_module_at_head` / `_head_get_value_bounds` `git show HEAD:…` the pre-change source and assert it differs from the working tree; HEAD is `5b94510`, which already **contains** the change they were written for, so `before == after`). Decisive: `src/envs/wrappers.py` is byte-identical to HEAD yet the wrapper differential still expects a difference, and the LoRA diff touches neither `src/envs/wrappers.py` nor the `get_value_bounds` body. They will keep failing until someone reworks or deletes them — worth a line in the record, but out of scope here.

### Honest fallbacks — what could not be verified, and why

1. **Real-checkpoint weight loading.** The `.*lora.*` back-fill, the `ShapeDtypeStruct` strip, and the `create_trained_policy` twin against the actual `gs://openpi-assets` π0.5 checkpoint — verified at the code and abstract-shape level only; the weights are not on this machine.
2. **VRAM.** The dominant unknown — the B·G KV-cache cotangent, analytic 4.6 GB at B·G=256 up to possible OOM at 1024 — is GPU-only. `scripts/exp_ogpo_unfrozen_backbone_memdiag.py` with `--backbone_lora` and its LORA=0 twin is the right instrument.
3. **Throughput.** F3's eval-side doubling and the per-query 111.5 MB zeros allocation are arithmetic, not measurements.
4. **Training behavior.** Whether the LoRA arm learns, and whether R1's deliberately frozen critic representation is the right call as the policy's representation drifts, is the experiment, not verification.
5. **`_unblock_adarms_gates` as a production proxy.** Structurally sound, but the real π0.5 gate values were not inspected.
6. **The recipes' GPU/EGL preamble** — not exercised by the dry inspections.
7. **`pytest tests/` still fails at collection** (OQ-4). Untouched by this change, correctly left alone.

---

## Dispositions (implementing session, post-verification)

| Finding | Action taken |
|---|---|
| F1 | Fixed via the verifier's option (a): `ws_bcbb_pipeline.sh` header now warns that an exported `LORA=1` reaches stage A through `env`'s inherited environment and silently converts it to a frozen+LoRA run. Option (b) (raise on a non-standard filter) was not taken — filter objects aren't comparable, and the unfrozen-backbone wrapper is the documented mechanism for custom filters. |
| F2 | **Fixed**: the guard now probes all 10 adapter paths (`config.py`), so any partial adapter freeze — FFN-only, single-factor `lora_b`-only — raises. The verifier's `test_v3` was inverted accordingly (`test_v3_guard_probe_catches_a_partial_adapter_freeze`, now also covering the `lora_b`-only variant) and passes. |
| F3 | Comment at the `_sample_action` split now states the eval-side cost explicitly; threading an eval flag through `_generate_actions` is recorded as a deferred optimization, to be motivated by the GPU measurement. |
| F4 | Recorded (DIFF.md addendum): the twin and guard are PaliGemma-variant-scoped; `action_expert_variant` LoRA fails loudly at learner construction via the strict `BaseModelConfig.load`, so no silent trap exists today. |
| F5 | Recorded (DIFF.md addendum): the DSRL `init_train_state` clone intentionally did not receive the R3 zeroing — DSRL is unwired, unreachable, and quarantined. |
| Ledger notes | Carried into `docs/ogpo_experiment_notes_libero44.md` in the step-4 doc pass (step-0 `lora_a` zero-gradient nuance; norm-series comparability). |
| Pre-existing suite failures | The three expired HEAD-differential tests in `test_verifier_alignment.py` are unrelated to this change and left alone (OQ-4 rule); flagged to the maintainer in the session summary. |

Post-fix test state: `test_backbone_lora_config.py` 10/10 +
`test_backbone_lora_verifier.py` 9/9 rerun green after the guard widening
(165 s, capped login-node run).

## GPU verification (2026-08-30, preempt A100-80GB)

Paired memdiag runs through the recipe
(`ENTRY=scripts/exp_ogpo_unfrozen_backbone_memdiag.py` via
`stability_study.sh`, `pi05_libero_online_ogpo_sft`, B=32, G=8, dedup on,
`store_prefix_rep` on, real `gs://openpi-assets` π0.5 weights, `N_STEPS=1000`
so ~11 policy updates run past the recipe's `training_start_step=900`
critic head start; jobs 10264410 `LORA=1` / 10264411 `LORA=0`, both
COMPLETED, ~10 min each; logs + JSONLs in `~/logs/lora_memdiag/`). This
doubles as the real-weights smoke: learner construction (the
`create_trained_policy` twin), the `.*lora.*` weight-loader back-fill, the
`lora_b` zeroing, the split collection-time prefix path, and the full
five-jit policy update all ran against the real checkpoint.

**Adapter gradients are live on real weights**: `actor/grad_norm_lora ≈
0.005` on the LoRA arm (the trained adaRMS gates make the backbone path
real, as predicted) and exactly `0.0000` on the baseline (the
zero-by-construction claim at scale).

**VRAM, measured (max live bytes inside each phase; LoRA vs baseline):**

| Phase | LoRA | base | Δ |
|---|---|---|---|
| overall peak | **43.7 GiB** | **30.5 GiB** | **+13.2** |
| collection (`sample_action`) | 43.6 | 30.5 | +13.1 |
| jit-1 `sampler_adv` | 20.6 | 20.2 | +0.4 |
| jit-2a `loss_grad_pg` | 20.7 | 20.1 | **+0.6** |
| jit-2b `bc_grad_acc` | **32.2** | 20.4 | **+11.8** |
| jit-3 `opt_tail` | 20.7 | 20.1 | +0.6 |
| critic update | 18.7 | 18.4 | +0.4 |

Three conclusions:

1. **The feared B·G KV-cache cotangent did not materialize.** The record's
   analytic worst case for jit-2a was 4.6–9 GiB at B·G=256; measured, the
   rescorer's backward with live adapters costs **+0.6 GiB** — XLA
   rematerializes the tiled cache under the `nothing_saveable` remat rather
   than storing its cotangent.
2. **The real policy-update cost is jit-2b** (+11.8 GiB): the BC anchor's
   `compute_loss` runs the joint prefix+suffix forward at B=32 with a live
   backbone backward. It stays below the collection peak, so it never sets
   the run's high-water mark.
3. **The run peak is set at collection** (+13.1 GiB): the R1 split's second,
   adapter-zeroed prefix forward — magnitude consistent with a second full
   f32 param-tree materialization around the standalone
   `_get_prefix_rep_with_model` executable, not with the forward itself.
   This is the one optimization target if 48 GB cards matter
   (43.7 GiB total is tight on an A6000-48GB; comfortable on A100-80GB).

Still unmeasured: throughput (the eval-side F3 doubling), and training
behavior — the experiment itself.
