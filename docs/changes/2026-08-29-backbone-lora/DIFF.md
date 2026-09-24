# DIFF — backbone LoRA for OGPO

What actually changed, file by file, against `PLAN.md`. Divergences from the
plan are marked ⚠ and explained at the end.

## Source

### `src/training/config.py`
- `_make_ogpo_freeze_filter(*, allow_lora: bool = False)` — parameterized in
  place (no fourth regex copy); `allow_lora=True` appends
  `nnx.Not(PathRegex(".*lora.*"))` to the LLM branch. The three existing call
  sites are untouched and produce a bit-identical filter. Docstring rewritten
  (the old "cast to bfloat16 by init_train_state" claim was dropped — the cast
  is a known no-op, see BLAST-RADIUS §Standing defects — and the LoRA
  semantics + the rejected `get_freeze_filter()` route documented).
- New field `OnlineTrainConfig.backbone_lora: bool = False` (after
  `default_prompt`), commented as a rewrite flag.
- `__post_init__` gains two blocks after the existing validations:
  1. the rewrite — raises for non-OGPO `rl` configs and non-`Pi0Config`
     models; otherwise `object.__setattr__`s `model` (via
     `dataclasses.replace(..., paligemma_variant="gemma_2b_lora")`) and
     `freeze_filter` (`_make_ogpo_freeze_filter(allow_lora=True)`). Idempotent.
  2. the guard — for any Pi0Config whose variant contains `"lora"`, probes the
     resolved freeze filter with the literal path
     `("PaliGemma","llm","layers","attn","q_einsum","lora_a")` via
     `nnx.filterlib.to_predicate` and raises (fix in message: "pass
     --backbone_lora") if the adapters would be frozen. Converts the live CLI
     trap from silent to loud.
- `dedup_group_prefix` comment gains the memory-load-bearing note.

### `src/rl/lora_utils.py` (new)
`LORA_FILTER` / `LORA_B_FILTER` (PathRegex), `zero_lora_params` /
`zero_lora_b_params` via a shared `_zero_matching`: `nnx.filter_state` +
`jax.tree.map(jnp.zeros_like)` + `nnx.merge_state`, returning the input
**unchanged** (`is`-identity) when nothing matches — the property that lets
every call site stay unconditional. Module docstring carries the R1 contract,
the dead-adapter-init rationale, and the `state_map`-is-a-no-op warning.

### `src/rl/filtered_sft_agent/filtered_sft_learner.py`
- import of `zero_lora_b_params, zero_lora_params`.
- `init_train_state`'s inner `init()`: `params = zero_lora_b_params(params)`
  immediately after `params = nnx.state(model)` (R3; fresh-init only — the
  resume path short-circuits before the jitted `init`, and the `eval_shape`
  pass is shape-only).
- `__init__`: `create_trained_policy` now receives a lora-less twin config —
  `dataclasses.replace(self._config, backbone_lora=False,
  model=replace(model, paligemma_variant=variant.removesuffix("_lora")))` (R2).
  ⚠ plan said hardcode `"gemma_2b"`; `removesuffix("_lora")` is the true no-op
  for every variant (incl. dummy test variants).
- `_sample_action` (R1 site 1): gated on `self._config.backbone_lora` — flag
  off keeps the single fused `return_prefix_rep=True` call bit-for-bit; flag
  on samples actions without the prefix rep and computes the critic prefix
  through `nnx.merge(model_def, zero_lora_params(params))` +
  `_get_prefix_rep_with_model` (one extra prefix-only forward per policy
  query at collection; cost quantified in the comment).
- `_attach_prefix_embeddings_to_episode_data` (R1 site 2): the composed EMA
  params are wrapped in `zero_lora_params(...)`.

### `src/rl/advantage_weighted_sft/advantage_weighted_sft_learner.py`
- import of `zero_lora_params`.
- `_get_policy_model` gains `*, zero_lora: bool = False` (stays
  `@staticmethod`); `_recompute_prefix_embedding` passes `zero_lora=True`
  (R1 site 3 — covers the critic-batch recomputes at `:348/:351` and the
  actor-batch recompute at `:376`). The `# TODO: this should be EMA` is
  scoped to the actor path (moot for the zero-lora prefix path).
- Best-of-N scoring block (R1 site 4): the compose is wrapped in
  `zero_lora_params(...)` and the variable renamed `policy_model` →
  `prefix_model` (it exists only for the prefix forward; the candidates come
  from `_sample_action`).

### `src/rl/best_of_n/best_of_n_learner.py` (clone family, R1 site 6)
Same two edits verbatim: `_get_policy_model(..., zero_lora=False)` keyword
(+ `zero_lora=True` from its `_recompute_prefix_embedding`), and the scoring
block's compose wrapped + renamed to `prefix_model`. `_get_on_policy_action`
(`:556`) keeps the un-zeroed model — action path.

### `src/rl/ogpo/update_actor.py`
- import of `LORA_FILTER, zero_lora_params`.
- jit-1 `critic_prefix is None` recompute (R1 site 5):
  `nnx.merge(policy_state.model_def, zero_lora_params(policy_state.params))`.
- jit-2b `loss_aux` gains `grad_norm_lora` / `grad_norm_rest`
  (`optax.global_norm` over the `LORA_FILTER` / `Not(LORA_FILTER)` splits of
  the combined grads); aux schema comment updated 25→27 keys / 36→38 info
  keys.
- Module docstring: frozen-backbone contract rewritten (adapters trainable
  under the flag; prefix forwards become live backward paths; critic prefix
  stays adapter-free).

### `src/rl/ogpo/sampling.py`
`sample_chain_with_logprob_grouped` docstring: dedup is memory-load-bearing
once the backbone carries trainable adapters.

## Recipes

### `scripts/ogpo_multitask_4task.sh`, `scripts/stability_study.sh`
`LORA` knob, emitted unconditionally in both branches
(`--backbone_lora` / `--no-backbone_lora` — the tyro flag-pair spelling; the
value form does not parse), after the `PER_TASK_CRITIC` block / in the
`EXTRA_FLAGS` block respectively. Comment covers: env-var authority for any
CONFIG_NAME (contract preserved), new-run-only, and the dedup memory note.
⚠ the plan's "one-line pointer at the dedup flag line" is impossible — bash
cannot carry a comment on a `\` continuation line (an attempt broke the
command and was reverted); the note lives in the knob block instead.
`_ref` + smoke recipes inherit via delegation.

### `scripts/ws_bcbb_pipeline.sh`
Header gains the reverse trap: a LoRA stage A feeding the non-LoRA stage B
silently drops the adapters via `_merge_params`.

## Tests

### New: `tests/ogpo/test_backbone_lora_config.py` (10 tests)
Filter polarity on the literal real-tree path list (lora-aware + the OLD
filter freezing adapters — pins the trap), lora-less-tree equivalence of the
two filter modes, the `__post_init__` rewrite (replace-not-restate), both
guard raises (fix-in-message asserted), all-registered-configs-off, the
policy-twin expression (pins the rewrites-it-back trap), and the mt4
`DRY=1 LORA=1/0/unset` resolution through the `_dry`/`_resolve` harness
(also the proof the flag spelling parses).

### New: `tests/ogpo/test_backbone_lora_init.py` (8 tests)
Dummy-width LoRA model via monkeypatched `gemma.get_config` (rank-4, no
submodule edit); lora-less twin built by copying base leaves. Certifies:
`zero_lora_params` `is`-identity on lora-less trees and exact zeroing of
exactly the adapter leaves; **the R1 key property** (zeroed-lora prefix ==
base-model prefix, bit-exact); prefix invariance to adapter values with an
unzeroed negative control; the R3 differential (chains AND prefix stream ==
the non-lora twin after `lora_b` zeroing); `lora_a` stays nonzero (dead-
adapter trap); `lora.Einsum` exact identity at `lora_b=0`; and a regression
pin that `nnx_utils.state_map` would NOT have zeroed anything.

### New: `tests/ogpo/test_backbone_lora_grads.py` (6 tests)
The production `allow_lora` filter on the dummy-lora model: adapters get
nonzero PG and BC grads and the frozen set is absent from the grad tree;
adapter grads match with `dedup_group_prefix` on/off (elementwise
atol 1e-3 / rtol 5% with the bf16+reassociation rationale stated, plus an
aggregate-norm check at 1%); `grad_norm² == grad_norm_lora² +
grad_norm_rest²`; `grad_norm_lora == 0.0` exactly without adapters; jit-1's
recompute invariant to adapter values (`v_mean` bit-equal) while the sampled
chains differ (non-vacuousness control).
⚠ **Fixture discovery**: pi05's adaRMS gate heads (`.*norm.*_1/Dense_0`) are
zero-initialized, so at random init the suffix stream bypasses attention
entirely — v_t is bit-independent of the KV cache, the images, and the
adapters (verified on dummy AND gemma_300m). The fixture randomizes the gate
heads (`_unblock_adarms_gates`); without it every adapter-gradient assertion
is vacuously false. This also means the pre-existing suite's backbone-grad
coverage at random init is weaker than it looks — flagged for the verifier.

### Edited: `tests/ogpo/test_split_equivalence.py`
`_SPLIT_ONLY_INFO_KEYS` gains `grad_norm_lora`, `grad_norm_rest` (the
reference monolith predates them). `_N_INFO_KEYS` stays 33 (asserted on the
frozen reference).

## Verification status at DIFF time

- `test_backbone_lora_config.py` 10/10, `test_backbone_lora_init.py` 8/8,
  `test_backbone_lora_grads.py` 6/6 — green on the login node under
  `taskset -c 0-7` (the XLA-vs-RLIMIT_NPROC abort otherwise kills
  model-backed tests there).
- Full `pytest tests/ogpo`: submitted to the preempt partition
  (job 10262117, `sbatch --partition=preempt --qos=preempt_qos
  scripts/run_ogpo_tests.sbatch`) — the login node aborts partway through
  the full suite even with affinity capping. Result recorded in
  VERIFICATION.md.
- GPU legs (memdiag VRAM, real-weights smoke) pending per PLAN.md
  §Verification.

## Addendum (2026-08-30, post-verification)

Applied in response to the verifier's findings (VERIFICATION.md):

- **Guard widened (F2)**: `__post_init__` now probes all 10 adapter paths
  instead of one, so a filter freezing any subset of the adapters (FFN-only,
  `lora_b`-only) raises. The verifier's `test_v3` was inverted to pin the
  widened guard.
- **`ws_bcbb_pipeline.sh` header (F1)**: warns that an exported `LORA=1`
  reaches stage A through `env`'s inherited environment and silently converts
  the unfrozen-backbone warmstart into a frozen+LoRA run.
- **`_sample_action` comment (F3)**: states that the second, adapter-zeroed
  prefix forward also fires on eval queries (where the prefix is discarded);
  an eval-flag optimization is deferred pending the GPU measurement.
- **Scope notes (F4, F5)**: the guard and the `create_trained_policy` twin are
  PaliGemma-variant-scoped — `--model.action_expert_variant gemma_300m_lora`
  is not silently trapped (it fails loudly at learner construction via the
  strict `BaseModelConfig.load`). The DSRL `init_train_state` clone
  (`dsrl_env.py:93-94`) deliberately did not receive the R3 zeroing — DSRL is
  unwired, unreachable, and quarantined; the pair is asymmetric by intent.
- **Suite-runner fix**: `jax.clear_caches()` teardowns added to the two new
  module fixtures — the first full-suite sbatch run segfaulted at ~61% because
  the new modules' resident compiled executables pushed RSS past the runner's
  32G cap right at the split-equivalence compiles.

## Divergences from PLAN.md

1. Twin variant via `removesuffix("_lora")` instead of hardcoded
   `"gemma_2b"` (robust to dummy/300m variants; still a valued no-op).
2. No comment on the recipes' dedup continuation lines (bash constraint);
   note carried in the knob blocks.
3. Dedup grad-equivalence tolerance is 1e-3/5% elementwise + 1% aggregate,
   not the plan's implicit split-equivalence 1e-6 — measured 2.4e-4 max
   elementwise difference from bf16 + backward reassociation; rationale in
   the test.
4. `_unblock_adarms_gates` in the grads fixture — unplanned, forced by the
   zero-init adaRMS gate discovery above.
5. `optax` import hoisted; `test_grad_norm_lora_is_zero_without_adapters`
   reuses the `test_grad_norm_decomposition` scaffold via importlib rather
   than duplicating it.
