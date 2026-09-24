# Backbone LoRA for OGPO — implementation plan (Tier 2, phase 3)

## Context

Make the PaliGemma LLM backbone trainable through rank-16 LoRA adapters in OGPO;
SigLIP and the base 2B weights stay frozen, action expert + heads stay fully
trainable. Motivation and full blast radius: `docs/changes/2026-08-29-backbone-lora/`
(README + BLAST-RADIUS). The libero44 ledger's unfrozen-backbone arms were the
healthy-PPO ones; LoRA is the affordable middle point (+27.87M params, +6.5%
trainable, ~+0.5 GB steady state; activation cost to be measured on GPU).

**Settled decisions (maintainer):**
- **R1**: critic prefix features come from the **untrained base backbone** —
  value and policy must not share the adapting representation. Implemented by
  zeroing all LoRA leaves for every critic-prefix forward; stored buffer
  prefixes stay a pure function of the observation, bit-identical to today's
  frozen-backbone features. Policy forwards (SDE sampling, chain rescoring, BC)
  use the full adapted model.
- **R4/R5**: delivery is a CLI flag `--backbone_lora` on `OnlineTrainConfig`
  whose `__post_init__` rewrites `model.paligemma_variant` + `freeze_filter`
  together. No new registered config names.
- **R6**: shared LR + shared global clip; add adapter-vs-rest grad-norm logging.
- R2 twin fix for `create_trained_policy`; R3 zero `lora_b` on fresh init; R7 no
  dedup assert (document memory-load-bearing); R8 knob in both recipes,
  both-branches emit; R9 rank/α fixed at 16/16.

**Three traps the implementation must not fall into:**
1. tyro renders the bool as a flag *pair* — recipes must emit
   `--backbone_lora` / `--no-backbone_lora`; the value form `--backbone_lora
   True` does not parse.
2. `nnx_utils.state_map` is a verified silent no-op under flax 0.10.6 — never
   use it for the zeroing helpers; use `nnx.filter_state` + `jax.tree.map` +
   `nnx.merge_state` (the `ema_utils.py` primitives).
3. The `create_trained_policy` twin must also pass `backbone_lora=False` in the
   `dataclasses.replace`, or `__post_init__` rewrites the variant right back.

## First action of the implementation pass

Write this plan verbatim to `docs/changes/2026-08-29-backbone-lora/PLAN.md`
before any source edit (CLAUDE.md pass structure).

## Step 1 — config surface (`src/training/config.py`)

1. Parameterize the existing filter factory in place (no fourth regex copy):
   `_make_ogpo_freeze_filter(*, allow_lora: bool = False)` — when set, append
   `nnx.Not(PathRegex(".*lora.*"))` to the LLM `nnx.All(...)` branch. The three
   existing call sites (`:706`, `:728`, `:766`) are untouched and bit-identical.
   Docstring: notes the +19→29 leaves / 430.10M→457.97M params delta and why
   `Pi0Config.get_freeze_filter()` was rejected (returns `nnx.Nothing` on a
   variant flip back to `gemma_2b` → silently unfreezes the whole backbone).
2. New field `backbone_lora: bool = False` on `OnlineTrainConfig` (after
   `default_prompt`, `:479`), commented as a rewrite flag.
3. Append to `__post_init__` (after `:530`, existing validations untouched):
   - `if self.backbone_lora:` raise unless `isinstance(self.rl,
     OGPOSFTLearnerConfig)` (on any other config `freeze_filter` is
     `nnx.Nothing` and the rewrite would freeze ~430M params that train today)
     and unless the model is a `Pi0Config`; then
     `object.__setattr__(self, "model", dataclasses.replace(self.model,
     paligemma_variant="gemma_2b_lora"))` and `object.__setattr__(self,
     "freeze_filter", _make_ogpo_freeze_filter(allow_lora=True))`.
     `dataclasses.replace` (not a restated `Pi0Config`) preserves
     `pi05/action_horizon/discrete_state_input`. Mutation precedent:
     `CollectionConfig.__post_init__` (`:457`, `:461`). Must be idempotent
     (tyro re-instantiates).
   - Guard, evaluated after the rewrite: if `"lora" in
     model.paligemma_variant` and `nnx.filterlib.to_predicate(freeze_filter)`
     freezes the probe path `("PaliGemma","llm","layers","attn","q_einsum",
     "lora_a")` → raise with the fix in the message ("pass --backbone_lora
     instead of setting --model.paligemma_variant by hand"). This converts the
     live CLI trap (randomly-perturbed, permanently-untrainable backbone with
     normal-looking metrics) from silent to loud.

## Step 2 — make a LoRA run constructible

**New module `src/rl/lora_utils.py`** (sibling of `ema_utils.py`; module
docstring carries the R1 contract and the state_map warning):
- `LORA_FILTER = PathRegex(".*lora.*")`, `LORA_B_FILTER = PathRegex(".*lora_b.*")`
  (matches exactly the 10 / 5 adapter leaves on the real tree; enumerated in
  BLAST-RADIUS).
- `zero_lora_params(params)` / `zero_lora_b_params(params)` via a shared
  `_zero_matching`: `nnx.filter_state` the matches; if empty, **return the
  input unchanged** (exact identity for every non-LoRA config — this is what
  makes all call sites unconditional); else
  `nnx.merge_state(filter_state(params, Not(f)), jax.tree.map(jnp.zeros_like, matched))`.

**R3 — `lora_b` zeroing** in `init_train_state`'s inner `init()`
(`filtered_sft_learner.py`): `params = zero_lora_b_params(params)` immediately
after `params = nnx.state(model)` (`:168`), before the (currently no-op) bf16
cast. Fresh-init only by construction: the resume path short-circuits at
`:189-190` and orbax restore overwrites the tree; the `jax.eval_shape` pass at
`:186` is shape-only and unaffected. Rationale comment: openpi inits BOTH
factors `normal(0.01)` (step 0 ≠ SFT policy); zeroing both would dead-end
adapter gradients (each factor's grad ∝ the other); a-random/b-zero is standard
LoRA init and makes the model bit-identical to the SFT policy at step 0.

**R2 — `create_trained_policy` twin** (`filtered_sft_learner.py:375-378`): call
with `dataclasses.replace(self._config, backbone_lora=False,
model=dataclasses.replace(self._config.model, paligemma_variant="gemma_2b"))`.
The loaded model is dropped at `:381`; the call exists only for
transforms/norm-stats, and the strict `BaseModelConfig.load` path (no
`missing_regex`) raises on the checkpoint's missing lora keys otherwise.
Unconditional (valued no-op for non-LoRA configs). Verified: nothing on this
path reads `paligemma_variant`.

## Step 3 — R1: the critic-prefix producers (six sites)

Prefer the keyword form where a shared model getter serves both actions and
prefixes: `_get_policy_model(..., *, zero_lora: bool = False)` applying
`zero_lora_params` before merge — never zero inside the shared getter
unconditionally (it also feeds the action path).

1. **`filtered_sft_learner.py:615-623` (`_sample_action`) — the only costly
   one.** Gate on `self._config.backbone_lora`: flag off → today's single fused
   `_sample_actions_with_model(..., return_prefix_rep=True)` call, bit-for-bit;
   flag on → actions from the full model with `return_prefix_rep=False`, then
   `base_model = nnx.merge(train_state.model_def, zero_lora_params(params))`,
   `base_model.eval()`, prefix from the existing `_get_prefix_rep_with_model`
   jit (`:362-369`). Comment quantifies the cost honestly: one extra 968-token
   stack-0 forward per policy query at collection (roughly doubles the
   backbone cost of a query; `compute_v_t` runs only the action expert). Keep
   the `sample_rng` derivation untouched so the action stream is identical. Do
   NOT cache a long-lived base model (`optimizer_tail` donates
   `policy_state`).
2. `filtered_sft_learner.py:757-765` (`_attach_prefix_embeddings_to_episode_data`):
   wrap the composed EMA params in `zero_lora_params(...)` before merge.
3. `advantage_weighted_sft_learner.py` `_get_policy_model` (`:308-314`, stays
   `@staticmethod`) gains the `zero_lora` keyword; `_recompute_prefix_embedding`
   (`:289-306`) passes `zero_lora=True`. Retires the `# TODO: this should be
   EMA` for the prefix path (the base-backbone prefix is invariant to
   params-vs-EMA).
4. `advantage_weighted_sft_learner.py:452-460` + `:542-548` (best-of-N
   Q-scoring): the `:542` prefix call uses a second, adapter-zeroed model; the
   `:471` action sampling keeps the adapted model.
5. `update_actor.py:183-190` (jit-1 `critic_prefix is None` recompute):
   `nnx.merge(policy_state.model_def, zero_lora_params(...))` for the prefix
   model only (dormant in production — OGPO threads the stored sidecar — but
   exercised by `test_split_equivalence` leg (d)).
6. `best_of_n_learner.py:235-245` + `:452-465`: same two edits verbatim
   (unreachable from OGPO configs but the identical clone family — do not
   leave a half-fixed pair; duplication sweep note in DIFF.md).

Consumers need no change (verified): critic TD update, OGPO advantage sidecar,
probes, and the `.shape`-only `get_prefix_rep` probes all read stored or
passthrough prefixes, now base-backbone by construction.

## Step 4 — R6: adapter-vs-rest grad-norm logging

In jit-2b's `loss_aux` construction (`update_actor.py:611-616`, the combined
tree that enters the global clip): add `grad_norm_lora =
optax.global_norm(nnx.filter_state(grads, LORA_FILTER))` and `grad_norm_rest =
optax.global_norm(nnx.filter_state(grads, nnx.Not(LORA_FILTER)))`. Both
explicit (the Pythagorean identity `grad_norm² = lora² + rest²` certifies the
filter split in tests). Scalars ride the existing aux dict → replicated
out-sharding broadcast; no jit signature or sharding changes. `grad_norm_lora`
is exactly 0.0 on a non-LoRA tree. Companion edits: the aux-key schema comment
(`update_actor.py:603-609`), and `tests/ogpo/test_split_equivalence.py:70`
`_SPLIT_ONLY_INFO_KEYS` gains both keys (this is the test that breaks —
`test_verifier_alignment.py:440-456` does NOT break under the flag route; no
registered names change). Update the `policy_param_norm` note: `actor/param_norm`
now includes adapters — LoRA-arm norm series are not comparable to prior arms.

## Step 5 — contract inversions (comments/docstrings)

`update_actor.py:27-29` (frozen-backbone claim); `config.py:253-258` +
`sampling.py:248-258` (`dedup_group_prefix` stays semantically
performance-only but is memory-load-bearing once the backbone trains — dedup
off pays G=8 backbone backwards per update); `ws_bcbb_pipeline.sh` header
(reverse trap: a LoRA stage A feeding the non-LoRA stage B silently drops the
adapters via `_merge_params`). Grep-verify no stale "backbone is frozen" claim
survives in `src/rl`.

## Step 6 — recipes (both, unconditional both-branches emit)

`scripts/ogpo_multitask_4task.sh` (after the `PER_TASK_CRITIC` block, ~`:185`)
and `scripts/stability_study.sh` (in the `EXTRA_FLAGS` block, after `:113`):

```bash
if [ "${LORA:-0}" = "1" ]; then EXTRA_FLAGS+=(--backbone_lora)
else EXTRA_FLAGS+=(--no-backbone_lora); fi
```

with the comment block covering: env-var-authoritative-for-any-CONFIG_NAME
(contract preserved because it's a real flag), new-run-only (orbax raises
across a LORA flip), the flag-pair spelling, and the dedup memory note. Add
`# LOAD-BEARING for memory when LORA=1` at each recipe's
`--rl.dedup_group_prefix` line. `_ref` and the smoke inherit via delegation.

## Step 7 — tests (written with each step; `pytest tests/ogpo` green throughout)

- **`tests/ogpo/test_backbone_lora_config.py`** (pure, no model; the
  `test_ema_utils` synthetic pattern + `to_predicate` on literal paths):
  lora-aware filter leaves adapters trainable / base+SigLIP frozen; the OLD
  filter freezes adapters (pins the trap); lora-less-tree equivalence of the
  two filter modes; `backbone_lora=True` rewrite preserves
  `pi05/action_horizon/discrete_state_input`; lora variant without the flag
  raises with `--backbone_lora` in the message; non-OGPO config raises; every
  registered config has the flag off; the policy-twin config is lora-less
  (pins trap 3); mt4 `DRY=1 LORA=1` resolves to `backbone_lora=True` via the
  `_dry`/`_resolve` harness (`test_verifier_alignment.py:501-541`) — also the
  proof the flag spelling parses.
- **`tests/ogpo/test_backbone_lora_init.py`** (dummy-width LoRA model via
  monkeypatched `gemma.get_config` returning the `"dummy"` config +
  `lora_configs` rank 4 — no submodule edit; build the lora-less twin by
  copying base leaves with `nnx.update`, never by re-seeding):
  `zero_lora_params` identity on lora-less tree (`is` check) and zeroes
  exactly the 10 leaves; **the R1 key property** — zeroed-lora prefix ==
  base-model prefix (exact; relax to a stated tolerance only if XLA fusion
  forces it); prefix invariant to adapter values + unzeroed negative control;
  the R3 differential — after `zero_lora_b_params`, `sample_actions` and
  `compute_loss` match the lora-less twin; `lora_a` stays nonzero (pins the
  dead-adapter trap); `lora.Einsum`/`FeedForward` exact-identity unit legs;
  a regression pin that `nnx_utils.state_map` would NOT have zeroed anything.
- **`tests/ogpo/test_backbone_lora_grads.py`** (clone the
  `test_grad_norm_decomposition.py:55-137` scaffold with the dummy-lora
  variant + `_make_ogpo_freeze_filter(allow_lora=True)`): adapters get nonzero
  PG grads and base stack-0 leaves are absent from the grad tree; adapter
  grads allclose with `dedup_group_prefix` on/off; nonzero BC grads;
  Pythagorean `grad_norm` identity (`rtol=1e-5`, fp32 accumulation);
  `grad_norm_lora == 0.0` exactly on the plain dummy config; jit-1 critic
  prefix invariant to adapter values (two policy states differing only in
  adapter leaves → identical `q_mean`/`v_mean`).
- Edits: `test_split_equivalence.py:70` key set (+2). No
  `test_verifier_alignment.py` changes (verified: `_REGISTERED` builds from
  `_CONFIGS`, not CLI parses).

## Step 8 — record + docs (step 4 of the workflow)

Write `DIFF.md` when implementation is done; launch the independent verifier
(fresh-context agent, **Opus at xhigh effort**, given the change spec + diff,
per CLAUDE.md step 3) and write its findings verbatim to `VERIFICATION.md`.
Doc updates scoped by the record's §Docs list, adjusted for the flag route:
`training.md` (config.py sections + `backbone_lora` field + guard),
`scripts.md` (LORA knob; note the flag route and why), `rl-ogpo.md` (jit-1
recompute wording → adapter-zeroed; prefix-purity note; live-backward memory
notes), `rl-learners.md` (`init_train_state` zeroing; `_load_weights_and_validate`
lora back-fill semantics; `_attach_prefix_embeddings`), `rl-core.md` (new
`lora_utils.py` row + section), `tests.md` (3 new files; count), `STYLE.md`
open-questions (registry-vs-wrapper resolved a third way; `state_map` defect
noted), `docs/ogpo_experiment_notes_libero44.md` (Backbone column gains LoRA;
norm-series comparability note).

## Verification summary

- CPU: the three new test files + full `pytest tests/ogpo` (326 baseline) at
  every step; `DRY=1 LORA=1` recipe resolution; `bash -n stability_study.sh`.
- **GPU verification on the preempt partition** (maintainer has authorized
  test-scale GPU jobs on preempt; full training arms still need separate
  approval). Short jobs, resubmit if preempted — nothing here depends on
  resume semantics:
  1. **Full suite off the login node**:
     `sbatch --partition=preempt scripts/run_ogpo_tests.sbatch` (the script's
     own header pins `general`; override on the command line).
  2. **VRAM measurement** — the one number no CPU test can settle (the B·G
     KV-cache cotangent, analytic range 4.6 GB → OOM):
     `scripts/exp_ogpo_unfrozen_backbone_memdiag.py` with
     `pi05_libero_online_ogpo_sft --backbone_lora` + the stability-study flag
     stack at B=32/G=8, run through the first collection + first policy-update
     burst (~1–2 h, `--time=02:00:00`, gpu:1). Needs no script edit under the
     flag route. Also run the LORA=0 twin for the paired baseline.
  3. **Real-weights smoke**: `LORA=1` through the recipe with a tiny
     `NUM_STEPS` (through one collection round + the first actor/critic
     updates) against the real `gs://openpi-assets` π0.5 checkpoint —
     exercises the `create_trained_policy` twin, the `.*lora.*` weight-loader
     back-fill, the `lora_b` zeroing, and the split collection-time prefix
     path at real shapes. Check in the logs: step-0 eval/collection behaves
     like the SFT policy (R3 identity at real weights), and
     `actor/grad_norm_lora` > 0 once PG is live.
- Honest fallbacks, stated in VERIFICATION.md: training *behavior* (does the
  arm learn, is the critic healthy over 10k+ steps) is out of verification
  scope — that is the experiment itself.

## Out of scope (unchanged from the record)

Molmo LoRA, action-expert LoRA, SigLIP training, rank/α knobs, separate
adapter LR/clip, the `state_map` bf16-cast defect (separate triage — but do
not *use* `state_map` anywhere in this change), warm-start across the LoRA
boundary.
