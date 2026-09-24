# Blast radius & change spec — backbone LoRA for OGPO

Discovery run 2026-08-29 by three independent read-only agents (openpi/config
layer; learner/runtime; recipes/tests/docs), findings cross-checked and the two
highest-stakes claims re-verified by the coordinating session with CPU probes
(`nnx.eval_shape` on the real config; no weights, no GPU). Facts below are
verified against source unless marked **analytic** or **not verified**.

## Intent

Make the PaliGemma LLM backbone (stack 0) trainable through rank-16 LoRA
adapters in OGPO, keeping SigLIP and the base 2B weights frozen and the action
expert + action heads fully trainable as today. Delivered as a new registered
config; every existing config stays bit-identical.

## The mechanics, verified

### The adapter tree

`Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m")`
creates exactly **10 new leaves, 27,869,184 params (106.3 MiB f32), all under
stack 0** — none under `_1` (action expert), none under `PaliGemma/img`
(SigLIP has no LoRA path at all, grep-verified):

```
PaliGemma/llm/layers/attn/q_einsum/lora_{a,b}         (18,8,2048,16) / (18,8,16,256)
PaliGemma/llm/layers/attn/kv_einsum/lora_{a,b}        (18,2,1,2048,16) / (18,2,1,16,256)
PaliGemma/llm/layers/attn/attn_vec_einsum/lora_{a,b}  (18,8,256,16) / (18,8,16,2048)
PaliGemma/llm/layers/mlp/gating_einsum_lora_{a,b}     (18,2,2048,16) / (18,2,16,16384)
PaliGemma/llm/layers/mlp/linear_lora_{a,b}            (18,16384,16) / (18,16,2048)
```

Adapted per layer: `q_einsum`, `kv_einsum`, `attn_vec_einsum`
(`openpi/src/openpi/models/gemma.py:185-198`, `:238-244` — the 2B takes the
q/kv branch because `num_kv_heads=1 != num_heads=8`, so there is no
`qkv_einsum`) and both FFN matmuls via `lora.FeedForward` (`gemma.py:319-324`).
Not adapted: `Embedder`, all `RMSNorm`s. `scaling_value = alpha/rank = 16/16 =
1.0` (`lora.py:28-30`). `PathRegex(".*lora.*")` matches these 10 leaves and only
these 10 (61-leaf tree enumerated; no base param name contains the substring).

### The freeze filter

`_make_ogpo_freeze_filter()` (`src/training/config.py:621-642`) **freezes every
adapter**: they match `.*llm.*` and not `.*llm.*_1.*`. The lora-aware filter is:

```python
gemma_params  = nnx_utils.PathRegex(".*llm.*")
action_expert = nnx_utils.PathRegex(".*llm.*_1.*")
img_params    = nnx_utils.PathRegex(".*PaliGemma/img.*")
lora_params   = nnx_utils.PathRegex(".*lora.*")
nnx.Any(
    nnx.All(gemma_params, nnx.Not(action_expert), nnx.Not(lora_params)),
    img_params,
)
```

Verified leaf-by-leaf on the abstract tree: trainable goes 19 → 29 leaves,
430,098,464 → 457,967,648 params (+6.48%); frozen 42 → 32; SigLIP and the 9
stack-0 base leaves stay frozen.

Rejected alternative: `cfg.model.get_freeze_filter()` +
`nnx.Any(..., img_params)`. openpi's own filter
(`openpi/src/openpi/models/pi0_config.py:79-108`) yields exactly the right LLM
branch, but it returns `nnx.Nothing` when the variant is flipped back to
`gemma_2b` — a variant flip would silently unfreeze the whole backbone. The
hand-written filter is inert-but-safe under the same flip. (This creates a
fourth copy of the same regex family — see §Duplication sweep.)

### LoRA init — a repo-side fix is required

`LoRAConfig.init_fn` defaults to `nn.initializers.normal(stddev=0.01)` and is a
**single field used for both factors** (`lora.py:20`, `:51-52`, `:113-120`;
re-verified by the coordinating session). Consequences:

- As shipped, `lora_b` is random ⇒ the model is **not** the loaded SFT policy at
  step 0. Measured on real shapes with proxy base weights (**analytic** — real
  π0.5 weights not on this machine): 1.8–5.8% relative output perturbation per
  adapted einsum, ×4 einsums ×18 layers, damped by nothing (scaling 1.0). The
  collection policy, EMA old-policy, BC anchor, and the prefix features the
  critic was trained on are all perturbed from step 0. openpi's own equality
  tests pass only because they override `init_fn=zeros`
  (`lora_test.py:37`, `:82`).
- `init_fn=zeros` is **not** the fix: it zeroes `lora_a` too, and
  `∂L/∂lora_a ∝ lora_b` and `∂L/∂lora_b ∝ lora_a`, so both gradients are
  identically zero forever — silently dead adapters, the repo's signature
  quiet-failure shape.
- **The fix**: zero the five `.*lora_b.*` leaves repo-side after model init
  (they never load from the checkpoint — see §Weight loading — so model init is
  the only writer). `lora_a` stays `normal(0.01)`, `lora_b = 0` ⇒ identity at
  init, gradients flow (standard LoRA init). This also turns "LoRA model at
  init ≡ non-LoRA model" into a true property and the natural differential test.
  Where to hook the zeroing (the config wrapper vs a step in `init_train_state`'s
  post-load path) is a plan-phase decision; it must run on the fresh-init path
  only, never on resume.

### Weight loading and checkpoints

- **Main path works**: `init_train_state` → `_load_weights_and_validate`
  (`src/rl/filtered_sft_agent/filtered_sft_learner.py:192-194`, `:110-121`) →
  `CheckpointWeightLoader.load` → `_merge_params(..., missing_regex=".*lora.*")`
  (`openpi/src/openpi/training/weight_loaders.py:50-54`, `:99-102`). The 10 lora
  keys are back-filled as `ShapeDtypeStruct`s, stripped by
  `_load_weights_and_validate:114-121`, and the model's own init survives
  `nnx.replace_by_pure_dict`. Simulated end-to-end at shape level: key sets
  match, `check_pytree_equality` passes; shape mismatches still raise loudly.
- **BLOCKER — learner construction crashes first.** `FilteredSFTLearner.__init__`
  calls `policy_config.create_trained_policy(...)` unconditionally
  (`filtered_sft_learner.py:372-378`) → `train_config.model.load(...)`
  (`openpi/src/openpi/policies/policy_config.py:57`) → `BaseModelConfig.load`
  (`openpi/src/openpi/models/model.py:233-240`), which has **no**
  `missing_regex` and was empirically shown to raise
  `ValueError: ... symmetric difference of key sets: {'lora'}` against a
  lora-less checkpoint. The pi05_libero checkpoint has no lora keys, so the
  first LoRA run dies in `__init__`. The loaded model is discarded three lines
  later (`_drop_policy_model()`, `:381`) — the load exists only for
  transforms/norm-stats. Fix options for the plan: construct that policy with a
  lora-less twin of the model config (cleanest — the model is dropped anyway),
  or route it through the tolerant loader. Not fixing it = the change does not
  run at all.
- **Resume is new-run only, enforced loudly.** Orbax `StandardCheckpointer` /
  `CheckpointManager` (pinned 0.11.13) were empirically shown to raise
  `ValueError` on both extra-key and shape mismatches. A distinct config name
  gets its own checkpoint dir (`openpi/src/openpi/training/config.py:545-550`),
  so `--resume`-by-default cannot collide LoRA and non-LoRA trees. The
  critic-side `_restore_rl_checkpoint` tree is unchanged by this change.
- **Reverse trap (document, don't fix)**: loading a LoRA-trained checkpoint into
  a non-LoRA config **silently drops the adapters** (`_merge_params:92-94` keeps
  only keys present in the reference tree; simulated: 10 tensors dropped, no
  error). This hits the `ws_bcbb_pipeline.sh` two-stage pattern directly: a LoRA
  stage A feeding a non-LoRA stage B would discard the adaptation with no
  signal. Stage-A config name is also hardcoded at `ws_bcbb_pipeline.sh:32,:44`.
- Replay-buffer shards restore *before* the train-state restore
  (`filtered_sft_learner.py:295-299`) and their schema is unchanged
  (`prefix_embedding` stays 2048-dim) — harmless today because the train-state
  restore raises next, but a fresh LoRA run warm-started from a prior run's
  params with copied shards would ingest stale prefixes with no guard.

### The CLI trap and the guard

`--model.paligemma-variant gemma_2b_lora` is a **live tyro flag today** on every
registered config (verified through `_config.cli()`; `freeze_filter` is
`tyro.conf.Suppress` and cannot be corrected from the CLI). Against the
unmodified OGPO config this yields: all 10 adapters frozen (still exactly 19
trainable leaves) but initialized random and never loaded from the checkpoint —
**a randomly-perturbed, permanently-untrainable backbone**, with KL, clipfrac,
alive-fraction, grad norms all normal. The change must add a `__post_init__`
guard on `OnlineTrainConfig` raising when `"lora" in model.paligemma_variant`
and the freeze filter freezes any `.*lora.*` leaf, following the existing
footgun-rejection pattern (`src/training/config.py:495-509`, `:510-527`).

### Dtype

`lora.Einsum.__call__` and `FeedForward._dot` cast base weights *and* LoRA
factors to the activation dtype (bf16 in the LLM) at op time (`lora.py:57-63`,
`:145-148`) — f32 params, bf16 compute, same mixed-precision scheme the base
weights already use. Sane; no action.

## Critic coupling — the research blast radius

Today the backbone is frozen, so `get_prefix_rep` is a pure function of the
observation and **every** producer/consumer of `PREFIX_EMBEDDING_NAME` is
bit-identical: collection-time EMA producers
(`filtered_sft_learner.py:616-631`, `:757-773`; best-of-N scoring
`advantage_weighted_sft_learner.py:542-546`), current-params recomputes
(`advantage_weighted_sft_learner.py:289-306` — with `# TODO: this should be
EMA` at `:311`; jit-1's `critic_prefix is None` branch,
`update_actor.py:183-190`), the buffer-stored copies
(`filtered_sft_learner.py:781-786`, `:902-914`), and the consumers (critic TD
update `advantage_weighted_sft_learner.py:344-353`; OGPO advantage sidecar
`ogpo_learner.py:922` → `update_actor.py:191-196`; four probe scripts). The
contract is invisible because it cannot be violated.

With trainable adapters it splits four ways:

1. Stored prefixes (`--collect.store_prefix_rep` is set in **both** live
   recipes: `stability_study.sh:156`, `ogpo_multitask_4task.sh:286`) freeze the
   collection-time EMA features into the buffer.
2. The critic trains on those stale features. At mt4 scale nothing is ever
   evicted (~120k inserted vs 500k capacity), so batch features span up to the
   full run's drift.
3. The OGPO advantage scores **freshly sampled** actions — drawn through the
   *current* backbone in jit-1 — with a Q fit on stale features. This is the
   stability study's "direction failure" made chronic instead of transient.
4. If `store_prefix_rep` were turned off, the recompute path uses **current
   params, not the EMA** — the AWR TODO at
   `advantage_weighted_sft_learner.py:311` becomes a live old/new mismatch.
   Best-of-N collection scoring (`n_samples=8` on the `_ref` config,
   `config.py:750`) inherits the same mismatch within each round.

**Plan-phase decision, not a follow-up**: prefix-recompute policy. Options —
(a) recompute every critic/actor batch (a backbone forward at critic batch
size; likely prohibitive), (b) lazy per-collection-round recompute + buffer
rewrite, (c) `store_prefix_rep` off + fix `_get_policy_model` to compose the
EMA, (d) accept the drift and instrument it (log ‖stored − recomputed‖ on a
held-out slice). The `ws_bcbb` header (`:1-18`) documents that the two-stage
design exists specifically to avoid this regime.

## Files to touch

| File | Change |
|---|---|
| `src/training/config.py` | New freeze-filter factory (lora-aware variant of `_make_ogpo_freeze_filter`); new registered config after the `_ref` entry (~`:768`); `__post_init__` lora/filter guard. **Footgun**: `make_base_libero_config` replaces `model=` wholesale (`:543-546`) — the new config must restate `pi05=True, action_horizon=10, discrete_state_input=False`. |
| `lora_b` zeroing hook (location per plan) | Zero `.*lora_b.*` leaves on the fresh-init path only. |
| `src/rl/filtered_sft_agent/filtered_sft_learner.py:372-381` | The `create_trained_policy` blocker fix (lora-less twin model for transform construction, or tolerant load). |
| `scripts/ogpo_multitask_4task.sh` (+ optionally `stability_study.sh`) | `LORA=1` knob selecting the config name. Note: a name-selecting knob is one-way — it breaks the "reproduce baseline by env vars alone" contract documented at `ogpo_multitask_4task.sh:186-196`; needs a comment. The mt4 recipe alone covers `_ref` and the smoke via delegation (`ogpo_multitask_4task_ref.sh:62,:80`). |
| `tests/ogpo/test_verifier_alignment.py` | **Breaks on any new registered name**: `:440-456` asserts exact dict equality over `_REGISTERED`; needs a row. `:186-189` is the natural place to assert the new name. |
| New tests (see §Verification plan) | Filter test, init-identity differential, guard test. |
| Docs (step 4) | See §Docs to update. |

**No edits** required on: the EMA path (10 slice/compose sites enumerated, all
filter-generic — `ema_utils.py:16-28`, `filtered_sft_learner.py:336-353`,
`:574-580`, `:701-703`, `:759-763`, `:990-993`,
`advantage_weighted_sft_learner.py:454-458`, `:891-894`,
`best_of_n_learner.py:241-243`, `ogpo_learner.py:124-127`, `:825-830`,
`update_actor.py:167-172`); the five OGPO jits (signatures, donation,
RNG-split arities all unchanged — trees grow by 10 leaves only; both grad
filters are `nnx.DiffState(0, config.trainable_filter)` at `update_actor.py:499`
and `:569`); sharding (both derived shardings are computed from the filtered
tree — `filtered_sft_learner.py:336-338`, `ogpo_learner.py:327-329`; FSDP rules
shard the 2048/16384 axes of the ≥4 MiB lora leaves cleanly, rank 16 is never
the sharded axis, no fallback warning); collection/eval (EMA-composed,
filter-generic).

## Duplication sweep (CLAUDE.md rule 1)

| Clone family | Reached? |
|---|---|
| `advantage_weighted_sft/update_critic.py` ↔ `best_of_n/update_critic.py` | **No.** Neither references freeze/trainable filters; critic trees unchanged. |
| `_pad_last_dim` + best-of-N scoring block (AWR ↔ BofN learner) | **No code edit**; the AWR copy (reachable from OGPO via `n_samples>1`) inherits the stale-feature mismatch behaviorally (§Critic coupling). |
| `init_train_state`: `filtered_sft_learner.py:142-205` ↔ `dsrl_env.py:78-108` | **One copy at most** (and zero if the change stays config-side). DSRL copy is filter-generic and unreachable (no dispatch branch). |
| RL-checkpoint block (AWR ↔ BofN) | **No.** Persists critics/normalizer only. |
| Recipe env preamble + checkpoint-mode block (`stability_study.sh` ↔ `ogpo_multitask_4task.sh`) | The knob touches the flag-assembly region of whichever recipes get it — **not** the duplicated preamble/checkpoint blocks. |
| **Freeze-filter regexes — the family this change extends**: `_make_ogpo_freeze_filter` (`config.py:621`) ↔ `_make_siglip_only_freeze_filter` (`scripts/exp_ogpo_unfrozen_backbone.py:48-51`) ↔ `Pi0Config.get_freeze_filter` (openpi) | The new factory is a **fourth** copy of the same three regexes. Consolidation (one factory with flags) is a plan-phase option; adding a fourth copy is the accepted-state default (Decisions log OQ-2). |

## Inheritance sweep (CLAUDE.md rule 2)

`init_train_state` is module-level with one call site
(`filtered_sft_learner.py:303-305`); no subclass overrides it. OGPO overrides
`update()` and owns its EMA advance (`ogpo_learner.py:825-830`, OQ-10) —
unaffected. All three `update()` EMA advances and every filter consumer take
`config.trainable_filter` as data, so the five subclasses of `FilteredSFTLearner`
are reached **only** if given the new config; under their existing configs the
param tree, filters, and behavior are bit-identical. Dispatch order in
`scripts/exp.py:74-85` is untouched (the new config is an
`OGPOSFTLearnerConfig`... **plan must confirm** whether it reuses that dataclass
— if so, dispatch just works; a new config *dataclass* would be a bigger tier-2
surface for no benefit).

## Tier-2 triggers, itemized

1. `trainable_filter` change → EMA slicing + `_ema_sharding` +
   `_trainable_params_sharding` + both `DiffState` grad paths (trees grow; no
   signature changes).
2. Config registry addition + `__post_init__` guard (config dataclass hierarchy).
3. Contract inversion: `update_actor.py:27-29` frozen-backbone docstring;
   `dedup_group_prefix` "performance-only" claims (`config.py:253-258`,
   `sampling.py:255`) become false — with a trainable backbone, dedup-off pays
   G=8 backbone backwards per update instead of one. Both live recipes already
   pass `--rl.dedup_group_prefix`; the plan should decide whether the LoRA
   config asserts it.
4. Checkpoint tree shape change (new-run only, loud).

## Gotchas checked (CLAUDE.md rule 3) + new ones found

- Known gotchas re-verified as unaffected: EMA-advance ownership (OQ-10), `del
  ema_dev` invariants (`ogpo_learner.py:713-734` — all three stated assumptions
  survive; only headroom changes), Molmo transform positional indexing (not
  touched), obs-preprocessing order (not touched), resume-manifest `extra`
  block (schema unchanged), `save_shard` non-determinism (schema unchanged).
- **New**: the CLI variant trap (§above). Guard required.
- **New**: metric comparability breaks — `actor/param_norm` now includes
  adapters (`update_actor.py:655-664` excludes only bias/scale/embeddings);
  every `grad_norm*` series reduces over 29 leaves; and
  `optax.clip_by_global_norm(1.0)` is one clip over the whole trainable tree,
  so adapter gradients change the action expert's effective step from step 1
  independent of any LoRA effect. Plan-phase: keep shared clip (comparable to
  nothing) vs per-subtree clip (precedent: per-task critic clipping, D5 of
  2026-08-21). Either way the LoRA arm's grad/param norms are not comparable to
  the existing arms — say so in the run ledger.
- **New**: upstream turns EMA off for LoRA fine-tuning
  (`openpi/.../config.py:698-699` "Turn off EMA for LoRA finetuning"); OGPO
  cannot (the EMA *is* the PPO old policy and the collection/eval weights).
  Deliberate divergence — LoRA runs here in a configuration upstream never
  exercises.
- **New**: upstream bug, inert here — `lora.FeedForward` never applies
  `scaling_value` (`lora.py:144-148`) while `lora.Einsum` does (`:63`).
  Harmless at α/rank = 1.0; any future rank/alpha tuning silently rescales only
  the attention adapters. Do not tune α or rank assuming uniformity.

## Standing defects discovered (separate triage — NOT this change)

1. **`nnx_utils.state_map` is a silent no-op under flax 0.10.6** — it tests
   bare path tuples against a set of `(path, value)` pairs
   (`openpi/src/openpi/shared/nnx_utils.py:69-72`). Re-verified independently
   with a concrete module: the bf16 cast in `init_train_state`
   (`filtered_sft_learner.py:169-174`) never fires, so **frozen params are f32
   today** — ~9.35 GiB for the stack-0 LLM where bf16 would be 4.67 GiB, in
   every learner, every run. The docstrings at `config.py:631` and
   `exp_ogpo_unfrozen_backbone.py:26-28` are false. Silver lining for this
   change: all existing memory-floor measurements already include f32 frozen
   params.
2. The "~11.3 GiB trainable-only EMA" figures (`ema_utils.py:4`,
   `ogpo_learner.py:118`, `:731-733`) were measured on the **fully-unfrozen**
   campaign (single source commit `4818b57`); the shipped frozen config's EMA is
   ~1.6 GiB. The referenced memory docs
   (`docs/plans/ogpo-memory/*`, `docs/ogpo_speed_memory_analysis.md`) are gone
   from the tree.
3. `docs/code/` line cites for `config.py`, `filtered_sft_learner.py`,
   `ogpo_learner.py` are systematically stale (content spot-checked correct);
   test count is 326, not the documented 302.

## Memory & compute

- **Steady state, computed exactly**: +111.5 MB params (f32), +223 MB Adam
  moments, +111.5 MB host EMA, +111.5 MB per grads tree, ~+334 MB
  `optimizer_tail` transients ⇒ ≈ **+0.5 GB**. Noise against the stated floor.
- **Activations — the real unknown (analytic only)**. Two newly-live backbone
  backwards per update (jit-2a prefix, jit-2b `compute_loss`). Prefix length
  968 tokens (768 image + 200 text), width 2048, depth 18, under
  `nn.remat(..., nothing_saveable)` + `nn.scan`: scan carry ≈ B × 71.4 MB
  (2.3 GB at B=32, 9.1 GB at B=128). The dominant term is the **KV-cache
  cotangent** in the rescorer — the cache is tiled to B·G
  (`update_actor.py:422`, `sampling.py:372-375`) and its cotangent goes live
  across the K=10 scan backward: ≈4.6 GB bf16 at B·G=256 (B=32,G=8),
  ≈18–37 GB at B·G=1024. XLA may schedule this better than the arithmetic —
  **only a GPU run settles it**. The measurement harness already exists:
  `scripts/exp_ogpo_unfrozen_backbone_memdiag.py` (per-phase JSONL,
  `peak_bytes_in_use`, pprof) needs only the new config name.
- `policy_grad_accum` sizing comment (`ogpo_learner.py:660-667`) explicitly
  says "NOT sized for unfrozen" — LoRA lands between its two cases; re-measure.

## Expected behavior after

- Every existing config: bit-identical (filters, trees, checkpoints, metrics).
- The new config: trains 457.97M params (existing 430.10M + 27.87M adapters);
  starts bit-identical to the loaded SFT policy (`lora_b = 0`); collection/eval
  ride the EMA'd adapters automatically; checkpoints/resume work within the
  config name and fail loudly across names.
- `--model.paligemma-variant gemma_2b_lora` against any non-LoRA-filter config:
  **raises at config construction** with the guard's message instead of
  silently perturbing the backbone.

## Residual decisions for the plan phase

| # | Decision | Notes |
|---|---|---|
| R1 | Stored-prefix staleness policy (options a–d in §Critic coupling) | The research-critical one. |
| R2 | `create_trained_policy` blocker fix shape | Lora-less twin model recommended (the loaded model is dropped at `:381`). |
| R3 | Where the `lora_b` zeroing hooks in | Fresh-init path only; never resume. |
| R4 | Registry entry vs wrapper-script registration | `best_practices.md:90-91` (registry) vs `STYLE.md:773-775` (wrapper, the `exp_ogpo_unfrozen_backbone.py` precedent). Recommend registry per the prescriptive doc; the STYLE.md contradiction goes to the open-questions record in step 4 rather than being resolved unilaterally. |
| R5 | Reuse `OGPOSFTLearnerConfig` for the new config (dispatch untouched) or subclass | Reuse recommended; subclassing re-opens `isinstance` order. |
| R6 | Global vs per-subtree grad clip | See §Gotchas, metric comparability. |
| R7 | Assert `dedup_group_prefix` on the LoRA config? | It is load-bearing for memory once the backbone trains. |
| R8 | Which recipes get the `LORA` knob | mt4 alone covers `_ref` + smoke via delegation; `stability_study.sh` has no `DRY` support for verification. |
| R9 | rank/α as config surface or fixed at 16/16 | If exposed, the FFN scaling omission must be documented at the knob. |

## Verification plan (step 3, independent agent)

Pytest-native, CPU-only, modeled on `tests/ogpo/`:

1. **Filter unit test** (pure, no model): the lora-aware filter against the
   enumerated real path list via `flax.nnx.filterlib.to_predicate` — adapters
   trainable, 9 stack-0 base leaves + SigLIP frozen, action expert + heads
   trainable, and the *old* filter shown to freeze the adapters (pins the trap).
   Model: `tests/ogpo/test_ema_utils.py`'s synthetic-tree pattern.
2. **Guard test**: lora variant + non-lora filter raises with the fix in the
   message; lora variant + lora filter constructs.
3. **Init-identity differential**: with `lora_b` zeroed, LoRA vs non-LoRA
   forward bit-identical at init. `lora.Einsum`/`FeedForward` level is cheap;
   the Pi0-level version needs a monkeypatched `gemma.get_config` returning a
   dummy-width config with `lora_configs` (no `dummy_lora` variant exists and
   `Pi0.__init__` hardcodes `get_config`; a submodule edit for a test variant is
   a separate decision — recommend the monkeypatch). Also assert gradients are
   **nonzero** on `lora_b` and zero-init does *not* hold for `lora_a` (pins the
   dead-adapter trap).
4. **Grad-flow test on the dummy fixture**: extend the `test_split_equivalence`
   fixture pattern; note the existing suite already runs with a trainable dummy
   backbone (`freeze_filter=".*PaliGemma/img.*"` in `_build_config`), so jit
   plumbing with backbone grads is exercised today — the new test pins that
   adapter leaves specifically receive nonzero grads through the prefix cache
   under `dedup_group_prefix` on and off.
5. **Registry tests**: the new row in `test_verifier_alignment.py:440-456`'s
   dict; name asserted at `:186-189`.
6. **Recipe verification**: `DRY=1` through `ogpo_multitask_4task.sh` with
   `LORA=1`, re-parsed via the existing `_dry`/`_resolve` harness
   (`test_verifier_alignment.py:500-541`).
7. **Honest fallbacks, stated as such**: weight loading against the real
   checkpoint, actual VRAM peak, throughput, and any training behavior cannot
   be verified on CPU. VRAM goes through `exp_ogpo_unfrozen_backbone_memdiag.py`
   on a GPU node (launch requires explicit permission).
8. `pytest tests/ogpo` (326 tests) must pass throughout.

## Docs to update (step 4 scope)

`docs/code/training.md` (config.py sections: freeze-filter bullet, factory
bullet, stale cites); `docs/code/scripts.md` (recipe knob list, config-variant
wrapper section — record which registration route was taken); 
`docs/code/rl-ogpo.md` (jit-1/jit-3 memory notes; the `compute_prefix_cache`
bullet already pre-authorizes an unfrozen backbone but must gain the
stored-prefix staleness note); `docs/code/rl-learners.md` (`init_train_state`,
`_load_weights_and_validate` — the `.*lora.*` back-fill semantics are currently
undocumented); `docs/code/rl-core.md` (ema_utils consumer note);
`docs/code/tests.md` (count, map); `STYLE.md` §6.4 (mixed-dtype note) + the
open-questions record (R4 registration-route contradiction; the `state_map`
defect if triaged); `docs/ogpo_experiment_notes_libero44.md` (the ledger's
Backbone column gains a "LoRA" value). `update_actor.py:27-29` docstring
rewritten as part of the implementation itself.
