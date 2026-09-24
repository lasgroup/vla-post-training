# Language-grounding probe (stage 1)

**Tier:** 1 (additive experiment code only — nothing under `src/`, `scripts/`,
or `openpi/` is modified).
**Date:** 2026-08-31 · **Branch:** `pranav_ref_ogpo`

## Why

The single-task LoRA run on `libero_90_31` ("put the black bowl on top of the
cabinet", KITCHEN_SCENE5) confidently performs a *different same-scene task*
(bowl into the top drawer + close it — the composite of `libero_90_29` and
`libero_90_28`). Hypothesis under test: the policy ignores the instruction and
infers the task from the image. The probe design decomposes WHERE language
dies: the VLM backbone (stage 1) vs the action expert reading the backbone's
cache (stage 2). Full design and interpretation guide:
`experiments/language_grounding/README.md`.

## What was built

New folder `experiments/language_grounding/`:

- `README.md` — the design: stage 1a (image-column K/V sensitivity to prompt
  swaps, calibrated by paraphrase/garbage), stage 1b (pooled text-rep cosine
  collapse vs the base checkpoint), stage 2 (cache-splicing hybrids — future),
  stage 3 (instruction-swap rollouts — future).
- `common.py` — checkpoint loading (LoRA-aware via `backbone_lora=True`
  config rewrite; strict `cfg.model.load(restore_params(...))`, the
  probe-family pattern), shard→observation loader (adapted from
  `scripts/probe_policy_candidate_spread.py` — shards store the full
  transformed obs), LIBERO registry lookup (true instruction + same-scene
  distractors), `PaligemmaTokenizer` variant tokenization, jitted
  `get_prefix_rep` forward (one call returns final hidden states AND the
  per-layer KV cache the expert consumes).
- `stage1_vlm.py` — the probe CLI: N shard states × variants
  {true, same-scene distractors, paraphrase, garbage, empty} in one B=V
  forward per state; per-layer image/text-column relΔ of K/V, final-hidden
  image relΔ, pooled-text cosine matrix; JSONL + summary table.
- `stage1_vlm.sbatch` — preempt GPU job looping {base, step checkpoints}.

## Blast radius

None by construction: every file is new, imports only. Verified: no edits to
tracked files outside `experiments/` and this record.

## Verification

- CPU smoke on the dummy Pi0 variant: full numeric path (B=V forward, column
  split, relΔ, cosine matrix) runs; the empty-prompt leg moves the image
  hidden states (relΔ ≈ 1.6), proving the bidirectional text→image path is
  mechanically live in the measurement, so ≈0 on real checkpoints is signal.
- LIBERO registry lookup verified on the login node: task 31 language +
  4 same-scene distractors (including the observed wrong behavior, 29 + 28).
- GPU run on real checkpoints/shards is the experiment itself (sbatch above).

## Addendum (2026-08-31): stage 2 built

`stage2_expert.py` + stage-2 helpers appended to `common.py` (`splice_cache`,
`sample_chain_with_cache` — the grouped sampler's SDE loop verbatim with an
injected cache — and `score_chain_with_cache` over the public `prefix_cache`
parameter). Still additive-only. CPU-smoked on the dummy model with unblocked
adaRMS gates: self-splice is a bit-exact identity, the sample/score round-trip
agrees to 9e-8, e2e/H1 conditions move actions and log-probs, the stochastic
floor is nonzero. Stage-1 results from the first campaign are recorded in the
probe logs (`~/logs/lang_grounding/`) and the memory file; headline — the VLM
conditions its scene encoding on language at base and is bit-identical under
the frozen run's checkpoint, so the grounding failure is action-expert-side.

## Addendum 2 (2026-08-31): stage 3 built

`stage3_rollouts.py` + `stage3_rollouts.sbatch` — instruction-swap rollouts,
the closed-loop discriminator (success matrix over env_task × prompt variant;
env tasks supply the native success predicates, the prompt is overridden via
the deployed action path's own `task_description` argument). Invoked through
`stability_study.sh` with `ENTRY=` (the Tier-C pattern) so the config and env
preamble are byte-identical to the checkpoint's run; knobs ride env vars
because that recipe has no `"$@"` passthrough. Carries Tier-C's `--resume`
wipe-guard and an `n_samples==1` guard. Paired seeds: every variant resets
from the same init states per (env_task, wave). Still additive-only.

## Addendum 3 (2026-09-02): stage 4 built — trained instructions in task 31's scene

Motivation: the BC set (`physical-intelligence/libero`, 40 tasks) has no
libero_90 demos; its only demo in the scene-4/5 bowl+white-cabinet layout is
libero_10's "put the black bowl in the bottom drawer of the cabinet and close
it" (KITCHEN_SCENE4 — same table, same cabinet pose, same bowl init region as
KITCHEN_SCENE5), which is the behaviour the base policy produces on task 31.
Stage 4 asks whether BC-*trained* instructions are followed in that scene:
"open the middle drawer of the cabinet" (libero_goal 19) and "put the wine
bottle on top of the cabinet" (libero_goal 14), against the true task-31
instruction as control.

`stage4_trained_instructions.py` + `.sbatch` (additive-only). Generates BDDLs
from the stock KITCHEN_SCENE5 file with the language/goal swapped (`s5`, init
from task 31's pruned init states) and a copy with `wine_bottle_1` at
KITCHEN_SCENE4's wine-bottle region (`s5wine`, init sampled from the BDDL
regions with np.random seeded — the pruned states have no bottle). The env
wrapper evaluates nine side predicates every step with LIBERO's own predicate
functions (bowl in / on, top drawer closed, middle/bottom open, wine in / on,
ketchup in, bowl on plate), so the "did the drawer thing instead" outcome is
recorded per episode. Paired seeds across prompts. GIFs + `stage4_results.json`
land in `<repo>/temp/` (maintainer-owned scratch, to be deleted).

Verification: login-node preflight (`S4_GEN_ONLY=1`) — all five BDDLs parse
with the intended language, goal, object set and open top drawer. The GPU run
is the experiment (job 10292215, preempt, MODE base via `ARM=stage4_base_scratch`).
Not verifiable locally: EGL rendering, the sampled-init pairing, GIF output.
