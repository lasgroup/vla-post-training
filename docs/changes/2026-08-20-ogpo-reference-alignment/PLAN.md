# Align OGPO with the reference implementation

## Context

`reports/findings.md` §10 established that OGPO's policy-gradient term is **net −10.6 points
against its own BC warmstart, paired, on ten of ten arms (t = −9.8)**. The cause chain: the
critic is pinned on the −200 Bellman fixed point (TD loss RMS 4.5 vs MC loss RMS 51 — the
error profile of a network emitting one constant), its within-state spread is ~0.3 on a
200-unit scale, and two locally-invented normalizers rescale that collapsed signal to a fixed
std of 0.36 so the resulting gradient is 7–12× the BC anchor regardless of critic quality.

`reports/ogpo_reference_divergence.md` compared this repo against the reference
(`/home/pchellap/Projects/SafeVADAR/OGPO_public`, closest recipe
`scripts/ogpo/square_image_paligemma.sh`). Several divergences map one-to-one onto that
failure — most importantly **our reward has no success bonus at all** (`wrappers.py:236`
gives `0.0` on success; the reference gives `+4`, up to `+36` total), so success is merely the
absence of a penalty and the −200 attractor has nothing pulling the critic off it.

**Intended outcome:** a parallel, reference-aligned OGPO stack that keeps the current one
runnable, so the two can be compared. This plan does **not** launch anything.

Full discovery record, with the evidence behind every decision (D1–D12):
`docs/changes/2026-08-20-ogpo-reference-alignment/{README,BLAST-RADIUS,HANDOFF}.md`.

## What planning changed vs. the discovery record

Three items collapsed or dropped once the code was read. **Update the change record to match
before implementing.**

| item | discovery said | actual |
|---|---|---|
| **A3** BoN collection | Group 3, "third copy of the scoring block" | **Zero code.** `AdvantageWeightedSFTLearner.sample_actions:313` already implements Best-of-N, gated on `rl.n_samples > 1`; `OGPOSFTLearnerConfig` inherits `n_samples` (`config.py:176`) and `OGPOAgentLearner` never overrides it. Becomes `--rl.n_samples 8`. |
| **A2** Q-target variance reduction | Group 3, port from reference | **Dropped (user).** Inapplicable: our TD target bootstraps `value_model(next_observation)` (`update_critic.py:255-258`), a separate V network — there is no next-action to average. The analogous defect (V regresses onto `Q(s, a_buffer)`, `update_critic.py:325-330`) would need policy sampling inside the critic jit at batch 1024 × 10 critic steps per policy step. Not affordable. C7 reduces variance on the same target instead. |
| **D4** parallel config | flagged the `isinstance` dispatch-order trap | **No trap.** The existing OGPO config registers `OGPOSFTLearnerConfig` directly (`config.py:626-634`), so a second entry needs **no new class** and no new branch in `exp.py:74-85`. |

## Implementation

### 1. `src/training/config.py` — one new field

Add beside `use_time_to_success_as_reward` (`:371`) on `CollectConfig`:

```python
success_reward_bonus: float = 0.0  # reward added on the terminating (success) step
```

**Default 0.0 is load-bearing (D12):** it reproduces today's reward bit-identically, so every
existing config, recipe and prior run is unchanged unless the new recipe opts in.

### 2. `src/envs/wrappers.py:230-237` — use it

`TimeToSuccessAsRewardWrapper` currently hardcodes `0.0 if terminate else -1.0`. Take the
bonus through `__init__` and emit `bonus if terminate else -1.0`.

`FilteredSFTLearner._make_env:62-63` is the only construction site; pass
`config.collect.success_reward_bonus`. **Do not read the config inside the wrapper** — it
takes an env, not a config, and the base learner builds it for all six learners.

### 3. `src/rl/value_distribution.py:129-134` — fix the bounds

`get_value_bounds` returns `upper = 0.0` for the time-to-success reward. With a positive bonus
Q exceeds 0 and that bound is wrong. Latent today (`num_value_bins = 1` ⇒ Gaussian, bounds
unused) but it becomes a silent trap the moment anyone sets `num_value_bins > 1`.

- `upper` → `max(0.0, success_reward_bonus)`.
- `lower` → **`reward/(1-discount)` = −200**, not the current `-(1-γ^T)/(1-γ)` = −173.
  −200 is the value that actually appears in the data, because `fix_mc_returns`
  (`filtered_sft_learner.py:749-751`) overwrites every failed episode's MC return with exactly
  that. This is a pre-existing latent bug; fix it here since we are in the function.

### 4. `src/rl/ogpo/ogpo_learner.py` — A4, success oversampling

After the `critic_utd` loop (`:399-425`), run **one additional** `_update_critics_jitted` call
on a success-only batch, mirroring the reference's `critic_update_sb`
(`ogpo/agents/ogpo.py:1581-1585`, on in 9/15 recipes including all three PaliGemma ones).

- The batch is **already built** for the BC anchor at `:471-481` — reuse
  `self._success_data_buffer.sample(...)` with the same `drop_obs_keys` gate the critic batch
  uses at `:373-381`.
- Gate on `self._success_data_buffer is not None and size >= critic_batch_size`, matching the
  BC path's guard at `:471-472`.
- Fresh `jax.random.split` for the extra call, consistent with the UTD loop.
- Emit its metrics under a distinct prefix (e.g. `critic_sb/`) so the existing `critic/` series
  stays comparable across the old and new stacks.
- No jit signature change — same function, second invocation.

### 5. `src/training/config.py` — register the parallel config (D4)

A second `make_base_libero_config(...)` entry beside `pi05_libero_online_ogpo_sft`
(`:626-634`), e.g. `pi05_libero_online_ogpo_ref`, same `freeze_filter`, with
`OGPOSFTLearnerConfig(...)` carrying the aligned defaults. No new dataclass → no dispatch
change.

### 6. `scripts/ogpo_multitask_4task_ref.sh` — the parallel recipe

Copy `scripts/ogpo_multitask_4task.sh` and change only the deltas below. **D10 constraint:
every knob stays an env var — a reviewer must be able to reproduce the current stack from the
new recipe by setting env vars alone.** Nothing is deleted.

| knob | today | new |
|---|---|---|
| `rl.advantage_combination` | `reduced` (`CONS=0`) | **`grpo_conservative`** (`CONS=1`) |
| `rl.normalize_group_advantage` | on | **off** (`NORM=0`) |
| `rl.normalize_advantage_per_task` | on | **off** (`MT_ADV=0`) |
| `rl.adv_clip_sym` | 4.0 | **off** (`CLIP_SYM=`) |
| `rl.critic.num_qs` / `num_vs` | 2 | **10** |
| `rl.critic.reduction` | `min` | **`mean`** |
| `rl.critic.td_weight_schedule` init/end | 1.0 | **0.95** (D11) |
| `rl.n_samples` | 1 | **8** (A3) |
| `collect.success_reward_bonus` | 0.0 | **72.0**, via `SUCC_BONUS` |

Explicitly unchanged, and worth a comment saying so: `clip_epsilon` **0.1** (D6 — measured
`ratio_max` ≤ 0.9888 < 0.990, so ε=0.01 puts *every* sample outside the trust region and
collapses the PG to positive-advantages-only) · `discount` **0.995** (D8 — γ=0.995 at our
295-step successes is the reference's γ=0.99 at ~150 steps; copying 0.99 would cut the
success/failure gap from 22.8% to 8.9%) · `group_num_samples` **8** (D5) ·
`post_collection_critic_steps` **1000** (D9) · `bc_coeff` 1.0 · `pg_start_step` 20000 /
`pg_ramp_steps` 5000 · actor LR 2.5e-5 constant (D3 — A5 is out; do not touch
`TrainState.tx` or `opt_state`).

## Verification

Per CLAUDE.md step 3, an independent fresh-context verifier (Opus 4.8, xhigh) receives the
change spec + diff and writes `VERIFICATION.md`.

**Pytest-native, on the `tests/ogpo/` pattern** (module-scoped fixtures, dummy PaliGemma
variants, explicit tolerances with a stated reason — see `tests/ogpo/test_sampling.py`):

1. `TimeToSuccessAsRewardWrapper` — `bonus=0.0` reproduces `0.0/-1.0` exactly (the
   regression guard for all five other learners); `bonus=72.0` emits `+72` on terminate and
   `-1.0` otherwise; termination behaviour unchanged.
2. `get_value_bounds` — `(-200.0, 72.0)` with a bonus, `(-200.0, 0.0)` without, `(0.0, 1.0)`
   when `use_time_to_success_as_reward=False`.
3. `fix_mc_returns` — **verify, don't assume**: failures stay all `-1` ⇒ still overwritten to
   −200; successes become `[-1,…,-1,+72]` ⇒ non-constant ⇒ untouched.
4. A4 — the success-batch path produces the documented metric keys and is skipped when the
   success buffer is under-full.
5. `summarize_critic_values` at `reduction="mean"`, `num_qs=10` (`update_critic.py:80`
   already implements it; this is a shape/contract test, not new behaviour).

**No differential test is required** — CLAUDE.md demands one for refactors of numeric code,
and nothing here refactors numeric code. A2, the only such item, was dropped.

**Command-line inspection:** `DRY=1 bash scripts/ogpo_multitask_4task_ref.sh` for every flag,
and `DRY=1 bash scripts/ogpo_multitask_4task.sh` to confirm the original recipe is byte-identical.

**Regression:** `pytest tests/ogpo` (23 tests, CPU, seconds). `pytest tests/` still fails at
collection (OQ-4) — out of scope.

**Cannot be verified locally, and will be stated as such:** any memory claim about C7
(`num_qs` 10 builds ten *separate* BRONet towers — `bronet_critic.py:118-121`, no `vmap` —
for Q **and** V at `critic.batch_size` 1024, against a ~34.7 GiB floor and a `d20` arm that
OOM'd at 150G); whether A3's inherited BoN path works on the OGPO learner, which has never
been exercised; and any run outcome.

**Two GPU smokes will be proposed and will wait for explicit permission** (CLAUDE.md
non-negotiable — no `sbatch`, no `uv run scripts/exp.py` without it): a short memory smoke for
C7, and a short collection smoke for A3.

## Change-record artifacts

`PLAN.md` (this file, verbatim) written as the **first action** of implementation, before any
source edit; then `DIFF.md`; then `VERIFICATION.md` with the verifier's findings verbatim,
failures included. Step 4 updates the affected sections of `docs/code/` only.

Two documentation notes: the **`_adv_scale`-not-checkpointed gotcha stays** — C4 is
recipe-level off, not removal, so it still bites anyone setting `NORM=1`. And the A3/A2/D4
corrections above must be folded back into the change record.
