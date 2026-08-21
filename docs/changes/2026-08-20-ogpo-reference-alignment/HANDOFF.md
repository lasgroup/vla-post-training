# Handoff — OGPO reference alignment

**For:** a fresh Claude Code session in
`/home/pchellap/Projects/OGPO-VLA/vla-post-training`.
**Written:** 2026-08-20. **State on disk: discovery pass done, decisions taken, zero source
files modified.**

---

## Read these first, in this order

1. `CLAUDE.md` — the working agreement. This is a **Tier 2** change.
2. `docs/changes/2026-08-20-ogpo-reference-alignment/README.md` — what the change is, the
   final in-scope item list, and the four decisions already taken (D1–D4).
3. `docs/changes/2026-08-20-ogpo-reference-alignment/BLAST-RADIUS.md` — verified blast
   radius, duplication + inheritance sweeps, gotcha interactions, per-item edit sites,
   verification plan.
4. `reports/ogpo_reference_divergence.md` — the full upstream-vs-here comparison this change
   is derived from.
5. `reports/findings.md` §10–§11 — the measurements that motivate it. §10 is the root-cause
   analysis; do not re-derive it.

The reference implementation is at `/home/pchellap/Projects/SafeVADAR/OGPO_public`. The
closest recipe to our setting is `scripts/ogpo/square_image_paligemma.sh` (frozen PaliGemma
encoder restored from a `pi05_libero` checkpoint). Its defaults live in
`ogpo/configs/algos/ogpo.yaml` + `common.yaml`; where the script overrides the yaml, **the
script value is the tuned one**.

## What you are picking up

Discovery (step 1 of CLAUDE.md's four steps) is **complete and recorded**. The four
decisions that were blocking are **answered**. Do not re-open them.

Your next action is the **plan phase**: enter plan mode, build the implementation plan
**from the change record** (`README.md` + `BLAST-RADIUS.md` — do not re-derive scope), and
present it. Plan-mode approval **is** the Tier 2 approval gate.

Then, in the implementation pass: write `PLAN.md` verbatim as your *first* action before any
source edit, then implement, then `DIFF.md`, then the independent verifier and
`VERIFICATION.md`.

## The decisions, restated so you cannot miss them

- **D1** — reward: keep −1/step, add a **+72 one-off bonus on the terminating step**. No
  `post_success_steps`. Termination behaviour unchanged.
- **D2** — scope: Groups 1, 2, 3. Group 4 (`pi_slow`/pessimism) is out.
- **D5–D9** — C9, C1 and C2 dropped; burst kept at 1000. See "FINAL change set" in
  `README.md`; the arithmetic behind D6 and D8 is recorded there.
- **D3** — **A5 is OUT.** Do not touch the actor optimizer, `TrainState.tx` or `opt_state`.
  (D2 nominally included A5; D3 is the more specific answer and wins. Final Group 2 = A4
  alone.)
- **D4** — parallel registered config + parallel shell recipe. Current stack stays runnable.

## In-scope items — FINAL

**Group 1, config only:** C3 `advantage_combination`→`grpo_conservative` ·
C4 `normalize_group_advantage` off · C5 `normalize_advantage_per_task` off ·
C6 `adv_clip_sym` off · C7 `num_qs`/`num_vs` 2→10 · C8 `reduction` min→mean

**Group 2:** A4 — second success-only TD batch per critic update.

**Group 3:** A1 — +72 terminal bonus (config-gated) · A2 — Q-target variance reduction over
8 sampled next-actions · A3 — Best-of-N as the collection behaviour policy.

**OUT — do not re-open:** C1 `clip_epsilon` (stays 0.1) · C2 `discount` (stays 0.995) ·
C9 `group_num_samples` (stays 8) · A5 optimizer reset · A7 pessimism. Each has recorded
evidence in `README.md`; C1 and C2 were dropped on measurement, not preference.

**BURST stays at 1000** in the new recipe (D9).

**`td_weight_schedule` init/end = 0.95** (D11) — 95% TD / 5% MC in the critic target. This is
a deliberate divergence from the reference, which has the mechanism (`mc_regression`) but sets
it `false` in every recipe. Do not "align" it away; the reasoning is in `README.md` §D11.

**Nothing is deleted (D10).** C4/C5/C6 are recipe-level *off*, not code removal — every
dataclass field and every `--rl.*` flag survives, and the current stack must be reproducible
from the new recipe by env vars alone. Corollary: the `_adv_scale`-not-checkpointed gotcha
stays in the docs, because `NORM=1` remains reachable.

## The five things most likely to trip you up

1. **A1 must be config-gated.** `TimeToSuccessAsRewardWrapper` is instantiated by the *base*
   learner (`src/rl/filtered_sft_agent/filtered_sft_learner.py:62-63`), so an unconditional
   edit changes the reward for **all six learners** — filtered SFT, AWR, MPO, FlowGRPO,
   Best-of-N, OGPO — and destroys the isolation D4 exists to provide. Gate the bonus behind a
   new `collect.*` field.
2. **`num_qs` / `reduction` live on the SHARED `CriticTrainingConfig`**
   (`src/training/config.py:124-155`), inherited by all nine registered configs. Set C7/C8 on
   the new OGPO config object or the recipe — **never** on the dataclass default.
3. **`get_value_bounds` returns `upper = 0.0`** (`src/rl/value_distribution.py:129-134`).
   A +72 bonus makes that wrong. Latent today (`num_value_bins = 1`) but fix it. The lower
   bound needs a call too: the formula gives −173 at T=400 while `fix_mc_returns` pins
   failures at exactly −200.
4. **A2 needs a policy sample inside the critic step ⇒ new RNG split arity.** Tier 2 by
   CLAUDE.md §6 regardless of diff size. It also has a **clone**:
   `src/rl/best_of_n/update_critic.py` mirrors `advantage_weighted_sft/update_critic.py` with
   eleven near-identical helpers. Apply to both or say in `DIFF.md` why not.
5. **C8 needs no code.** `summarize_critic_values` already threads `critic_reduction` and
   implements `"mean"` at `advantage_weighted_sft/update_critic.py:80`. It is a config value,
   not an edit.

## Feasibility gate before anything is launched

`BroNetStateActionCritic` (`src/rl/networks/bronet_critic.py:118-121`) builds `num_qs`
**separate** BRONet towers in a Python list — no `vmap`, no shared trunk. C7 is a literal 5×
in critic parameters, optimizer state and activations, for both Q and V, at
`critic.batch_size` 1024. **C9 was dropped (D5)**, so the actor's memory is unchanged and only the critic grows —
materially lower risk than when G=32 was in scope.

Context: the `d20` arm **OOM'd at 150G**, and `ogpo_learner.py:505`'s `del ema_dev` is
load-bearing for a ~34.7 GiB floor.

**A GPU memory smoke is required before C7 or C9 is committed to a run**, and per CLAUDE.md
that smoke is itself a run — **propose the command and wait for explicit permission.**

## Verification (CLAUDE.md step 3)

- Pytest-native, on the `tests/ogpo/` dummy-variant pattern: reward-wrapper bonus and
  termination; `get_value_bounds` under a positive reward; A4's two-batch metric schema;
  `summarize_critic_values` at `reduction="mean"`, `num_qs=10`.
- **A2 requires a differential test** against a verbatim copy of the pre-change TD target,
  which must be bit-identical at `q_vr_num_samples=1` — the
  `tests/ogpo/test_split_equivalence.py` pattern. Comparing a refactor to itself proves
  nothing.
- `DRY=1 bash scripts/<new recipe>.sh` for every Group-1 flag.
- `pytest tests/ogpo` (23 tests, CPU, seconds). `pytest tests/` still fails at collection
  (OQ-4) — out of scope.
- **Say plainly what could not be verified.** Memory claims and run outcomes cannot be.

## Hard constraints

- **Never launch a run without explicit permission** — no `sbatch`, no `nohup bash
  scripts/*.sh`, no `uv run scripts/exp.py`. Propose and wait.
- **Deviation protocol**: surface any non-raise error path (swallow-and-log, `return None`,
  silent default, `getattr(cfg, "x", default)`) and get an explicit OK before writing it.
- Do not commit or push unless asked.
- Report failing tests with their output.

## Two things that are NOT changes (already checked — don't redo this work)

1. **The SDE→ODE score correction exists here.** openpi's `_get_sde_dist`
   (`openpi/src/openpi/models/pi0.py:147-166`) carries the marginal-preserving `(σ_t²/2)·score`
   drift in pi0's reversed time convention. Equivalent to the reference's
   `sde_drift_correction`.
2. **The critic never seeing demonstration data is not a divergence.** The reference does the
   same and enforces it: `offline_ratio=0.0` in all 15 `scripts/ogpo/*.sh`, a hard raise at
   `ogpo/runners/online_rl_runner.py:374-375`, and `train_dataset` set to `None` and gc'd at
   `:383-386`.
