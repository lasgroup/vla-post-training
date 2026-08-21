# Align this repo's OGPO with the reference implementation

**Tier:** 2 (config dataclass hierarchy, critic ensemble shape, optimizer/`TrainState`,
reward semantics, ≥2 learner packages)
**Status:** discovery pass complete, decisions taken. **No source touched. No plan approved.**
Next step is the plan phase (plan mode) — see `HANDOFF.md`.
**Date:** 2026-08-20 · **Branch at time of writing:** `shashwat/stability-study`

## Why

`reports/findings.md` §10 establishes that OGPO's policy-gradient term is net **−10.6
points** against its own BC warmstart, paired, on **ten of ten** arms (t = −9.8), and traces
it to a critic pinned on the −200 Bellman fixed point whose within-state spread is ~0.3 on a
200-unit scale.

`reports/ogpo_reference_divergence.md` compares this implementation against the reference
(`/home/pchellap/Projects/SafeVADAR/OGPO_public`, closest recipe
`scripts/ogpo/square_image_paligemma.sh` — frozen PaliGemma encoder restored from a
`pi05_libero` checkpoint, MLP heads). It finds **6 mechanisms present upstream and absent
here**, **5 present here and absent upstream**, and a hyperparameter table in which the
absolute PPO trust region differs by **114×**.

Several of the divergences map one-to-one onto the measured failure. This change is to close
them deliberately, not to tune further.

## Resolved during discovery — NOT divergences

Two items from the divergence doc are now closed, and one earlier claim of mine is withdrawn.

1. **SDE→ODE score correction: we have it.** openpi's `_get_sde_dist`
   (`openpi/src/openpi/models/pi0.py:147-166`) computes
   `mean_t = (1 + σ_t²/(2t)·dt)·x_t + (1 + σ_t²/(2t)·(1−t))·v_t·dt`, which is the
   marginal-preserving `(σ_t²/2)·score` drift written in pi0's reversed time convention
   (t: 1→0). Equivalent to the reference's `sde_drift_correction`. **Drop item A8.**
2. **Noise schedule: both are tapered.** Ours is `σ₀·√(t/(1−t))` with `time` clipped to
   `1−|dt|` (`pi0.py:157-160`), so at `num_steps=10`, `noise_level=0.02` it is bounded at
   `3σ₀ = 0.06` at the noise end and → 0 at the data end. Same qualitative shape as the
   reference's `σ√(1−t)`. Different functional form and ~2× magnitude, but **not** the
   uncorrected-SDE bug I flagged as a caveat.
3. **`clip_bc` is out of scope.** It lives in `ogpo/runners/bc_runner.py:104-106`, i.e. the
   reference's *offline BC pretraining* phase (`bc_pi_steps=500000`). We have no such phase
   — we start from an SFT checkpoint. **Drop item A6.**
4. **Withdrawn:** my earlier remark that coupling the critic's batch to `online_ratio` is
   "arguably wrong". The reference makes the same choice and enforces it — `offline_ratio=0.0`
   in all 15 `scripts/ogpo/*.sh`, a hard raise at `online_rl_runner.py:374-375`, and
   `train_dataset` set to `None` and gc'd at `:383-386`. **We match upstream. Not a change.**

## The candidate change set

Grouped by cost. IDs match `reports/ogpo_reference_divergence.md`.

### Group 1 — config-only, no code

| id | change | ours → reference |
|---|---|---|
| C1 | `rl.clip_epsilon` | 0.1 → **0.01** |
| C2 | `rl.discount` | 0.995 → **0.99** |
| C3 | `rl.advantage_combination` | `reduced` → **`grpo_conservative`** (`CONS=1`) |
| C4 | `rl.normalize_group_advantage` off | ours-only (`NORM=0`) |
| C5 | `rl.normalize_advantage_per_task` off | ours-only (`MT_ADV=0`) |
| C6 | `rl.adv_clip_sym` off | ours-only (`CLIP_SYM=`) |
| C7 | `rl.critic.num_qs` / `num_vs` | 2 → **10** |
| C8 | `rl.critic.reduction` | `min` → **`mean`** |
| C9 | `rl.group_num_samples` (G) | 8 → **32** |

C7–C9 are config-only but **not** cost-free: see "Feasibility" below.

### Group 2 — small, contained code changes

| id | change | site |
|---|---|---|
| A4 | second success-only TD batch per critic update (2× success oversampling) | `ogpo_learner.update()` — the batch is **already built** at `:471-481` for the BC anchor and simply never reaches the critic |
| A5 | actor LR drop + optimizer reset (+ warmup/cosine) at `pg_start_step` | `ogpo_learner`, `TrainState.tx` |

### Group 3 — larger code changes

| id | change | site |
|---|---|---|
| A1 | **success bonus in the reward** | `src/envs/wrappers.py:230-237` |
| A2 | Q-target variance reduction — average `next_q` over 8 sampled next-actions | `advantage_weighted_sft/update_critic.py` (+ its clone) |
| A3 | Best-of-N as the collection behaviour policy | `ogpo_learner.sample_actions` (scoring code exists in `best_of_n_learner.py:326-516`) |

### Group 4 — deliberately out of scope unless asked

| id | change | why |
|---|---|---|
| A7 | `pi_slow` + χ²/KL pessimism, β annealed by Q-ensemble spread | off in every reference PaliGemma recipe; also requires C7 first (a 2-head spread is meaningless) |
| B4 | remove burst / `critic_utd` / `burst_use_mc_targets` | these compensate for a collection cadence the reference does not have and we cannot adopt |
| — | action chunk 10 → 4, `online_steps` 100k → 2M, 1-update-per-env-step | not reachable: chunk size is a π0.5 architecture property, and 2M env steps of a 3B VLA is out of budget |

## Feasibility notes

- **C7 (`num_qs` 10):** `BroNetStateActionCritic` (`src/rl/networks/bronet_critic.py:118-121`)
  builds `num_qs` **separate** BRONet towers in a Python list and `jnp.stack`s their outputs
  — no `vmap`, no shared trunk. Going 2 → 10 is a literal 5× in critic parameters, optimizer
  state and activations, for both Q and V (`num_vs` also 2 → 10) at `critic.batch_size` 1024.
  The `d20` arm already **OOM'd at 150G**, and `ogpo_learner.py:505`'s `del ema_dev` is
  load-bearing for a ~34.7 GiB floor. **Needs a memory smoke before committing.**
- **C9 (G = 32):** G is the PPO group; the actor samples G SDE chains per state through the
  3B action expert. 8 → 32 is 4× the actor forward *and* backward. Almost certainly the most
  expensive single item on the list.
- **A2:** 8 policy forward passes per critic step, at `critic.update_interval` = 1 and
  `critic.batch_size` 1024.

## Decisions taken (user, 2026-08-20)

The four open questions from the discovery pass are resolved. `BLAST-RADIUS.md` §5 holds the
framing that produced them.

| # | decision |
|---|---|
| **D1 — reward (A1)** | **Terminal bonus, ratio-scaled.** Keep −1/step; add a one-off bonus on the terminating step sized to preserve the reference's bonus-to-floor ratio: 36/100 × 200 ⇒ **+72**. Termination behaviour unchanged; **no** `post_success_steps`. |
| **D2 — scope** | **Groups 1, 2 and 3.** Group 4 (`pi_slow` / χ²-KL pessimism) is out. |
| **D3 — LR handoff (A5)** | **Out of scope.** Do not touch the actor optimizer, `TrainState.tx` or `opt_state`. |
| **D4 — compatibility** | **Parallel config + recipe.** Add a new registered config and a parallel shell recipe; the current OGPO stack stays runnable and all ten existing arms plus the BoN runs remain valid comparators. |

**D2 ∧ D3 ⇒ Group 2 reduces to A4 alone.** The final in-scope set is:

- **Group 1 (config):** C1 `clip_epsilon` 0.1→0.01 · C2 `discount` 0.995→0.99 ·
  C3 `advantage_combination` → `grpo_conservative` · C4 `normalize_group_advantage` off ·
  C5 `normalize_advantage_per_task` off · C6 `adv_clip_sym` off ·
  C7 `num_qs`/`num_vs` 2→10 · C8 `reduction` min→mean · C9 `group_num_samples` 8→32
- **Group 2:** A4 — second success-only TD batch per critic update
- **Group 3:** A1 — +72 terminal success bonus · A2 — Q-target variance reduction over 8
  sampled next-actions · A3 — Best-of-N as the collection behaviour policy

### Consequences of D1 that the plan must handle

- `get_value_bounds` (`src/rl/value_distribution.py:129-134`) returns `upper = 0.0`. With a
  +72 bonus, Q exceeds 0 and that bound is **wrong**. Latent today (`num_value_bins = 1`
  ⇒ Gaussian, bounds unused) but it must be corrected or it becomes a silent trap.
  The lower bound also needs a decision: the formula gives −173 at T=400, while
  `fix_mc_returns` pins failures at exactly **−200** = `reward/(1−γ)`. **−200 is the value
  that actually appears in the data.**
- `fix_mc_returns` (`filtered_sft_learner.py:749-751`) gates on
  `np.all(reward == reward[0])`. Successes become `[−1, …, −1, +72]` — still non-constant, so
  unaffected. Failures stay all −1 and are still overwritten to −200. **Verify, don't assume.**
- The terminal chunk bootstraps zero (`_discount = 0.0` when any step in the window
  terminates, `filtered_sft_learner.py:746`), so the success chunk's Q is essentially the
  bonus itself. That is the intent.

### D4 mechanics

- New registered config alongside the existing nine in `src/training/config.py:571-634`.
  **The `isinstance` dispatch-order sharp edge applies** (`scripts/exp.py:74-85`): if the new
  class subclasses `OGPOSFTLearnerConfig`, it must be checked *before* the OGPO branch, or it
  silently routes to the wrong learner.
- A parallel shell recipe. Note `ogpo_multitask_4task.sh` and `stability_study.sh` share a
  ~60-line env preamble that has **already diverged** (`BLAST-RADIUS.md` §1) — do not deepen
  the divergence without saying so in `DIFF.md`.
- A1 edits a wrapper used by the base learner (`filtered_sft_learner.py:62-63`), so it reaches
  **all six learners** regardless of D4. The bonus must therefore be **config-gated**, not
  hardcoded into `TimeToSuccessAsRewardWrapper`'s behaviour, or the parallel-config isolation
  D4 buys is defeated. This is the single most important structural constraint on the plan.

## Decision amendments (2026-08-20, later in the same session)

| # | amendment |
|---|---|
| **D5 — C9 dropped** | `group_num_samples` stays at **8**. G=32 is 4× the actor forward *and* backward through the 3B action expert; not affordable. |
| **D6 — C1 dropped** | `clip_epsilon` stays at **0.1**. Evidence below. |
| **D7 — C2 flagged** | `discount` 0.995 → 0.99 now looks **actively harmful** for our episode lengths. Evidence below. Open. |

### Why C1 (clip_epsilon) is dropped — measured, not argued

Measured over all post-PG steps of arm `ncb`:

| | mean | min | max |
|---|---|---|---|
| `actor/ratio_p05` | 0.8467 | 0.7053 | 0.9024 |
| `actor/ratio_p50` | 0.9350 | 0.8730 | 0.9573 |
| `actor/ratio_p95` | 0.9655 | 0.9166 | 0.9801 |
| **`actor/ratio_max`** | **0.9765** | 0.9352 | **0.9888** |

`clip_epsilon = 0.01` puts the lower bound at **0.990**. The largest ratio ever observed in
the run is **0.9888** — *every sample in every batch* would fall outside the trust region.
`clipfrac` → ~1.0, and since `min(ratio·A, clip(ratio)·A)` clips only `A < 0` samples when
ratio ≤ 1 (§10f), the PG would collapse to **positive-advantages-only**. That is a far larger
behavioural change than the number suggests, and it is not what the reference's ε=0.01 does
in *its* ratio distribution.

Worth recording: `OGPOSFTLearnerConfig.clip_epsilon`'s **dataclass default is already 0.01**
(`src/training/config.py:215`) — matching the reference. `ogpo_multitask_4task.sh:203`
overrides it to 0.1. The 10× loosening was a deliberate recipe choice, not a porting slip.

### Why C2 (discount) is flagged — the reference's γ does not transfer

Our successes take a measured **295 env steps** (`eval/mean_success_episode_length`, counted
in env steps at `collect.py:65-66`, `max_episode_steps = 400`). The success-vs-failure gap,
as a fraction of the value range:

| setting | γ^295 | failure floor | success return | gap | % of range |
|---|---|---|---|---|---|
| today (γ=0.995, no bonus) | 0.228 | −200 | −154.4 | 45.6 | **22.8%** |
| **D1 (γ=0.995, +72 bonus)** | 0.228 | −200 | −138.0 | 62.0 | **31.0%** |
| C2 (γ=0.99, +72 bonus) | 0.0517 | −100 | −91.1 | 8.9 | **8.9%** |

**C2 would cut the gap to well below today's.** At γ=0.99 the credit horizon is 100 steps
against a 295-step success, so the terminal event is discounted into near-invisibility.

The reference runs γ=0.99 on `square` with the same 400-step cap, but its successes arrive
sooner; at L≈150, γ^150 = 0.22 and its gap is ~30% — the same 0.22 discount factor our
γ=0.995 buys us at L=295. **γ=0.995 here *is* the equivalent of γ=0.99 there.** Matching the
number un-matches the mechanism.

### Correction to the D1 arithmetic

The option preview shown when D1 was taken read `success @295: −154.5 + 72 = −82.5, gap
117.5 (59%)`. **That was wrong — it failed to discount the bonus by γ^295 = 0.228.** The
correct figure is −138.0, gap 62.0, **31%**.

The decision itself stands: **+72 is still the right magnitude**, because 31% is what
reproduces the reference's ~30% relative gap. But the expected effect is **1.36×** today's
signal, not 2.6×.

---

## FINAL change set (all decisions closed, 2026-08-20)

| # | amendment |
|---|---|
| **D8 — C2 dropped** | `discount` stays at **0.995**. Matching the reference's *mechanism* (discount-to-success ≈ 0.22) rather than its *number*. See the table above. |
| **D9 — burst kept** | `post_collection_critic_steps` stays at **1000** in the new recipe, despite having no upstream counterpart and measuring inert three times (§5a). |

### In scope

**Group 1 — config only (no code):**

| id | change |
|---|---|
| C3 | `rl.advantage_combination` → **`grpo_conservative`** (`CONS=1`) |
| C4 | `rl.normalize_group_advantage` **off** (`NORM=0`) |
| C5 | `rl.normalize_advantage_per_task` **off** (`MT_ADV=0`) |
| C6 | `rl.adv_clip_sym` **off** (`CLIP_SYM=`) |
| C7 | `rl.critic.num_qs` / `num_vs` 2 → **10** |
| C8 | `rl.critic.reduction` `min` → **`mean`** (no code — `update_critic.py:80` already implements it) |

**Group 2:** A4 — second success-only TD batch per critic update.

**Group 3:** A1 — **+72** terminal success bonus (config-gated) · A2 — Q-target variance
reduction over 8 sampled next-actions · A3 — Best-of-N as the collection behaviour policy.

### Explicitly OUT

| id | why |
|---|---|
| C1 `clip_epsilon` 0.1 → 0.01 | measured `ratio_max` ≤ 0.9888 < 0.990 ⇒ `clipfrac` → 1.0, PG collapses to positive-advantages-only (D6) |
| C2 `discount` 0.995 → 0.99 | would cut the success/failure gap 22.8% → 8.9%; γ=0.995 here *is* γ=0.99 there (D8) |
| C9 `group_num_samples` 8 → 32 | 4× actor forward **and** backward through the 3B expert; not affordable (D5) |
| A5 LR drop / optimizer reset | out of scope; do not touch `TrainState.tx` or `opt_state` (D3) |
| A7 `pi_slow` / χ²-KL pessimism | off in every reference PaliGemma recipe (D2) |

### Preserved unchanged (ours-only, but staying)

`post_collection_critic_steps` = 1000 (D9) · `balance_success_buffer_tasks` on ·
`use_success_buffer` on (upstream has one too) · `pg_start_step` / `pg_ramp_steps` ·
`policy_grad_accum` · `dedup_group_prefix` (perf only, allclose-certified by
`tests/ogpo/test_group_dedup.py`) · `num_initial_rollouts` · `fix_mc_returns` ·
`noise_level` 0.02 · `critic_utd` = 1 · `burst_use_mc_targets` off · `bc_filtered_sft` off ·
`td_weight_schedule` TD-only.

### Feasibility, revised

**D5 removes the largest memory item.** With C9 gone the actor's memory is unchanged; only
the critic grows. C7 is a 5× on the Q **and** V BRONet towers
(`bronet_critic.py:118-121` — separate towers in a Python list, no `vmap`) at
`critic.batch_size` 1024, which is small relative to the 3B policy. A memory smoke is still
required before a run, but the risk is materially lower than when C9 was in scope.

---

## Amendments (2026-08-20, third round)

| # | amendment |
|---|---|
| **D10 — nothing is deleted** | C4/C5/C6 are **recipe-level off**, never code removal. Every field stays on `OGPOSFTLearnerConfig` and every env var stays in the new recipe. |
| **D11 — TD/MC blend kept at 0.95/0.05** | `rl.critic.td_weight_schedule` init = end = **0.95**, i.e. 95% TD / 5% MC in the critic target. A deliberate divergence from the reference. |

### D10 — preservation constraint (binding on the plan)

`normalize_group_advantage`, `normalize_advantage_per_task` and `adv_clip_sym` must remain:

- as dataclass fields on `OGPOSFTLearnerConfig` (`src/training/config.py:210-306`), unchanged;
- as `NORM` / `MT_ADV` / `CLIP_SYM` env vars in the new shell recipe, defaulting to **off**
  but flippable without editing source.

The same applies to everything in "Preserved unchanged" above. **No field is to be deleted
and no `--rl.*` flag is to disappear.** A reviewer should be able to reproduce the current
stack from the new recipe by setting env vars alone.

One consequence to carry into step 4 (docs): C4 being *off by default* rather than *removed*
means the **`_adv_scale`-is-not-checkpointed gotcha stays live** (`ogpo_learner.py:92-96`).
Do not delete that entry — it still applies to anyone who sets `NORM=1`.

### D11 — the TD/MC blend, and what the evidence actually says

Measured, post-PG (≥30k), all completed arms:

| arm | `td_weight` | post-PG mean | final | `q_mc_corr` | `q_value_mean` | `q_mc_loss` |
|---|---|---|---|---|---|---|
| `ncb` | 1.00 | **28.6** | **37.5** | 0.430 | −198.56 | 2571 |
| `d10` | 1.00 | **29.3** | 35.9 | 0.367 | −198.58 | 2652 |
| `mcblend` | 0.99 | 30.1 | 5.5 | 0.602 | −196.80 | 2039 |
| **`mcblend95`** | **0.95** | 24.6 | 20.3 | **0.899** | **−189.46** | **910** |
| `mcsched` | 0.99 | 20.3 | 11.7 | 0.630 | −198.98 | 2026 |

**Stated plainly: the 0.95 blend improved the critic, not the success rate.** On every
critic metric it is the best arm in the campaign — correlation 0.43 → 0.899, MC loss RMS
51 → 30, and it is **the only arm that measurably escaped the −200 fixed point** (−189.5 vs
−198.6). On policy outcome it is *below* the TD-only arms: post-PG 24.6 against 28.6 and
29.3, final 20.3 against 37.5 and 35.9. §5b's paired-by-step test over the whole MC family
is a null (TD − MC = +1.9 ± 9.8, t = +0.54) — not a harm, but not a gain either.

**Why keeping it is nevertheless the better bet now**, and this is the reason it is being
kept rather than the historical numbers:

§5b's mechanism for why MC targets bought nothing was that *MC returns are nearly a function
of state alone*, so regressing Q onto them pulls Q(s,·) flat in `a` — exactly the component
group centring keeps. **A1 breaks that premise.** With a +72 terminal bonus the MC return
depends sharply on *whether* the episode succeeds, not merely on *when* it ends, and whether
it succeeds is action-dependent. The blend and the bonus attack the −200 attractor from two
sides — the bonus moves the TD fixed point, the blend stops the critic sitting on it.

**Known confound, recorded deliberately:** A1 and D11 target the same failure. Run together,
a good result cannot be attributed to either. Accepted; the alternative (a separate arm) was
declined in favour of the stronger combined shot.

**Divergence from the reference, recorded so nobody "fixes" it:** upstream has the mechanism
— `mc_regression` + `mc_regression_coeff` (`ogpo/agents/modules/q_helper.py`
`compute_mc_regression_loss`) — but sets `mc_regression=false` in **every** recipe under
`scripts/`, with no exceptions. Upstream's form is also a *separate loss term*, not a blend
into the TD target as ours is. D11 is a deliberate, evidence-informed local choice, not an
alignment item.

### Revised final recipe deltas

Relative to `scripts/ogpo_multitask_4task.sh` today, the new recipe sets:

| knob | today | new |
|---|---|---|
| `rl.advantage_combination` | `reduced` (`CONS=0`) | **`grpo_conservative`** |
| `rl.normalize_group_advantage` | on | **off** |
| `rl.normalize_advantage_per_task` | on | **off** |
| `rl.adv_clip_sym` | 4.0 | **off** |
| `rl.critic.num_qs` / `num_vs` | 2 | **10** |
| `rl.critic.reduction` | `min` | **`mean`** |
| `rl.critic.td_weight_schedule` init/end | 1.0 / 1.0 | **0.95 / 0.95** |
| terminal success reward | 0.0 | **+72** (new config-gated field) |
| critic success oversampling (A4) | none | **on** |
| Q-target variance reduction (A2) | none | **8 samples, mean** |
| Best-of-N collection (A3) | none | **on** |

Unchanged and explicitly so: `clip_epsilon` 0.1 · `discount` 0.995 ·
`group_num_samples` 8 · `post_collection_critic_steps` 1000 · `bc_coeff` 1.0 ·
`pg_start_step` 20000 / `pg_ramp_steps` 5000 · `balance_success_buffer_tasks` on ·
`use_success_buffer` on · `noise_level` 0.02 · `critic_utd` 1 · actor LR 2.5e-5 constant.

---

## D12 — the terminal reward bonus is config-driven (explicit, 2026-08-20)

Restating D1's "config-gated" as a hard requirement, because it is the constraint most likely
to be lost in implementation:

- The bonus magnitude is a **new field on `CollectConfig`** (`src/training/config.py`, beside
  `use_time_to_success_as_reward` at `:371`), **not** a literal in
  `TimeToSuccessAsRewardWrapper`.
- Its default must **reproduce today's behaviour exactly** (bonus 0.0 ⇒ reward 0.0 on the
  terminating step), so that every existing config, recipe and prior run remains bit-identical
  unless the new recipe opts in.
- The new recipe sets it to **72.0**, ideally via an env var (e.g. `SUCC_BONUS`) so it is
  sweepable without editing source.
- Rationale: `TimeToSuccessAsRewardWrapper` is instantiated by the **base** learner
  (`filtered_sft_learner.py:62-63`), so it is shared by filtered SFT, AWR, MPO, FlowGRPO,
  Best-of-N and OGPO. A hardcoded bonus silently changes the reward for all six and destroys
  the isolation D4 exists to provide — including for the colleague's Best-of-N runs.

---

## Corrections from the plan phase (2026-08-20, applied)

Reading the code collapsed three items. `PLAN.md` and `DIFF.md` carry the detail.

| item | discovery record said | verified reality |
|---|---|---|
| **A3** Best-of-N collection | Group 3; "a third copy of the BoN scoring block" | **Zero code.** `AdvantageWeightedSFTLearner.sample_actions:313` already implements it, gated on `rl.n_samples > 1`; `OGPOSFTLearnerConfig` inherits `n_samples` (`config.py:176`) and `OGPOAgentLearner` never overrides `sample_actions`. It is `--rl.n_samples 8`. No BoN refactor, no risk to the Best-of-N runs. |
| **A2** Q-target variance reduction | Group 3; port from the reference | **Dropped (user).** Inapplicable: our TD target bootstraps `value_model(next_observation)` (`update_critic.py:255-258`) — a separate V network, so there is no next-action to average. The analogous defect (V regresses onto `Q(s, a_buffer)`, `:325-330`) needs policy sampling inside the critic jit at batch 1024 × 10 critic steps per policy step. C7 reduces variance on the same target affordably instead. |
| **D4** parallel config | flagged the `isinstance` dispatch-order sharp edge | **Not in play.** The existing OGPO config registers `OGPOSFTLearnerConfig` directly (`config.py:626-634`), so the second entry needs no new class and `scripts/exp.py:74-85` is untouched. |

One item was also **added** during implementation: `get_value_bounds`'s lower bound was
`-(1-γ^T)/(1-γ)` = −173.07, while `fix_mc_returns` pins every failed episode's MC return to
`reward/(1-γ)` = −200. A pre-existing latent bug (unused at `num_value_bins = 1`), fixed
while in the function since A1 required touching the upper bound anyway.
