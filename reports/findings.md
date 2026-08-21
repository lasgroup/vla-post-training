# OGPO 4-Task Multi-Task Campaign — Findings

**Date:** 2026-08-20 · **Branch:** `shashwat/stability-study` · **Task set:** LIBERO-90
`{79, 31, 82, 38}` · **Seed:** 0 for every run · **Total compute:** ~144 GPU-hours across
10 runs (all complete).

All numbers below are read directly from the offline wandb record files under
`run_store/wandb/wandb/`. Nothing here is from memory or from a previous summary.

---

## 0. Read this first: two measurement facts that change how the numbers are read

**(a) There are two different success-rate metrics, and they are not the same thing.**

| logged key | source | episodes per point |
|---|---|---|
| `eval/success_rate` | `evaluate_policy`, `collect.py:109` | `num_eval_rollouts × 4` = **128** |
| `success_rate` | `collect_data`, `collect.py:219` | `num_rollouts × 4` = **20** (5-rollout arms) |

`success_rate` is the *on-policy collection* rate — the rollouts that go into the replay
buffer, only 5 episodes per task on most arms. **Every eval number in this document is
`eval/success_rate` (n=128).** The two agree in the aggregate (paired Pearson
r = 0.677, n = 85, means 27.4 vs 28.7) but the collection metric is far too noisy to
compare arms with: at p ≈ 0.28 its binomial floor is 10.0 points versus 3.9 for eval.

**(b) The eval set is deterministic and effectively shared across all runs.**
`LiberoWrapper.__init__` hardcodes `seeding.np_random(0)` (`src/envs/libero.py:24`), the
eval env is rebuilt fresh at every eval (`scripts/exp.py:143-159`), and
`evaluate_policy` — unlike `collect_data`, which calls `env.seed(config.seed + step)` at
`collect.py:121` — **never re-seeds**. Every eval therefore draws initial states from the
same RNG stream. Combined with `seed=0` on all ten runs, arms are evaluated against
essentially the same episode set rather than independent samples.

This is good for paired comparison and bad for treating arms as independent replicates.
One caveat: which env worker resets when depends on episode lengths, which are
policy-dependent, so the consumption order can drift between arms. The streams are
identical; the realized episode sequences are only approximately so.

*Provenance note:* the offline reader used for this analysis initially used wandb's raw
`scan_record()`, which does not handle the 32KB block padding in the datastore format and
silently truncated every run to its first 509 records. It was rebuilt on `scan_data()`.
All figures here are from the corrected reader, verified against the job logs
(e.g. the in-flight run parses to step 37,225 against a log line reading 37.2k/100k).

---

## 1. The experiment matrix

Ten real runs. A full recursive diff over all 155 config keys shows the arms differ in
**exactly eight substantive fields** (plus `exp_name` and a 100000-vs-100001 step count);
everything else is byte-identical across all ten. The eight: `collect.num_rollouts`,
`rl.post_collection_critic_steps`, `rl.burst_use_mc_targets`,
`rl.critic.td_weight_schedule`, `rl.advantage_combination`, `rl.adv_clip_sym`,
`rl.normalize_advantage_per_task`, `rl.normalize_group_advantage`.

| # | arm | rollouts /task | burst | burst targets | `td_weight` | `advantage_combination` | adv norm | status |
|---|---|---|---|---|---|---|---|---|
| 1 | `v0` | 5 | 1000 | TD | 1.0 | **grpo_conservative** | full | done (99,975) |
| 2 | `ncb` | 5 | 1000 | TD | 1.0 | reduced | full | done 100k |
| 3 | `b` | 5 | 1000 | TD | 1.0 | reduced | **all off** | done 100k |
| 4 | `d10` | **10** | 1000 | TD | 1.0 | reduced | full | done 100k |
| 5 | `d20` | **20** | **3000** | TD | 1.0 | reduced | full | **OOM @ 40k** |
| 6 | `d10_b3000` | **10** | **3000** | TD | 1.0 | reduced | full | done 100k |
| 7 | `mcburst` | 5 | 1000 | **MC** | 1.0 | reduced | full | done 100k |
| 8 | `mcblend` | 5 | 1000 | **MC** | **0.99** | reduced | full | done 100k |
| 9 | `mcblend95` | 5 | 1000 | **MC** | **0.95** | reduced | full | done 100k |
| 10 | `mcsched` | 5 | 1000 | **MC** | **0.9 → 0.99 @20k** | reduced | full | done 100k |

"adv norm = all off" means arm `b` disables three things at once — `adv_clip_sym`
(4.0 → null), `normalize_advantage_per_task`, and `normalize_group_advantage`. It is a
confounded three-way change and cannot attribute an effect to any one of them.

Constant across every arm: G = 8, `clip_epsilon` 0.1, `bc_coeff` 1.0, `num_sde_steps` 10,
`noise_level` 0.02, γ = 0.995, batch 32, critic batch 1024, `num_qs` 2, reduction `min`,
`pg_start_step` 20000, `pg_ramp_steps` 5000, `policy.update_interval` 10,
`collect_interval` = `eval_interval` = 10000, per-task adv norm and task-balanced success
buffer both on (except arm `b`).

`update_interval` 10 over a 10k block means **1,000 policy updates between collections.**

---

## 2. Headline results

**The root cause is in §10; it was found after §§4–9 were written and it reframes them.**

0. **The policy gradient is net-negative on ten of ten arms.** Evals at 10k and 20k are
   BC-only (`pg_start_step` = 20000). Pre-PG mean 36.6%, post-PG mean 26.1%, paired
   **Δ = −10.6 ± 3.4, t = −9.8**, every arm negative. The step-averaged series crashes
   38.0 → 18.0 across the PG ramp and never returns to the warmstart level (§10a).
1. **The critic is a constant.** Reward −1/step and γ = 0.995 put the failure fixed point at
   exactly −200; measured `q_value_mean` is −197.6 to −201.7. TD loss RMS 4.5, MC loss RMS
   51 — the error profile of a network emitting one number for every (s, a) (§10b).
2. **The signal the PG consumes is 0.3 units wide on a 200-unit scale.** Read directly off
   arm `b`, the only run with normalisation off: raw group-centred advantage std 0.26 → 0.66
   (§10c).
3. **The normalisers hide that completely.** Per-task norm forces `advantage_std` to exactly
   1.000; `adv_scale` then fixes `advantage_std_final` at 0.36 for the whole run. A dead
   critic and a perfect critic produce the same-sized gradient — §7's result is right and its
   interpretation was backwards (§10d).
4. **The BC anchor is 1/7th–1/12th of the gradient**, i.e. cos(update, PG) = 0.99. There is
   effectively no anchor (§10e).
5. **No arm separates from any other**, and 39% of eval variance is a shared function of
   training step — the campaign as designed cannot resolve the arm effects it was built to
   measure (§4).
6. **Every knob tested had to be a null.** Bursts, MC targets and data volume all improve
   *between-state* accuracy, which group centring deletes identically (§5b, §10g).
7. **Collapses happen on frozen data with every guardrail reading normal** — necessarily so,
   because the normalisers make the surrogate's input well-conditioned regardless (§8, §10f).

---

## 3. Full eval series (`eval/success_rate`, n=128 per point)

| arm | 10k | 20k | 30k | 40k | 50k | 60k | 70k | 80k | 90k | 100k |
|---|---|---|---|---|---|---|---|---|---|---|
| `v0` | 31.2 | 34.4 | 13.3 | 15.6 | 28.9 | 26.6 | 35.2 | 10.9 | 35.9 | — |
| `ncb` | 28.9 | 39.8 | 23.4 | 14.8 | 37.5 | 26.6 | 40.6 | 12.5 | 35.9 | 37.5 |
| `b` | 33.6 | 37.5 | 25.8 | 28.1 | 35.9 | 24.2 | 26.6 | 22.7 | 47.7 | 3.9 |
| `d10` | 40.6 | 31.2 | 13.3 | 31.2 | 39.1 | 34.4 | 34.4 | 9.4 | 36.7 | 35.9 |
| `d20` | 35.2 | 39.8 | 21.9 | — | — | — | — | — | — | — |
| `d10_b3000` | 32.8 | 39.1 | 15.6 | 11.7 | 43.0 | 26.6 | 32.8 | 37.5 | 7.8 | 35.2 |
| `mcburst` | 39.8 | 43.8 | 23.4 | 43.8 | 29.7 | 25.0 | 39.8 | 15.6 | 30.5 | 21.1 |
| `mcblend` | 39.1 | 39.1 | 12.5 | 43.0 | 21.1 | 35.2 | 51.6 | 20.3 | 51.6 | 5.5 |
| `mcblend95` | 40.6 | 37.5 | 24.2 | 23.4 | 44.5 | 7.0 | 42.2 | 19.5 | 15.6 | 20.3 |
| `mcsched` | 30.5 | 37.5 | 6.2 | 24.2 | 22.7 | 1.6 | 50.0 | 26.6 | 19.5 | 11.7 |

### Post-PG summary (evals from 30k on, i.e. after PG has fully ramped in)

| arm | n | mean | sd | min | max | @20k | final | final − @20k | trend/40k |
|---|---|---|---|---|---|---|---|---|---|
| `v0` | 7 | 23.8 | 10.4 | 10.9 | 35.9 | 34.4 | 35.9 | +1.6 | +9.3 |
| `ncb` | 8 | **28.6** | 10.9 | 12.5 | 40.6 | 39.8 | 37.5 | −2.3 | +6.8 |
| `b` | 8 | 26.9 | 12.4 | 3.9 | 47.7 | 37.5 | 3.9 | −33.6 | −4.4 |
| `d10` | 8 | **29.3** | 11.4 | 9.4 | 39.1 | 31.2 | 35.9 | +4.7 | +4.6 |
| `d10_b3000` | 8 | 26.3 | 13.1 | 7.8 | 43.0 | 39.1 | 35.2 | −3.9 | +5.1 |
| `mcburst` | 8 | 28.6 | 9.4 | 15.6 | 43.8 | 43.8 | 21.1 | −22.7 | −5.2 |
| `mcblend` | 8 | 30.1 | 17.8 | 5.5 | 51.6 | 39.1 | 5.5 | −33.6 | +0.4 |
| `mcblend95` | 8 | 24.6 | 12.8 | 7.0 | 44.5 | 37.5 | 20.3 | −17.2 | −5.1 |
| `mcsched` | 8 | 20.3 | 15.0 | 1.6 | 50.0 | 37.5 | 11.7 | −25.8 | +3.6 |

**Pooled post-PG: n = 72, mean 26.5%, sd 12.4 — 3.2× the 3.9-pt binomial floor at n=128.**
The spread is real policy variation, not measurement noise.

Seven of nine completed arms end *below* their own 20k BC-warmstart value, five of them by
17–34 points. The 20k reading is the policy just before PG takes over; on this evidence
the PG phase is, on average, net-negative over 80k steps.

---

## 4. Result: the oscillation is common-mode, not knob-driven

Averaging each eval step across the eight arms with complete series:

| step | 10k | 20k | 30k | 40k | 50k | 60k | 70k | 80k | 90k | 100k |
|---|---|---|---|---|---|---|---|---|---|---|
| cross-arm mean | 35.7 | 38.2 | **18.1** | 27.5 | 34.2 | **22.6** | 39.7 | **20.5** | 30.7 | **21.4** |
| cross-arm sd | 4.8 | 3.5 | 7.1 | 11.7 | 8.8 | 12.1 | 8.5 | 8.8 | 15.4 | 13.7 |

Variance decomposition over all (arm, step) eval points:

- total variance **148.0** (sd 12.2)
- residual after removing the step mean **90.5** (sd 9.5)
- **fraction explained by step alone: 0.39**

Roughly 40% of the variation in this campaign is a function of *when* you look, not *what
you configured*. Eight arms that differ in critic target type, TD weight, burst length,
data volume, and advantage normalization all dip and recover together.

Consistent with this, the lag-1 autocorrelation of successive post-PG eval deltas (evals
from 30k on, all nine arms) is strongly negative in every single arm — −0.40 to −0.85
(mean ≈ −0.71). The series is a mean-reverting oscillator, not a trend plus noise.
Successive 10k blocks systematically undo each other.

**Consequence for the advisor conversation:** single-step arm-vs-arm comparisons in this
campaign are close to meaningless. An arm that "wins" at 70k and "loses" at 80k has told
you about the step, not the arm. With 8 evals/arm this design can rule out mean shifts of
roughly 10+ points; it cannot resolve 2–5 point effects.

The universal 30k dip (38.2 → 18.1, a 20.1-point drop shared by all ten arms) is the first
eval after PG fully engages — `pg_start_step` 20000 with a 5000-step ramp, confirmed in the
actor metrics, where `alive_fraction` goes 0.00–0.12 before 20k to 0.73–0.81 after 22.5k and
`advantage_std_final` goes 0.000 to ~0.35.

---

## 5. Result: the critic axis is closed

### 5a. TD bursts do nothing, at any length

`burst/q_mc_corr`, the Pearson correlation between head-mean Q and observed MC return,
measured at the end of each post-collection digestion burst:

| arm | burst steps | targets | corr per collection |
|---|---|---|---|
| `v0` | 1000 | TD | 0.28 0.31 0.43 0.47 0.49 0.35 0.39 0.42 0.21 |
| `ncb` | 1000 | TD | 0.37 0.40 0.23 0.43 0.38 0.51 0.53 0.49 0.47 0.57 |
| `b` | 1000 | TD | 0.46 0.37 0.47 0.34 0.38 0.25 0.43 0.49 0.42 0.31 |
| `d10` | 1000 | TD | 0.32 0.49 0.46 0.29 0.34 0.38 0.41 0.42 0.29 0.48 |
| `d20` | **3000** | TD | 0.34 0.46 0.27 |
| `d10_b3000` | **3000** | TD | 0.45 0.42 0.51 0.42 0.47 0.45 0.41 0.29 0.34 0.41 |
| `mcburst` | 1000 | **MC** | 0.91 0.78 0.85 0.88 0.85 0.90 0.89 0.88 0.88 0.90 |
| `mcblend` | 1000 | **MC** | 0.86 0.88 0.89 0.89 0.92 0.91 0.93 0.91 0.92 0.93 |
| `mcblend95` | 1000 | **MC** | 0.95 0.96 0.97 0.99 0.99 0.99 0.99 0.99 0.99 0.99 |
| `mcsched` | 1000 | **MC** | 0.98 0.95 0.96 0.95 0.96 0.96 0.96 0.94 0.93 0.94 |

Tripling the TD burst from 1000 to 3000 steps leaves correlation in the same 0.3–0.5 band —
two independent arms confirm it. Switching the *target* to MC at 1000 steps jumps it to
0.85–0.99 immediately.

The mechanism is visible in the loss values. Over `d10_b3000`'s ten completed 3000-step TD
bursts, burst `q_loss` runs 0.7–122.7 while `q_mc_loss` on the same batches runs
2,264–3,496 — three orders of magnitude apart. The TD objective is already essentially
solved: the critic sits at the −200 fixed point that `fix_mc_returns` writes for failed
episodes (`reward/(1−γ)` = −200 exactly), and `burst/q_value_mean` stays inside
−210.1 … −179.4 across every TD arm, i.e. within ~10% of that constant for the whole run.
More TD steps converge harder onto the same number. The burst is not under-trained; its
target carries no information to extract.

### 5b. Raising ranking quality buys nothing

`mcblend95` drove the critic's MC correlation from 0.003 to **0.948** and cut its MC
residual RMS from 67.2 to **22.2** — by far the best-fitting critic in the campaign. It
finished at 20.3% with a post-PG mean of 24.6%, *below* the `ncb` baseline's 28.6%.

**This is a null, not a harm — and the distinction took a paired test to establish.** At
100k the four MC arms average 14.6 against the four TD arms' 28.1, and across arms
corr(burst correlation, final eval) = −0.560, which reads as active damage. But a single
step is exactly the comparison §4 rules out. Repeating the family contrast at *every*
post-PG step gives TD − MC of +2.9, −12.1, +9.4, +10.7, −12.3, 0.0, +2.7, +13.5 — mean
**+1.9, sd 9.8, t = +0.54**. The 100k gap is one draw from a distribution that swings ±12
points. Across arms, corr(burst correlation, post-PG mean) = **−0.226** (n = 9).

Driving the critic's between-state accuracy to ceiling therefore moves policy performance
neither up nor down, by any measurement this campaign supports.

`mcsched`'s scheduled flip at 20k (`td_weight` 0.9 → 0.99) provides an internal control
confirming the mechanism fires as designed: training-time `q_mc_corr` moves 0.817 → 0.631
across the switch. The knob works. It just does not help.

**Why this is expected, not paradoxical:** OGPO's advantage is group-centered,
`advantage_{b,g} = Q(s_b,a_{b,g}) − mean_g Q(s_b,a_{b,g})` (`update_actor.py:296-297`), so
V(s) and any state-only component of Q cancel identically. `q_mc_corr` is computed over a
1024-transition batch with one action per state, so its variance is time-to-go and
success/failure — both state-level. It measures precisely the component the baseline
cancels. This does not make it worthless (it is upstream of within-state accuracy via the
bootstrap term `r + γV(s')`), but it is not the quantity the policy gradient consumes, and
these runs show the two are decoupled in practice.

**A mechanism worth testing before the next MC arm.** The MC return for a transition is
very nearly a function of the state alone — time-to-go plus eventual success — with almost
no dependence on which action was taken at that single step. Regressing Q hard onto MC
targets therefore pulls Q(s,·) toward a function of s, and a Q that is flat in `a` is
exactly the Q that group centering annihilates to zero signal. The MC target may be
optimizing the critic toward the degenerate case. This is testable for free with the
variance split in §12, and `mcblend95` is the natural positive control to point it at.

**We still have no direct measurement of within-state action-ranking quality.** That gap is
the main open item — see §12.

---

## 6. Result: more data does not help

| cell | arm | rollouts | burst | post-PG mean | verdict |
|---|---|---|---|---|---|
| baseline | `ncb` | 5 | 1000 TD | 28.6 | reference |
| more data | `d10` | 10 | 1000 TD | 29.3 | **clean null** vs `ncb` (+0.7) |
| **longer** burst, data fixed | `d10_b3000` | 10 | 3000 TD | 26.3 | **clean null** vs `d10` (−3.0) |
| **different** burst target | `mcburst` | 5 | 1000 MC | 28.6 | **clean null** vs `ncb` (0.0) |
| longer burst + 4× data | `d20` | 20 | 3000 TD | — | **OOM at 40k, not claimable** |

**Note on which contrast is controlled.** No arm runs a longer burst at the 5-rollout
baseline data level — the (5 rollouts, 3000 TD) cell is empty, and both 3000-step arms also
raised `num_rollouts`. The clean burst-*length* comparison is therefore `d10` → `d10_b3000`,
which a full 155-key config diff shows differ in exactly one substantive field
(`rl.post_collection_critic_steps` 1000 → 3000). `mcburst` is **not** a longer-burst arm: it
runs the same 1000 steps with `burst_use_mc_targets=True`, so it tests the burst *target*,
not its length.

Doubling rollouts per task moves the post-PG mean by 0.7 points against a per-arm sd of
~11. Buffer growth confirms the data knob took effect and scales exactly linearly with
`num_rollouts`: per-collection transitions are 6.9–7.6k for the 5-rollout arms,
14.4–15.0k for the 10-rollout arms, and 28.3–28.8k for `d20`.

Burst length is a null under direct control. Across `d10` → `d10_b3000` the burst
correlation moves 0.388 → 0.416 and the post-PG mean 29.3 → 26.3; the eight paired per-step
deltas are +2.3, −19.5, +3.9, −7.8, −1.6, +28.1, −28.9, −0.8, giving a paired
t of **−0.50**. Tripling the burst changes neither the critic nor the policy.

Burst *target* is the more informative axis, and it fails differently. At identical
1000-step budgets the TD arms sit at mean burst correlation ≈0.44 and `mcburst` at ≈0.87 —
and their post-PG means are 28.6 versus 28.6. The one burst configuration that provably
changes the critic changes nothing downstream, while tripling the step count changes
neither.

`d20` died of OOM at ~40k with `--mem=150G`; `d10` completed the same workload with 300G.
MaxRSS from `sacct` is unreliable here (13.9G reported for the 300G run, 17G for the OOM'd
run), so memory should be requested generously rather than tuned from reported usage. The
relaunched `d10_b3000` used 300G and completed 100k in 15.9 h.

**Caveat to state plainly:** these are n=8-eval nulls. They rule out ~10-point effects, not
2-point ones.

---

## 7. Result: advantage-scale drift is fixed

> **Read §10d alongside this section.** The measurement below is correct; the reading of it
> was backwards. The normaliser did not stabilise a useful advantage — it decoupled the
> gradient's magnitude from the critic's confidence, so a critic that has learned nothing
> and a critic that has learned everything both emit a unit-scale advantage.

The single-task study's headline pathology was advantage std growing ~13× over a run — a
silent 13× effective-LR increase. The EMA-quantile normalizer eliminates it.

`actor/adv_scale` across the run, all arms:

| arm | 0k | ~17k | ~33k | ~50k | ~66k | ~83k | ~99k |
|---|---|---|---|---|---|---|---|
| `v0` | 1.00 | 2.94 | 2.99 | 2.67 | 2.67 | 2.64 | 2.66 |
| `ncb` | 1.00 | 2.96 | 2.92 | 2.77 | 2.78 | 2.71 | 2.69 |
| `d10` | 1.00 | 3.03 | 3.05 | 2.84 | 2.81 | 2.73 | 2.77 |
| `d10_b3000` | 1.00 | 2.87 | 2.87 | 2.84 | 2.77 | 2.75 | 2.76 |
| `mcburst` | 1.00 | 3.02 | 3.02 | 2.94 | 2.74 | 2.66 | 2.76 |
| `mcblend` | 1.00 | 3.09 | 2.91 | 2.86 | 2.71 | 2.70 | 2.64 |
| `mcblend95` | 1.00 | 2.78 | 2.76 | 2.82 | 2.56 | 2.52 | 2.39 |
| `mcsched` | 1.00 | 2.71 | 3.01 | 2.90 | 2.82 | 2.81 | 2.72 |

After an initial rise to ~3.0 the scale is flat to slightly declining for the remaining 80k
steps — a total drift under 20%, against 13× before. **Report this as a solved problem.**

It also means scale drift can be eliminated from the list of suspects for the oscillation.

**Side finding — the symmetric clip is inert.** `adv_clip_sym` is set to 4.0, but
`advantage_std_final` settles at ~0.35, putting the clip roughly 11 σ into the tail.
Measured post-PG, `actor/adv_clip_sym_frac` has mean 0.0000 and a maximum of 0.0026 across
every arm that enables it — it fires on at most 0.26% of samples and typically never. The
`CLIP_SYM` intervention is not being tested by these runs at all; at its current threshold
it cannot do anything. If it is worth testing, the threshold has to be set relative to the
*post-normalization* advantage scale (~0.35), not in raw units.

---

## 8. Result: collapse happens on frozen data with normal guardrails

`mcblend`, the 90k → 100k block. Eval 51.6% → 5.5%, per-task 0.0 / 6.2 / 0.0 / 15.6 — all
four tasks collapsing together. The collection metric agrees independently: 50.0% → 0.0%.

Within-step ordering is collect → eval (`exp.py:101` then `:136`), so the policy took
~1,000 policy updates (10,000 steps at `update_interval` 10) against a buffer frozen at the
90k collection, and the 100k eval reflects that policy before it trained on any new data.

| step | approx_kl | alive_frac | clipfrac_lo | clipfrac_up | pg_loss | grad_norm | adv_std |
|---|---|---|---|---|---|---|---|
| 90000 | 0.0041 | 0.841 | 0.159 | 0.000 | −0.0009 | 0.24 | 0.373 |
| 92000 | 0.0052 | 0.736 | 0.264 | 0.000 | −0.0015 | 0.39 | 0.358 |
| 94000 | 0.0034 | 0.854 | 0.146 | 0.000 | −0.0016 | 0.48 | 0.371 |
| 96000 | 0.0053 | 0.768 | 0.232 | 0.000 | −0.0023 | 0.31 | 0.366 |
| 98000 | 0.0127 | 0.790 | 0.210 | 0.000 | −0.0033 | 0.27 | 0.379 |
| 100000 | 0.0068 | 0.773 | 0.227 | 0.000 | −0.0043 | 0.23 | 0.378 |

Every quantity is in its normal operating range for the entire collapse. KL never exceeds
0.013, alive fraction stays near 0.8, the upper clip never fires, gradient norms are flat,
advantage std is pinned at ~0.37 by the normalizer. **The policy destroyed itself while
every trust-region and scale diagnostic reported healthy.** This reproduces the single-task
study's central finding — the failure is directional, and none of the instrumented
quantities is sensitive to direction.

`pg_loss` does drift monotonically negative through this block (−0.0009 → −0.0043), which
looks like a candidate early-warning signal. It was tested and it fails: across all 55
post-PG blocks in the campaign, the correlation between a block's mean `pg_loss` and that
block's eval delta is **+0.021**. It is not predictive.

---

## 9. Result: the critic diffuses rather than converges

| arm | MC resid RMS @5k | @end | q_param_norm @5k | @end | v_param_norm @end | corr @5k | @end |
|---|---|---|---|---|---|---|---|
| `v0` | 49.3 | 53.1 | 150.2 | 445.9 | 552.0 | 0.314 | 0.352 |
| `ncb` | 57.6 | 50.8 | 147.3 | 401.4 | 488.3 | 0.270 | 0.444 |
| `b` | 56.3 | 53.5 | 148.1 | 343.2 | 388.6 | 0.327 | 0.368 |
| `d10` | 53.2 | 49.6 | 148.4 | 449.1 | 472.1 | 0.409 | 0.378 |
| `d10_b3000` | 57.1 | 49.9 | 148.3 | 419.5 | 568.0 | 0.471 | 0.360 |
| `mcburst` | 45.0 | 51.4 | 149.0 | 416.4 | 375.9 | 0.239 | 0.425 |
| `mcblend` | 44.7 | 44.0 | 151.2 | 424.2 | 583.4 | 0.460 | 0.527 |
| `mcblend95` | 40.0 | **24.4** | 152.9 | 404.5 | 349.9 | 0.686 | **0.938** |
| `mcsched` | 35.7 | 46.3 | 150.5 | 478.3 | 449.6 | 0.817 | 0.631 |

For every TD arm the MC residual is flat across 95k steps (≈50 → ≈50) while parameter norms
grow **2.3–3.2×**. The critic is not converging; it is wandering at constant error with
steadily growing weights. Only `mcblend95`, which optimizes MC directly, actually reduces
the residual.

The scale of that error can be pinned from measured quantities alone. The MC-fitted arms
give the mean return directly — `burst/q_value_mean` across 40 MC bursts averages
**≈ −178** (range −169 to −185). The TD arms sit at a near-constant −200.5 to −201.9. For a
predictor that is effectively the constant −200, the MC residual decomposes as
`RMS² ≈ σ_MC² + (μ_MC + 200)²`, so σ_MC ≈ √(50² − 22²) ≈ **45**.

A TD critic whose residual RMS (~50) exceeds the return's own standard deviation (~45)
explains approximately none of the return variance — consistent with its measured 0.35–0.45
correlation, and with predicting the unconditional mean being nearly as good.

---

## 10. Root cause: the policy gradient is net-negative, and the reason is measurable

Sections 4–9 searched for the knob that would stabilise OGPO. This section says the search
was mis-aimed. OGPO's policy-gradient term is not unstable-but-useful; on this task set it
is **negative on every single arm**, and the critic signal it consumes is ~0.3 units wide
on a value scale 200 units wide.

### 10a. Every one of ten arms is worse after PG turns on

`pg_start_step` = 20,000 and `pg_ramp_steps` = 5,000 on all ten runs. The PG term is muted
by `advantage * 0.0` (`ogpo_learner.py:531`) until step 20k and reaches full strength at
25k, so **the 10k and 20k evals are BC-only** and 30k is the first eval at full PG.

| arm | 10k | 20k | pre-PG mean | post-PG mean (≥30k) | Δ |
|---|---|---|---|---|---|
| `v0` | 31.2 | 34.4 | 32.8 | 23.8 | **−9.0** |
| `ncb` | 28.9 | 39.8 | 34.4 | 28.6 | **−5.8** |
| `b` | 33.6 | 37.5 | 35.5 | 26.9 | **−8.7** |
| `d10` | 40.6 | 31.2 | 35.9 | 29.3 | **−6.6** |
| `d20`\* | 35.2 | 39.8 | 37.5 | 21.9 | **−15.6** |
| `d10_b3000` | 32.8 | 39.1 | 35.9 | 26.3 | **−9.7** |
| `mcburst` | 39.8 | 43.8 | 41.8 | 28.6 | **−13.2** |
| `mcblend` | 39.1 | 39.1 | 39.1 | 30.1 | **−9.0** |
| `mcblend95` | 40.6 | 37.5 | 39.1 | 24.6 | **−14.5** |
| `mcsched` | 30.5 | 37.5 | 34.0 | 20.3 | **−13.7** |

\* `d20` OOM'd at 40k, so its "post-PG mean" is the single 30k eval.

**mean Δ = −10.6, sd 3.4, n = 10, paired t = −9.8.** Dropping `d20`: −10.0 ± 3.1, t = −9.7.
Ten of ten negative. No arm, and no knob setting tested, produces a policy gradient that
pays for itself.

The step-averaged series makes the shape plain (mean over all arms with an eval at that
step):

| step | 10k | 20k | 30k | 40k | 50k | 60k | 70k | 80k | 90k | 100k |
|---|---|---|---|---|---|---|---|---|---|---|
| mean eval | 35.2 | **38.0** | **18.0** | 26.2 | 33.6 | 23.0 | 39.2 | 19.4 | 31.2 | 21.4 |

38.0 → 18.0 is a **20-point crash across the ramp**, and no subsequent eval in any arm ever
returns to the BC-only level. It is not one task failing: across the nine arms with
per-task logging, all four drop together at 30k — `libero_90_79` 40.6 → 10.4,
`libero_90_31` 34.4 → 24.3, `libero_90_82` 37.2 → 19.8, `libero_90_38` 38.9 → 15.6.

The oscillation documented in §4 is what happens *after* this: the PG walks the policy
off, BC partially repairs it between collections, and the two never settle.

### 10b. The critic is a constant, and TD has no reason to move it

Per-chunk reward is Σ_{i<10} γ^i·(−1) = **−9.78** and the chunk discount is
γ^10 = 0.9511 (`filtered_sft_learner.py:725-746`, `TimeToSuccessAsRewardWrapper`
at `wrappers.py:230-237` gives −1 per step, 0 on the terminating step). The Bellman fixed
point for a policy that never succeeds is therefore

    Q* = −9.78 / (1 − 0.9511) = −200.0   exactly.

Measured `critic/q_value_mean`, post-PG mean: `ncb` −198.6, `d10` −198.6, `v0` −199.4,
`b` −200.6, `mcburst` −201.7, `d10_b3000` −197.6. Actor-side `q_mean` and `v_mean` agree to
within 0.2 of each other in every window.

The loss pair is the tell:

| arm | `q_td_loss` (RMS) | `q_mc_loss` (RMS) | `q_mc_corr` | `q_value_mean` |
|---|---|---|---|---|
| `ncb` | 20.2 (4.5) | 2571 (**50.7**) | 0.430 | −198.6 |
| `mcburst` | 19.4 (4.4) | 2370 (**48.7**) | 0.444 | −201.7 |
| `mcblend95` | 23.6 (4.9) | 910 (30.2) | 0.899 | −189.5 |

**The critic satisfies its own Bellman equation to ±4.5 while being wrong about realised
returns by ±51.** That is not a partially-trained critic — it is the exact error profile of
a network that emits the constant −199 for every (s, a). TD is self-consistent at that
constant, so gradient descent has no reason to leave it, which is why 1,000 and 3,000-step
bursts (§5a) and 2×/4× data (§6) changed nothing: none of them changes the objective's
fixed point. `mcblend95` is the control — 5% MC weight in the target moves
`q_value_mean` to −189.5 and `q_mc_corr` to 0.899, confirming the constant is a property
of the TD objective, not of the network's capacity.

### 10c. The within-state signal is ~0.3 wide on a scale of 200

Arm `b` is the only run with **all** advantage normalisation disabled, so its
`actor/advantage_std` is the raw group-centred advantage the PG would see:

| window | 0–10k | 20k | 30k | 40k | 50k | 70k | 90k |
|---|---|---|---|---|---|---|---|
| raw advantage std | 0.258 | 0.333 | 0.473 | 0.532 | 0.638 | 0.627 | 0.662 |
| q95 − q05 | 0.417 | 0.529 | 0.740 | 0.629 | 0.818 | 0.631 | 0.711 |
| `q_mean` | −200.0 | −200.7 | −201.1 | −200.5 | −200.9 | −201.6 | −200.8 |

Across the **eight action chunks sampled at one state**, Q varies by ~0.3–0.7 while sitting
at −200. The success-vs-failure difference on the same scale is ~45 (a success at the
logged mean length of ~295 steps returns −(1−γ²⁹⁵)/(1−γ) = −154.5; a failure is −200), and
the full value range is 200.

**The quantity the policy gradient consumes is 0.2–1.5% of the success/failure signal.**
Nothing in this campaign separates how much of that 0.3 is signal and how much is critic
noise — see §10h.

### 10d. The normalisers deliver that signal at full strength regardless of its quality

Two normalisers sit between the critic and the surrogate:

1. `normalize_advantage_per_task` (`update_actor.py:302-322`) divides by the per-task std.
   Measured `actor/advantage_std` = **1.0000** in every normalised arm, by construction.
2. `normalize_group_advantage` (`ogpo_learner.py:508-525`) divides by `adv_scale`, an EMA
   of q95 − q05. Measured `adv_scale` ≈ 2.79 and `advantage_std_final` ≈ **0.359**, flat
   across the whole run in `ncb` (0.3591), `d10` (0.3516), `mcblend` (0.3620).

So whether the critic's raw spread is 0.26 (early) or 0.66 (late), whether it has learned
anything or nothing, the number handed to the PPO surrogate has std 0.36. **§7's
measurement is correct and its interpretation was backwards: the normaliser did not
stabilise the advantage, it decoupled the gradient's magnitude from the critic's
confidence.** A dead critic and a perfect critic produce the same-sized policy gradient.

The `adv_clip_sym` = 4.0 guard never engages: `actor/adv_clip_sym_frac` = 3.3e-06 to
4.9e-06 post-PG, i.e. roughly one sample in 250,000.

### 10e. The BC anchor contributes ~1% of the update direction

During the warmstart `grads_pg` is an exact zero tree, so `actor/grad_norm` over 0–20k
**is** the BC gradient norm. Post-PG it is the norm of the sum.

| arm | \|BC\| (0–20k) | \|total\| (≥30k) | \|PG\| = √(tot²−bc²) | PG/BC |
|---|---|---|---|---|
| `v0` | 0.0488 | 0.4062 | 0.4033 | 8.3 |
| `ncb` | 0.0492 | 0.3634 | 0.3600 | 7.3 |
| `b` | 0.0484 | 0.5755 | 0.5734 | **11.9** |
| `d10` | 0.0486 | 0.3882 | 0.3852 | 7.9 |
| `mcburst` | 0.0467 | 0.4059 | 0.4032 | 8.6 |
| `mcblend` | 0.0502 | 0.3802 | 0.3769 | 7.5 |
| `mcblend95` | 0.0502 | 0.3353 | 0.3316 | 6.6 |
| `mcsched` | 0.0493 | 0.3665 | 0.3632 | 7.4 |
| `d10_b3000` | 0.0521 | 0.3688 | 0.3651 | 7.0 |

`bc_coeff` = 1.0 buys an anchor that is **1/7th to 1/12th of the gradient by norm**, and
cos(update, PG) = |PG|/|total| = **0.99**. There is effectively no BC anchor in the update
direction. (Arm `b`, which turns off both normalisers, has the largest PG — its unnormalised
advantage std averages 0.58 vs the normalised arms' 0.36 — and the worst final eval, 3.9%.)

Two further points about what the anchor is anchored *to*: `rl.online_ratio` = 1.0, so the
BC batch is drawn entirely from the **self-generated success buffer**, not the SFT dataset.
It anchors the policy to its own past successes, which is a weaker prior than the
checkpoint it started from (mitigated by the buffer retaining old data: 6.1k → 22.4k
entries over the run, never evicted).

### 10f. And none of it is visible in the diagnostics

Post-PG, every guardrail reads healthy in every arm: `clipfrac` 0.17–0.25,
`alive_fraction` 0.75–0.84, `approx_kl` 0.004–0.007, `grad_norm` flat to 3 significant
figures, `advantage_std` exactly 1.000, `adv_clip_sym_frac` ≈ 0. This is §8's finding
generalised: the guardrails cannot fail, because the normalisers make the input to the PPO
surrogate well-conditioned no matter what the critic says.

One structural point on the clip. `actor/log_ratio_max` is **negative in every 10k window
of every arm** (mean −0.020 to −0.030; the maximum over all ~40,000 logged steps in all ten
arms is +0.00004, against an upper clip bound of +0.0953). The ratio is below 1 for every
sample in every batch, so:

- `clipfrac_upper` is identically 0 and `clipfrac` ≡ `clipfrac_lower` — verified, exactly.
- With ratio ≤ 1, `min(ratio·A, clip(ratio)·A)` leaves every **A > 0** sample unclipped and
  clips only **A < 0** samples. PPO's trust region here only ever deletes "push away from
  the bad action" gradients; there is no upper guard on "move toward the action the critic
  likes", which is the direction that does the damage.

This is a consequence of the per-dimension log-prob normalisation (`update_actor.py:241-245`
divides by K·H·D = 10·10·32 = 3,200) combined with re-sampling: the per-sample log-ratio is
N(−½‖Δ‖², ‖Δ‖²), so its distance from zero in its own sigmas is ‖Δ‖/2 — the estimator
self-tightens as the policy moves, and after the /3,200 the mean offset (∝ c) dominates the
spread (∝ √c/56.6) for any per-dim divergence c ≳ 1.3e-3.

There is a weak dose-response. Within-arm z-scored, over the 10k window preceding each eval
(n = 71 window/eval pairs, 9 arms, ≥30k):

| window metric | r vs the eval at its end | t |
|---|---|---|
| `alive_fraction` | **+0.316** | +2.76 |
| `clipfrac` | **−0.316** | −2.76 |
| `advantage_std_final` | +0.295 | +2.41 |
| `approx_kl` | −0.163 | −1.37 |
| `grad_norm` | −0.097 | −0.81 |

Windows where the policy stays inside the trust region end better. The effect is real but
small; the dominant contrast by an order of magnitude is PG-off vs PG-on (t = −9.8).

### 10g. Why every knob tested had to be a null

The chain, end to end:

1. Reward −1/step with γ = 0.995 gives a Bellman fixed point of exactly −200 for failure,
   and TD is self-consistent there → the critic outputs a constant (§10b).
2. Group centring, `Q(s,a_g) − mean_g Q(s,a_g)`, then extracts the *within-state* variation
   of that constant: 0.3–0.7 units out of 200 (§10c).
3. Both normalisers rescale whatever comes out to std 0.36, so the gradient magnitude
   carries no information about whether step 2 produced signal or noise (§10d).
4. That gradient is 7–12× the BC anchor and 99% of the update direction (§10e).
5. It is applied **1,000 times between collections** (`policy.update_interval` = 10 ×
   `collect_interval` = 10,000) with no reality check.
6. Every diagnostic is well-conditioned by construction, so nothing fires (§10f).

Bursts, MC targets and more rollouts all act on step 1 or 2's *between-state* accuracy,
which group centring deletes identically (§5b). More data, longer digestion and better
return-correlation cannot reach the within-state term. **They were not underpowered tests;
they were tests of a quantity the algorithm discards.**

### 10h. What this does *not* establish

Stated plainly, because the recommendations in §11 depend on which way these go:

- **Whether the 0.3-wide within-state spread is signal or noise is still unmeasured.** It is
  the same gap §11 flagged; §10c gives its magnitude, not its quality. If it is 90% signal,
  a smaller/slower PG could work. If it is noise, the value-based path is closed.
- **Whether a better critic is achievable at all.** A 10-step chunk out of a ~295-step
  episode, discounted at γ = 0.995 (≈200-step horizon), may genuinely change the return by
  only ~0.3. If so the critic is *correct* and the problem is the estimator, not the
  training.
- **Whether BC-only keeps improving past 20k.** Two points, 35.2 → 38.0. Suggestive, not
  established; no run has ever trained BC-only past 20k.
- The `d20` Δ rests on a single post-PG eval.
- All ten runs are seed 0 (§0b), so these are paired contrasts within a shared eval stream,
  not independent replicates. The PG-off/PG-on contrast is *within* each run, which is the
  strongest design available here, but it is confounded with training step: something else
  that happens around 25k would produce the same signature. No knob in the config does.

### 10i. One secondary observation, flagged not quantified

`collect.replan_steps` = 5 but `model.action_horizon` = 10, and `collect.py:161-163` stores
only `env_action_chunk[:, :replan_steps]` — the second half of every policy query is
discarded, never executed. `_save_episode_in_buffer` then rebuilds 10-step windows by
sliding over the *executed* stream (`filtered_sft_learner.py:733`), so the critic is trained
on chunks that are the concatenation of two consecutive queries' first halves, while the PG
scores single-query 10-step samples. The two distributions are close but not identical. The
magnitude of this mismatch has not been measured and it is not offered as an explanation of
anything above.

---

## 11. Fastest paths to something that improves

Ordered by information per GPU-hour. Nothing here has been run.

### Free — no GPU, available now

**0. The BC-only baseline already exists: 20 evals across 10 arms at 35–38%.**
Post-PG pooled mean is 26.5%. That comparison is the reportable result and needs no new
compute. It also means the honest headline is not "OGPO is unstable in multi-task" but
"OGPO's policy gradient is net −10.6 points against its own warmstart, on ten of ten arms".

### ~1 GPU-hour, and it decides the whole direction

**1. Best-of-N eval with a trained OGPO critic.** Everything above hinges on one
unmeasured quantity: does Q rank actions *within* a state? `src/rl/best_of_n/` already
implements critic-only ranking with no policy training. Load a 90k OGPO checkpoint, sample
N = 8 chunks at each eval step, execute `argmax_a Q(s,a)`, compare against the same
checkpoint's ordinary eval.

- **BoN ≫ base** → the critic ranks and the PG is misusing a usable signal. Fix the PG
  (items 3–5) and keep the critic path.
- **BoN ≈ base** → the critic does not rank. No burst length, no MC target, no data volume
  will ever help, and the value-based path is closed. Stop tuning the critic.

It is eval-only, so it is the cheapest decisive experiment available. Note that it uses the
same G = 8 group and the same Q whose spread is 0.3, so a null is the *expected* outcome —
and a null is exactly the result worth having, because it is the one that redirects the
project.

### One-line instrumentation — land before the next run, whatever it is

**2. Log the within/between variance split.** At `update_actor.py:296-297`, before the
per-task norm at :302, emit `std(advantage_raw.reshape(B,G) − baseline)` (within-state) and
`std(baseline)` (between-state). Their ratio is the SNR the PG actually consumes. Today it
is visible only in arm `b`, and only because that arm turned the normalisers off by
accident of a different experiment.

**3. Finish `grad_norm_pg` / `grad_norm_bc` / `grad_cos_pg_bc`** — already written in the
working tree. It converts §10e's √(tot²−bc²) estimate into a direct read, and the cosine
answers whether BC is fighting the PG or merely being outvoted.

### Single runs, ~15 h each, in priority order

**4. `--rl.bc_coeff 8`** (needs a flag appended; the launcher has no `BC_COEFF` env var).
The measurement says the anchor is 1/8 of the gradient. Setting `bc_coeff` to the measured
ratio is the cheapest one-flag test of "is it the PG?". If the run recovers the ~36%
BC-only level *and* keeps a live PG, that is a working stack. If it merely reproduces
BC-only, the PG contributes nothing and item 0 becomes the final answer.

**5. Sparse-positive reward: `--collect.use_time_to_success_as_reward false`.**
This removes the −200 attractor at the root. `get_value_bounds`
(`value_distribution.py:129-134`) auto-switches the range to [0, 1] and the reward becomes
LIBERO's native +1-on-success. The critic then has to resolve 1–2 significant figures
instead of 3 (a discrimination of 0.01 on a range of 1, rather than 0.3 on a range of 200).
Caveat: this changes the MDP, so it is not a clean ablation of anything above — but the
current MDP demonstrably does not train.

**6. Shorten the blind window: `COLLECT_INT=2500`.** 250 PG updates per collection instead
of 1,000, directly attacking §10g step 5. Note `--collect.eval_interval 10000` is hardcoded
at `ogpo_multitask_4task.sh:185`, so eval cadence is unaffected; budget ~20 h for the extra
collection passes.

**Do not spend more compute on** burst length, MC targets, or rollout count. Three arms
already tested each, all are nulls, and §10g explains why they must be.

### If item 1 comes back null — the structural options

At that point the critic cannot rank actions and no amount of critic training fixes it,
because the underlying ΔQ across chunks really is ~0.3. The remaining moves all change the
problem rather than the tuning:

- **Denser reward** (subtask/progress), so a 10-step chunk's effect is locally observable
  instead of buried 295 steps downstream.
- **Return-based group advantage** — roll out G times from the *same* initial state and use
  `A_i = R_i − mean_g R_g`, no critic at all. The variance of R is the real success/failure
  variance rather than 0.3. **Note `src/rl/flow_grpo/` does not do this** — it also computes
  `advantage = q_value - value` (`flow_grpo/update_actor.py:110`) and inherits the same dead
  critic. A genuine return-based GRPO would be new work, and G× the collection cost.
- **Larger `action_horizon`**, so one decision carries more of the outcome.
- **Or accept it.** Filtered SFT on successes is the only thing in this campaign that
  demonstrably improves, it is at 38% and still rising at 20k, and it has never been run to
  100k. `PG_START=200000 NUM_STEPS=100001` on the existing launcher gives that baseline in
  one run.

---

## 12. What is ruled out, and what is left

**Ruled out as the cause of instability:**

- Advantage scale drift — measured stable (§7), though §10d shows stability here is
  guaranteed by construction and says nothing about the signal underneath it.
- Critic *between-state* ranking quality — driven to 0.99 with no benefit (§5b).
- Post-collection critic digestion, TD variant — no effect on the critic at 1000 or 3000
  steps (§5a). §10b explains why: the burst optimises an objective already at its fixed
  point.
- Data volume — 2× and 4× rollouts are nulls (§6).
- Trust-region violation — KL, clipfrac and alive fraction are normal throughout every
  collapse (§8) — necessarily, per §10f.
- `pg_loss` drift as an early-warning signal — correlation +0.021 over 55 blocks (§8).

**Established as the cause (§10):**

- The policy-gradient term itself is net-negative, −10.6 ± 3.4 points paired against each
  arm's own BC warmstart, t = −9.8, ten of ten arms (§10a).
- The critic sits on the −200 failure fixed point and its within-state spread is ~0.3 on a
  200-unit scale (§10b, §10c).
- Both normalisers rescale that to a fixed 0.36 regardless of quality, and the resulting
  gradient is 7–12× the BC anchor (§10d, §10e).

**Newly ruled out:**

- Value-bin quantisation. `num_value_bins` = 1 in every arm, i.e. a Gaussian/MSE critic —
  there is no discretisation destroying the 0.3-wide signal.
- The `adv_clip_sym` = 4.0 guard as a cause of anything: `adv_clip_sym_frac` ≈ 3e-06.
- One task dragging the aggregate: all four tasks crash together across the PG ramp (§10a).

**Never actually tested, despite being nominally enabled:**

- **The symmetric post-norm clip (`CLIP_SYM`).** Threshold 4.0 against a post-normalization
  advantage std of ~0.35; it fires on ≤0.26% of samples and usually 0.00%. Nine arms carry
  this flag and none of them exercised it (§7). Any claim that "symmetric clipping doesn't
  help" is unsupported by this campaign.

**Not yet tested:**

- **Within-state action ranking.** Nothing in the current instrumentation measures whether
  the critic orders *the 8 sampled chunks at a fixed state* correctly — which is the only
  component of Q that survives group centering and reaches the policy gradient. Three
  candidate metrics, in increasing cost and conclusiveness:
  1. *Variance split* (free, at `update_actor.py:296-297`, where `advantage_raw`, `B` and
     `G` are all in scope): `within = mean_b[std_g(adv_raw)]` versus
     `between = std_b[mean_g(adv_raw)]`. Must be logged there, before the per-task
     normalization at `update_actor.py:302` — the existing `advantage_std` is emitted after
     it at `update_actor.py:350`, which is why it reads exactly 1.000 in every run and
     carries no information.
  2. *Two-head rank agreement* (free): `num_qs` = 2 and `critic_values_per_head` is already
     in scope; per-state Pearson correlation and argmax agreement across the G=8 chunks,
     against a 1/8 = 0.125 chance baseline. Falsifies but cannot confirm.
  3. *Best-of-N eval* (costs eval time; the only ground truth) — **this is now §11's
     top-priority experiment**: choose actions by argmax-Q
     over N sampled chunks and compare against the N=1 eval. `BestofNLearner.sample_actions`
     (`best_of_n_learner.py:326`, scoring at `:470-500`) already implements exactly this scoring path, and the
     ordering is faithful — argmax over g of `Q − mean_g Q` equals argmax over g of Q, with
     `reduction="min"` in both.
- **Reduced off-policy update count.** Every arm runs 1,000 policy updates per collection.
  Shortening the block (`COLLECT_INT`) or raising `policy.update_interval` is untested and
  is the most direct attack on a directional failure that develops *within* a block.
- **Independent seeds.** All ten runs use seed 0. Given §4, seed replication is the
  cheapest way to find out how much of the remaining 54% of variance is arm effect at all.

---

## 13. Recommended framing for the advisor

1. **The multi-task setting is not learning, and we now know why.** OGPO's policy-gradient
   term is net −10.6 points against its own BC warmstart, on ten of ten arms
   (paired t = −9.8). Everything the campaign was tuning sits downstream of that.
2. **The critic has collapsed onto the Bellman fixed point of failure.** With −1/step
   reward and γ = 0.995 that point is exactly −200; the measured critic sits at −199 and
   satisfies its own TD equation to ±4.5 while mispredicting realised returns by ±51. TD
   has no gradient pushing it off that constant, which is why 3× longer digestion bursts
   and 4× more data changed nothing.
3. **The advantage the policy consumes is 0.3 units wide on a 200-unit scale**, and two
   layers of normalisation rescale it to a fixed std of 0.36 before it reaches the
   surrogate. The gradient's magnitude is therefore independent of whether the critic has
   learned anything. This is the mechanism behind the campaign's central puzzle — every
   guardrail (KL, clipfrac, alive fraction, advantage std) reads healthy through every
   collapse, because the normalisers guarantee it.
4. **The MC-target result is the strongest single finding**, precisely because it is a null
   with a mechanism: the manipulation provably worked (return-correlation 0.37 → 0.98), the
   outcome provably did not move (paired across steps, TD − MC = +1.9 ± 9.8, t = 0.54), and
   group centring explains why it could not — it deletes exactly the between-state component
   that better return-correlation improves.
5. **The one remaining unmeasured quantity is within-state action ranking**, and it is the
   one the algorithm actually consumes. A Best-of-N eval with a trained critic answers it in
   ~1 GPU-hour with no training, and its answer determines whether the value-based path stays
   open at all (§11, item 1).
6. **Caveats worth stating up front.** All ten runs are seed 0 against a deterministic,
   shared eval stream, so these are paired within-run contrasts, not independent replicates.
   The PG-off/PG-on contrast is within each run — the strongest design available here — but
   it is confounded with training step. And it is not established that a better critic is
   achievable: a 10-step chunk out of a ~295-step episode discounted at γ = 0.995 may
   genuinely change the return by only ~0.3, in which case the critic is correct and the
   estimator is the problem.

---

## 14. Run ledger

| run id | exp_name | last step | wall clock | outcome |
|---|---|---|---|---|
| `xqr360bw` | `mt4_v0_s0` | 99,975 | 15.5 h | complete |
| `nmcfk1oo` | `mt4_ncb_s0` | 100,000 | 14.8 h | complete |
| `0t78eqtk` | `mt4_b_s0` | 100,000 | 14.9 h | complete |
| `fiscke6b` | `mt4_ncb_d20_s0` | 39,950 | 6.4 h | **OOM** (150G) |
| `guqfpcfg` | `mt4_ncb_mcburst_s0` | 100,000 | 15.3 h | complete |
| `7ns7ffqq` | `mt4_ncb_d10_s0` | 100,000 | 16.0 h | complete |
| `zk1xdua1` | `mt4_ncb_mcblend_s0` | 100,000 | 15.0 h | complete |
| `37ljoihf` | `mt4_ncb_mcblend95_s0` | 100,000 | 15.3 h | complete |
| `rtado5pq` | `mt4_ncb_mcsched_s0` | 100,000 | 15.3 h | complete |
| `b9ogu237` | `mt4_ncb_d10_b3000_s0` | 100,000 | 15.9 h | complete |
| `d4dbowe9` | `mt4_ref_s0` | in progress | — | **running** (ref-aligned, CONS=1) |
| `e5wrybr0` | `mt4_ref_nc_s0` | in progress | — | **running** (ref-aligned, CONS=0) |
| `j4uf06gd` | `mt4_ref_nopg_s0` | in progress | — | **running** (ref-aligned, PG muted) |

Two further offline directories (`e08gg6vz`, `wvlcc94p`) are aborted launches with 0 and 14
logged rows and are excluded from all analysis.

---

## 15. Reference-aligned OGPO — interim results (2026-08-21)

**STATUS: RUNS IN PROGRESS.** Three arms, none complete. All numbers below are a
snapshot and will move. Written now because two results are already large enough to
act on.

| job | run id | arm | `advantage_combination` | PG | step at snapshot |
|---|---|---|---|---|---|
| 10178692 | `d4dbowe9` | `mt4_ref_s0` | `grpo_conservative` | on, from 20k | ~60k / 100k |
| 10178693 | `e5wrybr0` | `mt4_ref_nc_s0` | `reduced` | on, from 20k | ~52k / 100k |
| 10184622 | `j4uf06gd` | `mt4_ref_nopg_s0` | (inert) | **muted all run** | ~10k / 100k |

Shared config (`pi05_libero_online_ogpo_ref`): `success_reward_bonus` **90**,
`critic_success_oversample`, `num_qs = num_vs = 10`, `critic.reduction = mean`,
`td_weight` 0.95, `n_samples` **8** (best-of-N), all three local normalizers **off**
(`NORM=0`, `MT_ADV=0`, no `adv_clip_sym`), `clip_epsilon` 0.1, `discount` 0.995, G=8,
burst 1000, warmstart 20000/5000, seed 0, same 4 LIBERO tasks.

10178692 and 10178693 differ in **exactly one field**; 10184622 differs from 10178692 in
**exactly two tokens** (`exp_name`, `pg_start_step` 20000 → 200000), verified by diffing the
emitted command lines. 10184622's PG mute is `advantage * 0.0` at `ogpo_learner.py:581`,
which still runs jit-2a, so its RNG stream matches 10178692.

### 15a. The critic collapse (§10b) is fixed

This is the strongest and least confounded result. §10b established the critic was emitting
essentially one constant, pinned on the −200 Bellman fixed point that `fix_mc_returns`
writes into every failed episode.

| metric | `ncb` @100k | `ncb_d10` @100k | `d10_b3000` @100k | `mcblend` @100k | **ref_cons** ~57k | **ref_nocons** ~47k |
|---|---|---|---|---|---|---|
| `critic/q_mc_corr` | 0.485 | 0.370 | 0.360 | 0.548 | **0.950** | **0.959** |
| `critic/q_value_mean` | −199.3 | −199.7 | −197.4 | −195.7 | **−145.4** | **−159.8** |
| `critic/q_td_loss` | 19.4 | 21.4 | 20.7 | 19.9 | 11.5 | 8.4 |
| `critic/q_mc_loss` | 2641 | 2509 | 2494 | 2145 | **705** | **478** |
| `critic_sb/q_value_mean` | — | — | — | — | **−78.0** | **−75.0** |

Three things changed at once and they cannot yet be separated: the **+90 terminal reward
bonus**, **success oversampling** (one extra critic step per trainer step on a success-only
batch), and the **10-head `mean` ensemble**. Together they take `q_mc_corr` from 0.36–0.55 to
**0.95**, cut MC loss **3.5–5.5×**, and move `q_value_mean` **~40–55 points off** the −200
attractor.

The `critic_sb/` series is new and is the clearest single number: the critic now values
success-buffer states at **−75 to −78** against **−145 to −160** for the general replay
buffer — a **~70–90 unit separation** where §10c measured a within-state spread of **~0.3 on
a scale of 200**.

Almost all of this happens during the BC warmstart, before PG exists. In 10178692,
`grad_norm_pg` is exactly 0.000 through step 20k while `q_mc_corr` climbs 0.63 → 0.955 and
the success separation opens 12 → 70 units. **PG contributes nothing to the critic; it only
consumes it.**

### 15b. Confound: eval is Best-of-8 in these arms and was Best-of-1 in the ten

`--rl.n_samples 8` is on. `evaluate_policy` calls `agent.sample_actions`, which is the
overridden Best-of-N path (`advantage_weighted_sft_learner.py:313`), gated at
`critic.inference_start_step = 1`. So **eval itself is Best-of-8 with critic reranking**.
None of the ten earlier arms passed `--rl.n_samples`; the dataclass default is 1
(`config.py:176`).

**Absolute eval numbers in §15c are therefore NOT comparable to §3 or §10a.** Part of the
gap is an 8× more expensive inference procedure. The 10178692-vs-10178693 contrast is
unaffected — both are Best-of-8.

All eval below is also **train-tasks-only** (`eval_tasks = 4`). `HELDOUT=1`
(`ogpo_multitask_4task.sh:74`) appends 25 held-out tasks but is off; at `EVAL_ROLLOUTS=32`
it would cost 7.25× per eval and not fit the 48 h limit.

### 15c. Eval, at matched steps

Baseline row is the step-averaged series from §10a; `n = 128` per point in every run.

| step | ten arms (mean) | 10178692 `cons` | 10178693 `nocons` | new mean |
|---|---|---|---|---|
| 10k (BC only) | 35.2 | 48.4 | 52.3 | **50.4** |
| 20k (BC only) | **38.0** | 43.8 | 51.6 | **47.7** |
| 30k (PG on) | **18.0** | 43.0 | 40.6 | **41.8** |
| 40k | 26.2 | 51.6 | 41.4 | 46.5 |
| 50k | 33.6 | 52.3 | 37.5 | 44.9 |

The 20k → 30k step is the signature of §10a. The ten arms crash **38.0 → 18.0, −20 points**,
unanimously. The two new arms fall **47.7 → 41.8, −5.9**.

Pooled post-PG (≥30k): ten arms **26.5 ± 12.4** (n=72); new arms **44.4 ± 6.1** (n=6).

### 15d. PG's own contribution, each arm against its own BC phase

The §10a construction, applied to the new arms:

| arm | pre-PG (10k,20k) | post-PG (≥30k) | Δ |
|---|---|---|---|
| ten arms, mean | 36.6 | 26.5 | **−10.6** (sd 3.4, n=10, t = −9.8) |
| 10178693 `reduced` | 51.95 | 39.83 | **−12.1** |
| 10178692 `grpo_conservative` | 46.1 | 48.97 | **+2.9** |

**10178692 is the first arm in this campaign whose policy gradient is not negative.** At
n = 128 per eval, a two-point pre-PG mean has SE ≈ 3.1, so **+2.9 is indistinguishable from
zero** — the claim is "no longer harmful", not "helpful".

10178693 reproduces §10a almost exactly (−12.1 against −10.6) despite a critic that is now
demonstrably good. **A fixed critic is not sufficient for the PG term to pay for itself.**

### 15e. The conservative-gate result reverses at 10 heads

`scripts/submit_mt4_ref_arms.sh` was built to test whether the campaign's only clean
single-knob CONS result was a 2-head artifact. It was.

| comparison | heads | paired diff | n | t |
|---|---|---|---|---|
| `v0 − ncb` (campaign) | 2 | **−3.57** (sd 4.48) | 7 | −2.11 |
| `ref − ref_nc` (now) | 10 | **+9.13** (sd 6.27) | 3 | +2.52 |

Per-step: +2.4 (30k), +10.2 (40k), **+14.8 (50k)** — monotonically widening. Three points is
not conclusive (p ≈ 0.13) and the observed sd matches what eval noise alone would produce, so
this needs the remaining 50k steps. But the **sign flip** is the answer the paired design was
built for, and `cons_zero_frac` explains the mechanism: at `num_qs=2` the gate zeroed
8.7–22% of advantages; at 10 it zeroes **85%**.

### 15f. The gate halves the PG magnitude but does NOT arrest scale drift

With all three normalizers off, nothing bounds the advantage scale. Both arms drift, and
**at the same relative rate**:

| step | `advantage_std` cons | nocons | | `grad_norm_pg` cons | nocons |
|---|---|---|---|---|---|
| 20k | 0.621 | 1.400 | | 0.409 | 1.003 |
| 30k | 0.850 | 1.971 | | 0.991 | 2.133 |
| 40k | 1.148 | 2.505 | | 1.133 | 2.318 |
| 45k | 1.027 | 2.297 | | 0.976 | 2.115 |
| 55k | 1.422 | — | | 1.340 | — |

Over the matched 20k → 45k window: `advantage_std` grows **1.65×** (cons) and **1.64×**
(nocons); `grad_norm_pg` grows **2.4×** and **2.1×**. The conservative gate roughly **halves
the absolute PG scale** — it does not stop the drift. §7's scale-drift mechanism is live in
both arms and is the leading risk for the back half of both runs.

`cons_zero_frac` itself is stable: 0.816 (20k) → 0.850 (55k), flat for 35k steps. It rose
0.35 → 0.81 during the warmstart as the critic converged and within-group Q spreads shrank
toward the true advantage — i.e. **85% zeroed is the signature of a converged critic being
honest about eight near-equivalent action chunks**, not of a broken gate.

Three checks say 85% is not over-suppression: it is **75× above the 0.2% chance rate** for 10
independent heads, so the heads are informative; `grad_norm_pg / grad_norm_bc` is **21–27×**,
so PG still dominates the BC anchor; and since 85% are exact zeros,
`std_survive = std_all / √0.15 ≈ 2.6` — the surviving advantages are **larger** than
10178693's ungated 2.30, so the min-of-10 magnitude rule is selecting large-margin samples
rather than shrinking everything.

### 15g. What this does NOT establish

**Attribution is unresolved.** The +12.4-point pre-PG gap (36.6 → 49.0) is earned with PG
muted, so it belongs entirely to non-PG changes: the reward bonus, success oversampling, the
10-head critic, and Best-of-8. Only the extra +5.5 post-PG (gap 12.4 → 17.9) is attributable
to PG hurting less. **No part of the improvement is yet attributable to the policy gradient
doing something useful.**

Best-of-N is a critic-only method in this repo where the policy never trains
(`best_of_n_learner.py:590`, no `update_policy` branch). It is entirely possible that most of
the gain is BoN plus a working critic. **10184622 is the arm that decides this** — PG muted
for all 100k steps, everything else identical. At its 10k snapshot it already shows
`q_mc_corr` 0.956 and a 90-unit success separation, with `grad_norm_pg` exactly 0.000.

If 10184622 lands at or above 10178692's level, the conclusion is that the PG term is not
earning its place and the conservative gate was an expensive way of doing less of it.

Not separable within these arms: the bonus vs. oversampling (both on in all three);
`num_qs=10` vs. `reduction=mean` vs. `td_weight=0.95` (changed together).

### 15h. Follow-ups this suggests

1. **`ref_bononly`** — `policy.training_start_step` past the horizon freezes the actor
   entirely (no BC, no PG), giving the true BoN floor. One `sbatch --wrap`, no code change.
2. **`ref` + `NORM=1`** — the EMA-quantile normalizer against the drift measured in §15f,
   keeping the gate. Directly targets the one mechanism still unmitigated.
3. **`MT_ADV=1`** — per-task advantage scale normalization; the cheap proxy for the
   per-task-critic proposal, testable by env var rather than a Tier 2 change.
4. **`HELDOUT=1 EVAL_ROLLOUTS=8`** on the final checkpoints — 29 tasks × 8 = 232 rollouts,
   ~1.8× current eval cost, gives a generalization number these arms do not have.
