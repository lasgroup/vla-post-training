# Can the critic rank actions? Tier A–C and the V(s′) probe

**Date:** 2026-08-24 · **Checkpoint under test:** `mt4_ref_s0` @ step 100000
(`pi05_libero_online_ogpo_ref`, 4-task LIBERO) · **Controls:** `mt4_ref_nc_s0` (CONS=0),
`mt4_ref_nopg_s0` (no policy gradient), `init_untrained`.

## The question

OGPO's policy samples 8 candidate action chunks at every step, scores them with the
critic, and executes the highest-scoring one (`rl.n_samples = 8`). The same critic
supplies the policy-gradient advantage — with `advantage_combination="grpo_conservative"`
the advantage is per-head `Q_i − mean_G(Q_i)` over a group of `G = 8`
(`src/rl/ogpo/update_actor.py:64,273-286`), the identical construction Best-of-N ranks on.
So one question covers both: **can the critic tell which of several actions at the same
state is better?**

`reports/findings.md` §10 established that the PG term is net −10.6 against its own BC
warmstart, paired, on ten of ten arms (t = −9.8). This report is the attempt to find out why.

## Verdict

**No. Best-of-N buys nothing measurable, and the critic does not even agree with its own
training target about which candidate is best.**

| measurement | result | job |
|---|---|---|
| Δ = G(critic's pick) − G(mean), realised return | **+1.78 ± 2.99** (n=192) | 10199880-83 |
| success rate: critic's pick vs random pick | 0.438 vs 0.431 | 10199880-83 |
| Spearman(Q, realised) per state | −0.0002 (SE 0.033) | 10199880-83 |
| pooled within-state Pearson(Q, realised) | +0.042 (1024 pairs) | 10199880-83 |
| Q agrees with its own TD target on the argmax | **14.6%** (chance 12.5%) | 10200296 |
| pooled within-state Pearson(Q, TD target) | +0.018 | 10200296 |

Three independent statistics against realised outcomes, and two against the critic's own
training target, all sit on zero. The value did not move as n went 108 → 136 → 192.

## Tier A — the critic's action response surface (job 10191707)

**It has learned coarse action-conditioning but not fine.** The swap control pairs state *i*
with action *j* from another state:

| arm | own − foreign action Q | own is argmax of 256 | Q spread over states |
|---|---|---|---|
| `mt4_ref_s0` | **+22.74** (0.34 σ_across) | 8.2% (chance 0.4%) | 55.6 |
| `mt4_ref_nopg_s0` | +0.52 (0.007 σ) | 1.2% | 49.4 |
| `mt4_ref_nc_s0` (CONS=0) | **−6.65** | 0.0% | 49.4 |
| `init_untrained` | −0.009 | 0.0% | 0.07 |

Only the reference-aligned, PG-trained critic learned the state-action pairing at all. But
8.2% argmax means that 92% of the time some foreign action outscores the one actually taken.

**Input gradients — the action pathway exists, but is outvoted by width.** Per-dimension RMS:

| input | dims | per-dim | total norm |
|---|---|---|---|
| VLM prefix | 2048 | 0.01017 | **0.4600** |
| state (real) | 8 | 0.02178 | 0.0616 |
| action (real) | 70 | 0.01071 | **0.0896** |
| padding (dead-column null) | — | ~0.00105 | — |

Per dimension the action is treated like everything else and sits ~10× above the dead-column
floor. In aggregate the prefix carries **5.1×** the action's gradient, purely on width.

**Perturbation ladder** — Q's response to isotropic Gaussian action noise of per-dim size ε:

| ε | 0.01 | 0.03 | 0.1 | 0.3 | 1.0 |
|---|---|---|---|---|---|
| σ_within(Q) | 0.117 | 0.353 | 1.761 | 11.08 | 41.08 |

## Tier B — where the real candidates sit (job 10191876, 192 states)

| quantity | `ref` | `nopg` | `nc` |
|---|---|---|---|
| ε_equivalent (per-dim candidate spread) | 0.0741 | 0.0280 | 0.0738 |
| σ_within(Q) over the 8 candidates | 4.97 | 0.83 | 3.84 |
| σ_across (over states) | 65.1 | 63.5 | 52.3 |
| critic MC rmse | 31.61 | 25.70 | 31.68 |
| `cons_zero_frac` (advantage zeroed by unanimity gate) | **0.839** | 0.915 | 0.896 |
| between-head correlation | 0.366 | 0.253 | 0.267 |

**σ_across / σ_within = 11.2** — the critic separates *states* with a spread of 65 and
*actions at one state* with a spread of 5–6. It is a good V and a poor Q. It also
discriminates far better at states whose episode succeeded (σ 10.1, n=66) than failed
(σ 2.28, n=126) — weakest exactly where steering is most needed.

**The candidates are not near-copies.** ε 0.074 against a dataset-wide per-dim action std of
0.354 is 21% of the natural action scale, and isotropic noise of that same size moves Q by
only ~1.2 versus the candidates' 4.97 — so they differ along behaviourally meaningful
directions, not jitter.

## Noise sweep — `noise_level` is not a diversity knob (jobs 10194496, 10198381)

Reproduced on two independent runs to three decimals. Sweeping the SDE noise level in
`pi0.sample_actions` (`openpi/src/openpi/models/pi0.py:342`):

| noise_level | 0.0 (ODE) | 0.0697 (**reference-equivalent**) | 0.3 | 0.5 |
|---|---|---|---|---|
| ε_equivalent | 0.0762 | 0.0763 | 0.0860 | 0.1000 |
| ρ_noise | 0.194 | 0.200 | 0.219 | 0.217 |

At the reference's own noise level the candidate spread does not move. The SDE is
**marginal-preserving** — the `σ²/2·score` drift compensates the injected noise, so it
reshuffles which initial draw maps to which action without changing the distribution.

**Noise parameterisation:** reference uses `σ_i = σ_base·√(1−i/N)` directly as the Normal
scale; ours uses `noise_level·√(t/(1−t))·√|dt|`. Total injected std: reference 0.1173, ours
`1.682 × noise_level` ⇒ reference ≡ our `noise_level` **0.0697**.

**Acting divergence:** the reference acts with the SDE at collection and eval; we act with
the deterministic ODE (`_sample_action`, `filtered_sft_learner.py:511` — no `noise_level`
reaches openpi's sampler). Our candidate diversity comes entirely from the initial
`x_1 ~ N(0, I)` draw. The OGPO *actor update* does use the SDE at `rl.noise_level = 0.3`
(`sampling.py:208` raises if it is ≤ 0). This divergence is real but concerns trust-region
bookkeeping, not exploration.

**A recommendation retracted.** After Tier B I recommended raising `noise_level` for
candidate diversity. The sweep refutes it on two runs. Withdrawn.

## Tier C — counterfactual rollouts (calibrate 10198439; full 10199880-83)

Rewind to the same state 8 times by deterministic replay, force each candidate, continue
under π to termination.

**Determinism validated exactly:** 8 identically-driven envs returned
`-14.486206236334422`, all eight, `max_minus_min = 0.0`. Replay-to-state is sound.

**Continuation noise floor:** σ_cont = 26.26 over 10 calibrate states, range 0.0–48.0.

**Full phase, 192 probe states** (Δ table above, plus):

- **33.3% of states: all 8 candidates returned byte-identical values.** These are horizon
  timeouts — a failed episode's return is `-(1-γ^T)/(1-γ)`, a pure function of length — so
  this shows the action was irrelevant *to the outcome measure*, **not** that the actions
  were the same. The probe's built-in verdict block silently drops these, which inflates its
  Δ; the numbers here include them.
- **Outcomes are strongly correlated within a state**: at least one of eight succeeds in
  70.8% of states, where independent draws at the observed rate predict 98.9%. Success is
  mostly decided by the state, not the action.
- Oracle − mean = 38.02, against 37.39 expected from selecting the max of 8 pure-noise
  draws (`c₈·σ_cont`, c₈ = 1.4236). Excess **+0.63**.

### Correction: the size of the available prize is NOT established

An earlier read of this data put the action-caused return component at 13.0 and the prize
from a perfect ranker at ~18.5. On the full 192 states the point estimate is **8.54**, and a
20 000-draw bootstrap over both σ_within (192 states) and σ_cont (10 states) gives
**95% CI [0.00, 22.90], with 39.2% of resamples at exactly zero**. The estimate is a
difference of two similar squares (27.62 vs 26.26) resting on a σ_cont from only ten
heterogeneous states. **Whether there is a worthwhile prize is undetermined.** The earlier
figures were stated far more firmly than the data supports.

A related decomposition is on firmer ground but shares the σ_cont weakness: MC residual
30.75 ⊖ σ_cont 26.26 puts the critic's *own* error at ≈16.0 rather than 30.75 (the raw
residual double-counts irreducible environment noise), moving ρ_noise from 0.194 to 0.373.

## The V(s′) probe — where the chain breaks (job 10200296, 96 states)

With `td_weight = 0.95` the critic learns Q almost entirely from `r + γ·V(s′)`, and in LIBERO
`s′` is near-deterministic given `(s, a)` — so that target is the low-noise channel through
which any action signal reaches Q. This probe executes each candidate for exactly one chunk
and evaluates V on the resulting next-states.

| quantity | value |
|---|---|
| σ_TD / σ_Q | **0.82** |
| σ_Q | 9.28 |
| σ_V(s′) | 7.81 |
| σ_TD | 7.61 |
| Pearson(Q, TD target), pooled within-state | **+0.018** |
| argmax agreement (Q's pick = TD's pick) | **14.6%** (chance 12.5%) |

Two structural facts fell out. Within a state the chunk reward is *identical* across all
eight candidates (σ = 0.000 — same length, same −1 per step), so the TD target reduces
exactly to a rescaled `V(s′)`; `Pearson(V, TD) = 1.0000`. **All candidate discrimination must
come from V.** And V does separate them (σ 7.81).

**So the break is between the target and the Q head, not upstream.** The ratio ≈ 1 rules out
"Q is failing to express a much larger available signal" — its spread is the right size. But
its *ordering* is unrelated to the quantity it is trained on.

**The mechanism:** Q sees its target only at the one action that was executed. Nothing in the
loss forces `Q(s, a_k) ≈ r + γV(s′_k)` for the other seven, so Q's action-conditional shape
is unconstrained extrapolation — right on average (mean Q −144.1 vs mean realised −144.6 in
Tier B), arbitrary in direction.

## Defect found and fixed: Best-of-N ignored `critic.reduction`

Surfaced by this probe: mean Q over the candidates was −179.4 against a mean TD target of
−160.4 at the same states. Cause: both copies of the Best-of-N scoring block hardcoded
`scores.min(axis=0)` over the Q heads, ignoring `rl.critic.reduction`. Correct when the stack
ran `num_qs=2, reduction="min"`; the reference-alignment change moved the ref config to
`num_qs=10, reduction="mean"` (`config.py:734`) and these call sites did not follow.

It matters beyond the level offset: with `between_head_corr` at 0.42, the minimum over ten
weakly-correlated heads ranks each candidate by whichever head happens to dislike it most —
head-selection noise, where the mean would average it down by √10.

Fixed in **both** clone copies (`advantage_weighted_sft_learner.py`, `best_of_n_learner.py`),
routed through `summarize_critic_values`. Configs left at the `"min"` default are
bit-identical. Record: `docs/changes/2026-08-23-bon-reduction-fix/`. Suite run on the cluster
(job 10201675): 221 passed, 3 skipped; the defect-pin test was inverted into a guard; three
unrelated pre-existing failures traced to commit `5b94510`.

**This is not a fix for the ranking failure** — every measurement above was collected through
the min path, and changing a read-out does not change what Q learned. Whether mean-reduced
ranking scores better is **unmeasured**.

## What is solid, and what is not

**Solid.** Best-of-N buys nothing (three statistics, n=192, stable across n). Q's ordering is
inconsistent with its own TD target. The critic separates states ~11× better than actions.
`noise_level` is not a diversity knob. The critic is well calibrated in the mean.

**Not solid.** The magnitude of the available prize (CI spans zero). That the min/mean
reduction explains the −19 Q-vs-TD gap (direction and rough size fit; across-head spread was
not measured). Any claim about `nopg` using `ref`'s σ_cont.

**Not tested.** Whether the fix changes run outcomes. Whether mean-reduced ranking beats
min-reduced. Anything requiring a training run.

## Open questions and the experiments that would settle them

1. **Is there a prize at all?** Re-run Tier C `full` with a paired control — at each probe
   state, one extra group of eight continuations under a *common* action. Measures σ_cont and
   σ_within on identical states, removing both the estimation noise and the population
   mismatch (calibrate sampled depths 1–12 chunks; full probes sit at 20–80% of episode
   length). ~2× the per-state cost. **This should precede any loss work** — otherwise we would
   be fixing a ranker without knowing a fixed ranker is worth having.
2. **Does mean-reduction actually rank better?** Record per-head scores in `_bon_record` and
   run one Tier C job: both rankings on identical rollouts, paired, one run instead of two.
3. **Is exploration part of it?** The critic can only learn action-dependence from data where
   actions vary, and the buffer holds one action per state from a fairly tight policy — a
   plausible contributor. But the candidates are already 21% of the action scale apart and the
   critic cannot rank the diversity it has, so widening first gives a broken ranker more room
   to be wrong. The only knob that can widen the marginal is the initial-noise temperature
   (~2.2× needed); the SDE caps at 1.3×.

## Loss-side options, if (1) says there is a prize

The critic loss is one line — `update_critic.py:432`,
`loss = td_weight·td_loss + (1 − td_weight)·mc_loss` — over a batch holding **one action per
state**. That constraint shapes every option.

**A trap first:** regressing `Q(s,a) − V(s)` onto `G − V(s)` is algebraically identical to the
existing MC loss (a stop-gradiented `V(s)` cancels). It only bites if the *parameterisation*
changes to `Q ≜ V_θ(s) + A_φ(s,a)` with separate branches.

| option | needs | cost | fixes |
|---|---|---|---|
| Dueling `Q = V(s) + A(s,a)` | nothing new | architecture | action term gets own capacity/scale |
| Pairwise ranking on **repeated init states** | buffer only | ~free | within-state ordering |
| Pairwise ranking on **counterfactual rollouts** | side-collector | simulator time | within-state ordering, exactly |
| Ranking on **foreign-action negatives** | nothing | free | ⚠️ the axis already at +22.7 |
| Multi-rollout MC targets | counterfactual data | simulator time | target variance |

LIBERO resets are seeded and deterministic, so the same initial state recurs across episodes
— early-chunk transitions give genuine within-state pairs free from the existing buffer,
degrading as trajectories diverge. And the prefix embedding is cached in the batch, so scoring
extra actions at a state runs only the small critic head, not the VLM: any within-state
contrastive term is nearly free in compute. The bottleneck is targets, not FLOPs.

## Tooling

| script | what it measures |
|---|---|
| `scripts/probe_critic_action_sensitivity.py` | Tier A: swap control, input gradients, perturbation ladder |
| `scripts/probe_policy_candidate_spread.py` | Tier B: where real π0.5 candidates sit on that surface |
| `scripts/probe_noise_level_sweep.py` | candidate spread vs SDE `noise_level` |
| `scripts/probe_counterfactual_rollouts.py` | Tier C: `determinism` / `calibrate` / `full` |
| `scripts/probe_value_next_spread.py` | σ of V(s′) and the TD target across the candidates |

All are read-only, resume the checkpoint without saving, enforce `--resume` in code (without
it `initialize_checkpoint_dir` runs `rmtree` on the run directory), guard the mount with
`timeout 120 ls`, and flush JSON per state so a preemption costs only the state in flight.
They read the production selection path through the opt-in `_bon_record` hook rather than
cloning the ~130-line scoring block.
