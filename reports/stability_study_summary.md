# OGPO Stability Study — Experiment Summary
*(as of 2026-08-08 ~19:30 UTC; most arms at 90k/100k, finals pending on a few)*

## 1. The problem

The single-task campaign winner **(viii)_r2** (π0.5, frozen backbone, GRPO G=8,
ε=0.1, success-buffer BC, libero_90_44) reached a 100% final eval but its
collection SR oscillated violently mid-run (100→80→60→100→70→25 over
40k–90k). This study set out to (a) diagnose the oscillation, (b) fix it.

## 2. Diagnosis (from (viii)_r2 forensics + live-run telemetry)

1. **Scale drift (H1)**: the advantage = within-group spread of Q over 8
   sampled actions (V cancels under the group baseline). With no scale
   control, adv_std grew ~13× over the run — a silent ~13× effective-LR
   increase (PPO gradient is linear in the advantage).
2. **Direction failure (H7) — the deeper problem**: episodic collection dumps
   2–6k transitions into the buffer every 10k steps. The critic is
   uncalibrated on the new distribution and transiently **mis-ranks the
   freshly sampled actions** the advantage evaluates (double extrapolation:
   new states × never-seen actions). The policy then takes ~1,000
   well-scaled, well-clipped, statistically invisible wrong steps before the
   next reality check. Every crash observed (NC 97→16, 100→0; CAB 69→0;
   CA 41→0) happened with KL ≈ 0.006–0.013, alive ≈ 0.6–0.7, clipfrac normal.
3. **Floods are necessary, not sufficient (H6)**: every dip followed a large
   success-buffer flood within one round; large floods did not always dip
   ((viii)_r2's 30k/60k floods preceded its best rounds).

## 3. Interventions tested

| Code | Intervention | Attacks |
|---|---|---|
| N | EMA-quantile advantage normalizer (scale = EMA₀.₉₉ of q95−q05, floor 1) | H1 scale drift |
| C | symmetric clip ±4 post-normalization | H1 tails |
| E | actor EMA 0.999 (Marco's AWR setting) | smoothing (hypothesized) |
| G | policy grad accumulation ×2 (64 states/update) | update variance |
| CA | cons-GRPO: per-head Qᵢ−mean_G(Qᵢ), sign-unanimous across heads; V(s) never enters | H7 direction (trust filter) |
| B | **critic digestion burst**: 1,000 critic-only updates after each collection round, before any policy update | H7 direction (root cause) |
| q5 | Q/V ensemble 5 heads instead of 2 (with/without cons gate) | gate strength |

All arms = (viii)_r2 recipe + listed knobs, seed 0 unless `_s1/_s2`,
100k steps, **32-episode EMA eval every 10k** (the study's key instrument —
the original run only had a single final eval, so its "100%" partially
reflects when it was measured; NC would have shown 97% twice and then 0%).

## 4. Results — full eval trajectories (32-ep EMA evals, %)

### Winners

| Arm | 10k→90k evals | Late-window (50–90k) |
|---|---|---|
| **NCB** (norm+clip+burst) | 31, 81, 97, 12.5, **100, 100, 100, 100, 100** | **five consecutive 100s** |
| **B** (burst only) | 41, 41, 78, 75, **100, 100, 96.9, 100, 100** | **≥96.9 for five evals** |
| **CANC** (cons+norm+clip, no burst) | 59, 34, 41, 9.4, 84, 69, 59, **90.6, 96.9** | best no-burst arm, rising |
| CAB (cons+burst) | 63, 69, 0, 84, 97, 97, 9.4, 75, **100** | high but crashes twice |

### Everything else (latest/most representative)

| Arm | Trajectory | Verdict |
|---|---|---|
| NC (norm+clip) | 44, 66, 12.5, 12.5, **97**, 16, **97**, **100**, **0** | highest peaks, terminal crash — the H7 exhibit |
| N (norm) | 22, 66, 19, 22, 9, 34, 50, 0, 25 | oscillates |
| G (accum) | 47, 22, 6, 47, 3, 6, 94, 9, 94 | violent swings |
| B_s1 (burst, seed 1) | 22, 72, 44, 16, 100, 81, 69, 100, 62.5 | strong but swingy |
| B2k (burst 2000) | 25, 25, 53, 78, 66, 100, 12.5, 9, 100 | dose ↑ ≠ stability ↑ |
| CAB_s1 | 12.5, 25, 0, 0, 16, 31, 0, 25, 75 | weak |
| CA (cons only) | 25, 41, 19, 3, 0×5 | crashed, never recovered |
| CAN / CANB | ≤22 / ≈0 throughout | dead (CANB anomalous — see §6) |
| CANCB (everything) | 44, 44, 25, 44, 31, 0, 44, 19, 44 | capped ~44 |
| base_s1 / base_s2 | ≤34 / ≤28, ending 3 / 16 | weak, wobbly controls |
| E / ENC / ENCG (EMA .999) | ≤12.5 / ≤22 / ≤9 | **negative result** (killed at 75–82k) |
| CA_q5 | 38, 72, 22, 50, 22, 81, **97**, 12.5 | capable, unstable |
| CAB_q5 | 75, 9, 0, 38, 19, 41, 72 | unstable |
| CANB_q5 | 28, 59, 66, 88, 50, 9, 81 | mid, volatile |
| B_q5 | 34, 53, 56, 12.5, 100, 0, 97 | volatile |

## 5. Hypothesis scoreboard (preliminary — finals pending)

| H | Statement (short) | Verdict |
|---|---|---|
| H1 | un-normalized adv scale grows multi-fold | **Confirmed** (all non-N arms) |
| H2 | normalizer reduces oscillation | **Refuted as a cure** (N/NC still crash); helps speed |
| H3 | clip adds marginal stability over N | Weak/unclear (NC ≈ N in stability) |
| H4 | actor-EMA 0.999 hurts (starves PPO ratio) | **Confirmed negative** — all E-arms ≤22% ever |
| H5 | grad accum reduces SR variance | **Refuted** (G swings 6↔94) |
| H6 | dips follow floods; floods don't always dip | **Confirmed** (incl. (viii)_r2 history) |
| H7 | direction failure survives scale control | **Confirmed** (NC 97→16→97→100→0 with perfect actor health) |
| H8 | cons gating → shallower dips | **Mostly refuted at 2 heads** (CA crashed to 0; CAB crashed twice); 2 correlated heads share delusions |
| H9 | cons_zero_frac rises pre-crash | **Confirmed** (CAB: 0.20→0.30 over ~6k steps before its 0/20) — validated early-warning signal |
| H10 | critic digestion burst prevents post-flood dips | **Strongly supported**: every arm currently ≥96.9% late carries burst (B, NCB, CAB, B2k@90k); B and NCB sustained ceiling for 40k+ steps. Not sufficient alone on all seeds (B_s1, CAB_s1 swing) — bursts that END undigested (q_loss ~50) precede crashes (CAB 20k→30k autopsy) |
| H11 | K=2000 ≈ K=1000 | Mixed: B2k not more stable; dose isn't the axis — **completion** is (adaptive burst indicated) |
| H12 | gating and burst stack | **Refuted as stated**: CAB < B in stability; CANCB capped. NCB (norm+clip+burst, no cons) is the best stack |
| H13 | seed noise is large; effects must beat it | **Confirmed** (base_s1 vs s2; B vs B_s1) — and burst's effect does beat it in kind (all four ceiling arms carry it) but not in reliability |
| H14 | V(s) irrelevant to actor | **Confirmed by construction + unit test** (cons-GRPO is Q-only; offset-invariance verified) |

## 6. Key mechanistic findings

1. **The burst works when it completes.** End-of-burst q_loss is the tell:
   0.1–1.9 → next rounds rise; ~50 (undigested) → crash risk (CAB's 0/20
   followed exactly this signature). Fixed K is the flaw → **adaptive burst**
   (run until q_loss plateaus/threshold, K_min ∝ flood size, cap ~5k) is the
   designed next step.
2. **cons_zero_frac is a working crash predictor** (~6k steps of warning) even
   where the gate itself fails to protect — usable as a circuit-breaker
   (freeze actor, keep training critic when disagreement > ~0.25).
3. **Magnitude controls (N, C) are speed-ups, not cures.** They pin effective
   LR (by design) and the arms learn faster, but direction failures pass
   through untouched.
4. **Boom-bust is self-triggered**: success floods (mechanically ∝ SR) are
   the load; whether the gun fires depends on critic digestion at that
   moment. The instability is a structural price of episodic collection.
5. **Anomaly to investigate**: CANB (cons+norm+burst on bc0) flatlined ≈0
   while its Q5 twin (CANB_q5) and near neighbors (CAB, NCB) work — possibly a
   bad init/seed interaction or a real cons×norm interference; unexplained.

## 7. Collaborator branch survey (who else fought this)

Everyone hit the same instability; the repo contains 4 families of partial
fixes — none applied per-collection:
- **Critic-first scheduling**: offline critic pretraining (Ralf/Lenart:
  `num_offline_pretraining_steps`), warmup + optimizer/EMA reset at handoff
  (`critic.pre_training_steps`, on main), `policy.training_start_step=900`
  (we run it — it IS a t=0 burst), Ralf's current 100:1 critic:policy
  interval (`ralf/dev`), Marco's `training_end_step` (policy-window close).
- **Actor gating on critic confidence**: Mert's `min_advantage_std`
  (zero PG when the critic can't rank), `critic.inference_start_step`,
  Manan's original conservative per-head advantage.
- **Advantage robustness**: the EMA-quantile Normalizer (we ported it);
  Manan's babel_cfg comment independently states our H1 mechanism verbatim
  and replaces EMA with per-batch/per-task stats; relu/linear bounded weights.
- **Critic robustness**: BroNet (in use), SimbaV2 ports, C51/two-hot bounded
  values, Michal's truncation-bootstrap bugfix (critic was taught 0 future
  value at time-limit truncations — systematic pessimism on failure floods),
  Marco's `train_on_policy_value_function` (train Q on fresh policy actions —
  closes our double-extrapolation gap at the source), pzal's held-out critic
  validation buffer (the principled adaptive-burst stopping criterion).
- **Smoking gun**: branch `debug_without_new_reply_buffer` = a 5-seed A/B
  pinned before the ShardedReplayBuffer switch — someone suspected the
  new-data path itself.

**Our contribution**: identifying the per-collection recalibration gap and
showing that a post-collection critic-only burst — nobody's existing
mechanism — converts the boom-bust into sustained ceiling performance
(B: ≥96.9% × 5 evals; NCB: 100% × 5 evals over 40k steps).

## 8. Current best recipes

1. **NCB** — norm + clip ±4 + burst 1000: five consecutive 100% evals
   (50k–90k), one early dip at 40k. Best overall curve in the study.
2. **B** — burst 1000 alone: ≥96.9% for five evals, never crashed, simplest
   possible change (+3–5% wall-clock). Best simplicity/stability ratio.
3. Next iteration (designed, not yet run): **adaptive burst** (digest-until-
   converged) + optional disagreement circuit-breaker + Marco's
   on-policy critic training. Q5 gating: not earning its complexity so far.

## 9. Artifacts

- Pre-registered hypotheses: `reports/stability_study_hypotheses.md`
- B's headline curve vs (viii)_r2: `reports/sr_B_burst.png`
- All metrics: `run_store/checkpoints/stability_study/pi05_libero_online_ogpo_sft/stab_*/metrics.jsonl`
  on intern-{bc0,cons,dev,ablate}-shashabc
- Branch: `shashwat/stability-study` (flags: `normalize_group_advantage`,
  `adv_clip_sym`, `policy_grad_accum`, `advantage_combination=grpo_conservative`,
  `post_collection_critic_steps`, launcher `scripts/stability_study.sh`)
- 20 arms total: 8 bc0 + 4 cons + 4 dev + 4 ablate(Q5); E/ENC/ENCG killed
  at 75–82k to free GPUs for the cons family (CAN/CANC/CANB).
