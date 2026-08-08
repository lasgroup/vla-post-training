# Stability Study — Pre-registered Hypotheses (2026-08-08)

Written BEFORE results, to be scored when all arms finish. Base recipe =
abl_viii_r2 (frozen · GRPO G=8 · ε=0.1 · succBC). Oscillation metrics:
adv_std trajectory, worst eval after first ≥50% eval, dip depth/frequency,
final eval.

**H1 (scale drift).** In un-normalized arms, advantage std grows several-fold
over training (critic Q-spread growth = silent effective-LR increase), and
this growth is a driver of mid-run SR oscillation. *Test: base arms reproduce
(viii)_r2's adv_std climb; N-arms hold adv_std ≈ const.*

**H2 (normalizer).** EMA-quantile normalization of the advantage reduces
oscillation amplitude (shallower/less frequent post-peak dips) without
hurting — possibly accelerating — learning. *Test: N/NC vs base_s1/s2.*

**H3 (tail clip).** Symmetric ±4 clip on the normalized advantage adds
marginal stability over the normalizer alone by absorbing the EMA scale's
lag during spread spikes. *Test: NC vs N in spike windows and dip depth.*

**H4 (actor EMA 0.999 — expected NEGATIVE).** Slowing the old-policy anchor
to 0.999 starves PPO (stale denominator → half the batch clipped away) and
slows or kills learning rather than stabilizing it. *Test: E/ENC/ENCG
underperform their E-free counterparts throughout; alive_fraction lower.*

**H5 (state diversity).** Grad accumulation (2×32 states/update) reduces
round-to-round SR variance vs batch-32 baselines at matched steps. *Test:
G vs base arms; ENCG vs ENC adds nothing if H5 false.*

**H6 (flood → dip; necessary not sufficient).** Every SR dip follows a large
success-buffer flood (≥~600 new transitions) within one round; large floods
do not always cause dips. *Test: flood/dip contingency across ALL arms incl.
the original (viii)_r2 history.*

**H7 (direction, not scale).** Post-flood dips persist in scale-controlled
arms (N/NC) because the failure is transient Q mis-RANKING of freshly
sampled actions (critic uncalibrated on the new distribution), which no
magnitude control can fix. *Test: NC dips despite pinned KL/adv_std (already
observed once at 30k — needs to repeat to count).*

**H8 (ensemble gating).** Cons-GRPO (per-head Q−mean_G(Q), sign-unanimous)
zeroes exactly the untrustworthy samples, so CA-arms show shallower post-
flood dips than advantage-matched non-cons arms. *Test: CA vs base; CAB vs
B; CANCB vs NCB.*

**H9 (disagreement tracks miscalibration).** cons_zero_frac RISES in the
~1k steps after a success flood and relaxes as the critic digests — i.e.
head disagreement is a live measure of the H7 mechanism. *Test: cons_zero_
frac time-locked to collection rounds in CA/CAB/CANCB.*

**H10 (critic digestion burst).** K=1000 critic-only updates after each
collection round remove the uncalibrated window, so burst arms dip less
after floods than their no-burst counterparts, at negligible cost. *Test:
B vs base; CAB vs CA; NCB vs NC; burst/* q_loss should spike-then-settle
within each burst.*

**H11 (burst dose).** If H10 holds via the digestion mechanism, K=2000
adds little over K=1000 at mid-run buffer sizes (both exceed the ~30-visit
bar) but may help late-run. *Test: B2k vs B_s1/B trajectories, esp. >60k.*

**H12 (mechanism independence / stacking).** Gating (H8) and burst (H10)
attack the same failure through different channels and therefore stack:
CAB ≥ max(CA, B) and CANCB is the study's most stable arm. *Test: the
combined arms vs their components.*

**H13 (seed noise floor).** Same-config runs differ by large margins early
(≤30k) but converge in FINAL eval within ~±1 round of collection noise;
any intervention effect must exceed the base_s1-vs-base_s2 spread to count.
*Test: all seed pairs (base, N, E, ENCG, B, CAB replicates).*

**H14 (V-irrelevance).** Removing V(s) from the advantage (grpo_conservative
is Q-only) changes nothing about learning dynamics attributable to V —
V only ever mattered to the actor through critic-training side effects.
*Test: CA-family arms show no V-linked anomalies; v_mean diagnostics evolve
as in all other arms.*
