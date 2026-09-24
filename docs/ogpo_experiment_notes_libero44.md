# OGPO libero_90_44 experiment campaign — running notes

Task: `libero_90_44`, π0.5, batch 32, seed 0, 100k trainer steps, `subtract_v`
critic-baseline advantages (G=1) unless noted, `store_prefix_rep=true`,
ema_decay 0.99 (0.995 in the control). SR = 20-episode on-policy collection
round every 10k steps. Branch: `shashwat/cons-adv-success-bc` (on top of
`pranav/memory_optimisations`).

## Run ledger

Since 2026-08-29 the Backbone column has a third value, **LoRA** (`LORA=1` in
the recipes, `docs/changes/2026-08-29-backbone-lora/`): rank-16 adapters on
the 2B LLM stack 0, SigLIP + base frozen, critic prefix embeddings pinned to
the unadapted backbone. Two readings to get right on a LoRA arm: (1)
`actor/param_norm` and every `actor/grad_norm*` series include the adapters
and the shared global clip — NOT comparable to frozen/unfrozen arms
(`grad_norm_lora`/`grad_norm_rest` decompose it); (2) at step 0, `lora_a`
receives exactly zero gradient by construction (`lora_b` starts at zero and
each factor's grad is proportional to the other) — `lora_b` moves first and
`lora_a` comes alive from step 1; this is standard LoRA init, not a plumbing
bug.

| Run | Backbone | Advantage | ε | succBC | Result / status |
|---|---|---|---|---|---|
| control | unfrozen | reduced 2Q/2V | 0.01 | — | DONE: peak 85%@50k, plateau 70–85%, final ~55–80%. PPO term 100% clipped all run (dead) — learning was BC-driven. |
| A | unfrozen | reduced 2Q/2V | 0.1 | — | DONE: oscillated 10↔50%, terminal collapse, final eval **15.6%** |
| B | frozen | reduced 2Q/2V | 0.1 | — | DONE: same shape, 0% trough, final eval **12.5%** |
| C | frozen | **conservative 5Q/5V** | 0.03 | — | KILLED @58k: flat 0–5% throughout. See post-mortem below. |
| D | frozen | conservative 5Q/5V | 0.03 | ✅ | KILLED @50k: noisy 0–15% flatline. See post-mortem below. |
| (i) | frozen | conservative 2Q/2V | 0.03 | — | running (H200): 0→10→20% — PPO dead (clip), BC-only pace |
| (ii) | unfrozen | conservative 2Q/2V | 0.1 | — | running (H200): 5→35% @10k — healthy PPO (alive ~0.5–0.9) |
| (iii) | frozen | conservative 2Q/2V | 0.03 | ✅ | running (H200): 5→10→15% — PPO dead (clip) |
| (iv) | unfrozen | conservative 2Q/2V | 0.1 | ✅ | running (H200): 5→**100% (20/20 verified) @10k** — healthy PPO + succ anchor |
| (v) | frozen | conservative 2Q/2V | 0.1 | — | running (B200): isolates backbone vs (ii) |
| (vi) | frozen | conservative 2Q/2V | 0.1 | ✅ | running (B200): isolates backbone vs (iv) |
| (vii) | frozen | **GRPO: G=8, group-mean (vanilla), no critic baseline** | 0.1 | — | running (B200): critic-free baseline arm |
| (viii) | frozen | GRPO G=8 vanilla | 0.1 | ✅ | running (B200) |

## C/D post-mortem: why 5-head conservative advantage doesn't work

**Setup:** conservative advantage combine (per-head A_i = Q_i − V_i;
A = max(min_i A_i, 0) + min(0, max_i A_i)) with **num_qs = num_vs = 5**,
ε=0.03, frozen backbone. D additionally used the success-buffer BC anchor.

**Observed:** SR flat at 0–15% for 50–58k steps (vs 35–50%+ for every healthy
run by 20–30k). Actor telemetry: `advantage_max` pinned at **0.00–0.05** for
tens of thousands of steps (healthy runs: 0.3–1.5), `advantage_std` ~0.02,
`alive_fraction` ~0.00–0.17, clipfrac ~1.0. Critic TD losses were small and
normal — the critics themselves were fine.

**Hypothesis (high confidence): sign-unanimity across 5 heads is a
near-certain veto, so the advantage signal is annihilated at the combine, not
at the critics.** Each head's A_i = Q_i − V_i is a small number (post-TD
convergence, |A_i| ~ 0.1–1) with independent noise across heads. For the
combined advantage to be nonzero, ALL FIVE heads must agree on the sign. If
per-head sign agreement with the "true" advantage is p, unanimity occurs with
probability ≈ p⁵ + (1−p)⁵ — e.g. p=0.8 → ~33% of samples survive; p=0.7 →
~17%; and the surviving magnitude is the *most pessimistic head's* value,
which for 5 draws is far into the lower tail. Net effect: advantage ≈ 0 on
almost every sample → PPO gradient ≈ 0 → the run degenerates to pure BC on
the online batch — and the flat 0–5% matches exactly the no-RL trajectory
from a 5% base policy under plain BC-to-own-rollouts.

Contrast with 2 heads: unanimity is p² + (1−p)² (p=0.8 → 68% survive), and
the pessimistic-of-2 magnitude is only mildly shrunk. Runs (i)–(vi) confirm:
2-head conservative preserves advantage_max ~0.15–1.4 — usable signal — while
still crushing the 10–17-point spikes that destroyed A/B.

**Secondary factor:** ε=0.03 was independently fatal in C/D (as in (i)/(iii)
— the ratio spread needs ~±0.1), so C/D were doubly dead: no advantage signal
AND no clip window. Fixing either alone would not have saved them.

**Lesson:** conservatism must be dosed. The unanimity filter's strictness
grows exponentially in ensemble size; n=2 is the useful setting, n=5 is a
veto. If more ensemble smoothing is ever wanted, prefer soft penalties
(mean − β·std across heads) over unanimity at large n.

## Emerging conclusions (as of 2026-08-04 late)

1. **Raw (reduced min-min) Q−V advantages + live PPO = terminal collapse**
   (A, B). The critic ensemble's disagreement spikes (adv_max 10–17) get
   faithfully amplified by an unmuzzled surrogate.
2. **ε must match the ratio spread** (~±0.1 for this per-dim-normalized
   10-step chain ratio). ε=0.01 and 0.03 both produce 100% clipping → PPO
   contributes nothing (control, C, D, i, iii).
3. **Conservative-2Q + ε=0.1 is the working operating point** — bounded
   advantages AND a live clip window ((ii), (iv)).
4. **Success-buffer BC is a large amplifier**: (iv) vs (ii) at 10k is
   100% vs 35%. Anchoring BC to successes both floors collapse and
   compounds gains.
5. Backbone contribution and critic-baseline-vs-GRPO are pending
   ((v)/(vi) and (vii)/(viii) respectively).

## Bookkeeping

- wandb (far.wandb.io, project `shashabc/ogpo_sweep`): control synced;
  A=0an3ylk4, B=5p1dm2c3, C=opc0zj6p, D=29ifzjtx; H200 runs pending sync.
- Control-run artifacts (all checkpoints + log):
  `s3://far-research-internal/shashabc/backup_20260803_intern_dev/`
- Dashboards/analysis: `reports/ogpo_unfrozen_44_dashboard/`,
  `docs/ogpo_speed_memory_analysis.md`
