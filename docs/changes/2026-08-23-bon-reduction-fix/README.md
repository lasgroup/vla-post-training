# Best-of-N candidate scoring ignored `critic.reduction`

**Tier 1.** Approved by the maintainer ("fix the min to mean").

`sample_actions` scored Best-of-N candidates with a hardcoded `scores.min(axis=0)`
over the Q-ensemble heads, ignoring `rl.critic.reduction`. Correct when the stack
ran `num_qs=2, reduction="min"`; the reference-alignment change
(`2026-08-20-ogpo-reference-alignment`) moved the ref config to `num_qs=10,
reduction="mean"` (`config.py:734`) and this call site did not follow.

Found by `scripts/probe_value_next_spread.py` (job 10200296): mean Q over the
Best-of-N candidates was -179.4 against a mean TD target of -160.4 at the same
states — a ~19-point gap in the direction and rough magnitude a min-vs-mean
reduction would produce.

Why it plausibly matters beyond the level offset: `between_head_corr` is 0.417,
so the minimum over ten weakly-correlated heads ranks each candidate by whichever
head happens to dislike it most. That is head-selection noise; the mean averages
it down. See [[BLAST-RADIUS]] for scope and [[VERIFICATION]] for what was and
was not checked.

**This is a correctness fix, not a fix for the ranking failure.** Tier C
(10199880-83) measured Delta = -0.6 +/- 3.5 and within-state Pearson(Q, realised)
= +0.03; the vnext probe measured Pearson(Q, own TD target) = +0.018. Those
diagnose an unsupervised action ordering, which no read-out reduction addresses.
