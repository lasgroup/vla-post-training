# Critic action-sensitivity probes (Tier A / B / noise sweep / Tier C)

**Tier 1.** Read-only analysis tooling plus two small additive hooks that make
existing production behaviour observable. No algorithm, no jit signature, no
checkpoint layout, no config semantics change.

## Why

`reports/findings.md` §10 and the Tier A/B probes established that OGPO's
best-of-8 collection selects on a Q spread far below the critic's own error:

| | CONS+PG | no-PG | reduced+PG |
|---|---|---|---|
| candidate dispersion (eps_equivalent) | 0.0741 | 0.0281 | 0.0738 |
| sigma_within over the 8 candidates | 4.98 | 0.83 | 3.84 |
| critic MC rmse | 31.6 | 25.7 | 31.7 |
| rho_noise | 0.157 | 0.032 | 0.121 |

Two questions were left open, and this change builds the instruments for both:

1. **Would more sampling noise fix it?** Collection today samples the
   deterministic ODE (`create_trained_policy` with no `sample_kwargs`
   => `noise_level=0.0`), so all candidate diversity comes from the initial
   Gaussian draw. `probe_noise_level_sweep.py` re-runs the Tier B battery across
   a noise ladder that brackets both our recipe value (0.02) and the
   reference-equivalent (0.0697, derived below).
2. **Is the ranking CORRECT, not merely small?** Only counterfactual rollouts
   answer that. `probe_counterfactual_rollouts.py` executes each of the 8
   candidates from the same state and continues under pi to termination.

## Reference-noise calibration (answers a standing question)

All 15 `OGPO_public/scripts/ogpo/*.sh` recipes, the three PaliGemma ones
included, use `use_tapered_noise=true`, `constant_noise_std=0.05`,
`flow_steps=10`. `min_noise_std=max_noise_std=0.01` are **inert** there -- they
only parameterise the unused `NoiseInjectionNetwork`.

The two codebases scale noise differently:

| | per-step injected std | total over the chain |
|---|---|---|
| reference (`pg_helper.py`, `distrax.Normal(mean_next, sigma)`) | `0.05*sqrt(1-i/10)`, no `sqrt(dt)` | 0.1173 |
| ours (`pi0.py:160-166`) | `noise_level*sqrt(t/(1-t))*sqrt(|dt|)` | `1.682 * noise_level` |

Matching totals: **the reference's noise == our `noise_level` 0.0697**.
Our recipes pass `--rl.noise_level 0.02` (~3.5x less), and it applies only to the
actor's group sampling -- **collection and eval act on the deterministic ODE,
where the reference acts with the SDE** (`ogpo.py:1970` -> `sample_actions_with_noise`).
That divergence is not in `docs/changes/2026-08-20-ogpo-reference-alignment/`.

## Caveat carried into the sweep

The pi0 SDE is marginal-preserving (the drift carries the `sigma^2/2 * score`
term), so in the continuous-time limit `noise_level` does not change the terminal
action distribution at all; extra spread at `num_steps=10` is discretisation
error. Hence: measure, do not reason.
