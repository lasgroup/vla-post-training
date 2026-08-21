# DIFF — OGPO reference alignment

Implemented 2026-08-20 against the approved `PLAN.md`. `src/rl/ogpo/update_actor.py` and
`tests/ogpo/test_split_equivalence.py` also appear in `git diff` but are **pre-existing
uncommitted work from an earlier session** (the `grad_norm_pg`/`grad_norm_bc`/
`grad_cos_pg_bc` change) and were **not touched** by this change.

## Files changed

| file | lines | what |
|---|---|---|
| `src/training/config.py` | +52 | `CollectConfig.success_reward_bonus`; `OGPOSFTLearnerConfig.critic_success_oversample`; registered `pi05_libero_online_ogpo_ref` |
| `src/envs/wrappers.py` | +18/−4 | `TimeToSuccessAsRewardWrapper(success_bonus=...)` |
| `src/rl/filtered_sft_agent/filtered_sft_learner.py` | +4/−2 | pass the bonus at the single construction site |
| `src/rl/value_distribution.py` | +11/−3 | `get_value_bounds` lower = Bellman fixed point, upper tracks the bonus |
| `src/rl/ogpo/ogpo_learner.py` | +50 | A4 success-oversampling critic batch; fail-fast on the inert flag combination |
| `scripts/ogpo_multitask_4task.sh` | +17/−3 | six new env vars, all defaulting to today's values |
| `scripts/ogpo_multitask_4task_ref.sh` | new | thin wrapper, the aligned recipe |
| `tests/ogpo/test_reward_and_value_bounds.py` | new | 10 tests |

## Per-item

### A1 — terminal success bonus (config-driven, D12)

`CollectConfig.success_reward_bonus: float = 0.0`; the aligned recipe sets **90.0**
(corrected from 72.0 — see `VERIFICATION.md` D1: the reference pays `+5.0` per success step,
not +4.0, so the ratio-preserving value is 90, not 72). The wrapper takes it through `__init__`
rather than reading the config, because it receives an env, not a config, and
`FilteredSFTLearner._make_env:62-63` is its only construction site — shared by all six
learners. **Default 0.0 reproduces the previous reward exactly**, verified by
`test_zero_bonus_reproduces_the_original_reward_exactly` and `test_default_bonus_is_zero`.

`src/rl/dsrl/dsrl_env.py:486-487` also constructs the wrapper, positionally, with no bonus.
It is unaffected by the defaulted parameter and was left alone (quarantined tree).

### A4 — success oversampling into the critic

One extra `_update_critics_jitted` call on a success-only batch, after the `critic_utd` loop.
Same jit, second invocation — no signature change, no new RNG arity. Reuses the batch shape
and `drop_obs_keys` gate the online critic batch already uses. Metrics land under `critic_sb/`
so the existing `critic/` series stays comparable with the ten completed arms.

### Value bounds

Two changes, both in the `use_time_to_success_as_reward` branch:

- `upper` → `max(0.0, success_reward_bonus)`. Required by A1.
- `lower` → `-1/(1-discount)` (= −200.0) instead of `-(1-γ^T)/(1-γ)` (= −173.07).
  **This is a pre-existing latent bug, not a consequence of A1**: `fix_mc_returns`
  (`filtered_sft_learner.py:749-751`) overwrites every failed episode's MC return with
  exactly `reward/(1-γ)`, so the horizon-truncated value was never what the critic regressed
  onto.

  **CORRECTION (verifier D2): this is NOT inert.** The original claim here — "latent because
  `num_value_bins = 1` everywhere" — is withdrawn. Two committed launcher sweeps set
  `num_value_bins: 201` on `pi05_libero_online_best_of_n`:
  `scripts/configs/tuning/bofn/1_mc_vs_td_regression_vs_distributional/` and
  `scripts/configs/tuning/bofn/2_mlp_vs_bronet/` (libero + molmo). For those, bounds move
  `(-173.5011, 0.4327) → (-200.5, 0.5)` — **every one of the 201 bin centers shifts**, and the
  pinned −200 goes from outside the range (silently clipped at `value_distribution.py:34`) to
  inside it. The new bounds are correct, but **runs of those two sweeps completed before this
  change are not comparable with runs after it.** No OGPO config is affected.

### D4 — parallel config

`pi05_libero_online_ogpo_ref`, registered beside the existing OGPO config, reusing
`OGPOSFTLearnerConfig` with different defaults. **No new dataclass**, so `scripts/exp.py:74-85`
is untouched and the documented dispatch-order sharp edge is not in play.

## Divergences from PLAN.md

Three, all deliberate. None changes the approved scope.

1. **The recipe is a thin wrapper, not a copy.** `PLAN.md` §6 said "copy
   `ogpo_multitask_4task.sh`". Copying would have produced a **third** copy of the ~60-line
   env preamble that `BLAST-RADIUS.md` §1 flags as an already-diverged clone family. Instead
   the baseline recipe gained six env vars (`NUM_QS`, `CRITIC_RED`, `BON_N`, `SUCC_BONUS`,
   `SB_Q`, `TD_W`) plus `CONFIG_NAME`, each defaulting to today's value, and
   `ogpo_multitask_4task_ref.sh` sets them and `exec`s the baseline.
   *(The byte-identical-command-line claim originally made here is SUPERSEDED by the D3 fix
   below — see "Post-verification changes".)*
2. **A fail-fast was added that PLAN.md did not specify.**
   `critic_success_oversample=True` with `use_success_buffer=False` would have been a silent
   no-op. `OGPOAgentLearner.__init__` now raises, with the fix in the message, matching the
   `online_ratio` raise directly above it. Per the CLAUDE.md deviation protocol this is the
   fail-fast direction, so no approval was required — but it is recorded here because it was
   not in the approved plan.
3. **A2 and A3 required no code**, as `PLAN.md` already recorded from the planning pass.
   A3 is `--rl.n_samples 8` against the inherited
   `AdvantageWeightedSFTLearner.sample_actions:313`.

## Deliberate non-changes, each with a measurement

Recorded in the recipe header so they are not "fixed" later:

- `clip_epsilon` stays **0.1**. Measured `actor/ratio_max` ≤ 0.9888 over all post-PG steps of
  `ncb`; ε=0.01 sets the lower bound at 0.990, putting **every sample in every batch**
  outside the trust region and collapsing the PG to positive-advantages-only.
- `discount` stays **0.995**. Our successes take ~295 env steps, so γ²⁹⁵ = 0.228 here equals
  the reference's γ¹⁵⁰ at 0.99. Copying 0.99 would cut the success/failure gap 22.8% → 8.9%.
- `group_num_samples` stays **8** (reference 32): 4× the actor forward *and* backward on a 3B
  action expert.
- `post_collection_critic_steps` stays **1000** (D9), `bc_coeff` 1.0, warmstart 20000/5000,
  actor LR constant 2.5e-5 (D3 — A5 out of scope; `TrainState.tx`/`opt_state` untouched).

## Nothing removed (D10)

`normalize_group_advantage`, `normalize_advantage_per_task` and `adv_clip_sym` remain as
dataclass fields and as `NORM`/`MT_ADV`/`CLIP_SYM` env vars. The aligned recipe defaults them
off; its header documents the exact override string that reproduces the baseline stack. The
`_adv_scale`-not-checkpointed gotcha therefore **stays live** and must not be deleted from the
docs.

---

## Post-verification changes (2026-08-20)

Three defects from `VERIFICATION.md` were fixed after the diff above was first written:

1. **D1** — success bonus 72.0 → **90.0** everywhere (`config.py` comment,
   `ogpo_multitask_4task_ref.sh` default and header, the tests). The reference's constant is
   `+5.0` per success step, not `+4.0`.
2. **D2** — the "inert" claim about `get_value_bounds` withdrawn; see above.
3. **D3** — the five broken env-var guards (`NUM_QS`, `CRITIC_RED`, `BON_N`, `SB_Q`, `CONS`)
   in `ogpo_multitask_4task.sh` now emit their flag **unconditionally**, so the env var is
   authoritative regardless of which config is selected. The documented revert string in
   `ogpo_multitask_4task_ref.sh` now also resets `CONFIG_NAME`.

   **This gives up the byte-identical-DRY-output property** claimed earlier: the baseline
   recipe's command line now carries these flags at their baseline values. The replacement
   guarantee is stronger and machine-checked — both command lines are parsed through tyro and
   the **resolved configs** compared across 15 fields; the reverted ref recipe equals the
   baseline exactly.
