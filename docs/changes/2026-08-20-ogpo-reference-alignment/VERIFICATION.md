# VERIFICATION — OGPO reference alignment

Independent adversarial verifier (fresh context, Opus, xhigh), 2026-08-20. It received the
change spec + diff, not the implementing session's reasoning. Findings below are relayed
**verbatim** where quoted, including the failures. My resolution follows each.

An earlier verifier run was stopped by the user mid-flight and re-launched; the report below
is from the completed run.

---

## Verdict (verbatim)

> **No blocking code defect.** The core mechanics are correct: A1's default is bit-identical
> to the committed pre-change wrapper, A4 does not perturb the RNG stream when off, the new
> config resolves and dispatches correctly, and both recipes emit parseable command lines
> that produce exactly the documented config.
>
> **Three real defects found**, none of which will crash a run, but two of which should be
> settled before launching.

---

## D1 — the reference pays **+5.0**, not +4.0; the `+72` derivation used the wrong constant

Verbatim:

> The cite is right, the number is not.
> `/home/pchellap/Projects/SafeVADAR/OGPO_public/envs/robomimic_utils.py:457` (and `:467`,
> `:692`, `:702` — all four sites, both wrappers) reads **`reward += 5.0`**. `4.0` appears
> nowhere. […] The "36" in `36/100 × 200 ⇒ 72` is `4 × 9`. With the constant the source
> actually has, it is `5 × 9 = 45`, i.e. **45% of the −100 floor ⇒ +90 on our −200 floor**,
> not +72.
>
> **Failure scenario:** the single largest intervention in the change (A1 […]) is sized 20%
> below its stated target. Not a crash — a mis-set research hyperparameter whose written
> justification does not survive checking.

**Confirmed independently.** `robomimic_utils.py:436-457`: robomimic's raw success reward is
`1.0`, the `reward = reward - 1.0` shift lands it at `0.0`, then `reward += 5.0` ⇒ **+5.0** on
the success step. The implementing session's "−1 + 5 = +4" wrongly assumed the −1 time penalty
also applied on that step; it does not.

**RESOLVED — bonus corrected 72 → 90.** This executes decision D1's stated principle (preserve
the reference's bonus-to-floor ratio) with the correct constant; it is not a new decision.
Recomputed at γ = 0.995, 295-step successes:

| | success return | gap | % of range |
|---|---|---|---|
| no bonus (today) | −154.4 | 45.6 | 22.8% |
| +72 (as shipped, wrong) | −138.0 | 62.0 | 31.0% |
| **+90 (corrected)** | **−133.9** | **66.1** | **33.1%** |
| reference, +45 @ L=150 | −67.9 | 32.1 | **32.1%** |

Changed in `src/training/config.py` (comment + derivation), `scripts/ogpo_multitask_4task_ref.sh`
(`SUCC_BONUS` default and header), and `tests/ogpo/test_reward_and_value_bounds.py`.

## D2 — "inert because `num_value_bins == 1` everywhere" is **false**

Verbatim:

> Two checked-in launcher sweeps set 201 bins on `pi05_libero_online_best_of_n`, which is the
> one learner that builds a categorical head […] Measured (differential vs. the HEAD
> function): bounds go `(-173.4995, 0.43267) → (-200.5, 0.5)`. **Every one of the 201 bin
> centers moves**, and the pinned `-200` failure return goes from *outside* the range
> (silently clipped by `_discretize`, `value_distribution.py:34`) to inside it. The change is
> an improvement, but it is a **behavioural change to a reachable, committed config** and it
> breaks comparability with any already-completed run of that sweep.

**Confirmed independently**: `(-173.5011, 0.4327) → (-200.5, 0.5)`. Affected files:
`scripts/configs/tuning/bofn/1_mc_vs_td_regression_vs_distributional/…` and
`scripts/configs/tuning/bofn/2_mlp_vs_bronet/…` (libero + molmo).

**RESOLVED as a documentation correction, not a code change.** The new bounds are correct — the
old ones silently clipped the −200 that `fix_mc_returns` actually writes. But the claim
"inert" was wrong and is withdrawn:

> **The `get_value_bounds` lower-bound fix is NOT inert.** It changes all 201 bin centers for
> any `num_value_bins > 1` Best-of-N config. Runs of those two tuning sweeps completed before
> this change are not comparable with runs after it. No OGPO config is affected
> (`num_value_bins = 1`).

## D3 — the ref recipe violated D10: five knobs could not be reverted by env var

Verbatim:

> Each guard in `ogpo_multitask_4task.sh:144-152` appends a flag only when the var differs
> from the **baseline** default, while the ref **config** already carries the aligned value.
> […] `NUM_QS=2` → no flag → **stays 10**. Same for `CRITIC_RED`, `BON_N`, `SB_Q`, `CONS`.

This is the failure the compute-node suite surfaced
(`test_ref_recipe_documented_baseline_override_string_actually_reproduces_the_baseline`).

**RESOLVED — the five guards now emit their flag unconditionally**, so the env var is
authoritative for any config. The documented override string also now resets `CONFIG_NAME`,
because `clip_epsilon`, `discount` and `group_num_samples` live only in the config and have no
env var. Verified by parsing both emitted command lines through tyro and comparing **resolved**
configs across 15 fields — `revert == baseline` is `True`. The verifier's `xfail(strict=True)`
marker was removed; the test now passes for real.

Trade-off accepted: the baseline recipe's emitted command line is no longer byte-identical to
the pre-alignment script (it gains these flags at their baseline values). Resolved-config
equivalence is the stronger property and is what the tests now pin.

---

## Confirmed correct (verbatim highlights)

- **Backward compatibility, attacked hardest.** "**Differential against
  `git show HEAD:src/envs/wrappers.py`** (the committed file, not a transcription), 60
  randomized `(terminate, truncate)` scripts × 15 steps: reward value **and Python type**
  identical […] The old class raises `TypeError` on `success_bonus=`, so the bonus tests are
  not tautological." All ten registered configs have `success_reward_bonus == 0.0`; the
  positional `dsrl_env.py:487` call site still works.
- **`fix_mc_returns` premise verified, not assumed** — including a path the change record
  never mentioned: "`QueryFrequencyWrapper`'s post-termination `reward: 0.0` padding
  (`wrappers.py:194-206`) […] **every** entry in `get_max_steps_libero` (220/280/300/400/520)
  plus molmo's 450 is a multiple of `replan_steps=5`, so truncation lands on a chunk boundary
  and emits **no** padding → the array is exactly `[-1.0]*T` → constant → still overwritten."
  The degenerate 1-step-success case (`[72.0]` → MC return `+14400`) "is unreachable:
  `n_windows <= 0 → return`".
- **A4 RNG stream.** "AST: exactly one `critic_success_oversample` `If`, containing exactly 1
  split; 3 splits total in `update()`. Flag off ⇒ zero stream perturbation." Metric leak
  checked; success-buffer schema verified identical; `sample()` kwargs verified against
  `replay_buffer.py:134-138`; both critic states rebound, "required because the shared critic
  jit has `donate_argnums=(1, 2)`".
- **`_grpo_conservative_advantage` is exactly the reference's `_safe_max`** — "differential-
  tested it at n=2 and n=10 against a verbatim transcription of
  `OGPO_public/ogpo/agents/modules/pg_helper.py:455-461` + `:487-489`. Bit-for-bit within
  1e-5."
- **Nothing assumes `num_qs == 2`**; both BroNet ensembles instantiated at 10 on CPU.
- **The shipped tests are real, not tautological** — four of them verified to fail on revert.

## Risks recorded, not fixed

1. **The bonus is the only aligned knob not in the ref config**, so
   `uv run scripts/exp.py pi05_libero_online_ogpo_ref` alone reproduces none of A1. Asymmetric.
2. **Best-of-N collection ignores `critic.reduction`** — `advantage_weighted_sft_learner.py:475-477`
   hardcodes `scores.min(axis=0)`. Newly reachable: collection selects on **min-of-10** while
   the advantage and TD bootstrap use **mean-of-10**. Upstream also mins but with
   `subsample_bon=true` (min of 2 drawn from 10), "so ours is strictly more pessimistic".
3. **A4 goes silently inert after every requeue** — `runtime_state.py:81-91` persists only the
   online buffer; the success buffer rebuilds empty. The fail-fast checks config, not runtime
   availability.
4. **Step-size stacking** — the ref recipe disables `NORM`, `CLIP_SYM` and `MT_ADV` together
   while A1 widens the gap, keeping ε = 0.1 (10× upstream) and a constant LR, whereas upstream
   bounds the step with *both* ε = 0.01 *and* a cosine schedule (out of scope per D3). "The
   change record justifies each omission individually but never the combination."
5. Sign-unanimity at 10 heads is much stricter than at 2 — watch `cons_zero_frac`.
6. The burst does not oversample: success:all ratio during the 1000-step burst is 0.

## Could not verify (verbatim)

> - **Any memory claim.** […] Needs a GPU smoke — **I did not run one and did not request one.**
> - **That A3 works at all under OGPO.** `AdvantageWeightedSFTLearner.sample_actions:313-509`
>   has never executed on the OGPO learner. […] Everything past that — the transform ordering,
>   the prefix-embedding path, the actual Q scoring — needs real π0.5 weights and a simulator.
> - **The literal "byte-identical, 1831 bytes, diff clean" claim** […] the file is
>   **untracked**, so there is no pre-edit version to diff against.
> - **Any run outcome.**

On the third: the implementing session's byte-identity check was against a *reconstructed*
pre-edit script, not a git object. The verifier is right that this is weaker than stated. It is
now moot — D3's fix deliberately gives up byte-identity in favour of resolved-config
equivalence, which is machine-checked.

## Tests

The verifier appended 38 tests to `tests/ogpo/test_verifier_alignment.py` (now 51 functions /
119 parametrized cases / 1178 lines), including differentials against
`git show HEAD:src/rl/value_distribution.py` and `git show HEAD:src/envs/wrappers.py`.

Its final local run: `125 passed, 3 skipped, 1 xfailed`. After the D3 fix the xfail marker was
removed and that test passes normally.
