# OGPO debugging log — libero_90_44 regression

Session date: 2026-07-20. Written as a handoff so a fresh session doesn't
re-derive (or re-make) the same mistakes.

## Problem statement

OGPO (`scripts/ogpo_libero_babel.sh`) sits at ~5–10% success on `libero_90_44`
for 100k steps and never improves. AWR on the *same task*, same 100k steps,
same critic config, climbs to 65% (collection) / 46.9% (eval).

OGPO previously reached 62% — but on `libero_90_59`, a *harder* task, on
branch `origin/manan/ogpo_ablations`.

## ROOT CAUSE (found late in the session — read this first)

**OGPO never updated its EMA, and data collection uses the EMA. So OGPO
collected every rollout for 100k steps with the frozen, untrained SFT weights.**

```
filtered_sft_learner.py:310   self._ema = device_put(train_state.ema_params)   ← init ONLY
awr_learner.py:729            self._ema = self._ema_update_fn(self._ema, ...)  ← AWR updates
filtered_sft_learner.py:814   self._ema = self._ema_update_fn(self._ema, ...)  ← base updates
ogpo_learner.py               nothing — OGPO overrides update() and called neither
```

Collection reads it: `start_data_collection` (`filtered_sft_learner.py:763-765`)
attaches `self._ema` to `train_state.ema_params`, and `_sample_action`
(`:486-491`) prefers `ema_params` over `params`. `ema_decay=0.999` is set, so
`ema_params` is never `None` and the live policy is never used.
`evaluate_policy` also calls `agent.start_data_collection()` (`collect.py:39`),
so **eval measured the SFT policy too** — OGPO's trained policy was never
evaluated at all.

Origin: commit `32d5bc5 hotfix: ema parameters are not updated in awr update`
(2026-07-01) — **one file, one line**, applied to
`advantage_weighted_sft_learner.py`. OGPO overrides `update()` and was missed.

This explains every symptom without any of the other hypotheses:

| observation | explanation |
|---|---|
| `success_rate` flat 5–10% for 100k steps | SFT base rate; acting policy never changed |
| buffer distribution static | same policy generating data throughout |
| `value_loss` → 0.037, `q_grad_norm` decaying | critic converged on a static distribution |
| `q_mc_loss` falling 539→474 | Q converging to the *fixed* behavior policy's returns |
| AWR `value_loss` 18.31, `q_grad_norm` → 980 | AWR's EMA updates → data shifts → critic chases |

**Caveat:** on `ogpo_ablations` *nobody* updated the EMA (the hotfix postdates
that branch). So the "working" OGPO run was also collecting with frozen weights,
and its 0.5 → 0.625 may be SFT base-rate noise on `libero_90_59` rather than real
improvement. **Do not treat the `ogpo_ablations` config as a known-good target.**
This bug predates the merge; what changed is that AWR got fixed and pulled ahead.

## Reference points

| what | where |
|---|---|
| broken OGPO run | `diverse-data-synthesis/ogpo_sweep/zcyzcm63` |
| working AWR run | `diverse-data-synthesis/openpi/ngw61yte` ("Libero-44: AWR") |
| old working OGPO | `diverse-data-synthesis/ogpo_sweep/8n6j6ofi` (libero_90_59, 9100 steps, sr 0.625) |
| working branch | `origin/manan/ogpo_ablations` — **NOT** `manan/ogpo_agent` (that one is stale) |
| official OGPO impl | `/home/mananaga/OGPO`, see `scripts/ogpo/square.sh`, `square_pg_libero.sh` |

## Tooling

`tests/ogpo/wandb_quantiles.py` was broken and is now fixed.

**The bug:** `run.history(keys=[a, b])` **inner-joins** — it returns only steps
where *every* key is non-null. `actor/*` logs every 25 steps, `eval/*` every
`eval_interval`. Empty intersection → 0 rows → "no `_step` column" error.
`_fetch_history` now fetches per-metric and outer-merges on `_step`.

**Note:** run it via `uv run` from the repo root. A `wandb/` output directory
in the project root shadows the `wandb` package for a bare `python`.

**Metric naming** (`src/training/collect.py`):
- `success_rate/{task}` ← `collect_data:220`, logged every `collect_interval`.
  This is the **training curve**. Use this one.
- `eval/success_rate/{task}` ← `evaluate_policy:113`, every `eval_interval`.

---

## Key measurements

### Training curves (`success_rate/libero_90_44`, 10k buckets)

```
             0-10k  10-20  20-30  30-40  40-50  50-60  60-70  70-80  80-90  90-100k
AWR   works   0.0    0.10   0.40   0.55   0.15   0.50   0.40   0.55   0.60   0.65
OGPO  broken  0.05   0.10   0.10   0.05   0.15   0.0    0.05   0.10   0.0    0.10
```

AWR *started lower* (0.0 vs 0.05) and still took off. OGPO is flat at the SFT rate.

### Actor: working OGPO (5–10k) vs broken OGPO (75–100k)

| field | working | broken | |
|---|---|---|---|
| `clipfrac` | 0.0056 (0.17 early, max 0.997) | **0.0 — every step, 100k** | dead |
| `ratio_std` | 2.28e-3 | 9.11e-5 | 25× |
| `approx_kl` | 5.45e-6 | 1.59e-8 | 340× |
| `advantage_max` | +7.22 | +0.146 | 49× |
| `advantage_min` | −6.60 | −0.049 | 135× |
| `advantage_median` | +0.894 | +0.0028 | 320× |
| `bc_loss` | 0.0017 | 0.0168 | 10× **larger** |
| `pg_loss` | −1.045 (consistent sign) | −0.013 (sign oscillates) | |
| `grad_norm` | 0.079 | 0.0465 | |

PG:BC balance (`|adv_median| / bc_loss`): **526 → 0.17**, a ~3000× inversion.
`bc_coeff = 1.0` in both.

### Critic: AWR vs broken OGPO (last 25k)

| metric | AWR (works) | OGPO (broken) |
|---|---|---|
| `value_loss` | **18.31** | **0.037** ← V collapsed to a constant |
| `q_grad_norm` | 36 → **980** (grows) | 31 → **17.6** (decays) |
| `q_td_loss` | 5.7 → **32.6** (rises) | 6.8 → **6.8** (flat) |
| `q_mc_loss` | 550 → 1535 | 539 → 474 |
| `q_value_mean` | −198 → −195 (drifts) | −199.9 (flat) |

`q_loss == q_td_loss` exactly (td_weight=1, MC term logged but zero-weighted).

**Core symptom:** `Q(s,a) ≈ V(s)` for all `a`. Relative action-discrimination
(`advantage range / |V|`) went **27% → 0.098%**, a ~280× drop.

---

## RETRACTED — do not re-investigate

Each of these was proposed and then disproven this session.

1. **Missing `\` on `--rl.adv_strategy subtract_v`** — was real, user fixed it.
   But both dropped flags were no-ops: `batch_size` default is already 256
   (`config.py:411`), and `value_target_type` is inert (see #5).
2. **The `/3200` log-prob normalizer** — present on `ogpo_ablations` **and**
   matches official. Official has `normalize_denoising_horizon` /
   `normalize_act_space_dimension` (`ogpo.py:206-207, 2516-2519`); its total
   divisor is `(K+1)·H·D` vs our `K·H·D` (its `act_dim = action_dim ×
   horizon_length` already folds in H). Equivalent.
3. **`use_ema_as_old_policy` inert / `ratio ≡ 1`** — `ogpo_ablations:343` also
   offloads `ema_params` during updates. `ratio ≡ 1` is *normal* for this code.
4. **Reward flip `use_time_to_success_as_reward` False→True** — AWR uses the
   same default and works; AWR's critic is *also* pinned near −200.
   (−1/(1−0.995) = −200 is just the never-succeeds value under this reward.)
5. **`value_target_type` one_hot→two_hot** — inert. `make_value_distribution`
   (`value_distribution.py:231`) short-circuits to Gaussian when
   `num_value_bins <= 1`, and the script passes `num_value_bins 1`.
6. **Cold-start / no successes in buffer** — false. OGPO collects 5–15%
   successes throughout; thousands of successful transitions are in the buffer.
7. **`group_num_samples=1` zeroing the advantage** — with `adv_strategy=subtract_v`,
   `_group_baseline` returns **zeros** (`update_actor.py:54-56`), so G=1 is harmless.
8. **`store_success_episodes_only`** — branch default is `False` on *both*
   branches; the working run set it explicitly. Do **not** turn it on now — at
   ~5% success it would starve the buffer.
9. **`collect_interval` / `eval_interval` / `policy.update_interval` asymmetry**
   — these are **identical** in the AWR and OGPO scripts and runs (10000 /
   99999 / 10).

---

## VERIFIED CORRECT — do not re-audit

- **pi0 time direction is 1 → 0** (t=1 noise, t=0 clean), stated at
  `pi0.py:347-348` and consistent at every site (`x_t = t·noise + (1−t)·actions`,
  `dt = −1/num_steps`, `initial_time = ones`, `σ_t = nl·√(t/(1−t))`).
  Official OGPO is the **mirror** (t=0 noise, t=1 clean).
- **The SDE drift is marginal-preserving.** Expanding `pi0.py:162-164` gives
  `mean = x_t + [v_t − (σ²/2)·score]·dt`, which is exactly what Fokker–Planck
  requires for reverse-time (`dt<0`) integration. `scale_diag = σ·√|dt|` is
  correct Euler–Maruyama.
- **Sampling ↔ rescoring alignment is correct.** Both paths call
  `model._get_sde_dist` with the same `(x_t, time, dt, noise_level)`;
  `x_chain`/`times` are step-starts and `x_next_chain` is the sample.
  **The importance ratio is unbiased.**
- **Log-prob math is correct.** `MultivariateNormalDiag` sums over D internally;
  `sum_log_prob` sums over (steps, horizon); divided by `K·H·D`. Self-consistent
  and equivalent to official's reduction.

---

## CONFIRMED divergences from official OGPO

| variable | ours | official | status |
|---|---|---|---|
| `group_num_samples` | 1 | **32** | user: too compute-heavy to fix |
| `adv_strategy` | `subtract_v` (Q−V, separate V net) | **`vanilla`** (group-mean over 32 samples) | open |
| `noise_level` | 0.3 → **now 0.02** | tapered, `cns=0.01` | changed 2026-07-20 |
| rollout sampling | **deterministic** (ODE, `noise_level=0.0`) | **stochastic** (SDE) | open |
| `discount` | 0.995 | 0.99 | open |
| `clip_epsilon` | 0.01 | 0.01 | same |
| `bc_coeff` | 1.0 | 1.0 | same |
| `num_sde_steps` | 10 | 10 | same |

### Why `adv_strategy` matters

Official computes `A(s,aᵢ) = Q(s,aᵢ) − mean_j Q(s,aⱼ)` over 32 samples at the
same state — **same Q network**, so its bias cancels and spread is guaranteed.
Ours computes `Q(s,a) − V(s)` with a **separately trained V**, so the advantage
is the residual of two independently-trained ~200-magnitude predictors. That is
structurally why `advantage_std = 0.023`.

### noise_level: the σ comparison

| | per-step std | s=0.5 |
|---|---|---|
| ours | `nl·√(s/(1−s))·√0.1` | 0.0949 @ nl=0.3 |
| official | `cns·√s` | 0.00707 |

(s = noise fraction = our t = official's 1−t.) 13× gap at nl=0.3.
`log_ratio ∝ 1/σ²`, so that suppressed the ratio ~170× → `clipfrac = 0`.
`nl = 0.022` matches at mid-chain (within ~2× across the chain).

**Two shape caveats:** ours is `√(s/(1−s))` (diverges at the noise end, needs
the `clip(time, 0, 1−dt_abs)` guard); official's `√(1−t)` is bounded by design
so the score singularity cancels analytically. Also official uses σ as a direct
per-step std (**no `√dt`**) while applying `σ²/2` in the drift correction — so
the two σ values are **not the same kind of quantity** and `0.01` cannot be
ported directly.

---

## Minor issues found, not fixed

1. **`v_t` / coefficient time mismatch on step 1.** `compute_v_t(x_t, time)`
   uses `time = 1.0`; `_get_sde_dist` internally clips to `1 − dt_abs = 0.9`.
   Applied identically in both paths → **no ratio bias**. Cosmetic for OGPO.
2. **`preprocess_observation` asymmetry.** `sample_actions` preprocesses
   (`pi0.py:346`); `compute_prefix_cache` → `embed_prefix` does **not**
   (`sampling.py:57`). Dormant while images are already 224×224 with masks
   (consistent with observed `old_lp == new_lp`), but would silently bias the
   ratio if resolution/masks ever change upstream.
3. **RNG reuse**, `sampling.py:149-158`: `noise = normal(rng, ...)` then
   `sample_actions(rng=rng, ...)` — same key seeds both the initial noise and
   the scan. Should split first.

## Blast radius (what a change touches)

| change | affects |
|---|---|
| `rl.noise_level` | **OGPO only** (FlowGRPO has its own field) |
| `src/rl/ogpo/sampling.py` | **OGPO only** (imported by `ogpo/update_actor.py:39` alone) |
| `pi0._get_sde_dist` | **OGPO + FlowGRPO** |

`noise_level > 0` occurs only in `src/rl/ogpo/` and `src/rl/flow_grpo/`.
Every other algorithm — and **every rollout path including OGPO's** — uses the
`0.0` default and never executes `_get_sde_dist`.

---

## Config delta: working OGPO vs broken OGPO

All OGPO-specific hyperparameters are **identical** (`clip_epsilon`, `bc_coeff`,
`num_sde_steps`, `noise_level`, `adv_strategy`, `group_num_samples`, both
`normalize_*`, `use_ema_as_old_policy`, `entropy_coeff`, `adv_clip_min`, `beta`,
`use_bc_regularization`) **except `discount` (0.99 → 0.995)**.

The regression is therefore in the **shared critic + collection recipe**, which
was swapped to the AWR/best-of-N recipe during the merge and never re-validated:

| shared setting | working OGPO | current (= AWR) |
|---|---|---|
| `critic.td_weight_schedule.init_value` | 0 | 1 |
| `critic.pre_training_steps` | 900 | 0 |
| `critic.num_updates_per_batch` | 10 | 1 |
| `critic.use_ema` | False | True |
| `critic.inference_start_step` | 100 | 1 |
| `critic.batch_size` | None | 1024 |
| `collect_interval` | 300 | 10000 |
| `policy.update_interval` | 1 | 10 |
| `discount` | 0.99 | 0.995 |
| `ema_decay` | 0.995 | 0.999 |
| `fix_mc_returns` | False | True |

**AWR tolerates this recipe. OGPO has only ever been shown to work with the
left column.** Caveat: the working OGPO run was on `libero_90_59`, not
`libero_90_44` — but 44 is the *easier* task, which strengthens rather than
weakens the regression conclusion.

---

## Current state (end of session)

`scripts/ogpo_libero_babel.sh` now has `--rl.noise_level 0.02` (was 0.3).
Everything else unchanged. User plans to try `--rl.bc_coeff 0.02` next.

### WARNING: noise_level and bc_coeff multiply

`∇pg ∝ 1/σ²`. Dropping σ 15× (0.3 → 0.02) makes the PG term **~225× stronger
on its own**, before any `bc_coeff` change.

| config | PG:BC (approx) |
|---|---|
| broken | 0.17 |
| working | 526 |
| + `noise_level 0.02` alone | ~38 |
| + `bc_coeff 0.02` too | ~1900 |

`bc_coeff = 0.02` was sized from measurements taken at `noise_level = 0.3`;
those numbers are stale now. **Run the noise change alone first**, then read
`actor/clipfrac`:

- **0.005–0.2** (working run's band) → ratio healthy, may not need `bc_coeff`.
- **saturates near 1** → σ overshot; raise `noise_level` (try 0.05), leave `bc_coeff`.
- **still ≈ 0** → `bc_coeff` is the right lever, size it from fresh numbers.

### Strongly recommended regardless

```
--collect.collect_interval 1000     # currently 10000 → only 10 curve points
--collect.eval_interval 1000        # currently 99999 → 1 eval in 100k steps
```

`success_rate/*` only logs on collection. At 10000 you can't tell a good run
from a bad one until 10k steps in. The working run used 300.

### Risk to watch

BC is currently the only thing anchoring the policy. If `advantage` is mostly
critic noise (`advantage_std 0.023` vs critic drift ~3.0) rather than signal,
cutting `bc_coeff` lets noise drive the policy and success can fall *below* 5%.
A dense `collect_interval` catches that in ~2k steps.

## Ranked next steps (reordered after the ROOT CAUSE finding)

1. **Re-run with just the three learner fixes**, changing nothing else. Every
   hyperparameter conclusion in this log was measured on a policy that was never
   acting in the environment, so they all need re-measuring first. Add
   `--collect.eval_interval 1000` so the result is visible.
2. **Only then** revisit the tuning knobs, in this order:
   - `td_weight_schedule.init_value 0` + `pre_training_steps 900` — puts
     successful trajectories into the critic loss at all (with `td_weight=1`
     the MC term is multiplied by `1 - td_weight = 0`).
   - `noise_level` (already 0.02) — read `actor/clipfrac`; target the 0.005–0.2
     band. Saturating near 1 → raise it to 0.05.
   - `bc_coeff` — only after reading `clipfrac`. Note `noise_level` and
     `bc_coeff` multiply: `∇pg ∝ 1/σ²`, so 0.3 → 0.02 already gives ~225×
     more PG gradient on its own.
   - `adv_strategy vanilla` + `group_num_samples` > 1 — the structurally correct
     fix for a collapsed advantage; 8 or 16 if 32 is too expensive.
3. Fix the dormant `preprocess_observation` asymmetry and the RNG reuse.

**Do not tune anything before step 1 completes.** The measurements this log is
built on describe a frozen SFT policy, not OGPO.

## Structural differences found between the OGPO and AWR learners

`OGPOAgentLearner` inherits everything from `AdvantageWeightedSFTLearner` and
only swaps the actor train step — but it **overrides `update()`**, and three
things that live in AWR's `update()` were silently lost.

| # | difference | status |
|---|---|---|
| 1 | **EMA never updated** — see ROOT CAUSE above | **FIXED** |
| 2 | `rl.critic.batch_size` ignored; critic reused the actor's 256-sample batch instead of an independent 1024 one | **FIXED** |
| 3 | `pre_training_steps` critic-optimizer reset never ran | **FIXED** |

The critic *functions* (`train_q_step`, `train_value_step`) are byte-identical
between the two — OGPO calls the inherited `_update_critics_jitted`. Only the
batch fed in and the surrounding bookkeeping differed.

### Code changes made 2026-07-20

- `advantage_weighted_sft_learner.py` — extracted the `pre_training_steps`
  optimizer reset into `_maybe_reset_critic_optimizers()`. Pure refactor; AWR
  behavior unchanged.
- `ogpo_learner.py`:
  - calls `self._maybe_reset_critic_optimizers()`
  - samples an independent critic batch of `rl.critic.batch_size`
    (falls back to the policy batch size when unset)
  - **`self._ema = self._ema_update_fn(self._ema, self._train_state.params)`**
    after the policy update

All syntax-checked only — **none of this has been run.** Watch for OOM on the
first critic update: `store_prefix_rep=True` means 256 → 1024 quadruples the
per-step sampling, and OGPO's actor update is already the memory-heavy one.
`--rl.critic.batch_size 512` is the fallback.

## Open question

Largely closed by the ROOT CAUSE finding. The remaining unknown is simply
**what OGPO actually does once the EMA is live** — no run has ever measured its
trained policy. Re-run before drawing any further conclusions; most of the
tuning analysis in this log was performed on a policy that was never acting.
