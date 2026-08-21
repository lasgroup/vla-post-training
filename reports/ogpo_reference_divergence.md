# OGPO here vs. the reference implementation

**Date:** 2026-08-20 · **Reference:** `/home/pchellap/Projects/SafeVADAR/OGPO_public`
(`ogpo/agents/ogpo.py`, `ogpo/agents/modules/{pg_helper,q_helper}.py`,
`ogpo/configs/algos/ogpo.yaml`, `scripts/ogpo/square_image_paligemma.sh`) ·
**Ours:** `src/rl/ogpo/`, `src/training/config.py`, `scripts/ogpo_multitask_4task.sh`

The closest analogue to our setting in the reference is
`scripts/ogpo/square_image_paligemma.sh` — frozen PaliGemma/SigLIP vision backbone
restored from a `pi05_libero` checkpoint, image observations, MLP actor/critic heads on
top. Where that script overrides `ogpo.yaml`, the script value is the one quoted, because
it is the recipe that was actually tuned for a PaliGemma-encoded policy.

Everything below is read from source. Nothing is inferred from the paper.

---

## A. In the reference, absent here

### A1. The reward has an explicit success bonus. Ours does not.

Reference (`envs/robomimic_utils.py:436-468`):

```python
raw_obs, reward, done, info = self.env.step(action)
reward = reward - 1.0            # [0,1] -> [-1,0]: -1 per step
is_success_step = reward > -0.5
if is_success_step:
    reward += 5.0                # +4 net on the success step
    ...
# and repeated while steps_since_succ <= post_success_steps  (=8 in the paligemma script)
```

So the reference reward is **−1 per step, +4 on each of up to 9 success steps**, i.e. a
terminal bonus of up to **+36**. With γ = 0.99 its failure fixed point is −1/(1−γ) = −100,
so success is worth roughly 70–100% of the entire value range.

Ours (`src/envs/wrappers.py:236`):

```python
time_to_success_reward = 0.0 if terminate else -1.0
```

**Success is the mere absence of the penalty. There is no bonus.** With γ = 0.995 the
failure fixed point is −200 and a success at the logged mean length (~295 steps) returns
−154.5 — a **45-point gap on a 200-point scale, 22%**, against the reference's ~70–100%.

This is the direct cause of `findings.md` §10b: `q_value_mean` pinned at −199, TD loss RMS
4.5, MC loss RMS 51. We removed the part of the reward that carries the signal.

### A2. Q-target variance reduction over sampled next-actions

`q_variance_reduction=true`, `q_vr_num_samples=8`, `q_vr_reduction=mean`
(`ogpo.py:1297-1298`, `q_helper.reduce_q_over_samples`). The TD target's `next_q` is the
**mean over 8 sampled next-actions**, not a single sample. Default `false` in `ogpo.yaml`
but explicitly **on** in the PaliGemma recipe.

We have no equivalent — one next-action sample per target. This is target noise landing
directly on the 0.3-wide within-state signal of §10c.

### A3. Best-of-N is the *behaviour* policy during collection

`best_of_n=8`, `subsample_bon=true`; `online_rl_runner.py:517-518` swaps the rollout method
to `sample_actions_BON`. **The reference's replay buffer is filled with critic-selected
actions.** Ours is filled with a single unfiltered policy sample.

### A4. Successes are oversampled into the critic 2×

**First, what is *not* a divergence:** the reference's critic never sees demonstration
data either. `offline_ratio=0.0` in **all 15** `scripts/ogpo/*.sh` recipes;
`online_rl_runner.py:374-375` *raises* if a pure-online algorithm is given a nonzero
value; `train_dataset` is used only for a shape template and then set to `None` and
garbage-collected (`:383-386`); the replay buffer is created **empty** from that template
(`:405-406`). `calql_steps=0` and `q_warmup_steps=0` everywhere (two image recipes use
40,000 *online* env steps of critic warmup, not offline data), and `bc_q_steps=0` in all
three PaliGemma recipes. **We match upstream on this point.**

What *does* differ is the mixture. With `use_success_buffer_q=true`
(`ogpo.py:1567`, `1581-1585`) the critic runs **two TD updates per step** — one on the
ordinary replay-buffer batch and one on a **success-only batch** — via
`OGPOAgent.critic_update_sb(agent, (batch, batch_success), ...)`. Note the reference's
"success buffer" is not a separate buffer at all: it is a masked view of the same online
replay buffer (`_sample_success_batch` → `replay_buffer._traj_success_mask`,
`create_success_buffer_batch`, `online_rl_runner.py:151-162`).

So the reference **oversamples online successes into the critic by 2×**, directly
counteracting the failure-majority imbalance. On in 9 of 15 recipes, including all three
PaliGemma ones.

Here the success-masked batch feeds **only** the BC anchor (`ogpo_learner.py:471-481`);
the critic draws exclusively from `_online_data_buffer` (`:305-306`, `:374-375`,
`:406-407`) with no reweighting, so ~74% of its targets are failures pinned at exactly
−200 by `fix_mc_returns`.

### A5. Learning-rate drop and optimizer reset at the BC → online handoff

`online_rl_runner.py:420-421` calls `reset_optimizers_with_lr()` (`ogpo.py:2640-2685`) at
the start of the online phase:

- actor: **3e-4 → `ppo_lr` 4.5e-5** (6.7× drop), fresh optimizer, **cosine schedule**
  warmup 2000 / decay 50000 / end 2e-5;
- critic: cosine warmup 500 / decay 5000 / **end 1e-8**;
- Adam moment state is discarded.

Ours: a single **constant** 2.5e-5 actor LR and 1e-4 critic LR for the whole run, no reset,
no warmup, no decay. `pg_start_step` only multiplies the advantage by zero
(`ogpo_learner.py:531`). So at step 20,000 an Adam state whose second moments were
accumulated on a BC-only gradient of norm 0.049 suddenly receives a policy gradient of norm
~0.40 (§10e) at an unchanged LR. That is a plausible additional contributor to the
20-point crash at the first post-ramp eval.

*(Note: the reference's `ppo_lr` is a **phase** LR, not a per-term one — after the reset the
PG and BC gradients share one optimizer, as they do here.)*

### A6. `clip_bc`

`clip_bc: true`, `clip_bc_threshold: 0.45`, `clip_bc_wrt: "ODE"` — on by default in
`ogpo.yaml` and in the PaliGemma script. No equivalent here.

### A7. A slow reference policy and Q-uncertainty-driven pessimism

`pi_slow` (Polyak EMA at `tau_slow=5e-4`, ~100× slower than the target net) underpins
`chi_po` (χ²-pessimism), `kl_reg` (reverse KL) and `fwd_kl_reg` (forward KL via BC), with
β annealed by the **Q-ensemble spread** (`compute_chi_po_beta`: `β = β₀·std_M(Q)/target`).
All off in the PaliGemma recipe, but this is the built-in escape hatch for exactly our
failure mode — back off the policy gradient when the critic ensemble disagrees. We have no
such mechanism, and with `num_qs=2` we could not compute a meaningful ensemble spread even
if we wanted to.

### A8. SDE→ODE score correction

`error_correct_sde_to_ode: true` ("preserves the marginal of the BC ODE"). No counterpart
found in `src/rl/ogpo/sampling.py`.

---

## B. Here, absent in the reference (our inventions)

1. **`normalize_advantage_per_task`** (`update_actor.py:302-322`). No counterpart.
2. **`normalize_group_advantage` / `adv_scale`** — the EMA-quantile normalizer
   (`ogpo_learner.py:508-525`). A grep for `adv_scale`/quantile across `ogpo/agents/` in the
   reference returns **nothing**. The reference's only advantage normaliser is
   `normalize_group` (divide by group std), which is **`false` by default** and lives only
   in the `vanilla` branch of `compute_ogpo_advantages`.
3. **`adv_clip_sym`** — the reference has only `adv_clip_min` (default `null`), a one-sided
   floor.
4. **Post-collection critic burst**, **`critic_utd`**, **`burst_use_mc_targets`**,
   **`td_weight_schedule`** blending. The reference has `mc_regression` (a separate MC loss
   term, default off) but no burst concept — it has no collection flood to digest.
5. **`pg_start_step` / `pg_ramp_steps`** — an in-run advantage-muting ramp. The reference's
   analogue is a *phase* boundary: 500k offline BC steps, then the online phase with A5's
   optimizer reset.

Items 1–3 are what §10d measured: two normalisers, neither of which exists upstream,
rescaling a collapsed critic's output to a fixed std of 0.36.

---

## C. Same mechanism, different value

| knob | reference (PaliGemma recipe) | ours | comment |
|---|---|---|---|
| `num_qs` | **10** | **2** | 5× fewer heads |
| `q_agg` / `critic.reduction` | **mean** | **min** | mean reduces ensemble variance; `min` over 2 heads is the noisiest reduction available and adds pessimism bias |
| `group_num_samples` (G) | **32** | **8** | 4× smaller group |
| `adv_strategy` | **conservative** | vanilla (`CONS=0` in 9 of 10 arms) | we ran the reference's image-recipe default *off* |
| `clip_epsilon` | **0.01** | **0.1** | 10× looser per dimension |
| `discount` | 0.99 | 0.995 | credit horizon ~100 → ~200 steps |
| action chunk (`horizon_length`) | 4 | 10 | |
| action dim | 7 (square) | 32 | |
| ⇒ log-prob divisor K·H·D | 10·4·7 = **280** | 10·10·32 = **3,200** | 11.4× |
| ⇒ **absolute trust region** (ε × divisor) | 0.01·280 = **2.8 nats** | 0.1·3200 = **320 nats** | **114×** |
| actor LR, BC phase | 3e-4 | 2.5e-5 | |
| actor LR, PG phase | 4.5e-5, cosine → 2e-5 | 2.5e-5 constant | |
| critic LR | 3e-4, cosine → 1e-8 | 1e-4 constant | |
| `bc_coeff` | 1.0 | 1.0 | same value; the difference is A5, not this |
| SDE noise | tapered σ√(1−t), σ = 0.01 | σ ∝ √(t/(1−t)), `noise_level` 0.02 | different functional form **and** 2× magnitude — see caveats |
| online budget | **2,000,000** env steps | 100,000 train steps | ~20× |
| update cadence | **1 critic + 1 actor update per env step** (`utd_q=utd_pi=1`) | 1,000 actor updates per 10k-step collection flood, on frozen data | |
| `start_training` | 20,000 env steps before any update | policy from step 900 | |
| offline BC pretrain | 500,000 in-repo steps | π0.5 SFT checkpoint | |
| `clip_grad_norm` | 1000 (effectively off) | 1.0 | ours clips harder but never binds (measured grad norm 0.36) |

**Not a divergence:** `normalize_denoising_horizon` and `normalize_act_space_dimension` are
`true` in *both* (`ogpo.yaml`, `ogpo.py:185-186`). The per-dimension normalisation is
inherited. What diverges is what it is combined with — `clip_epsilon` and the action
dimensionality — which is why the absolute trust region differs by 114×.

---

## D. Mapping the divergences onto what we measured

| our measurement (`findings.md`) | most likely upstream cause |
|---|---|
| §10b — critic pinned at −199, MC RMS 51 | **A1** (no success bonus) — the reward carries ~22% of the signal the reference's does — compounded by **A4** (no success oversampling, so 74% of targets sit on the pinned value) |
| §10c — within-state advantage 0.3 wide on a 200 scale | **A1**, plus **C** `num_qs` 2/`min` and **A2** (no target variance reduction): three independent sources of critic noise the reference removes |
| §10d — normalisers deliver a fixed 0.36 regardless of quality | **B1–B2** — both normalisers are ours; the reference does not normalise the group advantage |
| §10f — `clipfrac_upper` ≡ 0, guardrails never fire | **C** — a 114× larger absolute trust region |
| §10a/§10e — PG 7–12× the BC anchor, −10.6 points | **A5** (no LR drop or optimizer reset at the handoff) and **C** (1,000 blind updates vs 1 per env step) |
| §5a/§5b/§6 — bursts, MC targets, data volume all null | **B4** — the burst is a local construct compensating for a collection cadence the reference does not have |

---

## E. Ranked changes, cheapest first

All config-level unless noted.

1. **Add the success bonus.** Mirror `robomimic_utils.py:451-468`: keep −1/step, add a
   positive terminal bonus, optionally sustained over `post_success_steps`. This is the
   root of §10b and nothing downstream can compensate for it. Note this is *stronger* than
   `--collect.use_time_to_success_as_reward false`, which replaces the time penalty rather
   than adding a bonus on top of it.
2. **`num_qs` 2 → 10, `critic.reduction` min → mean.** Needs a VRAM check — our heads are
   BRONet-1024 — but it is the reference default and it attacks §10c directly.
3. **`clip_epsilon` 0.1 → 0.01.** Brings the absolute trust region from 320 nats to 32; the
   reference's is 2.8.
4. **Turn off `normalize_group_advantage` and `normalize_advantage_per_task`** — or at
   minimum stop treating §7 as a validated result, since neither knob exists upstream.
   Arm `b` already ran with both off (post-PG 26.9 vs `ncb` 28.6, i.e. no worse).
5. **`adv_strategy` → conservative (`CONS=1`)**, but only *after* item 2 — sign-unanimity
   across 10 heads is a real gate; across 2 it is nearly a coin flip (we measured
   `cons_zero_frac` rising 8.7% → 22%).
6. **Drop the actor LR and reset the optimizer at `pg_start_step`**, with warmup. Small code
   change in `ogpo_learner`; mirrors A5.
7. **`q_variance_reduction`** — average the TD target over 8 sampled next-actions (A2). New
   code, but contained to `update_critic.py`.
7b. **Oversample successes into the critic (A4)** — a second success-only TD batch per
   update. We already build exactly that batch for the BC anchor
   (`ogpo_learner.py:471-481`); it just never reaches the critic.
8. **Shorten `COLLECT_INT`.** We cannot reach 1 update/env-step with a 3B VLA, but 1,000
   blind updates per collection is 3 orders of magnitude from the reference's 1.

## F. Caveats

- The reference's tuned recipes are **state / low-dim Robomimic** with small MLP actors, or
  image observations through a *frozen* PaliGemma encoder feeding **MLP heads** — the
  policy being optimised is small. We optimise a 3B π0.5 action expert. Not everything
  ports.
- **2,000,000 online env steps at 1 update/step is out of reach for us.** Some of the
  reference's stability may simply be a consequence of that budget and cadence, and no
  config change here will reproduce it.
- `num_qs=10` has a real memory cost; our 2 may have been a VRAM decision. Verify before
  changing.
- **The noise-schedule comparison is not settled.** The functional forms differ
  (σ√(1−t) vs σ√(t/(1−t))) and the time conventions run in opposite directions
  (`sampling.py:13-16`: dt = −1/num_steps, t: 1 → 0). They may be the same schedule
  written two ways, or not. This needs a proper check before anything is changed.
- I have not verified `clip_bc` (A6), `error_correct_sde_to_ode` (A8) or the `pi_slow`
  machinery (A7) beyond confirming they exist and have no counterpart here.
