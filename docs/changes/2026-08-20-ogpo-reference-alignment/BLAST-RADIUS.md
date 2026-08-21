# Blast radius — OGPO reference alignment

Seeded from `docs/code/`, **verified against source** with grep/reads on 2026-08-20.
Every `file:line` below was opened during this pass.

---

## 1. Duplication sweep (CLAUDE.md rule 1)

The known clone families, checked against this change set:

| clone family | hit by | note |
|---|---|---|
| `advantage_weighted_sft/update_critic.py` ↔ `best_of_n/update_critic.py` (11 near-identical helpers) | **A2**, **C8** | Any TD-target change must be applied to **both** or the BoN comparison silently diverges from OGPO. `dsrl/update_critic.py` is a third copy but DSRL is unwired (sharp edge) — mark, don't fix. |
| `summarize_critic_values` / `critic_values_per_head` (`advantage_weighted_sft/update_critic.py:72-95`) | **C7**, **C8** | Consumed by `ogpo/update_actor.py`, `flow_grpo/update_actor.py`, `advantage_weighted_sft/update_actor.py`, `dsrl/update_actor.py`. `critic_reduction` is threaded as an argument (`:72`), so `mean` is already supported at `:80` — C8 needs **no** code change, only a config value. |
| `stability_study.sh` ↔ `ogpo_multitask_4task.sh` (~60-line env preamble, already diverged) | all Group-1 items | The multitask script has **no** `UTD` var and hardcodes `--collect.eval_interval 10000` at `:185`. New knobs must be added to both or the single-task arms stop being comparable. |
| `best_of_n_learner.py:326-516` BoN scoring block ↔ `advantage_weighted_sft_learner.py` | **A3** | A3 is a port of this block into OGPO's `sample_actions`; it would create a **third** copy. Prefer extracting a shared helper — but that is itself Tier 2 (touches BoN). |

## 2. Inheritance sweep (CLAUDE.md rule 2)

- **`CriticTrainingConfig` (`src/training/config.py:124-155`) is shared**, not per-learner.
  `num_qs`, `num_vs`, `reduction` live there and are inherited by
  `AdvantageWeightedSFTLearnerConfig` → MPO → FlowGRPO → **OGPO**, and independently by
  `BestofNLearnerConfig`. **Changing the dataclass defaults changes all nine registered
  configs, including your colleague's BoN recipe.** C7/C8 must be set on the OGPO config
  object or via the shell recipe, never on the shared default.
- **`TimeToSuccessAsRewardWrapper` is applied in the base class**
  (`filtered_sft_learner.py:62-63`), so **A1 reaches every learner**: filtered SFT, AWR, MPO,
  FlowGRPO, Best-of-N, OGPO. There is no per-learner override point. This is the single
  widest-reaching item in the change set.
- **OGPO overrides `update()`** and therefore owns its own EMA advance (OQ-10,
  `ogpo_learner.py:596-601`). A5 rebuilds the actor optimizer inside that override; the EMA
  advance and `del ema_dev` (`:505`) must survive intact.
- `FilteredSFTLearner._make_env` (`:55-80`) is the only env-construction site — A1 edits the
  wrapper it instantiates, not the call site.

## 3. Gotchas checked (CLAUDE.md rule 3)

| documented gotcha | interaction |
|---|---|
| `isinstance` dispatch order is load-bearing (`scripts/exp.py:74-85`) | untouched — no new config classes proposed. |
| `del ema_dev` is load-bearing, ~34.7 → ~46 GiB without it (`ogpo_learner.py:505`) | **C7 raises the floor under it.** Memory smoke required. |
| `_adv_scale` is not checkpointed (`ogpo_learner.py:92-96`) | **C4 resolves this gotcha** — with `normalize_group_advantage` off there is no `_adv_scale`. If C4 lands, delete the gotcha entry per CLAUDE.md step 4. |
| Observation preprocessing order replicated in 3 places | untouched. |
| `replay_buffer.save_shard` non-determinism (`replay_buffer.py:200`) | untouched. |
| Molmo transform ordering is positional (`filtered_sft_learner.py:373-383`) | untouched. |

## 4. Per-item blast radius

### A1 — success bonus (widest)
- **Edit site:** `src/envs/wrappers.py:230-237` (7 lines).
- **Reaches:** all six learners (§2), every future run's reward semantics.
- **`fix_mc_returns` (`filtered_sft_learner.py:749-751`)** gates on
  `np.all(reward == reward[0])`. Failures stay constant → still overwritten to −200.
  Successes were already non-constant (0.0 on the terminating step) → unchanged behaviour.
  **No edit needed, but verify the assumption holds for whatever bonus shape is chosen.**
- **`get_value_bounds` (`src/rl/value_distribution.py:129-134`)** returns
  `upper = 0.0` for the time-to-success reward. **A positive bonus makes Q exceed 0 and this
  bound wrong.** Latent today (`num_value_bins = 1` → Gaussian, bounds unused) but it must be
  fixed or it becomes a silent trap the moment anyone sets `num_value_bins > 1`.
- **`post_success_steps` is structural, not a parameter.** `LiberoWrapper.step`
  (`src/envs/libero.py:55-58`) returns LIBERO's own `done`, which fires on success.
  Continuing past success requires suppressing termination in the wrapper chain. Treat as a
  separate, larger change.
- **Invalidates:** all 10 multitask runs, the stability study, and the BoN runs as
  comparators.

### A2 — Q-target variance reduction
- **Edit site:** `advantage_weighted_sft/update_critic.py` around `:255-257`
  (`q_backup_reduction`, `value_model(next_observation)`), **plus the BoN clone**.
- Needs a policy sample inside the critic step → **new RNG split arity** → Tier 2 by CLAUDE.md
  §6 even though the diff is small.
- **Cost:** 8 extra policy forwards per critic step at batch 1024.

### A3 — BoN collection
- **Edit site:** OGPO `sample_actions` (currently inherited from `FilteredSFTLearner`).
- Third copy of the BoN scoring block (§1). Changes the *collected data distribution*, so it
  invalidates run-to-run comparison the same way A1 does.

### A4 — success oversampling into the critic (cheapest real win)
- **Edit site:** `ogpo_learner.update()` only. The success batch already exists at
  `:471-481`; it needs routing to `self._update_critics_jitted` as a second call.
- No jit signature change (same function, second invocation). **Tier 1 in isolation.**
- Reference mechanism: `ogpo.py:1581-1585` `critic_update_sb(agent, (batch, batch_success),
  success_flag, rng2)`; on in 9/15 recipes including all three PaliGemma ones.

### A5 — LR drop / optimizer reset at the handoff
- **Edit site:** `ogpo_learner`, around the `pg_start_step` gate (`:531`).
- **Touches `TrainState.tx` and `opt_state` → checkpoint layout → Tier 2.**
- A resumed run that crosses `pg_start_step` must not double-reset. There is precedent for
  the reset pattern at `best_of_n_learner.py:591-614` (`pre_training_steps` optimizer reset).
- Reference: `reset_optimizers_with_lr` (`ogpo.py:2640-2685`), called once at
  `online_rl_runner.py:420-421`, guarded by `if not is_resumed`.

### C1–C6 — pure config
- No code. Every one is already an env var in `ogpo_multitask_4task.sh`
  (`CONS`, `NORM`, `CLIP_SYM`, `MT_ADV`) or a `--rl.*` flag. **C4/C5/C6 have already been run
  together as arm `b`** (post-PG 26.9 vs `ncb` 28.6 — no worse).

### C7/C8 — critic ensemble
- C8 (`reduction: mean`) is **config-only**; `summarize_critic_values:80` already implements it.
- C7 (`num_qs`/`num_vs` 10) is config-only *in code* but see Feasibility in `README.md`.
- Must be set on the OGPO config, **not** `CriticTrainingConfig`'s default (§2).

### C9 — G = 32
- Config-only. Interacts with `policy_grad_accum` and the five jits' batch shapes; a recompile,
  not a signature change. Cost is the concern, not correctness.

## 5. The four open decisions, framed

1. **A1 mechanism and magnitude.** The reference is −1/step **plus +4 on the success step**,
   repeated over `post_success_steps` (=8), with γ = 0.99 → failure floor −100, bonus up to
   +36 (~36% of the floor's magnitude, and it flips the sign of the return). Our γ = 0.995 →
   floor −200. Options: (a) terminal bonus only, magnitude scaled to preserve the reference's
   bonus/floor ratio; (b) literal +5 as upstream; (c) bonus **and** `post_success_steps`
   (structural, see A1 above); (d) sparse-positive instead
   (`use_time_to_success_as_reward=false`, bounds auto-switch to [0,1] at
   `value_distribution.py:131-134`). These are materially different MDPs.
2. **Scope.** Group 1 alone is a ~15 h single run and changes nothing structural. Groups 1+2
   is the natural "cheap alignment". Group 3 is where A1 lives — the item most likely to
   matter and the most invasive.
3. **A5 target LRs.** Our actor runs at a **constant 2.5e-5**, already *below* the
   reference's `ppo_lr` of 4.5e-5, while the reference's *BC-phase* LR is 3e-4 — 12× ours.
   So "match the reference" could mean (a) keep 2.5e-5, add only the optimizer reset +
   cosine warmup/decay; (b) mirror the phase structure — BC phase 3e-4, PG phase 4.5e-5; or
   (c) leave A5 out.
4. **Compatibility.** A1 and C1/C2/C7/C8 make new runs non-comparable with all ten existing
   arms *and* with the colleague's BoN runs. Options: change in place on this branch; or add a
   parallel registered config (e.g. `pi05_libero_online_ogpo_ref`) plus a parallel shell
   recipe, keeping the current stack runnable. The second costs a config-registration entry
   and a script, and preserves every existing comparator.

## 6. How this will be verified

Per CLAUDE.md step 3, once a plan is approved:

- **Pytest-native, on dummy variants** (`tests/ogpo/` patterns): the reward wrapper's
  bonus/termination logic; `get_value_bounds` under a positive reward; the A4 two-batch
  critic path's metric schema; `summarize_critic_values` under `reduction="mean"` with
  `num_qs=10`.
- **Differential test required for A2** — a verbatim copy of the pre-change TD target vs the
  variance-reduced one at `q_vr_num_samples=1`, which must be bit-identical
  (`tests/ogpo/test_split_equivalence.py` pattern).
- **`DRY=1 bash scripts/ogpo_multitask_4task.sh`** for every Group-1 flag.
- **Cannot be verified locally:** any memory claim about C7/C9, any claim about run outcomes.
  A GPU memory smoke is a separate, explicitly-requested step — **no run without permission.**
- `pytest tests/ogpo` (23 tests, CPU, seconds). `pytest tests/` still fails at collection
  (OQ-4) and is out of scope unless the change touches `tests/nnx_networks/`.
