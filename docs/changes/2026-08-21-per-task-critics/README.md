# Per-task critics for multi-task OGPO

**Tier:** 2 — new jit argument + sharding on `sample_and_advantage`, replay-buffer
schema change, critic `TrainState` / checkpoint layout change, config dataclass
change reaching four learner packages.
**Status:** discovery pass complete, design decisions taken. **No source touched.
No plan approved.** Next step is the plan phase (plan mode), which must build
`PLAN.md` from this record and `BLAST-RADIUS.md`.
**Date:** 2026-08-21 · **Branch at time of writing:** `shashwat/stability-study`

---

## Why

The multi-task frontier (`scripts/ogpo_multitask_4task.sh`) runs four LIBERO tasks
through **one shared Q ensemble and one shared V ensemble**. Both critics are fit on
a single uniformly-sampled buffer that interleaves all four tasks, and a single
scalar value head has to represent four different return distributions at once.

Two existing multi-task knobs mitigate the *symptoms* of that downstream, not the
cause:

- `normalize_advantage_per_task` (`update_actor.py:271-299`) rescales the **advantage**
  by the per-task std after the critic has already produced it. It cannot repair a
  Q that mis-*ranks* actions because it was pulled toward another task's return level
  — it only re-scales whatever ordering the shared critic emitted.
- `balance_success_buffer_tasks` (`ogpo_learner.py:632`) balances the **BC anchor's**
  batch. It never touches the critic.

That leaves the exact failure mode the stability study named as the dominant one.
`reports/stability_study_summary.md` traces the single-task crashes to **direction
failure** — the critic transiently mis-ranks freshly sampled actions after each
collection flood, and the policy then takes ~1,000 well-scaled, well-clipped,
statistically invisible *wrong* steps. A shared critic across tasks adds a second,
permanent source of exactly that mis-ranking: every gradient step from task B's
transitions moves the same parameters that rank task A's actions, and the tasks do
not share a return scale (they differ in length-to-success, hence in the
`-1/(1-γ)`-floored MC return they regress onto).

This change gives each task its own Q and its own V, with **no shared parameters**
and **no cross-task reuse at policy-extraction time**: the advantage for a sample of
task *t* is computed from task *t*'s critics alone.

## What is being built

`rl.critic.num_tasks = T` turns the single Q/V pair into T disjoint Q/V pairs held
in the existing two `TrainState`s. A per-sample `task_index` routes every critic
call — training and advantage — to its own task's ensemble. Everything downstream of
`create_critic(...)` keeps its current shapes, so `summarize_critic_values`,
`critic_values_per_head`, all three `advantage_combination` branches, the digestion
burst, `critic_utd` and the success-oversample update are unchanged.

`num_tasks = None` (the default) is today's code path, bit-identical.

## Decisions taken during discovery

Answered by the maintainer; these are settled inputs to the plan, not open questions.

| # | Decision | Chosen |
|---|---|---|
| D1 | Which critics split | **Both Q and V.** V is not optional: `train_q_step` bootstraps Q's TD target off `value_model(next_obs)` (`update_critic.py:255-258`), so a shared V would leak cross-task value scale into every Q backup even in the `grpo_conservative` arm where V never enters the advantage. |
| D2 | Packaging | **One module holding T disjoint ensembles; forward computes all T and gathers per sample by task index.** Single `TrainState`, single jit, single checkpoint entry. Constraints added by the maintainer — *no shared params*, and *never use another task's critic for a task's policy extraction* — are both structural properties of this shape. Costs T× critic FLOPs; scale target is the current 4-task set, not 40+. |
| D3 | Task identity | **New `task_index` transition field in the replay buffer**, assigned host-side by a `task_description → slot` registry, with the registry **persisted alongside the RL checkpoint**. Unpersisted, a resume would silently permute which critic owns which task — the `_adv_scale` trap (`ogpo_learner.py:92-96`) with a much worse blast radius. |
| D4 | Unknown task | **Fail fast**, with the fix in the message (name the unregistered task, say to raise `rl.critic.num_tasks`). Applies to training *and* to best-of-N collection/eval scoring. |
| D5 | Grad clip | **Per-task clipping.** `optax.clip_by_global_norm(1.0)` (`config.py:146`) is currently one global norm over the whole param tree; with T disjoint tasks in one tree it would make task A's gradient magnitude throttle task B's update. Replaced by one clip per task subtree so each task's optimizer behaves like a standalone critic. |
| D6 | Loss reduction | **Per-task mean, then average over the tasks present in the batch.** A task that is underrepresented in a batch still takes a full-size step. A task absent from a batch contributes exactly zero and does not shrink the others (needs a `max(count, 1)` guard). Uniform buffer sampling is kept — no task-balanced critic batches. |
| D7 | Config surface | **`rl.critic.num_tasks` on `CriticTrainingConfig`** (where the critic is actually constructed, in the AWR base) + a **new registered config** for the per-task OGPO arm + an env-var knob in `scripts/ogpo_multitask_4task.sh`. Default `None` keeps every existing config bit-identical. |
| D8 | Checkpoints | **New-run only; fail fast on structural mismatch.** No tiling of an existing single-critic checkpoint into the T slots. The running mt4 arms are unaffected. |

## Non-goals

- No change to `normalize_advantage_per_task` or `balance_success_buffer_tasks`.
  They stay independent knobs so the per-task-critic arm can be ablated against the
  existing ten multitask arms. (`normalize_advantage_per_task` becomes partly
  redundant — that is an experiment to run, not a default to flip.)
- No task-balanced **critic** batch sampling (D6).
- No warm-start path from a single-critic checkpoint (D8).
- No change to the other five learners' behavior. `num_tasks=None` is their path.
- No attempt to scale past ~8 tasks; the T× critic forward is accepted at T=4.
