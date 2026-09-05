# Privileged per-task critic for multi-task OGPO

**Status:** implemented. **Date:** 2026-08-23.
**Entry points:** `scripts/ogpo_privileged.sh`, config
`pi05_libero_online_ogpo_privileged`, learner
`src/rl/ogpo/privileged/ogpo_privileged_learner.py`.

---

## Why

The multi-task OGPO critic is suspected of being the binding constraint on
`scripts/ogpo_multitask_4task.sh`. Three properties bound it:

1. **State.** It sees a *mean-pooled* PaliGemma prefix embedding plus the
   policy's 8-d proprioceptive vector. Whatever the pooling destroys, the critic
   cannot recover — and it is the critic's *ranking* of actions, nothing else,
   that reaches the actor through the advantage.
2. **Sharing.** One Q ensemble and one V ensemble serve all four tasks, fit on a
   uniformly interleaved buffer. Every gradient step from task B moves the same
   parameters that rank task A's actions, and the tasks do not share a return
   scale.
A third property — `train_q_step` bootstraps through `V(s')`, and `V` is itself
regressed onto `Q(s, a_buffer)`, a SARSA-flavoured target — is **deliberately
left alone** on the default arm, so the experiment isolates the two above. It is
available as the `BACKUP=next_action_q` ablation (see below).

This change measures the headroom by removing the first two, using information a
real robot would not have. It is a **diagnostic upper bound, not a deployable
recipe**.

## What it does

| | baseline (`ogpo_multitask_4task.sh`) | privileged (`ogpo_privileged.sh`) |
|---|---|---|
| critic state | mean-pooled prefix ⊕ proprio | simulator state: `robot0_proprio-state` ⊕ `object-state`, zero-padded to 512 |
| critic count | 1 shared Q + 1 shared V | T independent Q + T independent V (T = distinct `collect.tasks`) |
| routing | — | per-sample one-hot gather on a task id recorded at env reset |
| grad clip | one global norm over the whole critic tree | one norm **per task subtree** |
| Q target | `r + γ·V(s')` | **identical** — the same shared `advantage_weighted_sft/update_critic` steps, not a copy |
| actor | — | **identical** |

Every actor-side hyperparameter is byte-identical between the two scripts
(verified by resolving both command lines and diffing the resulting configs:
the only differences are `store_prefix_rep`, `store_privileged_state`,
`privileged_backup`, and the run's name/paths). The critic train steps are
literally the same functions. Any difference in outcome is attributable to what
the critic sees and how it is parameterized.

## Design decisions

Where `docs/changes/2026-08-21-per-task-critics/` (a broader, unimplemented
`rl.critic.num_tasks` design) took a decision, this follows it unless noted.

- **D1 both Q and V split** — followed. Under `privileged_backup="value"` a
  shared V would leak cross-task value scale into every Q backup.
- **D2 one module holding T disjoint ensembles, forward-all-then-gather** —
  followed (`src/rl/networks/task_ensemble_critic.py`). Costs T× critic FLOPs;
  at T=4, `num_qs=2` and a 1024-wide BroNet that is ~4× a cheap MLP, negligible
  against the policy step. The gather is a hard route: the cotangent of every
  non-selected sub-critic is exactly zero (test:
  `test_only_the_selected_task_s_parameters_receive_gradient`).
- **D3 task identity** — **deviates, deliberately.** That design assigns slots
  from a first-seen `task_description → slot` registry and must persist it
  beside the checkpoint, or a resume silently permutes which critic owns which
  task. Here the slot is `collect.tasks`'s de-duplicated index, resolved from
  the config in `privileged_task_ids` and stamped into the observation by the
  env wrapper at reset. The mapping is a pure function of the config, so there
  is nothing to persist and nothing to permute.
- **D4 unknown task** — **deviates.** Held-out eval tasks get id `-1`, which
  one-hots to the zero vector, rather than raising. Eval never touches the
  critic and best-of-N collection is rejected outright (below), so there is no
  path on which a `-1` reaches a decision. Raising would make `HELDOUT=1` unusable.
- **D5 per-task gradient clipping** — followed
  (`clip_by_global_norm_per_task`). openpi is a quarantined submodule, so the
  AdamW core is rebuilt in `per_task_optimizer` from the same config fields
  rather than patching `optimizer.py`.
- **D6 per-task loss reduction** — **not implemented.** Under AdamW a uniform
  1/T rescale of every gradient is cancelled by the per-parameter
  normalization, so the plain batch mean and the per-task mean differ only in
  their transient. Not implementing it also keeps the default arm on the shared
  train steps, unmodified. Recorded here as a known deviation rather than
  silently skipped.
- **The Q backup is held at the baseline's** (`privileged_backup="value"`,
  the default). `next_action_q` — `r + γ·Q_ema(s', a'_π)` with `a'_π` read out
  of the stored trajectory — is built and tested but opt-in. It is *not* a
  policy rollout at TD time (the action was already taken during collection, so
  the runtime cost is one extra critic forward), but it does add a
  `next_actions` buffer column: one action chunk per transition, ~640 MB at
  capacity 500k. The `"value"` default allocates and writes nothing for it, and
  runs the inherited AWR train steps rather than the ones in
  `src/rl/ogpo/privileged/update_critic.py`.
- **D8 new-run only** — followed. The buffer gains two observation columns and a
  `next_actions` transition field, so `restore_shards` of a baseline shard
  directory fails loudly, which is the intent.

Additionally: **best-of-N collection (`rl.n_samples > 1`) is rejected.** That
path scores candidates from inside `sample_actions`, which assembles a
prefix-embedding critic observation this critic cannot read. The script pins
`--rl.n_samples 1` (the mt4 default) rather than exposing the knob.

## Blast radius

Shared code touched, all behaviour-preserving at the defaults:

| File | Change |
|---|---|
| `src/envs/wrappers.py` | `Pi0ObservationWrapper` gains opt-in `privileged_tasks` / `privileged_state_dim`. `None` (every pre-existing call) emits exactly the three policy keys. |
| `src/rl/filtered_sft_agent/filtered_sft_learner.py` | Two hooks: `_buffer_obs_passthrough_keys` (keys that bypass the repack transform) and `_extra_transition_fields`. Both empty in the base. Wires the wrapper's new args from `collect.store_privileged_state`. |
| `src/rl/advantage_weighted_sft/advantage_weighted_sft_learner.py` | Critic construction moved verbatim into an overridable `_build_critic_spec` hook returning a `CriticSpec`. |
| `src/rl/advantage_weighted_sft/update_critic.py` | Both init functions take an optional `tx=`; `None` builds it from the config as before. |
| `src/rl/ogpo/ogpo_learner.py` | The drop-keys gate, repeated at four call sites, moved into a `_critic_drop_obs_keys` hook. |
| `src/rl/ogpo/update_actor.py` | `critic_prefix` widened to accept a dict, used verbatim as the critic observation. Array/`None` branches unchanged. |
| `src/training/config.py` | Two inert `CollectionConfig` fields, `OGPOPrivilegedLearnerConfig`, one registered config. |
| `scripts/exp.py`, `scripts/eval.py` | Dispatch the new config **before** `OGPOSFTLearnerConfig` (it is a subclass). |

New: `src/rl/privileged_state.py`, `src/rl/networks/task_ensemble_critic.py`,
`src/rl/ogpo/privileged/`, `scripts/ogpo_privileged.sh`,
`tests/ogpo/test_privileged_critic.py`.

Evidence the default path is unchanged: resolving
`scripts/ogpo_multitask_4task.sh`'s emitted command line before and after this
work produces configs differing only by the two new `collect` fields at their
inert defaults; `tests/ogpo/test_verifier_alignment.py` has the same
pass/fail set as on HEAD (13 pre-existing failures from a missing `colorama` and
from git-HEAD differential tests, unrelated).

## What to read in the run

`critic/q_mc_corr` — Pearson correlation between the critic's Q and the observed
MC return over the batch. The advantage consumes the critic's *ordering*, so
this, not `q_value_mean`, is the number this experiment is about. The metric
schema is identical to the baseline arm's (same train steps), so the two series
compare step for step. Under `BACKUP=next_action_q` one extra key appears,
`critic/q_bootstrap_mean`, exposing the target's level — how a collapsed backup
(pinned at the never-succeeding fixed point `-1/(1-γ)`) shows itself.

## Ablations the script supports

- `BACKUP=next_action_q` — same critic, but the Q target bootstraps on the
  policy's action at s' instead of through V. Costs a `next_actions` buffer
  column (~640 MB at capacity 500k). Run it only after the default arm has
  shown whether the representation was the binding constraint.
- `PSTATE_DIM` — raise if a task's `object-state` overflows 512 (collection
  raises with the required width in the message rather than truncating).
- `CRITIC_HID`, `NUM_QS` — per-task width and ensemble size.

## Not verified here

No GPU and no simulator were available. The following are argued, not measured:
the LIBERO observation actually carrying `robot0_proprio-state` / `object-state`
at collection time (the extraction raises by name if not), end-to-end memory and
wall-clock, and anything requiring real π0.5 weights. Everything else — the
extraction, the wrapper, the task-ensemble routing and gradient disjointness,
the per-task clip, the buffer schema, the next-action alignment, the critic
train steps end to end, and the config/dispatch wiring — is covered by
`tests/ogpo/test_privileged_critic.py` (50 tests, CPU, ~30 s).
