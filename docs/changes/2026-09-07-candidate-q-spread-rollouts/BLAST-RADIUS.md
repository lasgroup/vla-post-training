# Change spec

## Intent

Read-only probe: N rollouts of a task from a restored checkpoint; at every
policy query sample M candidate chunks, score them with the critic, record the
variance of the M reduced Q-scores; execute argmax-Q (`QSPREAD_BON=1`) or
candidate 0 (`QSPREAD_BON=0`). Emit per-rollout plots, a mean-trace plot, and
a raw JSON, into `QSPREAD_OUT_DIR`.

## Files added

- `scripts/probe_candidate_q_spread.py` — the probe. Same ENTRY-swap contract
  as `probe_rollout_gifs.py` / `probe_counterfactual_rollouts.py`: takes
  `scripts/exp.py`'s exact tyro CLI, so the recipe supplies the checkpoint
  (`ARM`/`SEED`/`CONFIG_NAME` → `checkpoint_dir`), the task list
  (`TASKS` → `collect.eval_tasks`), M (`BON_N` → `--rl.n_samples`) and the
  episode length (`EP_MULT`) with no second copy of the flag block.
  Probe-only knobs are env vars (`QSPREAD_OUT_DIR` required, `QSPREAD_EPISODES`,
  `QSPREAD_SEED`, `QSPREAD_BON`), matching `ROLLOUT_*` in the gif probe.
- `scripts/probe_candidate_q_spread.sbatch` — preempt, **two GPUs /
  `FSDP=2`, matching the checkpoint's `fsdp_devices`** (see "Restore
  topology" below), otherwise mirrors `probe_rollout_gifs.sbatch` (mount-hang
  guard, `CKPT_MODE_FLAG=--resume`, `ENTRY` swap through
  `ogpo_multitask_4task_ref.sh`).
- `tests/ogpo/test_candidate_q_spread.py` — pytest-native tests for the two
  pure helpers (`candidate_q_variance`, `mean_over_alive`) and the
  per-group → per-env gather (`gather_records`). The script is loaded with
  `importlib` the way Tier B loads Tier A; the heavy imports are inside
  `main()`-adjacent functions or guarded so the helpers import on CPU.

## Mechanism (verified against source, not the docs)

- `AdvantageWeightedSFTLearner.sample_actions`
  (`src/rl/advantage_weighted_sft/advantage_weighted_sft_learner.py:412`)
  takes the best-of-N path when `rl.n_samples > 1` and
  `training_steps >= rl.critic.inference_start_step` (`:419-421`). The ref
  recipe passes `--rl.critic.inference_start_step 1` and a restored checkpoint
  has `training_steps >= 1`, so M candidates are scored on every query.
  `OGPOAgentLearner` does not override `sample_actions` (grep: only
  `best_of_n_learner.py:350` overrides it, and that learner is not used here).
- With `self._bon_record` set to a list, every prompt group appends a dict
  with `indices` (env slots in this group), `candidates` `(g, M, H, act)`
  float32 in absolute robot-space (`_sample_action` is called without
  `return_prefix_rep` at `:481`, so it passes through `_output_transform`,
  Unnormalize→AbsoluteActions, `filtered_sft_learner.py:629-637`; the comment
  at `:574-580` states it), `scores` `(g, M)` float32 after
  `summarize_critic_values` with `rl.critic.reduction` (`:606-611`),
  `best_idx` `(g,)` int32 (`:612`), plus `state` and `prefix` (and
  `task_index` under per-task critics) which the probe does not read
  (`:621-647`). The returned chunk is `group_actions[arange, best_idx]`
  (`:619`) stored as float32 (`:650`) from the same buffer the record holds
  (`:625`), so the probe's equality check is bit-exact. The probe arms the
  hook before each query and disarms it right after (under `try/finally`),
  as Tier C does at one state (`scripts/probe_counterfactual_rollouts.py:326-331`).
- Multi-group waves: envs are grouped by prompt (`:449-451`), so a wave whose
  envs all run one task yields one record; the probe still gathers by
  `indices` so a mixed wave would not misalign. It raises if the union of
  `indices` over the records is not exactly `range(env_num)`.
- `QSPREAD_BON=0`: the probe ignores the returned argmax chunk and steps with
  `candidates[:, 0]`. Observations are tiled `np.repeat(v, M, axis=0)`
  (`:471`) and the noise is one `jax.random.normal(rng, (g*M, H, D))` draw
  (`filtered_sft_learner.py:621-623`), so candidate 0 of env *e* is the first
  of that env's M iid initial-noise draws — what single-sample collection
  would have executed.
- Rollout driving copies `probe_rollout_gifs.py`'s `reset_wave` (per-env
  seeds `base_seed*7919 + w*G + i`), `rollout` (frame capture dropped,
  per-query score capture added), per-task `max_chunks` from
  `get_max_steps_libero(suite) * episode_steps_multiplier // replan_steps + 2`.
  `env_num = cfg.collect.env_num` by convention with the sibling probes; it
  is not a constraint here — `start_data_collection` sizes `_episode_storage`
  from it (`filtered_sft_learner.py:987`) but nothing in this probe calls
  `save_episode`/`add_data`, so that list is never indexed (verifier
  finding 3).
- Guards copied from the gif probe: `cfg.resume` must be set (otherwise the
  learner is constructed with `overwrite=True` and rmtrees the checkpoint);
  `agent._resuming` must be True (openpi silently downgrades `--resume` on an
  empty dir to a fresh start, which would roll out base weights under the
  trained run's name). Added guard: `cfg.rl.n_samples >= 2`, else there is
  nothing to take a variance over and `sample_actions` would silently take the
  single-sample path with no record.

## Restore topology (found during verification, not in the gif-probe record)

The gif probe's record claimed an FSDP=2 checkpoint restores on 1 GPU because
"orbax restores by the current process's sharding". It does not: openpi's
`restore_state` (`openpi/src/openpi/training/checkpoints.py:91-105`) passes
the `jax.eval_shape` train state as the restore target with **no shardings**
(`init_train_state` returns `state_sharding` separately and
`filtered_sft_learner.py:316-324` never hands it to orbax), so orbax falls
back to the *saved* sharding and maps its device ids onto `jax.devices()` by
id (`orbax/checkpoint/_src/metadata/sharding.py:119-131`). Ids `{0, 1}` on a
1-GPU job → "sharding passed to deserialization should be specified ... Got
None" — exactly how the gif probe's only run died (job 10324589,
2026-09-05, `/home/pchellap/logs/rollout_gif_10324589.out:96-171`). Every
successful restore in the logs is an FSDP=2 training resume. Device *kind* is
not compared, so any two GPUs work. The proper fix — annotating the restore
target with `state_sharding` — touches the sharding contract of every
learner's resume and is Tier 2; not done here. `probe_rollout_gifs.sbatch`
still carries the wrong claim and `--gres=gpu:1`; flagged, not edited.

## Expected behavior after

`sbatch scripts/probe_candidate_q_spread.sbatch` (or `--export=ALL,QSPREAD_BON=0`)
restores the checkpoint, logs the restored step, and writes under
`QSPREAD_OUT_DIR`:

- `<task>/ep<NN>_seed<seed>_<succ|fail>.png` — chunk index vs. variance of the
  M reduced Q-scores, one per rollout;
- `<task>/mean_trace.png` — top panel: mean variance over rollouts alive at
  each chunk index; bottom panel: alive count;
- `q_spread_results.json` — restored step, checkpoint dir, M, BoN flag, and
  per task per episode: seed, success, env steps, chunk count, the
  `(chunks, M)` score matrix, the executed candidate index per chunk, and the
  variance trace. Flushed after every wave so a preempted job leaves partial
  results.

Exactly `QSPREAD_EPISODES` episodes per task: `ceil(N / env_num)` waves are
run and the results are cut to N (the surplus envs in the last wave are
stepped but not recorded — a whole wave costs the same wall time as a partial
one).

## Duplication sweep

- `reset_wave` / `policy_chunk` / `rollout` — third divergent copy after
  `probe_counterfactual_rollouts.py` and `probe_rollout_gifs.py`, per this
  repo's duplication-by-copy convention (OQ-2); neither source exposes them
  as an importable utility. `rollout` here differs materially (arms the hook,
  chooses the executed candidate, records scores).
- The `_bon_record` consumer: Tier C reads `rec[0]` and asserts one group;
  this probe gathers by `indices`. Not a copy.
- No new critic scoring code — the production block is reused via the hook,
  which is the whole point (a reimplementation could diverge from the
  normalize→pad order the critic was trained on).
- Clone families in CLAUDE.md (`update_critic.py` pair, `_pad_last_dim`,
  `_get_on_policy_action`, `init_train_state`, RL-checkpoint block, recipe
  preambles): none touched.

## Inheritance sweep

None. No learner method is overridden or edited; the probe drives
`OGPOAgentLearner` through `sample_actions` / `start_data_collection` /
`end_data_collection` and the pre-existing `_bon_record` attribute. No call to
`update()`, `save_checkpoint`, `add_data`, or `save_episode`.

## Gotchas checked

- `scripts.md` checkpoint-mode precedence: an explicit `CKPT_MODE_FLAG` wins;
  the sbatch sets `--resume` explicitly like the gif probe.
- `rl-learners.md` / CLAUDE.md: `del ema_dev`, EMA ownership, dispatch order —
  not in play (no `update()`).
- Per-task critics (`PER_TASK_CRITIC=1` checkpoints): `sample_actions` then
  requires `task_id`; `policy_chunk` passes it, as in the gif probe.
- `store_prefix_rep` makes `sample_actions` return a tuple; `policy_chunk`
  unwraps it.

## Blast radius

Three new files (`scripts/` ×2, `tests/ogpo/` ×1) and this record. Nothing in
`src/` changes; no config field; no jit, sharding, RNG, or checkpoint contract
touched. Docs: add the probe family row to `docs/code/scripts.md`'s module
map (the family is currently undocumented there) and a tests row in
`docs/code/tests.md`.

## How it will be verified

- `pytest tests/ogpo/test_candidate_q_spread.py` (CPU, seconds) and the full
  `pytest tests/ogpo` for collection health.
- `python -m py_compile` on the script; `bash -n` on the sbatch.
- `DRY=1` render of `ogpo_multitask_4task_ref.sh` with the sbatch's env
  block, confirming `ENTRY`, `--resume`, `--rl.n_samples 8`, the task list.
- GPU + simulator + checkpoint execution cannot run from the login node; the
  verifier states that plainly. The sbatch is proposed, not submitted.
