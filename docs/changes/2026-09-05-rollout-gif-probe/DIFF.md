# Diff

Two new files, no edits to existing code.

## `scripts/probe_rollout_gifs.py` (new)

Restores `OGPOAgentLearner` from `_config.cli()` (hard guard: raises if
`not cfg.resume`, and a second guard raises if `not agent._resuming` — added
after verification found openpi silently downgrades a `--resume` against an
empty/missing checkpoint dir to a fresh start, which would otherwise roll out
untrained base weights labeled as the trained checkpoint), builds the
multi-task env the same way `scripts/exp.py` does
(`make_env(cfg, cfg.collect.eval_tasks, num_devices=1)` +
`filtered_sft_wrap_env(..., env_num=cfg.collect.env_num)` — task switching
goes through `LiberoWrapper.reset(options={"task_id": ...})` rather than a
synthetic scene). Note `env_num=cfg.collect.env_num`, NOT `eval_env_num` —
`exp.py`'s own eval loop uses `eval_env_num`, but this probe must use
`collect.env_num` because `start_data_collection` sizes `_episode_storage`
from it (the same constraint `probe_counterfactual_rollouts.py` guards on
explicitly). The two happen to both be 8 in this run's recipe, so this has no
effect here, but the two are NOT the same knob (verification finding 2).

For each task in `cfg.collect.eval_tasks`, runs `ROLLOUT_EPISODES // env_num`
waves of `env_num` parallel episodes, each env in a wave seeded DIFFERENTLY
(`base_seed*7919 + w*env_num + i`, matching `stage4_trained_instructions.py`'s
convention) — an earlier version reused Tier C's single-shared-seed
`reset_wave`, which is correct for Tier C's controlled-comparison purpose but
here made every episode in a wave start from the same initial state, i.e. the
episode count was cosmetic (verification finding 1, fixed). Captures the
agentview frame (`obs["observation/image"][e][-1]`) every step and writes one
GIF per episode plus a `rollout_results.json` summary (step restored,
per-task per-episode success/steps/gif path), flushed after every wave.
`max_chunks` is computed per task from that task's own LIBERO suite (parsed
from the task id), not hardcoded to `libero_90` — `TASKS` is an exposed
override and a non-`libero_90` task has a different max-step map
(verification finding 4, fixed).

`policy_chunk`/`reset_wave` mirror `probe_counterfactual_rollouts.py`'s
same-named helpers (candidate-return tracking dropped — this probe wants
frames, not counterfactual returns). `rollout`'s frame-capture loop is
adapted from `stage4_trained_instructions.rollout` (side-predicate judging
dropped, task-id-based reset instead of BDDL-path-based). `write_gif` is
copied verbatim from the same source.

No calls to `agent.update()` or any checkpoint-write path — read-only
w.r.t. the restored checkpoint, same guarantee Tier C and stage 4 give.

## `scripts/probe_rollout_gifs.sbatch` (new)

`preempt`, `--gres=gpu:1`, mirrors `probe_counterfactual_rollouts.sbatch`'s
`SRC` mount-hang guard (`timeout 120 ls -d "$SRC"`) against
`$REAL_BASE/pi05_libero_online_ogpo_ref/mt4_mt2_b128_ep2_s0`. Sets
`ENTRY=scripts/probe_rollout_gifs.py`, `CKPT_MODE_FLAG=--resume`, `GPU=0`,
`ARM=mt2_b128_ep2 SEED=0 TASKS="libero_90_38 libero_90_82" EP_MULT=2
BATCH=128`, and the four `ROLLOUT_*` knobs (all overridable via
`--export=ALL,...`), then `exec bash scripts/ogpo_multitask_4task_ref.sh`.

## Divergence from PLAN

No `PLAN.md` — this is a Tier 1 change (single new script + wrapper, no
approval gate), so the one-pass flow applies and there is no separate
plan-mode artifact to diverge from. Implementation matches `BLAST-RADIUS.md`
exactly.
