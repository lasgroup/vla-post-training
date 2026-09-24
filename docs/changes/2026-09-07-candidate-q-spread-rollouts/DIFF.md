# Diff

Three new files, no edits to existing code.

## `scripts/probe_candidate_q_spread.py` (new)

Layout mirrors `probe_rollout_gifs.py`, with two deliberate differences:

1. **Env / learner imports are inside `main()`**, not at module top.
   `src.envs.libero` pulls in LIBERO, robosuite/MuJoCo and torch; the gif
   probe imports them at top and has no tests. Moving them lets
   `tests/ogpo/test_candidate_q_spread.py` load the module with `importlib`
   (Tier B's pattern for Tier A) and exercise the pure helpers and the rollout
   loop on a CPU login node. `best_practices.md` §9 ("heavy or optional
   dependencies inside the function that needs them").
2. **Nothing is copied from the critic scoring block.** The production path
   is reused through `agent._bon_record`, armed before and disarmed after
   every `sample_actions` call, exactly as Tier C does at a single probe
   state (`probe_counterfactual_rollouts.py:326-331`).

Pure helpers (module top, numpy only):

- `candidate_q_variance(scores)` — `(n_env, M)` → `(n_env,)`, population
  variance (ddof=0; the M candidates are the whole set the selector chose
  between). Raises on rank ≠ 2 or M < 2.
- `gather_records(records, env_num)` — the per-prompt-group hook dicts →
  `(candidates (env_num, M, H, act), scores (env_num, M), best_idx)` in env
  order. Raises if the groups' `indices` do not partition `range(env_num)`,
  if shapes disagree across groups, or if the list is empty (the message
  names `n_samples` / `inference_start_step` as the cause).
- `mean_over_alive(traces)` — ragged per-episode traces → `(mean over the
  episodes alive at each chunk index, alive count)`.

Env driving:

- `reset_wave`, `policy_chunk` — copied from the gif probe (per-env seeds).
- `rollout(env, agent, obs, info, task_ids, max_chunks, use_bon)` — arms the
  hook under `try/finally` (disarmed even if the forward raises; nothing is
  swallowed), calls `policy_chunk`, gathers, **raises on a non-finite chunk
  with its own message**, then **checks that `candidates[arange, best_idx]`
  equals the returned chunk** (pins the indices→env alignment the BoN=0 arm
  depends on), executes `best_idx` or 0,
  appends the `(M,)` score row and the executed index for every live env,
  then the gif probe's `(live, replan)` termination bookkeeping and
  `jax.tree.map` scatter. Returns `(succ, steps, per-env (chunks, M) score
  matrices, per-env executed indices)`.

Plots (`matplotlib`, `Agg`, imported lazily): `plot_episode` — one line,
chunk index vs. variance, title carries task / episode / seed / SUCCESS or
FAILURE / env steps / arm / M. `plot_mean` — two stacked panels sharing x:
mean trace on top, alive count as a step plot beneath. One axis per panel;
no second y-axis.

`main()`:

- guards, in order: `cfg.resume` (else the learner would rmtree the
  checkpoint), `QSPREAD_OUT_DIR` set, `QSPREAD_EPISODES >= 1`,
  `QSPREAD_BON ∈ {0, 1}`, `rl.n_samples >= 2`; after construction
  `agent._resuming` (openpi's silent fresh-start downgrade) and
  `training_steps >= rl.critic.inference_start_step` (else `sample_actions`
  takes the single-sample path, AWR:420). All raise with the fix named.
- env: `filtered_sft_wrap_env(make_env(cfg, cfg.collect.eval_tasks,
  num_devices=1), env_num=cfg.collect.env_num)`;
  `agent.start_data_collection(step=None)` opens the EMA window.
- per task: `max_chunks` from that task's suite, `ceil(N / env_num)` waves
  with seeds `base_seed*7919 + w*G + i`, records cut to exactly N episodes,
  one PNG per episode, JSON flushed after every wave (tmp file +
  `os.replace`, so a preemption mid-write cannot lose earlier waves), then the
  mean-trace PNG and its arrays into the task cell. Per-wave log line carries
  the recorded count, SR and the min / median / max of the variances over the
  **recorded** envs of that wave only.
- `agent.end_data_collection(); env.close()`. No `update()`, `add_data`,
  `save_episode`, or save path anywhere.

Output layout: `<out>/<task>/ep<NN>_seed<seed>_<succ|fail>.png`,
`<out>/<task>/mean_trace.png`, `<out>/q_spread_results.json`.

## `scripts/probe_candidate_q_spread.sbatch` (new)

Copy of `probe_rollout_gifs.sbatch` with: **`--gres=gpu:2`, `GPU=0,1`,
`FSDP=2`** — the checkpoint's own topology, because the restore path cannot
re-shard onto fewer devices (BLAST-RADIUS "Restore topology"; the gif sbatch's
1-GPU claim is wrong and that job died in restore); job name `q_spread`;
`QSPREAD_BON` in the default output directory name so the two arms never
collide;
`CONFIG_NAME` exported and used in the mount-guard path (it was hardcoded in
the gif sbatch); `TASKS` default `libero_90_38` (one task, per the request);
`BON_N` exported explicitly as the M knob (the ref recipe's default is 8
anyway); the three `QSPREAD_*` knobs. Everything else — preempt, 300G, 4 h,
`ENTRY` swap, `CKPT_MODE_FLAG=--resume`, `EP_MULT=2`, `BATCH=128` — is
unchanged from the gif probe.

## `tests/ogpo/test_candidate_q_spread.py` (new)

12 pytest-native tests, CPU, ~2 s: variance value and ddof, rank / M guards;
gather placement by index across two shuffled groups, missing / duplicate /
empty / shape-mismatch rejection; mean-over-alive values and alive counts on
ragged input, empty rejection; the rollout loop against a fake agent that
reproduces the `_bon_record` contract (two prompt groups, tuple return under
`store_prefix_rep`) and a fake `(live, replan)` env — for both arms it checks
the executed chunk is the argmax / candidate-0 one, the score matrices match
the fake's scores query by query, chunk counts and env-step counts follow the
termination sub-step, success vs. truncation is told apart, and the hook is
disarmed after every query; a no-record agent and a swapped-indices agent
raise; both plot functions write non-empty PNGs.

## `tests/ogpo/test_candidate_q_spread_verifier.py` (new, written by the verifier)

21 tests. The strongest ones drive the **real** `AdvantageWeightedSFTLearner.
sample_actions` body on a stub `self` (the `object.__new__` + attribute
injection pattern of `test_per_task_critics_verifier2.py:333`) with the real
`pi05_libero_online_ogpo_ref` config, faking only the pi0.5 forward, the
transforms and the two `nnx` critic models — so the hook *contract* the probe
relies on (field set, shapes, dtypes, bit-exact argmax chunk, candidate 0 =
first tiled noise draw, tuple return under `store_prefix_rep`, configured
reduction, empty record on the single-sample path) is pinned against the
production code, not against the implementer's imitation of it. The rest:
float64 variance from float32 scores, NaN propagation, numpy-index gathers,
zero-length traces, `max_chunks` truncation, dead envs queried-but-not-
recorded, obs scatter of live rows only, terminated-beats-truncated, JSON
round-trip of the outputs, PNG magic bytes, and the two behaviour pins
rewritten after the fixes (hook disarmed on raise; NaN chunk reported as
non-finite).

## Verification run here

- `python -m py_compile scripts/probe_candidate_q_spread.py`: OK.
- `bash -n scripts/probe_candidate_q_spread.sbatch`: OK.
- `pytest tests/ogpo/test_candidate_q_spread.py`: 12 passed in 2.13 s.
- `DRY=1` render of `ogpo_multitask_4task_ref.sh` with the sbatch's env
  block (`CKPT_BASE_DIR` / `STORE_ROOT` redirected to scratch): the command
  carries `uv run scripts/probe_candidate_q_spread.py`,
  `--exp_name mt4_mt2_b128_ep2_s0`, `--resume` (once), `--rl.n_samples 8`,
  `--rl.critic.reduction mean`, `--collect.tasks libero_90_38`,
  `--collect.eval_tasks libero_90_38`, `--collect.episode_steps_multiplier 2`,
  `--rl.critic.inference_start_step 1`.

## Post-verification changes

Applied after the independent verifier's report (`VERIFICATION.md`):

1. `rollout` arms the hook under `try/finally` (finding 1).
2. Separate `np.isfinite` check on the returned chunk with its own message,
   before the alignment check (finding 2).
3. `flush()` writes `q_spread_results.json.tmp` then `os.replace` (finding 6).
4. Per-wave log restricted to the recorded envs (finding 7).
5. Corrected `file:line` cites in the script's comments and docstring
   (finding 5) and the `env_num` comment (finding 3).
6. sbatch: 2 GPUs / `FSDP=2` (restore topology, found while checking the
   gif probe's only run); header carries the measured envelope from job
   10349712 (215 GB RSS, ~24 min startup, ~28 min per 8-episode wave).
7. Two verifier tests that pinned the pre-fix behaviour were rewritten to
   assert the fixed behaviour (`test_rollout_disarms_the_hook_when_sample_actions_raises`,
   `test_rollout_reports_a_nan_chunk_as_non_finite_not_misaligned`).

Known, accepted: the module runs `mp.set_start_method("spawn", force=True)`
and a `sys.path.insert` at import, so loading it in a test process sets the
process-wide start method (finding 8; same as `exp.py`, harmless for
`tests/ogpo`). Every env is queried on every chunk including finished ones
(finding 9; inherited from the gif probe — a wave costs the longest episode
at the full `G×M` batch).

## Multi-pass candidates (added 2026-09-07, maintainer request: 16 episodes, 32 candidates)

`QSPREAD_PASSES` (default 1): `rollout` calls `sample_actions` `passes`
times per query, each armed/gathered/alignment-checked separately, and
concatenates to M = passes × `rl.n_samples`. BoN=1 executes the argmax over
the union; BoN=0 executes candidate 0 of the first pass. Exact w.r.t. a
single pass of M: the candidates are iid initial-noise draws and the critic's
score is deterministic per (state, candidate). Motivation: the observation
is tiled `rl.n_samples` times BEFORE the VLM prefix pass (AWR:471), so a
single call with M=32 on 8 envs is a 256-row VLM batch; 4 passes of 8 keep
the peak at the collection-validated 64 rows for the same total compute. For
`passes == 1` and BoN=1 the probe additionally checks that the union argmax
equals production's `best_idx` (same float32 scores, same `np.argmax`), so
the recorded scores are provably the ones selected on. JSON gains
`n_samples_per_pass` and `passes`; `n_samples` is the total M. Tests: the
rollout test is parametrised over `passes ∈ {1, 2}` (union concatenation
order, argmax over the union, candidate 0 of the first pass, call count),
plus `passes=0` rejection and the union-vs-`best_idx` check. Both wrappers
expose the knob; the stab wrapper defaults to 4 × 8 and 16 episodes.

## Divergence from PLAN

No `PLAN.md` — Tier 1, one pass. Implementation matches `BLAST-RADIUS.md`
as amended above.
