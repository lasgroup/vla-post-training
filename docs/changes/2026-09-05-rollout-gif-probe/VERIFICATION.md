# Verification

Independent agent: Opus, xhigh reasoning effort, fresh context (change spec +
diff only). Findings relayed verbatim below, then the outcome of each.

## Execution disclaimer (from the verifier, and true for this session too)

Nothing was executed against real weights, a simulator, or the checkpoint.
`/data` is not mounted on the login node used for this change (`ls` on the
checkpoint root returns ENOENT — the sbatch's own
`timeout 120 ls -d "$SRC"` guard would fire here), and there is no GPU. What
*was* run: `python3 -m py_compile scripts/probe_rollout_gifs.py` (passes),
`bash -n scripts/probe_rollout_gifs.sbatch` (passes), and a `DRY=1` render of
`ogpo_multitask_4task_ref.sh` with the sbatch's exact env block and
`CKPT_BASE_DIR`/`STORE_ROOT` redirected to local scratch (confirms
`--exp_name mt4_mt2_b128_ep2_s0`, `--collect.episode_steps_multiplier 2`,
`--collect.tasks/--collect.eval_tasks libero_90_38 libero_90_82`, `--resume`,
`--rl.n_samples 8`). Everything else is source reading. GPU/simulator
execution remains unverified — say so plainly rather than claiming success.

## Verified correct (no finding, no change)

- `imgs[row, j]` / `_scatter` indexing: correct. `SubprocVectorEnv` with
  `wait_num=None` is synchronous (`src/envs/venv.py:824-832`) and stacks
  results in `id` order, so row *r* of a returned array is env `live[r]`
  (`venv.py:865-868`).
- `obs["observation/image"][e][-1]`: correct. `Pi0ObservationWrapper` emits a
  single agentview view; `QueryFrequencyWrapper` stacks `replan_steps` along a
  new axis and tiles it on reset; the vector env prepends the env axis.
- `term.shape[1] == replan_steps`; `if done[e]: break` correctly excludes
  `QueryFrequencyWrapper`'s post-termination padding frames.
- No hang: `max_steps = 400*2 = 800`, `replan_steps = 5`
  (`config.py:404`), `max_chunks = 162` vs. 160 needed to reach `TimeLimit`
  truncation; `TimeLimit` sits outside `WarmUpOnResetWrapper`, so the 10
  warm-up steps don't count against the bound. `episode_steps_multiplier`
  does reach `TimeLimit` (`libero.py:102`).
- `agent.training_steps` genuinely reports the restored step
  (`filtered_sft_learner.py:330`, from `self._resume_state.step`);
  `_checkpoint_manager.all_steps()` exists (`:430`).
- `sample_actions` / `start_data_collection(step=None)` /
  `end_data_collection()` signatures all match the probe's calls
  (`filtered_sft_learner.py:723-726, :985, :993`); OGPO overrides none of
  them.
- Best-of-N flows through identically at probe time and at the run's own
  in-training eval time: `--rl.n_samples 8` activates BoN in
  `AdvantageWeightedSFTLearner.sample_actions` because
  `training_steps >= critic.inference_start_step` (`AWR:420`); the tuple
  return under `store_prefix_rep` is already handled (`probe.py:63`,
  unchanged by the fixes below).
- `json.dumps(..., default=float)`'s fallback is never exercised — every
  value written into `out` is already `int`/`bool`/`str`/`float` via explicit
  casts.
- Read-only holds: no `update()`, no `save_checkpoint`, no `add_data`;
  `--resume` appears once in the render; the `not cfg.resume` guard precedes
  learner construction.
- `BATCH=128` is genuinely inert at eval — `_refresh_train_step` only builds
  the `jax.jit` wrapper, no eager trace.
- `write_gif`'s `parents=True` always runs before the first `flush()`, so
  `out_dir` exists in time even though it's never `mkdir`'d directly.

## Findings and outcomes

1. **[CONFIRMED, fixed]** All episodes in a wave shared one seed
   (`reset_wave(env, task_id, seed)` broadcast the same seed to every env),
   and `LiberoWrapper.reset` picks its init state from that seeded rng
   (`src/envs/libero.py:26-28,47-50`) — so with the default
   `ROLLOUT_EPISODES=8` and `env_num=8`, `waves=1`, meaning all 8 "episodes"
   per task were the SAME initial state, differing only in flow-sampling
   noise, while the JSON and filenames implied 8 independent trials. Fixed:
   `reset_wave` now takes a per-env seed list; the main loop seeds each env
   in a wave as `base_seed*7919 + w*env_num + i`, matching
   `stage4_trained_instructions.py`'s convention.
2. **[CONFIRMED, doc-only]** The claim "the exact factory `exp.py` uses for
   its own in-training eval" was imprecise: `exp.py`'s eval env uses
   `env_num=eval_env_num`; this probe intentionally uses
   `env_num=collect.env_num` because `start_data_collection` sizes
   `_episode_storage` from `collect.env_num`
   (`filtered_sft_learner.py:987`) — the same constraint Tier C guards on
   explicitly. No code change (the code was already right); corrected the
   docstring/DIFF.md/BLAST-RADIUS.md wording.
3. **[CONFIRMED, doc-only]** "Byte-identical config" was false for 4 flags
   that the ENTRY-swap renders from recipe defaults rather than the training
   job's actual `--export=ALL` shell env
   (`--collect.num_rollouts` 5 vs. trained 20, `--collect.num_eval_rollouts`
   32 vs. 64, `--num_train_steps` 100000 vs. 100001, `--max_runtime` 169200
   vs. 165600 — see `[[mt2-batch128-run]]`). Verified all four are inert for
   this probe (`grep -rn "num_train_steps\|max_runtime" src/` returns
   nothing; the other two are read only by `src/training/collect.py`, never
   invoked here). No code change; corrected the "byte-identical" wording in
   the sbatch comment and script docstring.
4. **[CONFIRMED, fixed]** `max_steps` was computed from a hardcoded
   `"libero_90"` while `TASKS` is an exposed override; a `libero_10` task
   (max-step map 520) would have been truncated by the probe's own 162-chunk
   bound (built for 800 steps) before the episode could finish, silently
   recorded as `success: false` with a short GIF, no error. Fixed: `max_steps`
   / `max_chunks` are now computed per task from that task's own suite name
   (parsed the same way `make_env_libero` does).
5. **[CONFIRMED, fixed]** No guard against `--resume` silently downgrading to
   a fresh start (`openpi/training/checkpoints.py:56-61`, when the checkpoint
   dir is absent or empty) — the probe would have logged step 0 and an empty
   `checkpoint_steps_present`, then written a full set of GIFs and a JSON
   labeled `mt4_mt2_b128_ep2_s0` off the UNTRAINED base weights. Fixed: added
   a hard guard on `agent._resuming` immediately after construction, raising
   with the checkpoint dir and a pointer at ARM/SEED/CONFIG_NAME/
   checkpoint_base_dir to check.
6. **[unverified by the reviewer, fixed anyway]** Missing
   `mp.set_start_method("spawn", force=True)` — `scripts/exp.py` and
   `probe_counterfactual_rollouts.py` (the file this probe's env-driving
   helpers are copied from) both set it before constructing any
   `SubprocVectorEnv`; this probe didn't. The reviewer could not confirm this
   is fatal from source alone (stage 4 has the identical omission and
   reportedly ran), but it's an unexplained, zero-cost-to-fix divergence from
   both named models. Fixed: added at the top of the script, matching
   `exp.py`'s placement and comment.
7. **[unverified by the reviewer, mitigated]** `--mem=128G` vs. the full
   replay + success buffer restore that `OGPOAgentLearner(cfg)` does
   unconditionally on `--resume`, regardless of whether this read-only probe
   ever uses them; at step 40k this run held 83,251 + 48,656 rows and will
   hold substantially more by its final step, while the runs that produced
   this checkpoint used `--mem=300G`. The reviewer could not confirm the
   actual peak without running it. Mitigated rather than root-caused: bumped
   `--mem` to 300G (matching the training run's own allocation) rather than
   measuring the probe's true footprint, which needs a real run to observe.

## What remains unverified

Everything that requires GPU + simulator + the actual checkpoint: whether the
job schedules and completes within the 4h walltime, whether 300G is in fact
sufficient (finding 7), whether the `spawn` start-method fix actually mattered
(finding 6), and whether orbax's cross-mesh restore (FSDP=2-saved →
FSDP=1-loaded) behaves as expected. These can only be settled by running the
probe with the maintainer's explicit go-ahead (this change adds no new
training-launch or experiment-campaign authorization — CLAUDE.md's
launch-permission rule still applies to the sbatch submission itself).
