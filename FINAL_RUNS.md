# Tasks to finish for final submission

Status of each split. Append "RUNNING" once all 3 seeds are submitted, "DONE" once all
3 seeds have their final eval. Last updated 2026-09-10.

## General Rules
Number of training steps:
- Single-task: 10 rounds of 10k critic steps (100 actor steps if FSFT is involved)
- Multi-task: 10 rounds of 50k critic steps (500 actor steps if FSFT is involved)
Evaluation:
- For Libero after round 10, for Molmo after rounds 5 and 10
Single-task splits are 4 separate single-task runs per seed:
- Libero: libero_90_79 / _31 / _82 / _38
- Molmo:  molmo_267 / molmo_366 / molmo_138 / molmo_143
  (the original BoN-family sweeps only ran molmo_138; the `_rest` groups add the
  other three so every method averages over the same four tasks)

Results dir / index per cell: `scripts/results_registry*.yaml`.

### BoN Tests
- Impact of N on compute time: Evaluate the inference time of BoN for different values of N: 1, 2, 4, 8, 16, 32. Based on 20 forward passes per value. Use env_num=1. The idea is to show deployment-latency cost, for the full candidate-generation + critic-scoring block
- Impact of N on performance: For Molmo single-task (all four tasks), re-run the main BoN jobs (training + eval) with different values of N: 2, 4, 8, 16. Eval used the same N the run trained with.
- Does BoN+FSFT amortize BoN? Take the round-5 BoN+FSFT single-task Molmo checkpoints and run the mid eval again, but with N=1 (i.e., just the policy). The idea is to then compare this with pure FSFT. For BoN+FSFT (full fine-tuning) and BoN+FSFT (EO)

### BoN (scripts/configs/multitask/bofn)
- Libero:
  - 1 task - DONE
  - 4 tasks - DONE
  - 16 tasks - DONE
  - 64 tasks - DONE
- Molmo:
  - 1 task - DONE (final eval only)
  - 4 tasks (4 inst. x cat. 1) - DONE (mid + final eval)
  - 8 tasks (8 inst. x cat. 1) - DONE (final eval only)
  - 8 tasks (1 inst. x cat. 1-8) - DONE (final eval only)
  - 16 tasks (16 inst. x cat. 1) - DONE (mid + final eval)
  - 64 tasks (8 inst. x cat. 1-8) - DONE (final eval only)
  - 128 tasks - TODO (split not defined)
  - Mid-eval pass for 1 / 8 / 64 tasks - TODO (separate eval-only run, not a resubmit).
    Blocked: ~9 of these runs have a manifest ahead of their last consistent checkpoint,
    and `eval.py --resume` takes its step from the manifest, so each needs a rewind first.
  - Re-eval on the new split [molmo_666, molmo_602, molmo_543, molmo_273, molmo_916,
    molmo_505, molmo_769, molmo_675] - DONE (2026-09-11). Final checkpoints of Molmo 1
    (4 single tasks), 4, 8 ("8 inst. x cat. 1", index 0) and 16 = 21 evals, all COMPLETED.
    Results are in each run's metrics.jsonl under the `eval_newtasks/` prefix AND as new
    W&B runs in groups bofn_molmo_tasks{1,4,8,16}. See "Re-eval mechanics" below.

### BoN critic ablation (scripts/configs/tuning/bofn/paper_ablation)
Grid: {TD,MC} x {regression,classification} x {Bronet,MLP}, vs the TD+reg+Bronet baseline.
All submitted cells are complete (36/36).
- Libero 1 task - DONE. Caveat: the 12 non-baseline cells are the superseded 512-wide /
  500k-buffer runs, not comparable to the 1024/250k cells; marked DO NOT PLOT.
- Libero 4 tasks - DONE. Baseline td_reg_bronet seeds 1/2 are colleague runs, wandb-only,
  no local logs and no verified config.
- Molmo 1 task - DONE (all cells at bronet_hidden_dim 1024 / buffer 250k).
- Molmo 8 tasks (1 inst. x cat. 1-8) - DONE (mid + final eval).

### BoN critic-proprioception ablation (scripts/configs/tuning/bofn/proprio)
Does the critic need proprioception? Standard best recipe (TD+reg+Bronet, 1024/250k) with
`rl.critic.use_proprio=False`, so the critic sees only the policy prefix embedding, against
the main-sweep BoN runs. Only the BoN learner honours the flag; the AWR/BoN+FSFT learner
still passes `state` to its critic.
- Molmo 1 task (all 4 single tasks), seed 0 - DONE. Seeds 1/2 not run.

### BoN+FSFT (scripts/configs/multitask/bofn_plus_fsft)
- Libero:
  - 1 task - DONE
  - 4 tasks - DONE
  - 16 tasks - DONE
  - 64 tasks - BLOCKED (OOM under current code; never started)
- Molmo:
  - 1 task - DONE (mid + final eval)
  - 4 tasks (4 inst. x cat. 1) - DONE (2026-09-11; all 3 seeds have the mid-eval at 249999
    and the final at 499998)
  - 8 tasks (8 inst. x cat. 1) - DONE (mid + final eval)
  - 8 tasks (1 inst. x cat. 1-8) - DONE (mid + final eval)
  - 16 tasks (16 inst. x cat. 1) - DONE (2026-09-13; all 3 seeds have the mid-eval at
    249999 and the final at 499998)
  - 64 tasks (8 inst. x cat. 1-8) - BLOCKED (OOM under current code; never started)
  - 128 tasks - TODO (split not defined)

### BoN+FSFT - Expert only (scripts/configs/multitask/bofn_plus_fsft_eo)
- Libero:
  - 1 task - DONE
  - 4 tasks - DONE (index 0 of the tasks4-16 dirs)
  - 16 tasks - DONE (index 1 of the tasks4-16 dirs). The 249999 eval in these logs is a
    stale artifact of a pre-fix config, not a mid-eval.
  - 64 tasks - DONE (2026-09-13; jobs 3241551/52/53, all 3 seeds have the final eval
    at 499999)
- Molmo:
  - 1 task - DONE (mid + final eval)
  - 4 tasks (4 inst. x cat. 1) - DONE (mid + final eval)
  - 8 tasks (8 inst. x cat. 1) - DONE (mid + final eval)
  - 8 tasks (1 inst. x cat. 1-8) - DONE (mid + final eval), index 1 of the tasks8 dirs.
    249999 appears twice in the eval history because a rewind resumed exactly on the eval
    step; use the later one.
  - 16 tasks (16 inst. x cat. 1) - DONE (mid + final eval)
  - 64 tasks (8 inst. x cat. 1-8) - DONE (2026-09-25; all 3 seeds trained to 500k, mid + final
    split evals merged into metrics.jsonl and published to W&B group
    splitval_tasks64_bofn_plus_fsft_eo_molmo_v0; final eval jobs 3492766-69, 3507900, 3507914).
    History: a round is collect ~8.5h + train 3.2h = ~11.7h, which never fit the 12h
    partition limit, so the group lost six allocations to TIMEOUT with nothing banked.
    BoN 64 fits easily at ~7.2h/round because it trains only the critic (0.45h vs 3.2h).
    collect.env_num 8 -> 16 was tried and REVERTED: slower (8.46h -> 10.6h on the same
    round) and it exhausted the node's 450G, killing all 3 seeds with host OOM.
    exp.py's mid-round requeue (b533001) then exposed a latent bug: save_epoch_state
    waited only on the POLICY checkpoint manager, never on the separate async RL-state
    checkpointer, so sys.exit for requeue truncated the critic write. The manifest
    advertised a mid-round step whose rl_state was only an orbax tmp dir, and all 3 seeds
    died at the 3/3 retry ceiling, unable to resume. Fixed 2026-09-13 by waiting on
    `agent._rl_state_checkpointer` in save_epoch_state. The 3 seeds were rewound to
    boundary+1 (50001 / 150001 / 100001), which keeps the already-collected round: the
    policy is aliased from the boundary checkpoint, identical because the actor updates
    every 100 steps and boundary+1 is not a multiple. Broken mid-round saves moved to
    <subrun>/broken_midround_save_<step>/.
    Seed 0 was separately rewound to 50k on 2026-09-10 after a policy checkpoint at 100000
    was never written; see <subrun>/rewound_to_50k_20260910/.
    Eval-fits-the-wall problem (2026-09-15): the EO policy fails more than BoN, so eval
    episodes run to the 450-step cap and the 2560-rollout eval (32 rollouts x 80 tasks) takes
    ~14.6h -- past the 11:59 wall, and it is atomic so a timeout banks nothing and re-loops.
    eval_env_num does NOT help: Best-of-N scoring (32 critic candidates/step) is GPU-compute-
    bound, so 16 envs cost ~2x/step and net wall time is unchanged (measured: env8 19.7s/it,
    env16 20.5s/it). free_buffer_before_eval also collides on the resume-on-eval-step path
    (pre-eval save hits an existing rl_state/<step>). Both discarded.
    Decision (2026-09-15): keep 32 rollouts, split the eval. In-loop eval DISABLED on the 3
    training jobs (eval_interval 100000000) so they train straight to 500k banking round-5
    (250000) and round-10 (500000) checkpoints; jobs 3408476/77/78. Mid + final evals run as
    separate eval-only jobs, each covering 40 of the 80 tasks (~7.3h, fits), results merged.
    KEEP round-5 and round-10 policy+rl_state checkpoints (needed for the split eval).
    Built by scripts/_reeval/build_molmo64_eo_split_eval.py (+ merge_*.py). Mid eval (round 5,
    step 250000) submitted 2026-09-17 for seeds 1/2 as jobs 3421174-77 (2 halves each,
    measured ~7h/half). Seed 0 mid pending its 250000 bank; all finals pending round 10.
  - 128 tasks - TODO (split not defined)
  - Re-eval on the new split [molmo_666, molmo_602, molmo_543, molmo_273, molmo_916,
    molmo_505, molmo_769, molmo_675] - DONE (2026-09-11). Mid AND final checkpoints of
    Molmo 1 (4 single tasks), 4, 8 ("8 inst. x cat. 1", index 0) and 16 = 41 evals, all
    COMPLETED. Results are in each run's metrics.jsonl under the `eval_newtasks/` prefix
    AND as new W&B runs in groups bofn_plus_fsft_eo_molmo_tasks{1,4,8,16}, where each run
    carries `eval/*` (final) and `eval_mid/*` (round 5) together.
    GAP: Molmo 4 seed 1 has no final re-eval - an early prune pass
    (.ckpt_delete_list.txt) deleted policy@499998 and the 500000 save had been
    interrupted, leaving only a 120 KB orbax tmp dir with no weights, so that cell has
    2 of 3 seeds. Mid is fine for all 3, and the run's ORIGINAL final eval is unaffected.

## Re-eval mechanics (new task split)
Driven by `scripts/_reeval/build_reeval_jobs.py` (`--apply` writes, no args = dry run).

Why it needs staging rather than a flag: `eval.py --resume` takes its step solely from
`runtime_state/resume_state.json`, and restores BOTH the policy and `rl_state/<step>` at
that step. There is no `--eval_step`. Every finished Molmo run is unusable as-is, because
policy checkpoints survive only at round boundaries (100000 / 500000, rounds 5 and 10 kept
by the prune passes) while the terminal critic sits at the eval step (99999 / 99998 /
499998 / 499999), and the manifest points at the round number where no critic exists.

So per (subrun, step) the builder stages `<subrun>/reeval_<mid|final>/<config>/<exp>/` with
symlinks to the real policy and critic, a manifest at the target step, a copy of
wandb_id.txt (so results reach the original wandb run) and an EMPTY replay_shards dir
(`--resume` otherwise loads 12-25 GB the eval never samples). The source run dirs are not
modified at all, and mid + final can run concurrently on one subrun.

Policy and critic are 0-2 steps apart. For BoN that pairing is exact: the BoN learner never
reassigns `_train_state`, so its policy is frozen and every policy checkpoint is identical.
For expert-only it is at most one actor update out of 5000.

Jobs need ~2.2 h: 8 tasks x 32 rollouts = 256 at ~30 s/rollout with eval_env_num 8,
measured on a 16-rollout smoke test of this task split. The earlier 14.6 s/rollout figure
came from the 64-task eval, whose episodes are shorter; the new split fails more often and
failures run to the 450-step cap. Submitted with `--time=04:00:00`.
`EVAL_METRIC_PREFIX=eval_newtasks` keeps them from overwriting the original
`eval/success_rate` at the same wandb step. Each job also drops
`eval_metrics_eval_newtasks_step<S>.json` in its staging dir; those still need merging
into each run's `metrics.jsonl` in one serial pass (concurrent appends to one file are not
safe on Lustre).

Outcome (2026-09-11): all 62 jobs COMPLETED, 62 results written and merged into the
source runs' metrics.jsonl via `scripts/_reeval/merge_reeval_metrics.py --apply` (that
merge is idempotent; re-running reports 62 already_present). One job died mid-eval with
the known Molmo renderer EOFError and was resubmitted by hand - the generated re-eval
scripts have no crash-retry wrapper, unlike the training jobs, so any failure needs a
manual resubmit. Jobs ran 0.5-2.2 h each, so the 2 h limit originally used was too tight
for the slowest four; use 4 h for a re-eval pass of this shape.

W&B publishing (2026-09-11): writing the results into the ORIGINAL training runs FAILED
silently - a resumed run drops history at an already-committed step, so all 62 in-place
`wandb.log` calls were discarded and nothing appeared. The values were never lost (sidecars
+ metrics.jsonl). They were instead published as 42 NEW runs following the FSFT
`different_instances_eval` pattern, verified field-by-field against a live FSFT run:
one run per evaluated training run, group per (recipe, task-count) with seeds pooled,
one history row at _step=0 mirrored into the summary, `collect.tasks` holding the TRAINING
tasks as the join key, and `weight_loader.params_path` / `rl_state_path` / `eval_step`
recording the exact checkpoint pair (policy and critic are 1-2 steps apart; for BoN that
pairing is exact because its policy is frozen). Built by
scripts/_reeval/build_wandb_manifest.py + publish_reevals_to_wandb.py, ledger in
.reeval_wandb_published.json. No compute node was needed: wandb installs into a plain
python3.11 venv on the login node, which can reach the W&B API.
