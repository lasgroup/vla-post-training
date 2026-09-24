# Verification

Self-verified (Tier 1, shell-only, no numerics). No separate verifier agent: the
change adds no code path — every knob is `${VAR:-<previous literal>}` and the
default-equivalence below is a stronger check than an adversarial read would be.

## 1. Syntax — PASS

`bash -n` clean on all three files.

## 2. Defaults unchanged — PASS

`DRY=1 GPU=0 bash scripts/ogpo_multitask_4task.sh` with no overrides:

```
[mt4] node=login4 gpu=0 arm=v0 seed=0 tasks=4 eval_tasks=4 rollouts/task=5
--collect.tasks libero_90_79 libero_90_31 libero_90_82 libero_90_38
--collect.eval_tasks libero_90_79 libero_90_31 libero_90_82 libero_90_38
--batch_size 32
```

Task set, eval set, banner count and batch size all identical to pre-change.

## 3. Overrides take effect — PASS

`TASKS="libero_90_44 libero_90_79" BATCH=128 FSDP=2`:

```
[mt4] node=login4 gpu=0,1 arm=v0 seed=0 tasks=2 eval_tasks=2 rollouts/task=5
--fsdp_devices 2
--collect.tasks libero_90_44 libero_90_79
--collect.eval_tasks libero_90_44 libero_90_79
--batch_size 128
```

`EVAL_TASKS` and the banner's `tasks=` both follow `TASKS` — the three downstream
consumers a CLI override would have missed.

## 4. Full-command differential against the real baseline — PASS

Diff of the complete emitted `uv run scripts/exp.py …` line, `mt4_ref` (the stack
job 10178692 ran) vs. the proposed 2-task arm. Exactly seven changed tokens, all
intended:

```
exp_name          mt4_ref_s0 -> mt4_mt2_b128_s0
fsdp_devices      1 -> 2
collect.tasks     libero_90_79/31/82/38 -> libero_90_44 libero_90_79
collect.eval_tasks  (same)
num_rollouts      5 -> 20
num_eval_rollouts 32 -> 64
batch_size        32 -> 128
```

Byte-identical otherwise: `pg_start_step 20000`, `pg_ramp_steps 5000`,
`grpo_conservative`, `post_collection_critic_steps 1000`,
`num_initial_rollouts 10`, `balance_success_buffer_tasks`, `num_qs/num_vs 10`,
`reduction mean`, `n_samples 8`, `success_reward_bonus 90`,
`critic_success_oversample`, `td_weight 0.95`, `lr 2.5e-5`, `ema_decay 0.99`,
`buffer_capacity 500000`, `clip_epsilon 0.1`, `group_num_samples 8`,
`noise_level 0.02`, `critic.batch_size 1024`.

Emitted-but-inert: `--rl.critic.num_tasks None`, added by the uncommitted per-task
critic work. `pi05_libero_online_ogpo_ref` already defaults `critic.num_tasks=None`,
so the resolved config is unchanged. Present in both sides of the diff.

## 5. Tyro repeated-flag semantics — PASS (background fact, relied on nowhere)

Directly exercised: `tyro.cli` with `["--tasks","a","b","c","d","--x","32","--tasks","p","q"]`
returns `tasks=('p','q')` — last occurrence wins. This is *why* a CLI override would
have produced a correct config with a wrong banner, and hence why the env knob was
the right fix. The shipped change does not depend on it.

## 6. Test suite — PASS (one pre-existing red)

`pytest tests/ogpo/test_verifier_alignment.py -k "dry or script or recipe or emit or flag"`
→ **56 passed, 1 failed** in 70.8 s.

```
FAILED tests/ogpo/test_verifier_alignment.py::test_head_wrapper_differential_over_a_randomized_flag_script
```

**Confirmed pre-existing**, not caused by this change: `git stash`ing all three
edited files and re-running that single test reproduces the same failure
(`1 failed, 119 deselected`). It is one of the three known reds recorded in
CLAUDE.md. Not investigated further — out of this change's blast radius.

## 7. 2-GPU / batch-128 memory smoke — PASS (job 10227513, babel-m9-16)

`TASKS="libero_90_44 libero_90_79" BATCH=128 FSDP=2`, `--gres=gpu:2 --mem=200G`,
`NUM_STEPS=1400 N_ROLLOUTS=2 INIT_ROLLOUTS=6 PG_START=1000 PG_RAMP=100`.
**COMPLETED, ExitCode 0:0, 01:36:37.**

```
[smoke] peak GPU 75.9 GiB / 95.6 GiB   peak host 159.9 GB
```

- **No `RESOURCE_EXHAUSTED` / OOM / traceback** (grep count 0) through the actor
  jit compile at `policy.training_start_step=900` and past `pg_start_step=1000`
  into full PG. B=128 with `group_num_samples=8` — 1024 SDE chains — fits on two
  96 GiB cards. Caveat unchanged from §"could not verify": 75.9 GiB is close to
  JAX's `0.75 x 95.6 = 71.7` GiB preallocation plus the EGL context, so this is
  "did not OOM inside the arena", not a headroom measurement.
- **FSDP=2 works.** First 2-GPU run of this stack. `--fsdp_devices 2` with
  `CUDA_VISIBLE_DEVICES=0,1` reaching the process — the fix in §DIFF item 2 is
  what made this possible.
- **Both warmstart phases exercised.** PG-phase metrics all live and sane:
  `approx_kl=0.0055`, `clipfrac=0.29-0.38`, `ratio_mean=0.905`,
  `cons_zero_frac=0.37`, `advantage_std=0.53`, non-zero `pg_loss` and `bc_loss`.
- **Throughput: 1.00-1.10 s/it**, flat across 350 PG-on steps. `mt4_ref` at
  B=32/1 GPU ran 0.80 s/it *amortized over collection and eval*, so pure-training
  is ~1.3x slower for 4x the batch. Projects to ~33 h for 100k steps.
- **Host RAM 159.9 GB exceeds the 150G every prior arm used** — and that is at
  step-0 collection, before the buffer fills. `ShardedReplayBuffer` preallocates
  `max_capacity` with `np.zeros` (`replay_buffer.py:57`), so RSS grows as pages
  are touched, and `N_ROLLOUTS=20` fills it ~2x faster than `mt4_ref` did.
  **Main run sized at `--mem=300G`.**

Cost breakdown, for whoever schedules the next one: 40 min `uv` venv import over
NFS (cold mount), 5 min weight restore + buffer alloc + EGL init, 36 min step-0
collection of 16 episodes (first-time MuJoCo env build x `n_samples=8` tiling),
14 min for the 1400 training steps. Only the last is paid repeatedly.

## Main run launched

Job **10228019** — `mt2_b128`, `--gres=gpu:2 --time=48:00:00 --mem=300G`,
`N_ROLLOUTS=20 EVAL_ROLLOUTS=64 NUM_STEPS=100001 MAX_RUNTIME=165600`, everything
else `mt4_ref`. `--exclude=babel-m9-16` deliberately **not** used: m9-20 and m9-28
were both GPU-full with two 8-GPU jobs pending, the 08-23 NIC fault looks resolved
(job 10220195 stable there 22 h, plus this smoke), and `exp.py:190` checkpoints at
every collect/eval boundary so a node fault costs <=10k steps rather than the run.

## What could NOT be verified here

All three open items (VRAM at B=128, FSDP=2, throughput) were **closed by the
smoke** in §7. What remains open:

- **Whether host RAM holds for 100k steps.** The smoke only reached step 1400, so
  the buffer was barely touched. 300G is sized from an estimate, not a measurement.
- **Whether the science is any good.** The smoke says nothing about learning —
  1400 steps, 50 policy updates, no eval ran (`eval_interval=10000`).
