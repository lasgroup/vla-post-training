#!/bin/bash
# ---------------------------------------------------------------------------
# mt4_ncb_mcburst_s0 — post-collection Q warmup regressed on MONTE-CARLO
# returns instead of TD targets. Single-variable ablation against mt4_ncb_s0
# (job 10010995): same NCB base, same 5 rollouts/task, same 1000-step warmup,
# only --rl.burst_use_mc_targets added.
#
# WHY. Our launcher pins the critic to pure TD for the entire run
# (--rl.critic.td_weight_schedule init/end 1, switch 999999), so across
# mt4_v0/ncb/b the critic has NEVER received an MC gradient — q_mc_corr has
# been a purely out-of-objective diagnostic, and it sat at 0.2-0.57 in all 30
# warmup rounds. TD loss is a self-consistency residual: a critic can drive it
# to ~1 and still rank returns at 0.4 (we measured exactly that decoupling —
# rounds ending at q_loss ~1 scored no better on corr than rounds ending at
# 37). MC targets are the one lever that grounds the critic in realised
# returns, so this tests whether the corr ceiling is an objective problem
# rather than a data or optimisation one.
#
# EXPECT A LARGE PERTURBATION. Logged scales at 100k: q_td_loss ~14 (RMSE ~4)
# vs q_mc_loss ~2830 (RMSE ~53) against q_value_mean ~-200. The first MC warmup
# will move the critic hard; a jump in q_value_mean and a disturbed policy in
# the round after are expected, not necessarily a failure. Watch q_mc_corr
# (the target metric), then eval.
#
# PLUMBING. burst_use_mc_targets is implemented (config.py:284,
# ogpo_learner.py:234 — a second jit with td_weight pinned to 0, burst-only).
# ogpo_multitask_4task.sh forwards "$@" to exp.py (line 213) but the .sbatch
# wrapper does a bare `exec bash`, so we submit via --wrap and call the inner
# script directly. Header below mirrors ogpo_multitask_4task_maxlab.sbatch.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

sbatch --partition=maxlab --qos=maxlab_qos --job-name=mt4_ogpo \
  --gres=gpu:1 --constraint=VRAM_96GB --cpus-per-task=16 --mem=150G \
  --time=48:00:00 --output=/home/pchellap/logs/%x_%j.out \
  --export=ALL,ARM=ncb_mcburst,SEED=0,CONS=0,NUM_STEPS=100001 \
  --wrap='export GPU="${CUDA_VISIBLE_DEVICES:-0}" FSDP=1 && exec bash scripts/ogpo_multitask_4task.sh --rl.burst_use_mc_targets'

squeue -u "$USER" -o "%.10i %.14j %.8T %.10M %.20R"
