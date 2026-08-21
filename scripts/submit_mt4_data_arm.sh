#!/bin/bash
# ---------------------------------------------------------------------------
# mt4_ncb_d20_s0 — the data arm: 4x collection data + 3x post-collection Q
# warmup, on the NCB base (= mt4_ncb_s0 plus N_ROLLOUTS 5->20, BURST 1000->3000).
#
# WHY DATA, NOT BURST LENGTH. The 100k-step NCB/B arms showed the critic stuck
# at q_mc_corr ~0.2-0.57 in every one of 20 warmup rounds, and — decisively —
# endpoint q_loss and q_mc_corr were DECOUPLED across those rounds (rounds that
# digested to q_loss ~1 scored no better on corr than rounds ending at 37).
# The critic also already trains 10k steps/round against a near-static buffer,
# so the 1k-step warmup is only ~9% of its training: burst length is a small
# perturbation to something that is not step-starved. What the warmup rounds
# DID track was the collection round that fed them (B's worst collection round,
# SR 0.10 @70k, produced its worst warmup, q_loss 68) — i.e. data-limited.
#
# WHAT 4x DATA BUYS
#   - per-task coverage: 5 -> 20 episodes/task/round (4 value landscapes in one
#     critic was the structural suspect; this is the direct fix)
#   - zero-success rounds essentially vanish: at the observed per-task
#     collection SR ~0.2, P(no success in a round) = 0.8^5 = 33% -> 0.8^20 = 1%.
#     task-79 posted 0.00 rounds in every arm; those rounds feed the success
#     buffer nothing and leave that task's Q head unconstrained.
#   - more groups per batch for the group-relative advantage.
#
# COST (measured from job 10010995 log timestamps): a 10k round is ~90 min, of
# which collection is only ~4 min and eval ~8 min — training dominates. So
# 20 -> 80 episodes/round adds ~12.6 min/round (~2.1 h over 10 rounds) and
# BURST 1000->3000 adds ~2.4 min/round (~24 min). ~17.5 h total vs 14.8 h,
# well inside the 48 h limit. Buffer: ~920 episodes x ~360 transitions ~= 330k,
# under the 500k capacity, so no eviction.
#
# HELD CONSTANT vs mt4_ncb_s0 (so the comparison stays interpretable): CONS=0
# base, warmstart (PG_START=20000, PG_RAMP=5000), MT_BAL, MT_ADV, seed 0, and
# eval (32 ep/task every 10k — eval_interval is hard-coded, so it does NOT
# scale with collection and the eval series stays directly comparable).
#
# CONFOUND, ACKNOWLEDGED: this moves two knobs (data and warmup length). Given
# the corr/q_loss decoupling above, data is the likely active ingredient if it
# works; a data-only control (BURST=1000) decomposes it in one follow-up run.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

sbatch --export=ALL,ARM=ncb_d20,SEED=0,CONS=0,N_ROLLOUTS=20,BURST=3000,NUM_STEPS=100001 \
  scripts/ogpo_multitask_4task_maxlab.sbatch

squeue -u "$USER" -o "%.10i %.14j %.8T %.10M %.20R"
