#!/bin/bash
# ---------------------------------------------------------------------------
# Submit the stability study's two winning recipes in the multi-task setting,
# as controlled ablations against mt4_v0_s0 (= CANCB + warmstart, job 9967191):
#
#   mt4_ncb_s0  NCB (norm + clip + burst, NO conservative gating)
#               = v0 minus CONS. The study's best stack (5 consecutive 100%
#               evals, 50k-90k single-task). Isolates the cons-gating effect.
#
#   mt4_b_s0    B (burst only) = NCB minus all advantage normalization
#               (EMA-quantile norm, +-4 clip, AND per-task norm — MT_ADV is
#               advantage normalization too, so a faithful B drops it).
#               The study's simplest sustained-ceiling arm (>=96.9% x 5 evals).
#
# Both arms KEEP (constant vs v0, so each comparison is single-variable):
#   - filtered-BC warmstart (PG_START=20000, PG_RAMP=5000) — post-study
#     colleague addition, showed no harm in v0 (best eval was the 20k handoff)
#   - task-balanced success-buffer BC (MT_BAL=1) — BC-side, orthogonal to
#     the advantage machinery being ablated
#   - seed 0, 5 rollouts/task, 32-ep eval on the 4 train tasks every 10k
#
# NUM_STEPS=100001: loop reaches step 100000 => the final checkpoint gets an
# in-run eval (v0's step-100k checkpoint shipped unevaluated).
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

sbatch --export=ALL,ARM=ncb,SEED=0,CONS=0,NUM_STEPS=100001 \
  scripts/ogpo_multitask_4task_maxlab.sbatch

sbatch --export=ALL,ARM=b,SEED=0,CONS=0,NORM=0,CLIP_SYM=0,MT_ADV=0,NUM_STEPS=100001 \
  scripts/ogpo_multitask_4task_maxlab.sbatch

squeue -u "$USER" -o "%.10i %.12j %.8T %.10M %.20R"
