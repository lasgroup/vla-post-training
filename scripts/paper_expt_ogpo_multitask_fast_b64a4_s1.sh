#!/bin/bash
#SBATCH --partition=maxlab
#SBATCH --qos=maxlab_qos
#SBATCH --job-name=paper_ogpo_mt4_fast_b64a4_s1
#SBATCH --gres=gpu:1
#SBATCH --constraint=VRAM_96GB
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/pchellap/logs/%x_%j.out
#SBATCH --error=/home/pchellap/logs/%x_%j.out
# ---------------------------------------------------------------------------
# Paper experiment: OGPO multitask, the FAST config derived on 2026-09-20,
# on a SINGLE RTX PRO 6000 (maxlab's only GPU model -- VRAM_96GB is
# unambiguous here, unlike `general`'s mixed 48GB-tier fleet that
# confounded the single-task probe series). Seed 1 of 3
# (paper_expt_ogpo_multitask_fast_s{0,1,2}.sh). Replaces the earlier
# paper_expt_ogpo_multitask_s{0,1,2}.sh (old config: update_interval=10,
# n_samples=8, collect_interval=10000, fsdp_devices=1 batch=32/no accum
# tuning) -- those queued jobs (10512865/66/67) were killed before ever
# starting.
#
# fsdp_devices=1, batch_size=32, rl.policy_grad_accum=8: NOT FSDP=2 like
# the single-task fast recipe. batch=32 x group_num_samples=8 = 256 local
# chains was already validated safe on a 96GB card at the ORIGINAL
# single-task baseline (75.8/95.6 GiB peak, 10-head critic, job 10178533)
# -- with a lighter 2-head critic here, this should have MORE headroom, so
# a single GPU (no FSDP sharding at all) should comfortably fit without
# needing the 2-GPU workaround the 48GB L40S/6000Ada cards required.
#
# Corrections vs. the earlier "fast" derivation, checked directly against
# origin/marco/parl:scripts/configs/policy_extraction/libero/
# parl_libero_tasks4_seed0_v0.yaml (2026-09-20):
#   - collect.collect_interval=50000, NOT 10000 (ogpo_multitask_4task.sh's
#     own script default) -- the real yaml collects every 50000 steps, 10
#     rounds total over 500000, not 50. collect.eval_interval is ALSO
#     overridden to 50000 here: ogpo_multitask_4task.sh hardcodes eval at
#     10000 regardless of COLLECT_INT, which would otherwise fire 50 evals
#     against only 10 collection rounds -- inconsistent.
#   - num_train_steps=500001 (matches the yaml's 500000, +1 so the loop
#     reaches step 500000 itself and gets its eval, same reasoning as the
#     other _maxlab.sbatch wrappers).
#
# Carried over from the single-task fast derivation (paper_expt_ogpo_
# single_task_fast_s0.sh) -- same reasoning applies, task-count-
# independent:
#   - rl.policy.update_interval=100 (not the script's 10): matches
#     parl_libero_tasks4_seed0_v0.yaml's literal rl.policy.update_interval
#     =100. Over 500001 steps that's ~5000 actor updates instead of
#     ~50000.
#   - rl.n_samples=1 (not 8, not the paperconfig family's 32): disables
#     best-of-N entirely (advantage_weighted_sft_learner.py:420),
#     collapsing collect/eval cost to a single on-policy sample per query.
#   - rl.critic.num_qs/num_vs=2, rl.critic.reduction=min: the paper
#     excerpt's stated critic design (2-head ensemble, min-aggregation for
#     both the TD target and best-of-N scoring).
#   - rl.buffer_capacity=250000 (not the script's own multitask default of
#     500000), rl.noise_level=0.3 (not the script's 0.02).
#   - rl.pg_start_step LEFT DISABLED (PG_START=0, script's own default here
#     is 20000/5000 -- the mt4 reference-alignment warmstart convention --
#     turned off to match what the single-task fast runs actually use, not
#     because the source yaml has an opinion; AWR/PARL has no such
#     concept at all).
#
# Projected: training-loop ~5000 x ~40s (actor) + ~495000 x ~0.25s
# (critic-only) ~= 90h; collect+eval ~10 rounds x ~25.6min ~= 4.3h
# (using the n_samples=1 measurement from the single-task probe, job
# 10513903: 388s/20-episode collection, 287s/32-episode eval, scaled to
# multitask's 128-episode eval and 20-episode collection per round).
# Total projected ~94h ~= 3.9 days. UNVERIFIED against a real multitask
# run -- first real measurement of this recipe.
#
# Everything else (task set libero_90_79/31/82/38, in-distribution eval
# only via HELDOUT=0 default, N_ROLLOUTS=5/task, CONS=1/grpo_conservative,
# NUM_QS-independent critic knobs inherited from pi05_libero_online_ogpo_
# ref's own defaults, TD_W=0.95, SUCC_BONUS=90, SB_Q=1, BURST=1000,
# PER_TASK_CRITIC=0/shared critics) is ogpo_multitask_4task_ref.sh's
# unmodified reference-aligned recipe.
#
# Submit: sbatch scripts/paper_expt_ogpo_multitask_fast_s0.sh
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_DIR=/home/pchellap/Projects/OGPO-VLA/vla-post-training
cd "$PROJECT_DIR"

export ARM="paper_expt_ogpo_multitask_fast"
export SEED="1"
export NUM_STEPS="500001"
export BON_N="1"
export NUM_QS="2"
export CRITIC_RED="min"
export COLLECT_INT="50000"
export PG_START="0"
export BATCH="64"

export GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"

echo "[paper-mt4-fast-b64a4-s1] node=$(hostname) job=${SLURM_JOB_ID:-?} arm=$ARM seed=$SEED"
echo "[paper-mt4-fast-b64a4-s1] gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo n/a)"

child_status=0
bash scripts/ogpo_multitask_4task_ref.sh \
  --project_name vla-post-training \
  --group_name policy_extraction_tasks4_ogpo_fast_v0 \
  --rl.policy.update_interval 100 \
  --rl.policy_grad_accum 4 \
  --rl.buffer_capacity 250000 \
  --rl.noise_level 0.3 \
  --collect.eval_interval 100000 \
  || child_status=$?

if [[ "$child_status" -eq 42 ]]; then
  echo "[$(date --iso-8601=seconds)] Job ${SLURM_JOB_ID} requested requeue." >&2
  if scontrol requeue "${SLURM_JOB_ID}"; then
    echo "[$(date --iso-8601=seconds)] Requeue submitted for job ${SLURM_JOB_ID}." >&2
    exit 0
  fi
  echo "[$(date --iso-8601=seconds)] Failed to requeue job ${SLURM_JOB_ID}." >&2
  exit "$child_status"
fi

exit "$child_status"
