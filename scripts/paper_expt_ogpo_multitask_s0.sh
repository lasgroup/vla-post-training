#!/bin/bash
#SBATCH --partition=maxlab
#SBATCH --qos=maxlab_qos
#SBATCH --job-name=paper_ogpo_mt4_s0
#SBATCH --gres=gpu:1
#SBATCH --constraint=VRAM_96GB
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/pchellap/logs/%x_%j.out
#SBATCH --error=/home/pchellap/logs/%x_%j.out
# ---------------------------------------------------------------------------
# Paper experiment: OGPO, reference-aligned (pi05_libero_online_ogpo_ref),
# 4-task LIBERO (libero_90_79/31/82/38 -- same set as
# pi05_libero_online_ogpo_sft_pertask), evaluated IN-DISTRIBUTION (the same 4
# tasks only). Seed 0 of 3 (paper_expt_ogpo_multitask_s{0,1,2}.sh). Multitask
# counterpart of paper_expt_ogpo_single_task_s{0,1,2}.sh.
#
# Source: scripts/configs/policy_extraction/libero/parl_libero_tasks4_seed0_v0.yaml
# on origin/marco/parl (config_name=pi05_libero_online_aw_sft), same recipe
# family as the single-task port -- see that script's header for the full
# AWR/PARL background (rl.awr_loss_weight/filtered_sft_weight dead for OGPO,
# rl.filter_sft_by_success/rerank_buffer_actions nonexistent in this repo).
#
# Launch vehicle: scripts/ogpo_multitask_4task_ref.sh (CONFIG_NAME=
# pi05_libero_online_ogpo_ref, wraps ogpo_multitask_4task.sh), NOT
# stability_study_ref.sh -- this is the multitask analog, with its own
# defaults that partly overlap and partly diverge from the single-task
# script's. Decided with the maintainer 2026-09-14, diffed directly against
# the source yaml (not just against the single-task choices, which is where
# the single-task port's unflagged gaps came from):
#
#   MATCHES SOURCE:
#     num_train_steps 500001 (source: 500000; +1 so range(0,500001) reaches
#       step 500000 itself -- eval_interval=10000 would otherwise miss the
#       last eval, same reasoning as stability_study_ref_maxlab.sbatch:56).
#     rl.n_samples 32 (BON_N=32, not the script's own default of 8).
#     rl.buffer_capacity 250000 (not the script's own multitask default of
#       500000 -- overridden via trailing --rl.buffer_capacity).
#
#   DIFFERS FROM SOURCE (deliberately, same class of decision as the
#   single-task port -- these are OGPO-internal knobs the AWR source has no
#   analog for, or where the script's own established convention was kept):
#     rl.policy.update_interval/training_start_step 10/900 (script's hardcoded
#       values, not source's effective 100/100) -- SAME open question as the
#       single-task runs, resolved the same way here.
#     rl.critic.td_weight_schedule 0.95/0.95 via TD_W (not source's [1,1] "TD
#       regression") -- same as single-task.
#     rl.critic.num_qs/num_vs 10 via NUM_QS (not source's unset/2) -- same as
#       single-task.
#     rl.noise_level 0.3 (not the script's hardcoded 0.02) -- matches the
#       single-task runs' override, not a source value (source has no
#       noise_level concept, AWR-only recipe never had this field).
#     max_runtime 169200 (47h, the script's OWN default, unlike the
#       single-task recipe where we set 172800==48h==wall with no margin).
#       Kept at the script's safer default this time.
#
#   OGPO-ONLY, NO SOURCE ANALOG, SCRIPT DEFAULTS KEPT AS-IS:
#     rl.pg_start_step/pg_ramp_steps: PG_START=0 (DISABLED) -- the script's
#       own default is 20000/5000 (BC-only warmstart, ON), matching the mt4
#       reference-alignment convention used elsewhere in this repo. We turned
#       it OFF here specifically to match what the single-task runs actually
#       ran with (no warmstart), not because the source has an opinion --
#       AWR has no such concept at all (config.py has no pg_start_step field
#       on AdvantageWeightedSFTLearnerConfig).
#     rl.post_collection_critic_steps 1000 (BURST, script's own default,
#       kept).
#     collect.eval_tasks: in-distribution only (HELDOUT=0, script's own
#       default) -- NOT the source's 4-train + 25-held-out = 29 tasks. Same
#       lesson as the single-task eval-cost blowup, applied proactively here.
#     collect.collect_interval 10000 (script's own default, NOT the source's
#       50000).
#     Everything else this script doesn't override (CONS=1/grpo_conservative,
#       NORM=0, MT_ADV=0, MT_BAL=1, SB_Q=1/critic_success_oversample,
#       SUCC_BONUS=90, group_num_samples=8, clip_epsilon=0.1,
#       use_success_buffer, dedup_group_prefix, PER_TASK_CRITIC=0/shared
#       critics) is the script's unmodified reference-aligned recipe --
#       inherited from pi05_libero_online_ogpo_ref's own registered defaults
#       or ogpo_multitask_4task_ref.sh's own overrides, same as the
#       single-task port.
#
# CKPT_BASE_DIR is NOT overridden here: ogpo_multitask_4task.sh already
# defaults it to group storage (/data/group_data/maxlab/common_datasets/
# $USER/vla-post-training/checkpoints/ogpo_multitask_4task), unlike
# stability_study.sh which needed an explicit redirect in the single-task
# maxlab wrapper.
#
# Submit:  sbatch scripts/paper_expt_ogpo_multitask_s0.sh
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_DIR=/home/pchellap/Projects/OGPO-VLA/vla-post-training
cd "$PROJECT_DIR"

export ARM="paper_expt_ogpo_multitask"
export SEED="0"
export NUM_STEPS="500001"
export BON_N="32"
export PG_START="0"

# SLURM renumbers the allocated GPUs inside the cgroup; take the whole list
# so a --gres=gpu:2 allocation is actually visible to JAX.
export GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"

echo "[paper-expt-mt4] node=$(hostname) job=${SLURM_JOB_ID:-?} arm=$ARM seed=$SEED"
echo "[paper-expt-mt4] gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo n/a)"

# exp.py exits 42 when it has saved a resumable epoch and wants the wall
# clock back (scripts/exp.py:5, :191-193); requeue then continues from that
# checkpoint. Plain `bash`, not `exec` -- the exit status has to come back
# to this shell.
child_status=0
bash scripts/ogpo_multitask_4task_ref.sh \
  --project_name vla-post-training \
  --group_name policy_extraction_tasks4_ogpo_ref_v0 \
  --rl.buffer_capacity 250000 \
  --rl.noise_level 0.3 \
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
