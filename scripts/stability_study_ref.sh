#!/bin/bash
# ---------------------------------------------------------------------------
# Single-task OGPO, ALIGNED WITH THE REFERENCE IMPLEMENTATION.
#
# Thin wrapper over scripts/stability_study.sh: the single-task analogue of
# scripts/ogpo_multitask_4task_ref.sh (which wraps ogpo_multitask_4task.sh the
# same way), so the ~90-line env preamble is not cloned a third time.
#
# UNLIKE the multitask recipe, most reference-alignment fields (num_qs/num_vs,
# critic.reduction, critic_success_oversample, n_samples, advantage_combination,
# normalize_group_advantage, normalize_advantage_per_task, adv_clip_sym) are
# baked into pi05_libero_online_ogpo_ref's own dataclass defaults
# (src/training/config.py:843-872) and reach the run untouched here --
# stability_study.sh's gates for these (QS, CONS, NORM, CLIP_SYM) are all
# optional and default off/unset, so CONFIG_NAME alone already gets them
# right (verified against source, not assumed from the multitask recipe's
# shape -- see docs/changes/2026-09-07-single-task-ref-recipe/BLAST-RADIUS.md).
#
# Two fields are the exception -- stability_study.sh's fixed flag block
# emits them unconditionally, clobbering the ref config's intent regardless
# of CONFIG_NAME, so they need setting here explicitly:
#   TD_W          rl.critic.td_weight_schedule.{init,end}_value: reference
#                 config default is 0.95 (95% TD / 5% MC).
#   SUCC_BONUS    collect.success_reward_bonus: not baked into ANY config's
#                 own default; reference recipe wants 90.
#
# Reference: docs/changes/2026-08-20-ogpo-reference-alignment/,
#            reports/findings.md section 15.
#
# NOT a working "reproduce the baseline by env vars alone" recipe (unlike
# ogpo_multitask_4task_ref.sh): num_qs/num_vs and adv_clip_sym still answer to
# stability_study.sh's QS/CLIP_SYM gates, but CONS=0 is a SILENT NO-OP here
# (advantage_combination reaches grpo_conservative via the ref config's own
# default, not a flag CONS can override) and critic.reduction/rl.n_samples/
# rl.critic_success_oversample have no knob at all -- stability_study.sh never
# emits those unconditionally, so nothing can override the ref config's
# defaults except a trailing CLI flag, e.g.
# `... bash scripts/stability_study_ref.sh --rl.advantage_combination reduced`.
#
# Usage:  ARM=REF_libero_90_31 TASK=libero_90_31 GPU=0 bash scripts/stability_study_ref.sh
#         ARM=REF_libero_90_31 TASK=libero_90_31 GPU=0 DRY=1 bash scripts/stability_study_ref.sh
# ---------------------------------------------------------------------------
set -euo pipefail

export CONFIG_NAME="${CONFIG_NAME:-pi05_libero_online_ogpo_ref}"
export TD_W="${TD_W:-0.95}"
export SUCC_BONUS="${SUCC_BONUS:-90}"

exec bash "$(dirname "$0")/stability_study.sh" "$@"
