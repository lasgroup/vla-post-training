#!/bin/bash
# ---------------------------------------------------------------------------
# OGPO 4-task LIBERO, ALIGNED WITH THE REFERENCE IMPLEMENTATION.
#
# Thin wrapper over scripts/ogpo_multitask_4task.sh: it sets only the knobs that
# differ and delegates everything else, so the ~60-line env preamble is not
# cloned a third time (stability_study.sh <-> ogpo_multitask_4task.sh have
# already diverged; see docs/code/ gotchas).
#
# Reference: /home/pchellap/Projects/SafeVADAR/OGPO_public, recipe
#            scripts/ogpo/square_image_paligemma.sh (frozen PaliGemma encoder
#            restored from a pi05_libero checkpoint).
# Evidence for every value: docs/changes/2026-08-20-ogpo-reference-alignment/
# Measurements motivating it:  reports/findings.md §10,
#                              reports/ogpo_reference_divergence.md
#
# WHAT CHANGES vs. the baseline recipe
#   CONS=1        advantage_combination -> grpo_conservative  (reference default
#                 for its image recipe; we ran it OFF in 9 of 10 arms)
#   NORM=0        normalize_group_advantage OFF   } no counterpart upstream; §10d
#   MT_ADV=0      normalize_advantage_per_task OFF} shows they rescale a collapsed
#   CLIP_SYM=     adv_clip_sym OFF                } critic to a fixed std of 0.36
#   NUM_QS=10     num_qs/num_vs 2 -> 10           (reference: 10)
#   CRITIC_RED=mean  reduction min -> mean        (reference: q_agg=mean)
#   TD_W=0.95     95% TD / 5% MC critic target    (the only measured lever that
#                 moved q_value_mean off the -1/(1-gamma) fixed point: -189.5)
#   BON_N=8       best-of-N collection            (reference: best_of_n=8)
#   SB_Q=1        extra success-only critic batch (reference: use_success_buffer_q)
#   SUCC_BONUS=90 terminal success reward         (reference pays +5.0 on each of up
#                 to 9 success steps = +45, i.e. 45% of its -100 floor; the same ratio
#                 on our -200 floor is 90. Gives a 33.1% success/failure gap against
#                 the reference's 32.1%; today's no-bonus reward gives 22.8%.)
#
# WHAT DELIBERATELY DOES NOT CHANGE (each measured, not assumed)
#   clip_epsilon 0.1  NOT the reference's 0.01: measured ratio_max <= 0.9888 < 0.990,
#                     so 0.01 puts EVERY sample outside the trust region and collapses
#                     the PG to positive-advantages-only.
#   discount 0.995    NOT the reference's 0.99: our successes take ~295 env steps, so
#                     gamma^295 = 0.228 here == the reference's gamma^150 at 0.99.
#                     Copying 0.99 would cut the success/failure gap 22.8% -> 8.9%.
#   G = 8             NOT the reference's 32: 4x the actor fwd+bwd on a 3B expert.
#   BURST = 1000      no upstream counterpart, kept for continuity with the 10 arms.
#   actor LR          constant 2.5e-5; the reference's optimizer reset is out of scope.
#
# Every knob above is still an env var, so this recipe reproduces the baseline stack
# by overriding them back. CONFIG_NAME must be reset too: clip_epsilon, discount and
# group_num_samples live only in the config and have no env var.
#
#   CONFIG_NAME=pi05_libero_online_ogpo_sft NUM_QS=2 CRITIC_RED=min TD_W=1 BON_N=1 \
#     SB_Q=0 SUCC_BONUS=0 NORM=1 MT_ADV=1 CLIP_SYM=4.0 CONS=1 \
#     GPU=0 bash scripts/ogpo_multitask_4task_ref.sh
#
# (CONS=1 is the baseline recipe's own default. The ten completed multitask arms
# passed CONS=0 explicitly; use that to reproduce those instead.)
#
# Usage:  GPU=0 bash scripts/ogpo_multitask_4task_ref.sh
#         GPU=0 DRY=1 bash scripts/ogpo_multitask_4task_ref.sh   # print, don't run
# ---------------------------------------------------------------------------
set -euo pipefail

export ARM="${ARM:-ref}"
export CONFIG_NAME="${CONFIG_NAME:-pi05_libero_online_ogpo_ref}"

# Advantage pipeline: conservative gate on, all three ours-only normalizers off.
export CONS="${CONS:-1}"
export NORM="${NORM:-0}"
export MT_ADV="${MT_ADV:-0}"
export CLIP_SYM="${CLIP_SYM-}"

# Critic: 10-head ensemble reduced by mean, 5% MC in the target, success oversampling.
export NUM_QS="${NUM_QS:-10}"
export CRITIC_RED="${CRITIC_RED:-mean}"
export TD_W="${TD_W:-0.95}"
export SB_Q="${SB_Q:-1}"

# Collection: best-of-8 with the critic, and a terminal success bonus.
export BON_N="${BON_N:-8}"
export SUCC_BONUS="${SUCC_BONUS:-90}"

exec bash "$(dirname "$0")/ogpo_multitask_4task.sh" "$@"
