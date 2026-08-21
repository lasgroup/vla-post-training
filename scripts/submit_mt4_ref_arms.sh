#!/bin/bash
# ---------------------------------------------------------------------------
# Reference-aligned OGPO: the CONS paired comparison, at num_qs=10.
#
# WHY TWO ARMS. `v0` vs `ncb` is the only clean single-knob test of the
# conservative advantage gate in the whole campaign, and it came back NEGATIVE:
# paired over the 7 common post-PG eval steps,
#
#     v0 - ncb  =  -3.57  (sd 4.48, n=7, t = -2.11)
#
# i.e. CONS=1 was 3.6 points WORSE, and |t| = 2.11 is the largest single-knob
# signal the campaign produced -- every other contrast (bursts, MC targets, data
# volume) landed at |t| < 1.
#
# BUT `v0` ran the gate at num_qs=2, where sign-unanimity across two heads is
# close to a coin flip (measured cons_zero_frac 8.7% -> 22% over that run). The
# aligned stack runs 10 heads, which is what the reference uses, and the smoke
# measured cons_zero_frac ~0.50 there. Whether a stricter gate helps (less noise
# survives) or hurts (half the gradient discarded) is genuinely unknown.
#
# So: run both, at 10 heads, everything else byte-identical. This is the paired
# design the campaign lacked -- same seed, same shared deterministic eval stream,
# differing in exactly one field.
#
#   mt4_ref_s0     CONS=1  advantage_combination = grpo_conservative
#   mt4_ref_nc_s0  CONS=0  advantage_combination = reduced
#
# Everything else in both: success_reward_bonus=90, num_qs=num_vs=10,
# reduction=mean, td_weight=0.95, n_samples=8 (best-of-N collection),
# critic_success_oversample, all three local normalizers OFF, clip_epsilon=0.1,
# discount=0.995, G=8, burst=1000, warmstart 20000/5000.
#
# Cost: 2 GPUs x ~16 h.
#
# Submit:  bash scripts/submit_mt4_ref_arms.sh
# Inspect: DRY=1 bash scripts/submit_mt4_ref_arms.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SBATCH_SCRIPT=scripts/ogpo_multitask_4task_ref_maxlab.sbatch
RUN=(sbatch)
[ "${DRY:-0}" = "1" ] && RUN=(echo sbatch)

# Arm 1 -- conservative gate ON (the reference's own choice for its image recipe)
"${RUN[@]}" --export=ALL,ARM=ref,SEED=0,CONS=1,NUM_STEPS=100001 \
  --job-name=mt4_ref "$SBATCH_SCRIPT"

# Arm 2 -- conservative gate OFF (matches the campaign's best-performing setting)
"${RUN[@]}" --export=ALL,ARM=ref_nc,SEED=0,CONS=0,NUM_STEPS=100001 \
  --job-name=mt4_ref_nc "$SBATCH_SCRIPT"
