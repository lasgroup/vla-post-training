#!/bin/bash
# Wave 1 of the single-task stability study on intern-bc0-shashabc (8x H200).
# One arm per GPU, all nohup'd; logs at ~/stab_<arm>.log.
#
#   GPU  ARM     Interventions                                   Tests
#   0    base_s1 none, seed 1                                    noise floor
#   1    base_s2 none, seed 2                                    noise floor
#   2    E       ema 0.999                                       old-policy/eval anchor
#   3    N       quantile normalizer                             advantage-scale growth
#   4    NC      normalizer + sym clip 4.0                       tail spikes over N
#   5    G       grad accum M=2                                  state diversity alone
#   6    ENCG    ema .999 + norm + clip 4 + accum 2              the package
#   7    ENC     package minus accum                             is accum needed?
#
# Baseline seed note: seeds 1/2 are the noise floor; the original abl_viii_r2
# (B200) was seed 0 but is NOT directly comparable across hardware, hence two
# fresh baselines here.
set -euo pipefail
cd ~/vla-post-training

launch() {
  local gpu="$1"; shift
  local arm="$1"; shift
  echo "[wave1] gpu=$gpu arm=$arm $*"
  env GPU="$gpu" ARM="$arm" "$@" nohup bash scripts/stability_study.sh \
    > "$HOME/stab_${arm}.log" 2>&1 &
  sleep 20   # stagger so uv/JAX cache races and EGL init don't collide
}

launch 0 base_s1 SEED=1
launch 1 base_s2 SEED=2
launch 2 E       EMA=0.999
launch 3 N       NORM=1
launch 4 NC      NORM=1 CLIP_SYM=4.0
launch 5 G       ACCUM=2
launch 6 ENCG    EMA=0.999 NORM=1 CLIP_SYM=4.0 ACCUM=2
launch 7 ENC     EMA=0.999 NORM=1 CLIP_SYM=4.0

sleep 5
pgrep -af stability_study.sh | grep -v pgrep || true
echo "[wave1] all launched"
