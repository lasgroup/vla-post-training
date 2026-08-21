#!/bin/bash
#SBATCH --partition=maxlab
#SBATCH --qos=maxlab_qos
#SBATCH --job-name=mt4_ref_smoke
#SBATCH --gres=gpu:1
#SBATCH --constraint=VRAM_96GB
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=02:00:00
#SBATCH --output=/home/pchellap/logs/%x_%j.out
#SBATCH --error=/home/pchellap/logs/%x_%j.out
# ---------------------------------------------------------------------------
# MEMORY + PATH SMOKE for the reference-aligned OGPO stack. NOT a science run.
#
# Same resources as scripts/ogpo_multitask_4task_maxlab.sbatch (the ten completed
# arms): 1x VRAM_96GB, 16 cpus, 150G host RAM. Only --time is shortened.
# Note the `d20` arm OOM'd at this exact 150G, so a host-OOM here is a RESULT,
# not a misconfiguration.
#
# WHAT IT IS TESTING (nothing else has ever exercised these):
#   1. num_qs = num_vs = 10 -- BroNetStateActionCritic builds TEN separate towers
#      for Q and ten for V (bronet_critic.py:118-121, no vmap) at critic batch
#      1024, against a ~34.7 GiB floor.
#   2. Best-of-N collection at N=8 on the OGPO learner. The code is inherited from
#      AdvantageWeightedSFTLearner.sample_actions:313 and has NEVER run under OGPO
#      (n_samples was 1); it tiles the policy forward 8x per query.
#   3. The A4 success-oversampling critic batch actually firing.
#
# WHAT IS DELIBERATELY SHRUNK (and why it does not affect the memory question):
#   N_ROLLOUTS=2 INIT_ROLLOUTS=2 -> 16 episodes at step 0 instead of 60. Collection
#   COUNT does not change peak memory; it only changes how long step 0 takes. 16
#   episodes still yields ~5600 buffer windows (>1024 critic batch) and ~1400
#   success windows (>1024), so both the critic path and A4 still engage.
#   NUM_STEPS=1200 clears policy.training_start_step=900, so the actor jits compile
#   and run -- that is where peak memory lives. eval_interval=10000 and exp.py
#   skipping the step-0 eval mean no eval runs, which is the point.
#
# Everything memory-relevant is untouched: num_qs 10, critic batch 1024, batch_size
# 32, G=8, BON_N=8, num_sde_steps 10, dedup_group_prefix on.
#
# PG stays muted throughout (pg_start_step=20000 > 1200). That is fine for memory:
# jit-2a still runs with a zeroed advantage -- "sampling/rescoring still run,
# keeping rng streams and jit structure identical" (config.py). It does mean this
# smoke says NOTHING about whether the change helps.
#
# Submit:  sbatch scripts/ogpo_ref_smoke_maxlab.sbatch
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_DIR=/home/pchellap/Projects/OGPO-VLA/vla-post-training
cd "$PROJECT_DIR"

export ARM="${ARM:-refsmoke}"
export NUM_STEPS="${NUM_STEPS:-1200}"
export N_ROLLOUTS="${N_ROLLOUTS:-2}"
export INIT_ROLLOUTS="${INIT_ROLLOUTS:-2}"

# SLURM renumbers allocated GPUs inside the cgroup.
export GPU="${GPU:-0}"

echo "[smoke] node=$(hostname) job=${SLURM_JOB_ID:-?} gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo n/a)"
echo "[smoke] arm=$ARM steps=$NUM_STEPS rollouts/task=$N_ROLLOUTS init=$INIT_ROLLOUTS"

# Sample GPU + host memory every 15s so a peak is recorded even if the job is killed.
MEMLOG="/home/pchellap/logs/${SLURM_JOB_NAME:-mt4_ref_smoke}_${SLURM_JOB_ID:-0}_mem.csv"
(
  echo "unix_s,gpu_used_MiB,gpu_total_MiB,host_used_MB"
  while true; do
    G=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    H=$(free -m | awk '/^Mem:/{print $3}')
    echo "$(date +%s),${G:-,},${H}"
    sleep 15
  done
) > "$MEMLOG" &
MEMPID=$!
# Always print the peak, even on OOM/timeout -- that is the whole point of the run.
trap 'kill $MEMPID 2>/dev/null || true;
      echo "[smoke] ==== PEAK MEMORY ====";
      awk -F, "NR>1 && \$2!=\"\" {if(\$2+0>g)g=\$2+0; if(\$4+0>h)h=\$4+0} END{printf \"[smoke] peak GPU %.1f GiB / %.1f GiB   peak host %.1f GB\n\", g/1024, \$3/1024, h/1024}" "$MEMLOG";
      echo "[smoke] memlog: $MEMLOG"' EXIT

bash scripts/ogpo_multitask_4task_ref.sh
