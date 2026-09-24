#!/bin/bash
#SBATCH --partition=maxlab
#SBATCH --qos=maxlab_qos
#SBATCH --job-name=paper_ogpo_pc_s0
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
# Paper experiment: OGPO, single train task libero_90_82, matching the
# hyperparameter budget described in the paper excerpt the maintainer pasted
# 2026-09-19 (M=10 policy-improvement iterations, K=20 episodes/task/epoch,
# 2 Q/V critic heads aggregated by min) crossed with marco/parl's own
# reference factory defaults (make_base_libero_config: batch_size=256,
# num_train_steps=10_000) for the fields the paper excerpt didn't state.
# Seed 0 of 3 (paper_expt_ogpo_single_task_paperconfig_s{0,1,2}.sh).
#
# NEW checkpoint dir (ARM includes "paperconfig") -- deliberately NOT the
# same exp_name as paper_expt_ogpo_single_task_s0.sh: that run's checkpoint
# has a 10-head critic (num_qs=10) and batch_size=32; this run's 2-head
# critic (num_qs=2) and batch_size=256 have incompatible parameter shapes,
# so resuming into the old checkpoint would fail (or silently mismatch).
#
# REVISION 2026-09-19: the first cut of this recipe used num_train_steps
# =10000 / collect+eval every 1000 steps, reading marco/parl's own
# make_base_libero_config factory default (num_train_steps=10_000) as "the"
# reference value. That default is generic to every algorithm on that
# branch, not specific to this paper's policy-adaptation experiments -- and
# the ACTUAL parl_libero_tasks1_seed0_v0.yaml (this port's real source,
# same one paper_expt_ogpo_single_task_s{0,1,2}.sh is ported from) overrides
# num_train_steps to 100000, moving AWAY from that 10000 default. Corrected
# back to 100001 steps / eval every 10000, matching our other single-task
# runs and the yaml's own stated value, on the maintainer's explicit call.
#
# Decided with the maintainer 2026-09-19, changes FROM the established
# paper_expt_ogpo_single_task_s{0,1,2}.sh recipe:
#   - rl.critic.num_qs/num_vs 2, rl.critic.reduction min (not the _ref
#     config's 10/mean) -- the paper excerpt explicitly states "an ensemble
#     of two Q and V-heads... aggregate them by taking the minimum."
#   - batch_size 32 (the script's own default, NOT marco/parl's factory
#     batch_size=256) + rl.policy_grad_accum 8, giving the SAME effective
#     256-states-per-optimizer-step marco/parl's own factory default implies,
#     without a single actor call ever exceeding the validated 32 x
#     group_num_samples=8 = 256-chain footprint. A literal batch_size=256
#     call would expand to 256 x 8 = 2048 chains through the full ~3B-param
#     pi0.5 backbone+action-expert -- the measured ceiling for 256 chains
#     (10-head critic) is already 75.8/95.6 GiB peak on a single 96GB GPU
#     (mt4 smoke, job 10178533); 2048 chains would need ~600+ GiB, nowhere
#     close to fitting on one GPU. marco/parl's batch_size=256 was
#     calibrated for AWR's own actor step, which has no group_num_samples
#     -style G-fold expansion -- reusing it verbatim against OGPO's PPO
#     surrogate isn't an apples-to-apples port. grad_accum trades wall-clock
#     (8x more sequential micro-batches) for staying within the single-GPU
#     memory budget instead.
#
# NOTE, not yet resolved: the paper excerpt also states "<100 gradient
# steps" for single-task policy adaptation and "100+" for the critic. This
# recipe does NOT enforce that -- rl.policy.update_interval/training_start_
# step stay at the script's 10/900 (same as our other single-task runs),
# which is on the order of ~9000 actor updates over 100000 steps, and the
# critic updates roughly every step by default. Flagged to the maintainer;
# not changed without an explicit decision on how "gradient steps" should
# map onto update_interval here.
#
# UNCHANGED from paper_expt_ogpo_single_task_s{0,1,2}.sh (not part of this
# discussion, kept as previously decided): rl.n_samples 32, rl.noise_level
# 0.3, task libero_90_82, in-distribution eval only, lr_schedule.value
# 2.5e-5 (NOT marco/parl's factory 5e-5 -- only batch_size was requested from
# that reference), TD_W 0.95, SUCC_BONUS 90, rl.policy.update_interval 10 /
# training_start_step 900, group_num_samples 8, clip_epsilon 0.1,
# use_success_buffer, critic_success_oversample, max_runtime 169200 (moot at
# this step count -- the run should finish in well under the 48h wall).
#
# Submit:  sbatch scripts/paper_expt_ogpo_single_task_paperconfig_s0.sh
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_DIR=/home/pchellap/Projects/OGPO-VLA/vla-post-training
cd "$PROJECT_DIR"

export ARM="paper_expt_ogpo_single_task_paperconfig_s0"
export TASK="libero_90_82"
export SEED="0"
export N_STEPS="100001"
export QS="2"

# SLURM renumbers the allocated GPUs inside the cgroup; take the whole list
# so a --gres=gpu:2 allocation is actually visible to JAX.
export GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"

# Checkpoints default to GROUP STORAGE, not $HOME (100G-quota NFS mount).
export CKPT_BASE_DIR="${CKPT_BASE_DIR:-/data/group_data/maxlab/common_datasets/${USER:-pchellap}/vla-post-training/checkpoints/stability_study}"

echo "[paper-expt-pc] node=$(hostname) job=${SLURM_JOB_ID:-?} arm=$ARM task=$TASK seed=$SEED"
echo "[paper-expt-pc] gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo n/a)"

child_status=0
bash scripts/stability_study_ref.sh \
  --project_name vla-post-training \
  --group_name policy_extraction_tasks1_ogpo_paperconfig_v0 \
  --rl.n_samples 32 \
  --rl.noise_level 0.3 \
  --rl.critic.reduction min \
  --rl.policy_grad_accum 8 \
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
