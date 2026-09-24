#!/bin/bash
#SBATCH --partition=maxlab
#SBATCH --qos=maxlab_qos
#SBATCH --job-name=paper_ogpo_st_s0
#SBATCH --gres=gpu:1
#SBATCH --constraint=VRAM_96GB
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=48:00:00
#SBATCH --requeue
# A requeue reuses %j, and sbatch's default open mode truncates, so without
# this each attempt erases the log of the attempt it is recovering from
# (scripts/launcher.py:219-220 pairs the two flags for the same reason).
#SBATCH --open-mode=append
#SBATCH --output=/home/pchellap/logs/%x_%j.out
#SBATCH --error=/home/pchellap/logs/%x_%j.out
# ---------------------------------------------------------------------------
# Paper experiment: OGPO, reference-aligned (pi05_libero_online_ogpo_ref),
# single train task libero_90_82, evaluated IN-DISTRIBUTION (libero_90_82
# only). Seed 0 of 3 (paper_expt_ogpo_single_task_s{0,1,2}.sh).
#
# REVISION 2026-09-13: dropped the original 26-task LIBERO-90 generalization
# eval (libero_90_82 + 25 held-out tasks) back to TASK alone -- that eval
# round alone cost ~10h/round (26x the episode count, plus rl.n_samples=32
# best-of-N reranking applying to every eval query too, since evaluate_policy
# (src/training/collect.py:55) calls the same agent.sample_actions as
# collection). At 48h wall / max_runtime, that blocked ever completing a
# training round. The 3 original jobs (10416778/79/80) were cancelled at
# step 10000, mid-first-eval -- no training progress lost, the step-10000
# checkpoint (weights/optimizer/buffer) predates that eval and is unaffected.
# These resubmits --resume from it.
#
# Ported from an AWR+BestofN recipe (project_name=vla-post-training,
# group_name=policy_extraction_tasks1_parl_libero_v0_seed0) onto OGPO.
# Two fields from that recipe (rl.awr_loss_weight, rl.filtered_sft_weight)
# were dropped: dead for OGPO, never read by ogpo/update_actor.py. Two more
# (rl.filter_sft_by_success, rl.rerank_buffer_actions) do not exist anywhere
# in this repo or its git history and were dropped rather than guessed at.
#
# Deviations from scripts/stability_study_ref.sh's own baseline recipe,
# decided with the maintainer 2026-09-12:
#   - rl.n_samples 32, not the _ref config's own registered default of 8:
#     4x the collection-time critic-scoring compute, untested at this scale
#     for OGPO before this run (pi05_libero_online_ogpo_sft_ref's only prior
#     use of n_samples>1 is 8, config.py:855).
#   - rl.noise_level 0.3, not the script's unconditionally-emitted 0.02
#     (stability_study.sh:263) -- the OGPOSFTLearnerConfig class default
#     (config.py:240), which the script would otherwise silently override
#     regardless of CONFIG_NAME.
#   - max_runtime 172800 (48h, EQUAL to this job's SLURM wall), not the
#     script's own 169200 (47h, kept deliberately under the wall per its
#     comment at stability_study_ref_maxlab.sbatch:82-83). This removes the
#     1h graceful exit-42-and-requeue margin: if the run hits the wall
#     mid-step, SLURM SIGKILLs it instead of a clean checkpoint+resume.
#     --resume from the last completed collect/eval (10k-step) boundary is
#     still safe either way -- this only risks losing partial progress since
#     the last boundary.
#   - rl.policy.update_interval / training_start_step LEFT AT the script's
#     fixed 10 / 900 (stability_study.sh:247-248) -- NOT the _ref config's
#     own registered 20 / 100 (config.py:845).
#   - collect.eval_tasks is left at the script's own default (TASK alone,
#     libero_90_82) -- no override, as of the 2026-09-13 revision above.
#   - project_name / group_name overridden away from the script's own
#     ogpo_stability / stability_study (group_name still says
#     policy_extraction_..., a holdover from the dropped generalization
#     framing -- ask the maintainer before renaming, wandb history already
#     has runs under it).
#
# Everything else (collect.collect_interval/num_rollouts 10000/20,
# collect.env_num/eval_env_num 8/8, collect.eval_interval 10000,
# collect.store_prefix_rep, rl.discount 0.995, rl.online_ratio 1.0,
# rl.buffer_capacity 250000, rl.critic.{batch_size,pre_training_steps,
# use_bronet,bronet_hidden_dim,inference_start_step} 1024/0/True/1024/1,
# rl.critic.td_weight_schedule {0.95,0.95,999999} via TD_W, the _ref config's
# own group_num_samples=8/clip_epsilon=0.1/advantage_combination=
# grpo_conservative/use_success_buffer=True/critic_success_oversample=True,
# lr_schedule.value 2.5e-5, log_interval 25, batch_size 32) is the script's
# unmodified reference-aligned recipe.
#
# Submit:  sbatch scripts/paper_expt_ogpo_single_task_s0.sh
#          DRY=1 bash scripts/stability_study_ref.sh ... to preview first
#          (see the command below with ARM/TASK/SEED exported and DRY=1 set).
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_DIR=/home/pchellap/Projects/OGPO-VLA/vla-post-training
cd "$PROJECT_DIR"

export ARM="paper_expt_ogpo_single_task_s0"
export TASK="libero_90_82"
export SEED="0"

# stability_study.sh's own N_STEPS default (100000) misses the 10th
# eval_interval=10000 boundary at step 100000 itself -- same reasoning as
# stability_study_ref_maxlab.sbatch:56.
export N_STEPS="${N_STEPS:-100001}"

# SLURM renumbers the allocated GPUs inside the cgroup; take the whole list
# so a --gres=gpu:2 allocation is actually visible to JAX.
export GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"

# Checkpoints (rl_state + runtime_state/replay+success shards) default to
# GROUP STORAGE, not $HOME -- home is a 100G-quota NFS mount and a
# 100k-step run's replay buffer + optimizer state does not fit two arms of
# it comfortably (stability_study_ref_maxlab.sbatch:63-74).
export CKPT_BASE_DIR="${CKPT_BASE_DIR:-/data/group_data/maxlab/common_datasets/${USER:-pchellap}/vla-post-training/checkpoints/stability_study}"

echo "[paper-expt] node=$(hostname) job=${SLURM_JOB_ID:-?} arm=$ARM task=$TASK seed=$SEED"
echo "[paper-expt] gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo n/a)"

# In-distribution eval only (libero_90_82): no --collect.eval_tasks override
# here, so it falls through to stability_study.sh's own default of TASK
# alone (stability_study.sh:236).
#
# exp.py exits 42 when it has saved a resumable epoch and wants the wall
# clock back (scripts/exp.py:5, :191-193); requeue then continues from that
# checkpoint. Plain `bash`, not `exec` -- the exit status has to come back
# to this shell.
child_status=0
bash scripts/stability_study_ref.sh \
  --project_name vla-post-training \
  --group_name policy_extraction_tasks1_ogpo_ref_v0 \
  --rl.n_samples 32 \
  --rl.noise_level 0.3 \
  --max_runtime 172800 \
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
