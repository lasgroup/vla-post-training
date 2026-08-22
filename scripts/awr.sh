#!/bin/bash
#SBATCH --partition=maxlab
#SBATCH --qos=maxlab_qos
#SBATCH --nodelist=babel-m9-16
#SBATCH --job-name=awr_libero
#SBATCH --gres=gpu:4
#SBATCH --constraint=VRAM_96GB
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --output=/home/mananaga/logs/%j/.out
#SBATCH --error=/home/mananaga/logs/%j/.out

set -euo pipefail

PROJECT_DIR=/home/mananaga/VLA/manan_babel/vla-post-training
STORE_ROOT=/data/group_data/maxlab/common_datasets/mananaga/vla-post-training
EXP_NAME=pi05_libero_online_aw_sft_libero_90_44_seed0_nonorm
CKPT_BASE_DIR=$STORE_ROOT/checkpoints/awr_multitask_with_bon_cfg

cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/openpi/packages/openpi-client/src:$PROJECT_DIR/openpi/src:$PROJECT_DIR/openpi/packages/openpi-client:$PROJECT_DIR/molmospaces"

export OPENPI_DATA_HOME="$STORE_ROOT/cache/openpi"
export HF_HOME="$STORE_ROOT/cache/huggingface"
export LIBERO_CONFIG_PATH="$HOME/.libero"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

export NCCL_CUMEM_ENABLE=0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
# Leave real VRAM headroom on the shared GPUs so MuJoCo/EGL offscreen framebuffers
# if the crash still bites.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES%%,*}"

mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME" "$CKPT_BASE_DIR"

echo "[awr] node=$(hostname) job=${SLURM_JOB_ID:-none} exp=${EXP_NAME}"

# Schedule is matched to BOTH baselines at once, which is possible because AWR
# gates its critic and policy updates off the same step counter independently
# (advantage_weighted_sft_learner.py:605-612):
#   100000 steps / policy.update_interval 20 = 5000 policy updates  -> == fsft_libero_babel.sh
#   100000 steps / critic.update_interval  1 = 100000 critic updates -> == bon_libero_babel.sh
#   100000 steps / collect_interval    10000 = 10 collection rounds  -> == both
# So vs FSFT the only algorithmic difference is the loss weight (FSFT uses w=1
# on a success-only buffer, AWR uses w=exp(A/beta) on all data), and vs BoN the
# critic gets an identical training budget — a BoN win can't be written off as
# its critic simply having been trained 20x longer.
#
# Do not set policy.update_interval back to 1 without also dropping
# num_train_steps: it would cost 100000 full pi0.5 gradient steps.
#
# The critic recipe is copied from bon_libero_babel.sh so the two runs learn
# their critics identically. The defaults differ from BoN in two ways:
#   td_weight_schedule 1/1/999999 -> pure TD from step 0, instead of the default
#     0->1 at step 1000 (MC regression for the first 1000 steps).
#   pre_training_steps 0 -> no q/v optimizer reset + EMA re-seed at step 1000
#     (advantage_weighted_sft_learner.py:578).
#
# Advantage normalization is OFF. The weight is exp(A / beta) on raw Q - V,
# with no per-task baseline: update_actor.py:147-152 takes the else branch and
# the learner feeds it bias=0, scale=1 (advantage_weighted_sft_learner.py:762).
# awr_normalization.sh is the same recipe with the per-task quantile baseline,
# kept as the comparison arm.
#
# What that baseline was removing, and now is not:
#   - the per-task offset in Q - V (critic level error). At beta=1 an offset of
#     2.3 (1.2% of the value scale, well under the critic's own TD RMSE of 4.8)
#     is a 10x weight ratio between tasks, so one task takes most of the batch's
#     weight mass. That is the suspected cause of every previous multi-task run
#     improving only libero_90_31 at beta = 0.05, 1 and 10 alike, while the same
#     config works single-task.
#   - the common-mode level, which moves several value units between actor
#     updates because the critic takes update_interval steps in between, so the
#     mean weight (and the effective lr) drifts with it.
# Both now land in the weights. Watch actor/weight_share/<task>: if it leaves
# ~1/4 the multi-task collapse is back, and that is the result this run is for.
#
# beta is back in raw value units, hence 1.0 rather than the normalized 0.5.
#
# weight_clip 3.0 caps the exponent, so the largest weight is exp(3) ~= 20 --
# the usual AWR cap. Advantages are heavy-tailed here (advantage_max ran 47-70
# at std 1.2, ~50 sigma), so at beta=1 the positive tail sits against the clip
# and selectivity comes from the clip, not beta. Uncentered, the negative half
# of the batch is exponentially suppressed instead of merely down-weighted, so
# expect a lower ESS/N than the 0.6 the normalized run held.
#
# advantage_scale 1.0 is a placeholder: uncentered weights straddle 1 rather
# than sitting above it, so the ~3.5x gradient inflation the q05 centering
# caused is gone. Read actor/weight_mean off the first few hundred steps and
# set this to it, or the lr is no longer matched to fsft_libero_babel.sh.
#
# The normalizer EMA state is bypassed entirely, but _per_group_stats still
# runs, so actor/normalizer_{bias,scale}/<task> keep logging as diagnostics.
#
# Untouched AWR-only knobs worth revisiting:
# --rl.advantage_weight_type relu \
# --rl.advantage_combination conservative \
# --rl.critic.per_critic_value_target \
# --rl.critic.q_bootstrap_reduction mean \

exec uv run scripts/exp.py \
  pi05_libero_online_aw_sft \
  --project_name openpi \
  --group_name awr_bon_recipe_babel \
  --exp_name "$EXP_NAME" \
  --checkpoint_base_dir "$CKPT_BASE_DIR" \
  --seed 0 \
  --fsdp_devices 4 \
  --overwrite \
  --log_interval 25 \
  --save_interval 100000 \
  --num_train_steps 100000 \
  --lr_schedule.value 2.5e-5 \
  --max_runtime 169200 \
  --collect.tasks libero_90_79 libero_90_31 libero_90_82 libero_90_38 \
  --collect.eval_tasks libero_90_79 libero_90_31 libero_90_82 libero_90_38 \
  --collect.store_prefix_rep \
  --collect.collect_interval 10000 \
  --collect.num_rollouts 20 \
  --collect.env_num 8 \
  --collect.eval_env_num 8 \
  --collect.eval_interval 99999 \
  --rl.discount 0.995 \
  --rl.online_ratio 1.0 \
  --rl.buffer_capacity 500000 \
  --rl.policy.update_interval 20 \
  --rl.policy.training_start_step 0 \
  --rl.critic.no-use_distributional_critic \
  --rl.critic.num_value_bins 1 \
  --rl.critic.batch_size 1024 \
  --rl.critic.pre_training_steps 0 \
  --rl.critic.td_weight_schedule.init_value 1 \
  --rl.critic.td_weight_schedule.end_value 1 \
  --rl.critic.td_weight_schedule.switch_step 999999 \
  --rl.critic.use_bronet \
  --rl.critic.bronet_hidden_dim 1024 \
  --rl.beta 1.0 \
  --rl.advantage_scale 1.0 \
  --rl.weight_clip 3.0 \
  --rl.no-normalize_advantages \
  --batch_size 256
