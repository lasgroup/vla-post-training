#!/bin/bash
#SBATCH --partition=maxlab
#SBATCH --qos=maxlab_qos
#SBATCH --nodelist=babel-m9-16
#SBATCH --job-name=awr_group_adv
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
EXP_NAME=pi05_libero_online_aw_sft_libero_90_44_seed0_groupadv
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

echo "[awr-group] node=$(hostname) job=${SLURM_JOB_ID:-none} exp=${EXP_NAME}"

# Group-relative AWR. Same schedule and same critic recipe as awr.sh -- the only
# change is what the actor scores and what it regresses onto
# (update_actor_group.py):
#   awr.sh:  A = Q(s, a_buffer) - V(s), CFM target = a_buffer
#   this:    draw G chunks from the current policy, A_i = Q(s,a_i) - mean_j Q(s,a_j),
#            CFM target = a_i, weight = exp(A_i / beta)
#
# The group mean is an exact per-state baseline, so the per-task Q-V offset and
# the common-mode level drift that rl.normalize_advantages was built to remove
# both cancel inside the group. V is no longer in the actor path at all (it is
# still trained, so critic/value_* stays comparable to awr.sh). Nothing here
# sets rl.normalize_advantages -- the group path ignores it.
#
# WHAT THIS BUYS AND WHAT IT COSTS
# Q is now evaluated at actions it was never trained on, and exp(A/beta) is a
# soft-argmax over G, i.e. a directed search for the critic's largest error at
# that state. Unlike BoN collection -- where a bad pick is corrected by the
# environment's reward on the next collect -- that error goes straight into the
# policy gradient unchecked.
#
# This run takes that risk deliberately and unguarded, so that every knob below
# matches awr.sh and the only difference left is the group advantage itself.
# Two knobs exist for it and are both at the awr.sh value on purpose:
#   policy.training_start_step 0: the actor starts against an untrained critic.
#     In awr.sh that is harmless -- an untrained critic gives near-uniform
#     weights on buffer actions, which degrades to plain BC. Here uniform
#     weights land on self-samples instead, which is self-distillation, so the
#     first few thousand steps are the most likely place to see damage. Raise
#     this to ~5000 if early actor/grad_norm or eval success falls apart.
#   filtered_sft_weight 0.0: AWR's fixed point is anchored to the buffer only
#     because the buffer is fixed data. Regressing onto our own samples removes
#     that anchor, and nothing here replaces it -- entropy is free to collapse.
#     Set this to ~0.5 to add a BC term on successful transitions, computed on
#     the un-repeated batch, if the policy narrows.
#
# BATCH SHAPE
# The actor step repeats each state G times, so the learner subsamples to
# batch_size/G states first and the backward pass stays at 256 chunks. At G=8
# that is 32 distinct states per actor update, 8x fewer than awr.sh -- on a
# 4-task batch some updates will miss a task entirely. If actor/grad_norm is
# too noisy, drop G to 4 before touching batch_size.
#
# beta IS UNSET UNTIL THE FIRST DIAGNOSTIC READ
# The advantage is now a within-state quantity, not a cross-batch one, so
# awr.sh's raw-unit beta does not transfer. The scale that matters is
# actor/within_group_adv_std (mean over states of the std of Q across the G
# samples). Read it in the first few hundred actor steps and set beta to about
# that value, so the exponent spans roughly [-1, 1] within a group. 1.0 below is
# a placeholder, not an estimate.
#
# THE EXPERIMENT THIS RUN IS FOR
# Whether within-group Q spread carries signal at all. Compare
# actor/within_group_adv_std against critic/q_td_loss: if the spread across G
# policy samples at one state is below the critic's own error, the weights are
# ranking critic noise and no amount of beta tuning fixes it. That single
# comparison decides whether this direction is worth pursuing -- check it before
# reading any success number.
# Also watch actor/weight_ess_frac (1.0 uniform, 1/G a hard argmax) and
# actor/weight_share/<task>, which should now sit near 1/4 by construction.
#
# Sampling uses the probability-flow ODE (group_noise_level 0.0), so the G
# chunks are drawn from the policy's own distribution via G different initial
# noise draws. flow_grpo uses the SDE at 0.3 only because it needs per-step
# log-probs for its policy gradient; nothing here does.

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
  --rl.group_advantage \
  --rl.group_size 8 \
  --rl.group_num_steps 10 \
  --rl.group_noise_level 0.0 \
  --batch_size 256
