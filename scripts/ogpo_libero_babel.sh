#!/bin/bash
#SBATCH --partition=maxlab
#SBATCH --qos=maxlab_qos
#SBATCH --nodelist=babel-m9-16
#SBATCH --job-name=ogpo_libero
#SBATCH --gres=gpu:4
#SBATCH --constraint=VRAM_96GB
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=/home/mananaga/logs/%j/.out
#SBATCH --error=/home/mananaga/logs/%j/.out

# Babel replication of the ORIGINAL OGPO run, i.e. the exact flag set that
# scripts/ogpo_agent/launcher.py produced on swiss-ai. Only the Babel-specific
# environment differs (paths, EGL, NCCL, CUDA headroom). Do not "modernize" the
# flags here -- this script is the baseline the main/babel branch is compared
# against, one change at a time.

set -euo pipefail

PROJECT_DIR=/home/mananaga/VLA/manan_ogpo/vla-post-training
STORE_ROOT=/data/group_data/maxlab/common_datasets/mananaga/vla-post-training
# Reuse the babel branch's already-built venv: pyproject.toml is identical
# between the two branches, and the venv contains no editable install of this
# repo (vla-post-training / openpi / molmospaces all resolve via PYTHONPATH
# below, which points at *this* checkout). Invoke the interpreter directly
# rather than via `uv run` -- uv would re-resolve against this branch's stale
# uv.lock and mutate the shared venv, breaking the babel branch too.
PYTHON=/home/mananaga/VLA/manan_babel/vla-post-training/.venv/bin/python
TASK=libero_90_59
EXP_NAME=pi05_libero_online_ogpo_sft_${TASK}_seed1
CKPT_BASE_DIR=$STORE_ROOT/checkpoints/ogpo_ablations_babel_repro

cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/openpi/packages/openpi-client/src:$PROJECT_DIR/openpi/src:$PROJECT_DIR/openpi/packages/openpi-client:$PROJECT_DIR/molmospaces"

export OPENPI_DATA_HOME="$STORE_ROOT/cache/openpi"
export HF_HOME="$STORE_ROOT/cache/huggingface"
# NOT $HOME/.libero -- that one still points at the pre-move /home/mananaga/vla-post-training
# venv and resolves to dead bddl_files/init_states paths. This copy points at the
# babel venv's libero package (the one $PYTHON uses) and is verified to resolve.
export LIBERO_CONFIG_PATH=/home/mananaga/.libero_ogpo_repro

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

export NCCL_CUMEM_ENABLE=0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
# Leave real VRAM headroom on the shared GPUs for MuJoCo/EGL offscreen
# framebuffers (exp.py setdefaults this to 0.9, which starves them).
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES%%,*}"

export WANDB_ENTITY="${WANDB_ENTITY:-RL-experiments}"

mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME" "$CKPT_BASE_DIR"

echo "[ogpo] node=$(hostname) job=${SLURM_JOB_ID:-none} exp=${EXP_NAME}"

# Flags below mirror launcher.py exactly:
#   - defaults dict + applicable_configs grid (grid wins on conflicts)
#   - rl.critic_pre_training_steps := rl.policy_training_start_step (900)
#   - rl.use_ema_critic := False, since td_weight_schedule.switch_step
#     (1e6) >= num_train_steps (10k), so the critic trains on MC returns
#     throughout -- parity with AWR.
exec "$PYTHON" scripts/ogpo_agent/exp.py \
  pi05_libero_online_ogpo_sft \
  --project_name libero_59 \
  --exp_name "$EXP_NAME" \
  --checkpoint_base_dir "$CKPT_BASE_DIR" \
  --overwrite \
  --seed 1 \
  --log_interval 25 \
  --batch_size 256 \
  --num_train_steps 10000 \
  --lr_schedule.value 2.5e-5 \
  --ema_decay 0.995 \
  --fsdp_devices 4 \
  --collect.tasks "$TASK" \
  --collect.eval_tasks "$TASK" \
  --collect.env_num 1 \
  --collect.eval_env_num 1 \
  --collect.num_rollouts 1 \
  --collect.num_initial_rollouts 5 \
  --collect.collect_interval 300 \
  --collect.eval_interval 300 \
  --collect.num_eval_rollouts 8 \
  --collect.use_time_to_success_as_reward \
  --rl.buffer_capacity 250000 \
  --rl.online_ratio 1.0 \
  --rl.policy_update_interval 1 \
  --rl.policy_training_start_step 900 \
  --rl.critic_pre_training_steps 900 \
  --rl.num_critic_updates_per_batch 10 \
  --rl.td_weight_schedule.switch_step 1000000 \
  --rl.no-use_ema_critic \
  --rl.store_success_episodes_only \
  --rl.group_num_samples 1 \
  --rl.clip_epsilon 0.01 \
  --rl.bc_coeff 1.0 \
  --rl.num_sde_steps 10 \
  --rl.noise_level 0.3 \
  --rl.adv_strategy subtract_v
