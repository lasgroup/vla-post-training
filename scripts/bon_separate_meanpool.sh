#!/bin/bash
#SBATCH --partition=maxlab
#SBATCH --qos=maxlab_qos
#SBATCH --nodelist=babel-m9-16
#SBATCH --job-name=bon_seppool
#SBATCH --gres=gpu:4
#SBATCH --constraint=VRAM_96GB
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --output=/home/mananaga/logs/%j/.out
#SBATCH --error=/home/mananaga/logs/%j/.out

set -euo pipefail

# Copy of scripts/bon_libero.sh with ONE change: the critic input is the
# PaliGemma prefix mean-pooled over image tokens and over language tokens
# SEPARATELY, concatenated (2 x 2048 instead of 2048), so the critic gets a
# distinct language representation for instruction following. Selected by
# --collect.prefix_pooling image_text_mean (default "mean" = bon_libero.sh).
# Ported from manan_babel's scripts/bon_libero_babel.sh, with two changes:
# it runs THIS checkout's code (PROJECT_DIR below), and the task set is
# narrowed to libero_90_38 to match scripts/fsft_libero_babel.sh here.
PROJECT_DIR=/home/mananaga/VLA/ogpo/vla-post-training
STORE_ROOT=/data/group_data/maxlab/common_datasets/mananaga/vla-post-training
EXP_NAME=pi05_libero_online_best_of_n_libero_90_38_seed0_imgtxtpool
CKPT_BASE_DIR=$STORE_ROOT/checkpoints/bon_separate_meanpool

# Reuse the babel venv rather than building one here: pyproject.toml and
# uv.lock are byte-identical between the checkouts, and that venv has no
# editable install of vla-post-training/openpi/molmospaces -- all three come
# from PYTHONPATH below, so the interpreter runs THIS tree's code. Call the
# interpreter directly, never `uv run`: uv re-resolves against the branch
# lockfile and mutates the venv, which would break the babel checkout too.
PY="${PY:-/home/mananaga/VLA/manan_babel/vla-post-training/.venv/bin/python}"
[ -x "$PY" ] || { echo "[bon] no interpreter at $PY" >&2; exit 1; }

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
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES%%,*}"

mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME" "$CKPT_BASE_DIR"

echo "[bon] node=$(hostname) job=${SLURM_JOB_ID:-none} exp=${EXP_NAME}"

# Best-of-N never updates the policy — BestofNLearner.update() only steps the Q
# and V critics. All the gain comes at collection time: rl.n_samples action
# chunks are sampled per env and the highest-Q one is executed. So the
# FSFT/AWR knobs that shape the policy loss (lr_schedule, rl.beta,
# rl.policy.*, batch_size for the actor) have no effect here, and the schedule
# follows scripts/configs/multitask/bofn/libero/bofn_libero_tasks4-16_seed0_v0.yaml
# instead: many cheap critic steps (100k) with sparse collection (every 10k),
# not FSFT's 5k/500.
#
# store_prefix_rep is required: the critics consume cached prefix embeddings,
# and with train_on_policy_value_function=False the learner then drops images
# from the online buffer (~45x smaller).
#
# inference_start_step 1 skips best-of-N at step 0, when the critic is still
# random and N-way sampling would just cost collection time.

exec "$PY" scripts/exp.py \
  pi05_libero_online_best_of_n \
  --project_name openpi \
  --group_name bon_multitask_babel \
  --exp_name "$EXP_NAME" \
  --checkpoint_base_dir "$CKPT_BASE_DIR" \
  --seed 0 \
  --fsdp_devices 4 \
  --overwrite \
  --log_interval 25 \
  --save_interval 100000 \
  --num_train_steps 100000 \
  --max_runtime 169200 \
  --collect.tasks libero_90_38 \
  --collect.eval_tasks libero_90_38 \
  --collect.store_prefix_rep \
  --collect.prefix_pooling image_text_mean \
  --collect.collect_interval 10000 \
  --collect.num_rollouts 20 \
  --collect.env_num 8 \
  --collect.eval_env_num 8 \
  --collect.eval_interval 99999 \
  --rl.discount 0.995 \
  --rl.online_ratio 1.0 \
  --rl.buffer_capacity 250000 \
  --rl.n_samples 32 \
  --rl.critic.batch_size 1024 \
  --rl.critic.pre_training_steps 0 \
  --rl.critic.inference_start_step 1 \
  --rl.critic.td_weight_schedule.init_value 1 \
  --rl.critic.td_weight_schedule.end_value 1 \
  --rl.critic.td_weight_schedule.switch_step 999999 \
  --rl.critic.no-use_distributional_critic \
  --rl.critic.num_value_bins 1 \
  --rl.critic.use_bronet \
  --rl.critic.bronet_hidden_dim 1024 \
  --batch_size "${BATCH_SIZE:-256}"
