#!/bin/bash
#SBATCH --partition=maxlab
#SBATCH --qos=maxlab_qos
#SBATCH --nodelist=babel-m9-16
#SBATCH --job-name=awr_libero
#SBATCH --gres=gpu:1
#SBATCH --constraint=VRAM_96GB
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=/home/mananaga/logs/%j/.out
#SBATCH --error=/home/mananaga/logs/%j/.out

set -euo pipefail

# ----------------------------------------------------------------------------
# Knobs (override via environment when calling sbatch / bash)
# ----------------------------------------------------------------------------
DRY_RUN="${DRY_RUN:-0}"
PROJECT_DIR="${PROJECT_DIR:-/home/mananaga/vla-post-training}"
CONFIG_NAME="${CONFIG_NAME:-pi05_libero_online_aw_sft}"
SEED="${SEED:-0}"
TASK="${TASK:-libero_90_44}"          # matches scripts/configs/awr.yaml
PROJECT_NAME="${PROJECT_NAME:-openpi}"          # matches awr.yaml's wandb project
GROUP_NAME="${GROUP_NAME:-awr_bon_recipe_babel}" # awr.yaml's group + _babel suffix
FSDP="${FSDP:-1}"                     # shard train state across N GPUs; keep == --gres=gpu:N.
                                      # NOTE: FSDP>1 (NCCL) currently crashes on
                                      # these Blackwell cards -- see NCCL block below.

# Stable experiment name -> stable checkpoint dir so requeue resumes instead of
# starting over (do NOT put a timestamp here).
EXP_NAME="${EXP_NAME:-${CONFIG_NAME}_${TASK}_seed${SEED}}"
STORE_ROOT="${STORE_ROOT:-/data/user_data/mananaga/vla-post-training}"
CKPT_BASE_DIR="${CKPT_BASE_DIR:-${STORE_ROOT}/checkpoints/${GROUP_NAME}}"

# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------
cd "$PROJECT_DIR"

# openpi + openpi-client + molmospaces are imported via PYTHONPATH (they are git
# submodules, not pip-installed). This mirrors the Dockerfile's PYTHONPATH.
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/openpi/packages/openpi-client/src:$PROJECT_DIR/openpi/src:$PROJECT_DIR/openpi/packages/openpi-client:$PROJECT_DIR/molmospaces"

# Caches on /data (not $HOME, which is small). Base pi05 weights + norm stats are
# fetched anonymously from gs://openpi-assets and cached here on first run.
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${STORE_ROOT}/cache/openpi}"
export HF_HOME="${HF_HOME:-${STORE_ROOT}/cache/huggingface}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.libero}"   # pre-created, avoids LIBERO's interactive prompt

# Headless MuJoCo / robosuite rendering for LIBERO rollouts.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Force glvnd to use NVIDIA's EGL ICD only. Babel compute nodes have mesa-libEGL
# installed alongside libEGL_nvidia, and glvnd tries mesa's DRI2 path first,
# which fails ("/dev/dri/cardN: Permission denied") since DRM nodes are
# restricted to the video group. Pinning to the NVIDIA ICD skips that attempt.
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
# Pin EGL device to the first visible CUDA device, but only if one is set —
# robosuite errors on an empty MUJOCO_EGL_DEVICE_ID instead of falling back.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES%%,*}"
fi

# Multi-GPU (FSDP) NCCL collectives. NCCL 2.26's cuMem (CUDA VMM) allocator
# clashes with JAX's allocator and crashes at clique init with "illegal memory
# access" while creating NCCL buffers (seen on both P2P and SHM transports) ->
# NCCL_CUMEM_ENABLE=0 is the fix. IB disabled (no InfiniBand within one node).
# Set NCCL_DEBUG=INFO to debug.
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# NOTE (2026-07-09): Multi-GPU (FSDP>1) is currently broken on these Blackwell
# RTX PRO 6000 cards. The bundled NCCL (2.26.2+cuda12.2 from jaxlib 0.5.3) hits
# an illegal memory access on a basic cudaHostAlloc during clique init on the
# cuda12.9 driver -- it dies at critic-init (update_critic.py) with a cuSolver
# orgqr / CUDA_ERROR_ILLEGAL_ADDRESS. NCCL_CUMEM_ENABLE=0 and P2P/SHM disable do
# NOT help. Until jax/jaxlib+NCCL are upgraded to a cuda12.9-capable build, run
# single-GPU (FSDP=1), which avoids NCCL entirely. Verified working.

# Pin JAX to the first $FSDP visible GPU(s) when the launcher didn't (e.g. running
# via bare `bash` after ssh-ing onto the node, where CUDA_VISIBLE_DEVICES is unset).
# With FSDP=1 this keeps JAX on one device and skips NCCL init altogether.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((FSDP - 1)))"
fi

mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME" "$CKPT_BASE_DIR"

# ----------------------------------------------------------------------------
# Training args (single-task AWR-SFT; values follow scripts/configs/awr.yaml,
# single seed instead of the [0, 1, 2] sweep).
# Nested fields use tyro dotted paths; bool flags are --flag / --no-flag.
# ----------------------------------------------------------------------------
ARGS=(
  "$CONFIG_NAME"

  # bookkeeping / checkpointing
  --project_name "$PROJECT_NAME"
  --group_name "$GROUP_NAME"
  --exp_name "$EXP_NAME"
  --checkpoint_base_dir "$CKPT_BASE_DIR"
  --seed "$SEED"
  --fsdp_devices "$FSDP"

  # resume-safe if you manually re-submit after a failure; harmless otherwise
  --resume
  --no-overwrite

  --log_interval 25
  --save_interval 100000
  --num_train_steps 100000
  --lr_schedule.value 2.5e-5
  --max_runtime 169200            # ~47h; keep above --time so exp.py never self-exits early

  # data collection
  --collect.tasks "$TASK"
  --collect.eval_tasks "$TASK"
  --collect.store_prefix_rep
  --collect.collect_interval 10000
  --collect.num_rollouts 20
  --collect.env_num 8
  --collect.eval_env_num 8
  --collect.eval_interval 99999

  # AWR
  --rl.beta 0.05
  --rl.discount 0.995
  --rl.online_ratio 1.0
  --rl.buffer_capacity 250000
  --rl.policy.update_interval 10
  --rl.policy.training_start_step 900

  # critic (TD/regression target, not distributional)
  --rl.critic.td_weight_schedule.init_value 1
  --rl.critic.td_weight_schedule.end_value 1
  --rl.critic.td_weight_schedule.switch_step 999999
  --rl.critic.no-use_distributional_critic
  --rl.critic.num_value_bins 1
  --rl.critic.batch_size 1024
  --rl.critic.pre_training_steps 0
  --rl.critic.use_bronet
  --rl.critic.bronet_hidden_dim 1024
  --rl.critic.inference_start_step 1
  --rl.critic.value_target_type one_hot
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'uv run scripts/exp.py'
  printf ' %q' "${ARGS[@]}"
  printf '\n'
  exit 0
fi

echo "[awr] node=$(hostname) job=${SLURM_JOB_ID:-none} exp=${EXP_NAME}"
echo "[awr] checkpoints -> ${CKPT_BASE_DIR}/${CONFIG_NAME}/${EXP_NAME}"

exec uv run scripts/exp.py "${ARGS[@]}"
