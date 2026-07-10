#!/bin/bash
# AWR (advantage-weighted SFT) online post-training of pi05 on LIBERO, on the
# babel cluster. Single-seed, single-run launcher for scripts/configs/awr.yaml.
#
# This is the babel analogue of the SwissAI/CSCS launcher.py workflow
# (scripts/launcher.py + scripts/configs/awr.yaml). Instead of going through
# srun --environment=<enroot container>, we run directly inside the project's
# uv venv (.venv). It mirrors the launcher's requeue behaviour: scripts/exp.py
# exits with code 42 when it wants to be requeued (either because max_runtime
# was exceeded, or right before an eval if --requeue_before_eval is set), and
# we resubmit via `scontrol requeue` while reusing the same checkpoint dir so
# the run resumes (resume=True, overwrite=False) from where it left off.
#
# Modeled directly on scripts/bofn_libero_babel.sh (the working best_of_n
# babel launcher) -- see that script for more background on the env vars.
#
# Usage:
#   sbatch scripts/awr_libero_babel.sh                # submit with defaults (seed 0, libero_90_44)
#   SEED=1 TASK=libero_90_59 sbatch scripts/awr_libero_babel.sh
#   DRY_RUN=1 bash scripts/awr_libero_babel.sh         # print args, no run
#
# GPU sizing: the pi05 train state (~50 GiB: params + AdamW moments + EMA)
# plus this config's critic (BRONet, hidden_dim=1024, its own AdamW state) is
# a fixed cost that does not fit a 48 GB card, so we need a 96 GB GPU. Set
# FSDP>1 (and --gres=gpu:N to match) to shard the state across GPUs instead.
#SBATCH --partition=maxlab
#SBATCH --qos=maxlab_qos
#SBATCH --nodelist=babel-m9-16
#SBATCH --job-name=awr_libero
#SBATCH --gres=gpu:1
#SBATCH --constraint=VRAM_96GB
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=11:59:00
#SBATCH --requeue
#SBATCH --open-mode=append
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
FSDP="${FSDP:-1}"                     # shard train state across N GPUs; keep == --gres=gpu:N

# Stable experiment name -> stable checkpoint dir so requeue resumes instead of
# starting over (do NOT put a timestamp here).
EXP_NAME="${EXP_NAME:-${CONFIG_NAME}_${TASK}_seed${SEED}}"
STORE_ROOT="${STORE_ROOT:-/data/user_data/mananaga/vla-post-training}"
CKPT_BASE_DIR="${CKPT_BASE_DIR:-${STORE_ROOT}/checkpoints/${GROUP_NAME}}"

REQUEUE_EXIT_CODE=42

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

  # requeue-safe: start fresh the first time, resume after a requeue
  --resume
  --no-overwrite

  --log_interval 25
  --save_interval 100000
  --num_train_steps 100000
  --lr_schedule.value 2.5e-5
  --max_runtime 39600            # 11h; scripts/exp.py self-exits (code 42) before the SLURM wall-clock hits

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

# ----------------------------------------------------------------------------
# Run with requeue handling (mirrors launcher.py's sbatch wrapper).
# ----------------------------------------------------------------------------
child_status=0
uv run scripts/exp.py "${ARGS[@]}" || child_status=$?

if [[ "$child_status" -eq "$REQUEUE_EXIT_CODE" ]]; then
  echo "[$(date --iso-8601=seconds)] Job ${SLURM_JOB_ID:-?} requested requeue." >&2
  if [[ -n "${SLURM_JOB_ID:-}" ]] && scontrol requeue "${SLURM_JOB_ID}"; then
    echo "[$(date --iso-8601=seconds)] Requeue submitted for job ${SLURM_JOB_ID}." >&2
    exit 0
  fi
  echo "[$(date --iso-8601=seconds)] Failed to requeue (no SLURM_JOB_ID or scontrol error)." >&2
  exit "$child_status"
fi

exit "$child_status"
