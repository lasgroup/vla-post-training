#!/bin/bash
#SBATCH --partition=maxlab
#SBATCH --qos=maxlab_qos
#SBATCH --gres=gpu:1
#SBATCH --constraint=VRAM_96GB
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=00:45:00
#SBATCH --job-name=probe_ogpo_b64a4
#SBATCH --output=/home/pchellap/logs/%x_%j.out
#SBATCH --error=/home/pchellap/logs/%x_%j.out
# ---------------------------------------------------------------------------
# PROBE, not a campaign run: does batch_size=64/rl.policy_grad_accum=4
# (still 256 effective states/step, same as the 32x8 recipe) fit on ONE
# 96GB RTX PRO 6000 (fsdp_devices=1), and is it faster per actor-update
# step than the 32x8 combo currently used by all single-task fast scripts?
#
# Clone of paper_expt_ogpo_single_task_fast_libero31_s0.sh with:
#   - batch_size 32 -> 64, rl.policy_grad_accum 8 -> 4 (the thing under test)
#   - num_train_steps 100001 -> 1101: just enough to clear
#     rl.policy.training_start_step=900 and catch 1-2 actor updates
#     (steps ~900/1000/1100) at update_interval=100. collect.collect_interval
#     stays at its recipe default (10000), which is > 1101, so only the
#     one unavoidable initial buffer-seeding collection round runs -- no
#     mid-probe collection, no eval (eval_interval=10000 also > 1101).
#   - log_interval 25 -> 10 for finer per-step timing resolution in
#     metrics.jsonl / stdout.
#   - save_interval/keep_period left at 100000 (never reached) -- no
#     checkpoint I/O to keep this a clean timing measurement.
#   - distinct exp_name/group_name (probe_*), no --requeue (a probe that
#     dies just dies; nothing here is worth resuming).
#   - --time=00:45:00: initial 20-rollout collection measured ~388s
#     (job 10513903, n_samples=1) + a couple ~40s actor updates is well
#     under this; bounded tightly since this is throwaway.
#
# Reads: metrics.jsonl step-duration field for the actor-update steps
# (~900-1100) vs the running 32x8 campaigns' own measured step time, and
# nvidia-smi peak memory from the job log, to answer OOM + speed both.
#
# Submit: sbatch scripts/probe_ogpo_single_task_b64a4_maxlab.sh
# ---------------------------------------------------------------------------
set -uo pipefail

PROJECT_DIR=/home/pchellap/Projects/OGPO-VLA/vla-post-training
cd "$PROJECT_DIR"

STORE_ROOT="$PROJECT_DIR/run_store"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/openpi/packages/openpi-client/src:$PROJECT_DIR/openpi/src:$PROJECT_DIR/openpi/packages/openpi-client:$PROJECT_DIR/molmospaces"
export OPENPI_DATA_HOME="$STORE_ROOT/cache/openpi"
export HF_HOME="$STORE_ROOT/cache/huggingface"
export LIBERO_CONFIG_PATH="$STORE_ROOT/libero"
export UV_CACHE_DIR="$STORE_ROOT/cache/uv"
export TORCH_HOME="$STORE_ROOT/cache/torch"
export TRITON_CACHE_DIR="$STORE_ROOT/cache/triton"
export MPLCONFIGDIR="$STORE_ROOT/cache/matplotlib"
export XDG_CACHE_HOME="$STORE_ROOT/cache/xdg"
export XDG_CONFIG_HOME="$STORE_ROOT/config/xdg"
export WANDB_MODE="offline"
export WANDB_DIR="$STORE_ROOT/wandb"
export WANDB_CACHE_DIR="$STORE_ROOT/cache/wandb"
export WANDB_CONFIG_DIR="$STORE_ROOT/config/wandb"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="$MUJOCO_GL"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
  export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
elif ldconfig -p 2>/dev/null | grep -q libEGL_nvidia; then
  mkdir -p "$STORE_ROOT/egl"
  printf '%s\n' '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
    > "$STORE_ROOT/egl/10_nvidia.json"
  export __EGL_VENDOR_LIBRARY_FILENAMES="$STORE_ROOT/egl/10_nvidia.json"
fi
export NCCL_CUMEM_ENABLE=0
export NCCL_IB_DISABLE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.75"

export CUDA_VISIBLE_DEVICES="0"
export MUJOCO_EGL_DEVICE_ID="0"

mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME" "$LIBERO_CONFIG_PATH" "$UV_CACHE_DIR" \
         "$TORCH_HOME" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" \
         "$XDG_CONFIG_HOME" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

if [ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]; then
  printf 'n\n' | uv run python -c "import libero.libero" >/dev/null 2>&1 || true
fi

echo "[probe-b64a4] node=$(hostname) job=${SLURM_JOB_ID:-?} cuda=$CUDA_VISIBLE_DEVICES mujoco_egl=$MUJOCO_EGL_DEVICE_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

CKPT_BASE_DIR="/data/group_data/maxlab/common_datasets/${USER:-pchellap}/vla-post-training/checkpoints/stability_study"
mkdir -p "$CKPT_BASE_DIR"

child_status=0
uv run scripts/exp.py pi05_libero_online_ogpo_ref \
  --project_name vla-post-training \
  --group_name policy_extraction_probes_v0 \
  --exp_name probe_ogpo_single_task_b64a4_s0 \
  --checkpoint_base_dir "$CKPT_BASE_DIR" \
  --seed 0 \
  --fsdp_devices 1 \
  --resume \
  --log_interval 10 \
  --save_interval 100000 \
  --keep_period 100000 \
  --num_train_steps 1101 \
  --ema_decay 0.99 \
  --lr_schedule.value 2.5e-5 \
  --max_runtime 2400 \
  --collect.tasks libero_90_31 \
  --collect.eval_tasks libero_90_31 \
  --collect.store_prefix_rep \
  --collect.collect_interval 10000 \
  --collect.num_rollouts 20 \
  --collect.env_num 8 \
  --collect.eval_env_num 8 \
  --collect.eval_interval 10000 \
  --rl.beta 0.05 \
  --rl.discount 0.995 \
  --rl.online_ratio 1.0 \
  --rl.buffer_capacity 250000 \
  --rl.policy.update_interval 100 \
  --rl.policy.training_start_step 900 \
  --rl.critic.td_weight_schedule.init_value 0.95 \
  --rl.critic.td_weight_schedule.end_value 0.95 \
  --rl.critic.td_weight_schedule.switch_step 999999 \
  --rl.critic.no-use_distributional_critic \
  --rl.critic.num_value_bins 1 \
  --rl.critic.batch_size 1024 \
  --rl.critic.pre_training_steps 0 \
  --rl.critic.use_bronet \
  --rl.critic.bronet_hidden_dim 1024 \
  --rl.critic.inference_start_step 1 \
  --rl.group_num_samples 8 \
  --rl.clip_epsilon 0.1 \
  --rl.bc_coeff 1.0 \
  --rl.num_sde_steps 10 \
  --rl.noise_level 0.3 \
  --rl.adv_strategy vanilla \
  --rl.dedup_group_prefix \
  --rl.use_success_buffer \
  --rl.critic.value_target_type one_hot \
  --batch_size 64 \
  --collect.success_reward_bonus 90 \
  --rl.critic.num_qs 2 \
  --rl.critic.num_vs 2 \
  --no-backbone_lora \
  --rl.n_samples 1 \
  --rl.critic.reduction min \
  --rl.policy_grad_accum 4 \
  || child_status=$?

exit "$child_status"
