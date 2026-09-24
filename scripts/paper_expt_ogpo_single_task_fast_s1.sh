#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --gres=gpu:2
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --job-name=paper_ogpo_fast_s1
#SBATCH --output=/home/pchellap/logs/%x_%j.out
#SBATCH --error=/home/pchellap/logs/%x_%j.out
# ---------------------------------------------------------------------------
# Paper experiment: OGPO, single train task libero_90_82, the FAST config
# derived from the 2026-09-20 probe series. Seed 1 of 3
# (paper_expt_ogpo_single_task_fast_s{0,1,2}.sh). Same paper-critic design
# as paper_expt_ogpo_single_task_paperconfig_s{0,1,2}.sh (2-head, min-
# reduction critic) but a DIFFERENT checkpoint (NEW exp_name) and a much
# faster schedule -- do not confuse the two families.
#
# Derivation, in order:
#   1. batch_size=256 (marco/parl's own factory default) needs
#      256 x group_num_samples=8 = 2048 chains per fsdp_devices=1 GPU --
#      measured ceiling for 256 chains alone is 75.8/95.6 GiB on a 96GB
#      card (10-head critic), so 2048 chains needs ~600+ GiB. Nowhere close
#      to fitting on any single GPU in this cluster.
#   2. rl.policy_grad_accum trades wall-clock for peak memory: batch=32,
#      grad_accum=8 gives the same 256-effective-states-per-step target at
#      a per-call footprint of 32 x 8 = 256 chains (fsdp_devices=1) --
#      still needs a 96GB+ card, confirmed OOM on 48GB L40S.
#   3. fsdp_devices=2 halves the per-GPU chain count to 128 (batch=32
#      splits 16/GPU) -- confirmed safe on 48GB cards (jobs 10513431,
#      10514270). fsdp_devices=4 (128->64 chains, tried as batch=64/
#      accum=4) also fit, but cross-GPU sync overhead made it ~2.3x SLOWER
#      per step than fsdp_devices=2 (11.05s/step vs 4.85s/step, jobs
#      10513307 vs 10513431) -- more GPUs was not faster here.
#      fsdp_devices=1 at the same 128-chain footprint (batch=16,
#      grad_accum=16) also fit but was slower still (6.3s/step, job
#      10513672) -- doubling grad_accum's sequential-call count cost more
#      than removing FSDP's cross-GPU sync saved. fsdp_devices=2/batch=32/
#      grad_accum=8 (this script) is the best of the four combinations
#      tried.
#   4. THE BIG ONE: rl.policy.update_interval. Decomposing the ~4.85s/step
#      measured at update_interval=10: pre-900 critic-only steps run
#      ~0.25s/step, so of the ~48.5s per 10-step block, ~46.2s is the ONE
#      actor update alone (grad_accum=8 sequential passes through the
#      ~3B-param model) -- critic-only steps are nearly free by
#      comparison. update_interval=10 gives ~9910 actor updates over the
#      full 100001-step run; the paper excerpt says <100 total gradient
#      steps for single-task adaptation. update_interval=1000 gives ~100
#      actor updates (matching that budget) and was confirmed on L40S (job
#      10514270, babel-m5-20): step 1000's actor update measured 39.7s,
#      then straight back to ~4.0it/s critic-only. Training-loop total:
#      100 x 39.7s + 100001 x 0.25s ~= 8.0h, vs ~5.6 days at
#      update_interval=10 -- ~16.8x, matching the predicted ~16x.
#
# GPU MODEL MATTERS: --constraint=VRAM_48GB let jobs land on RTX 6000 Ada,
# L40, AND L40S interchangeably (all tagged VRAM_48GB, not necessarily
# identical real usable memory) -- this confounded early speed comparisons
# and caused at least one spurious-looking OOM. Pinned to --constraint=L40S
# specifically here, matching what was actually validated.
#
# rl.n_samples stays at 8 (the _ref config's own default, NOT the 32 the
# paperconfig family uses) -- n_samples only affects collect_data/
# evaluate_policy (sample_actions), never the training step, so it's
# orthogonal to everything above. A separate rl.n_samples=1 probe OOM'd on
# a mixed-model card (job 10513755) before producing a clean result; not
# folded in here pending a clean rerun.
#
# Bypasses stability_study.sh directly: that script sets
# MUJOCO_EGL_DEVICE_ID="$GPU" unconditionally, with no comma-list
# truncation for multi-GPU (stability_study.sh:134, unlike
# ogpo_multitask_4task.sh's ${GPU%%,*}, :151) -- feeding it "0,1" breaks
# MuJoCo's EGL context. CUDA_VISIBLE_DEVICES and MUJOCO_EGL_DEVICE_ID are
# set independently below instead.
#
# Submit: sbatch scripts/paper_expt_ogpo_single_task_fast_s0.sh
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

export CUDA_VISIBLE_DEVICES="0,1"
export MUJOCO_EGL_DEVICE_ID="0"

mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME" "$LIBERO_CONFIG_PATH" "$UV_CACHE_DIR" \
         "$TORCH_HOME" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" \
         "$XDG_CONFIG_HOME" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

if [ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]; then
  printf 'n\n' | uv run python -c "import libero.libero" >/dev/null 2>&1 || true
fi

echo "[paper-fast-s1] node=$(hostname) job=${SLURM_JOB_ID:-?} cuda=$CUDA_VISIBLE_DEVICES mujoco_egl=$MUJOCO_EGL_DEVICE_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

CKPT_BASE_DIR="/data/group_data/maxlab/common_datasets/${USER:-pchellap}/vla-post-training/checkpoints/stability_study"
mkdir -p "$CKPT_BASE_DIR"

# exp.py exits 42 when it has saved a resumable epoch and wants the wall
# clock back; requeue then continues from that checkpoint.
child_status=0
uv run scripts/exp.py pi05_libero_online_ogpo_ref \
  --project_name vla-post-training \
  --group_name policy_extraction_tasks1_ogpo_fast_v0 \
  --exp_name stab_paper_expt_ogpo_single_task_fast_s1 \
  --checkpoint_base_dir "$CKPT_BASE_DIR" \
  --seed 1 \
  --fsdp_devices 2 \
  --resume \
  --log_interval 25 \
  --save_interval 100000 \
  --keep_period 100000 \
  --num_train_steps 100001 \
  --ema_decay 0.99 \
  --lr_schedule.value 2.5e-5 \
  --max_runtime 79200 \
  --collect.tasks libero_90_82 \
  --collect.eval_tasks libero_90_82 \
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
  --batch_size 32 \
  --collect.success_reward_bonus 90 \
  --rl.critic.num_qs 2 \
  --rl.critic.num_vs 2 \
  --no-backbone_lora \
  --rl.n_samples 1 \
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
