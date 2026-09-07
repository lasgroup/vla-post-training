#!/bin/bash
#SBATCH --partition=maxlab
#SBATCH --qos=maxlab_qos
#SBATCH --nodelist=babel-m9-16
#SBATCH --job-name=ogpo_molmo
#SBATCH --gres=gpu:4
#SBATCH --constraint=VRAM_96GB
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --output=/home/mananaga/logs/%j/.out
#SBATCH --error=/home/mananaga/logs/%j/.out
# ---------------------------------------------------------------------------
# OGPO launcher for the MolmoSpaces (FrankaPickDroidMiniBench) domain.
#
# Molmo twin of scripts/ogpo_libero.sh: the OGPO recipe (warmstart -> PG
# ramp-in, grpo_conservative advantage, EMA-quantile normalizer + symmetric
# clip, post-collection critic burst, per-task advantage normalization,
# task-balanced success-buffer BC, 500k buffer) is IDENTICAL. Only the
# domain-side pieces change, mirroring what differs between the libero and
# molmo FSFT/BofN YAMLs in scripts/configs/multitask/{fsft,bofn}:
#
#   config      pi05_libero_online_ogpo_sft -> pi05_molmo_online_ogpo_sft
#               (pi05_droid_jointpos base, action_horizon 15, domain=molmo,
#               max_episode_steps 450 -- same base as the molmo fsft/bofn cfgs)
#   tasks       libero_90_* -> molmo_* (train task + the 16-task held-out
#               eval block shared by every molmo multitask YAML)
#   env         no LIBERO config seeding; MolmoSpaces asset/cache/benchmark
#               dirs come from the MLSPACES_* env vars below instead
#
# Everything else (lr, ema, batch sizes, critic, PPO knobs, env_num 8, eval
# cadence) is kept byte-for-byte from ogpo_libero.sh so the two runs differ
# only in the domain.
#
# PER-TASK SEMANTICS: collect.num_rollouts, num_initial_rollouts and
# num_eval_rollouts are all PER TASK and multiply by len(tasks). See
# ogpo_libero.sh for the calibration behind the defaults.
#
# Env vars (defaults = the libero recipe; set =0 / empty to disable):
#   GPU           - CUDA device index or comma list (defaults to SLURM's cards)
#   FSDP          - fsdp_devices (default: number of GPUs in GPU)
#   ARM           - run name suffix (default v0)
#   SEED          - default 0
#   N_ROLLOUTS    - rollouts PER TASK per collection round (default 5)
#   INIT_ROLLOUTS - EXTRA per-task episodes at step 0 only (default 10; empty to skip)
#   COLLECT_INT   - collection interval (default 10000)
#   PG_START      - BC-only warmstart length (default 20000; 0 disables)
#   PG_RAMP       - linear PG ramp-in after handoff (default 5000)
#   CONS          - 1 => grpo_conservative advantage (default 1)
#   NORM          - 1 => EMA-quantile advantage normalizer (default 1)
#   CLIP_SYM      - symmetric post-norm clip (default 4.0; empty to skip)
#   BURST         - critic-only steps after each collection round (default 1000)
#   MT_ADV        - 1 => per-task advantage normalization (default 1)
#   MT_BAL        - 1 => task-balanced success-buffer BC (default 1)
#   HELDOUT       - 1 => eval also on the 16-task held-out block (default 0:
#                   eval on the train tasks only). With HELDOUT=1 consider
#                   EVAL_ROLLOUTS=8 -- eval cost is EVAL_ROLLOUTS x 17 episodes.
#   EVAL_ROLLOUTS - eval episodes PER TASK (default 32)
#   NUM_STEPS     - train steps (default 100000; 100001 to get a final in-run eval)
#   SAVE_INT      - save_interval (default 200000 = never write epoch state;
#                   100000 => one final checkpoint)
#   CKPT_BASE_DIR - checkpoint root (default: your user_data dir, see below)
#   MLSPACES_BENCHMARK_DIR - FrankaPickDroidMiniBench json benchmark dir
#                   (REQUIRED off CSCS: src/envs/molmo.py defaults to the
#                   /capstor path, which does not exist on babel)
#   MLSPACES_ASSETS_DIR / MLSPACES_CACHE_DIR - MolmoSpaces assets + resource
#                   cache (default: under $STORE_ROOT; populate once with
#                   scripts/install_molmo_assets.py)
#   DRY           - 1 => print the final command instead of running it
#
# Usage:  GPU=0 bash scripts/ogpo_molmo.sh
#         GPU=1 ARM=noburst BURST=0 bash scripts/ogpo_molmo.sh
# ---------------------------------------------------------------------------
set -euo pipefail

GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
ARM="${ARM:-v0}"
SEED="${SEED:-0}"
_NGPU=$(awk -F, '{print NF}' <<<"$GPU")
FSDP="${FSDP:-$_NGPU}"

# Train task: molmo_138, the single-task pick of the bofn_molmo_tasks1 YAML
# (the fsft tasks1 seeds sweep molmo_267/366/138/143). The 8-task set of the
# tasks8 YAMLs is
#   molmo_164 molmo_86 molmo_70 molmo_267 molmo_366 molmo_138 molmo_143 molmo_212
TASKS=(molmo_138)
# Held-out eval block shared by every molmo multitask YAML.
HELDOUT_TASKS=(
  molmo_8 molmo_273 molmo_222 molmo_73 molmo_346 molmo_49 molmo_21 molmo_379
  molmo_140 molmo_356 molmo_345 molmo_155 molmo_357 molmo_163 molmo_208 molmo_226
)
EVAL_TASKS=("${TASKS[@]}")
[ "${HELDOUT:-0}" = "1" ] && EVAL_TASKS+=("${HELDOUT_TASKS[@]}")

export PATH="$HOME/.local/bin:$PATH"

PROJECT_DIR="${PROJECT_DIR:-/home/mananaga/VLA/ogpo/vla-post-training}"
STORE_ROOT="${STORE_ROOT:-/data/user_data/mananaga/vla-post-training}"
EXP_NAME="molmo_${ARM}_s${SEED}"
CKPT_BASE_DIR="${CKPT_BASE_DIR:-$STORE_ROOT/checkpoints/ogpo_molmo}"

# Same interpreter contract as ogpo_libero.sh: reuse the babel venv, call
# python directly, never `uv run`. molmospaces is imported from PYTHONPATH.
PY="${PY:-/home/mananaga/VLA/manan_babel/vla-post-training/.venv/bin/python}"
[ -x "$PY" ] || { echo "[molmo] no interpreter at $PY" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="$GPU"
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/openpi/packages/openpi-client/src:$PROJECT_DIR/openpi/src:$PROJECT_DIR/openpi/packages/openpi-client:$PROJECT_DIR/molmospaces"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$STORE_ROOT/cache/openpi}"
export HF_HOME="${HF_HOME:-$STORE_ROOT/cache/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$STORE_ROOT/cache/uv}"
export TORCH_HOME="${TORCH_HOME:-$STORE_ROOT/cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$STORE_ROOT/cache/triton}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$STORE_ROOT/cache/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$STORE_ROOT/cache/xdg}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$STORE_ROOT/config/xdg}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-$STORE_ROOT/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$STORE_ROOT/cache/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$STORE_ROOT/config/wandb}"

# MolmoSpaces resource dirs (molmospaces/molmo_spaces/molmo_spaces_constants.py
# reads these). The benchmark dir has no usable default off CSCS.
export MLSPACES_CACHE_DIR="${MLSPACES_CACHE_DIR:-$STORE_ROOT/cache/molmo-spaces-resources}"
export MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-$STORE_ROOT/molmospaces/assets}"
if [ -z "${MLSPACES_BENCHMARK_DIR:-}" ]; then
  _BENCH_DEFAULT="$MLSPACES_ASSETS_DIR/benchmarks/molmospaces-bench-v1/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231"
  [ -d "$_BENCH_DEFAULT" ] && export MLSPACES_BENCHMARK_DIR="$_BENCH_DEFAULT"
fi
if [ "${DRY:-0}" != "1" ] && [ ! -d "${MLSPACES_BENCHMARK_DIR:-/nonexistent}" ]; then
  echo "[molmo] MLSPACES_BENCHMARK_DIR is unset or missing (${MLSPACES_BENCHMARK_DIR:-<unset>}); point it at the FrankaPickDroidMiniBench json benchmark dir" >&2
  exit 1
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="$MUJOCO_GL"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ "$MUJOCO_GL" = "egl" ] && [ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ]; then
  if [ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
  elif ldconfig -p 2>/dev/null | grep -q libEGL_nvidia; then
    mkdir -p "$STORE_ROOT/egl"
    printf '%s\n' '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
      > "$STORE_ROOT/egl/10_nvidia.json"
    export __EGL_VENDOR_LIBRARY_FILENAMES="$STORE_ROOT/egl/10_nvidia.json"
  fi
fi
export NCCL_CUMEM_ENABLE=0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}"
export MUJOCO_EGL_DEVICE_ID="${GPU%%,*}"

EXTRA_FLAGS=()
# --- stability recipe (identical to ogpo_libero.sh) ---
PG_START="${PG_START:-20000}"
PG_RAMP="${PG_RAMP:-5000}"
[ "${PG_START}" != "0" ] && EXTRA_FLAGS+=(--rl.pg_start_step "$PG_START" --rl.pg_ramp_steps "$PG_RAMP")
if [ "${CONS:-1}" = "1" ]; then
  EXTRA_FLAGS+=(--rl.advantage_combination grpo_conservative)
else
  EXTRA_FLAGS+=(--rl.advantage_combination reduced)
fi
[ "${NORM:-1}" = "1" ] && EXTRA_FLAGS+=(--rl.normalize_group_advantage)
CLIP_SYM="${CLIP_SYM-4.0}"
[ -n "$CLIP_SYM" ] && [ "$CLIP_SYM" != "0" ] && EXTRA_FLAGS+=(--rl.adv_clip_sym "$CLIP_SYM")
BURST="${BURST:-1000}"
[ "$BURST" != "0" ] && EXTRA_FLAGS+=(--rl.post_collection_critic_steps "$BURST")
INIT_ROLLOUTS="${INIT_ROLLOUTS-10}"
[ -n "$INIT_ROLLOUTS" ] && EXTRA_FLAGS+=(--collect.num_initial_rollouts "$INIT_ROLLOUTS")
# --- multi-task knobs ---
[ "${MT_ADV:-1}" = "1" ] && EXTRA_FLAGS+=(--rl.normalize_advantage_per_task)
[ "${MT_BAL:-1}" = "1" ] && EXTRA_FLAGS+=(--rl.balance_success_buffer_tasks)
# --- reference-alignment knobs, always emitted (see ogpo_libero.sh) ---
NUM_QS="${NUM_QS:-2}"
EXTRA_FLAGS+=(--rl.critic.num_qs "$NUM_QS" --rl.critic.num_vs "$NUM_QS")
CRITIC_RED="${CRITIC_RED:-min}"
EXTRA_FLAGS+=(--rl.critic.reduction "$CRITIC_RED")
BON_N="${BON_N:-1}"
EXTRA_FLAGS+=(--rl.n_samples "$BON_N")
SUCC_BONUS="${SUCC_BONUS:-0}"
EXTRA_FLAGS+=(--collect.success_reward_bonus "$SUCC_BONUS")
if [ "${SB_Q:-0}" = "1" ]; then
  EXTRA_FLAGS+=(--rl.critic_success_oversample)
else
  EXTRA_FLAGS+=(--rl.no-critic_success_oversample)
fi
TD_W="${TD_W:-1}"
CONFIG_NAME="${CONFIG_NAME:-pi05_molmo_online_ogpo_sft}"

N_ROLLOUTS="${N_ROLLOUTS:-5}"
COLLECT_INT="${COLLECT_INT:-10000}"
EVAL_ROLLOUTS="${EVAL_ROLLOUTS:-32}"
NUM_STEPS="${NUM_STEPS:-100000}"
SAVE_INT="${SAVE_INT:-200000}"

mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME" "$CKPT_BASE_DIR" \
         "$UV_CACHE_DIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" \
         "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$MLSPACES_CACHE_DIR" \
         "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

echo "[molmo] node=$(hostname) gpu=$GPU arm=$ARM seed=$SEED tasks=${#TASKS[@]} eval_tasks=${#EVAL_TASKS[@]} rollouts/task=$N_ROLLOUTS ckpt=$CKPT_BASE_DIR"
echo "[molmo] benchmark=${MLSPACES_BENCHMARK_DIR:-<unset>} assets=$MLSPACES_ASSETS_DIR"
echo "[molmo] extra=${EXTRA_FLAGS[*]:-none}"

RUN=("$PY" scripts/exp.py)
[ "${DRY:-0}" = "1" ] && RUN=(echo "$PY" scripts/exp.py)

"${RUN[@]}" \
  "$CONFIG_NAME" \
  --project_name ogpo_multitask \
  --group_name mt_molmo \
  --exp_name "$EXP_NAME" \
  --checkpoint_base_dir "$CKPT_BASE_DIR" \
  --seed "$SEED" \
  --fsdp_devices "$FSDP" \
  --overwrite \
  --log_interval 25 \
  --save_interval "$SAVE_INT" \
  --keep_period 100000 \
  --num_train_steps "$NUM_STEPS" \
  --ema_decay 0.99 \
  --lr_schedule.value 2.5e-5 \
  --max_runtime 169200 \
  --collect.domain molmo \
  --collect.tasks "${TASKS[@]}" \
  --collect.eval_tasks "${EVAL_TASKS[@]}" \
  --collect.store_prefix_rep \
  --collect.collect_interval "$COLLECT_INT" \
  --collect.num_rollouts "$N_ROLLOUTS" \
  --collect.num_eval_rollouts "$EVAL_ROLLOUTS" \
  --collect.env_num 8 \
  --collect.eval_env_num 8 \
  --collect.eval_interval 10000 \
  --rl.beta 0.05 \
  --rl.discount 0.995 \
  --rl.online_ratio 1.0 \
  --rl.buffer_capacity 500000 \
  --rl.policy.update_interval 10 \
  --rl.policy.training_start_step 900 \
  --rl.critic.td_weight_schedule.init_value "$TD_W" \
  --rl.critic.td_weight_schedule.end_value "$TD_W" \
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
  --rl.noise_level 0.02 \
  --rl.adv_strategy vanilla \
  --rl.dedup_group_prefix \
  --rl.use_success_buffer \
  --rl.critic.value_target_type one_hot \
  --batch_size 32 \
  "${EXTRA_FLAGS[@]}" \
  "$@"
