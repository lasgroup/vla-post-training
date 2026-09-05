#!/bin/bash
#SBATCH --partition=maxlab
#SBATCH --qos=maxlab_qos
#SBATCH --nodelist=babel-m9-16
#SBATCH --job-name=ogpo_priv
#SBATCH --gres=gpu:4
#SBATCH --constraint=VRAM_96GB
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --output=/home/mananaga/logs/%j/.out
#SBATCH --error=/home/mananaga/logs/%j/.out
# ---------------------------------------------------------------------------
# PRIVILEGED-CRITIC OGPO: the critic-capacity upper bound for the 4-task LIBERO
# benchmark set (libero_90_79/31/82/38).
#
# This is scripts/ogpo_multitask_4task.sh with the CRITIC swapped and nothing
# else. Same tasks, same rollout budget, same warmstart + PG ramp, same
# advantage combination / normalizer / clip, same digestion burst, same policy
# schedule, same optimizer. Run the two side by side and every difference is
# attributable to the critic.
#
# WHAT CHANGES (src/rl/ogpo/privileged/, src/rl/privileged_state.py):
#   1. STATE   The critic reads the simulator's own state — robot proprioception
#              (robot0_proprio-state) plus every object's pose relative to the
#              world and the gripper (object-state) — instead of the mean-pooled
#              PaliGemma prefix + the policy's 8-d proprio vector. Zero-padded
#              to PSTATE_DIM so one buffer schema covers tasks with different
#              object counts.
#   2. HEADS   One INDEPENDENT critic per training task (4 here), selected per
#              sample from the task id recorded at reset. No capacity is spent
#              telling the tasks apart, and one task's transitions cannot move
#              another task's critic.
#
# WHAT DOES NOT CHANGE: the TD target. Q still bootstraps through V(s'), and V
# is still regressed onto Q(s, a_buffer), through the SAME shared critic train
# steps the baseline uses — so the default arm isolates the critic's INPUTS and
# PARAMETERIZATION from everything else.
#   BACKUP=next_action_q is an opt-in ablation that instead backs up through
#   Q_ema(s', a'), a' = the action the policy took at s'. That action is read
#   out of the stored trajectory at collection (no policy rollout at TD time —
#   the runtime cost is one extra critic forward), but it does add a
#   `next_actions` column to the replay buffer: one action chunk per
#   transition, ~640 MB at --rl.buffer_capacity 500000. The default allocates
#   nothing for it.
#
# The privileged vector does not exist on a real robot: this measures the
# headroom the current critic is leaving on the table, it is not a deployable
# recipe. Read critic/q_mc_corr against the baseline run's — the advantage only
# ever consumes the critic's ORDERING of actions, so that correlation, not
# q_value_mean, is the number this experiment is about.
#
# NOTE ON store_prefix_rep: deliberately NOT passed. The privileged critic reads
# no prefix rep, so the per-collection-step PaliGemma prefix forward and the
# prefix column in the buffer are both dropped (the learner rejects the flag
# rather than silently paying for it).
#
# Env vars (defaults = the mt4 recipe; set =0 / empty to disable):
#   GPU           - CUDA device index or comma list (default: SLURM's allocation)
#   FSDP          - fsdp_devices (default: number of GPUs in GPU)
#   ARM           - run name suffix (default v0)
#   SEED          - default 0
#   BACKUP        - value (default, the baseline's V bootstrap) | next_action_q
#   PSTATE_DIM    - privileged state width, zero-padded (default 512)
#   CRITIC_HID    - per-task BroNet width (default 1024, same as mt4)
#   N_ROLLOUTS    - rollouts PER TASK per collection round (default 5)
#   INIT_ROLLOUTS - EXTRA per-task episodes at step 0 only (default 10)
#   COLLECT_INT   - collection interval (default 10000)
#   PG_START      - BC-only warmstart length (default 20000; 0 disables)
#   PG_RAMP       - linear PG ramp-in after handoff (default 5000)
#   CONS          - 1 => grpo_conservative advantage (default 1)
#   NORM          - 1 => EMA-quantile advantage normalizer (default 1)
#   CLIP_SYM      - symmetric post-norm clip (default 4.0; empty to skip)
#   BURST         - critic-only steps after each collection round (default 1000)
#   MT_ADV        - 1 => per-task advantage normalization (default 1)
#   MT_BAL        - 1 => task-balanced success-buffer BC (default 1)
#   NUM_QS        - critic ensemble size PER TASK (default 2)
#   CRITIC_RED    - ensemble reduction (default min)
#   SUCC_BONUS    - reward on the terminating step (default 0)
#   SB_Q          - 1 => success-oversampled extra critic update (default 0)
#   TD_W          - TD/MC blend for the critic loss (default 1 = pure TD)
#   HELDOUT       - 1 => eval also on the 25-task held-out block (default 0)
#   EVAL_ROLLOUTS - eval episodes PER TASK (default 32)
#   NUM_STEPS     - train steps (default 100000)
#   SAVE_INT      - save_interval (default 200000 => no epoch state written)
#   CKPT_BASE_DIR - checkpoint root
#   DRY           - 1 => print the final command instead of running it
#
# Usage:  GPU=0 bash scripts/ogpo_privileged.sh
#         GPU=1 ARM=qbackup BACKUP=next_action_q bash scripts/ogpo_privileged.sh
# ---------------------------------------------------------------------------
set -euo pipefail

GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
ARM="${ARM:-v0}"
SEED="${SEED:-0}"
_NGPU=$(awk -F, '{print NF}' <<<"$GPU")
FSDP="${FSDP:-$_NGPU}"

# The 4-task LIBERO train set — identical to ogpo_multitask_4task.sh, so the
# two runs are directly comparable. This list ALSO defines the critic's head
# index space: one critic per distinct entry, in this order.
TASKS=(libero_90_79 libero_90_31 libero_90_82 libero_90_38)
HELDOUT_TASKS=(
  libero_90_34 libero_90_37 libero_90_4 libero_90_71 libero_90_66
  libero_90_52 libero_90_75 libero_90_11 libero_90_18 libero_90_23
  libero_90_81 libero_90_10 libero_90_69 libero_90_48 libero_90_7
  libero_90_13 libero_90_0 libero_90_26 libero_90_77 libero_90_19
  libero_90_72 libero_90_22 libero_90_2 libero_90_68 libero_90_57
)
EVAL_TASKS=("${TASKS[@]}")
# Held-out tasks have no critic head; their task id is recorded as -1, which
# one-hots to zero. Harmless because collection-time critic scoring (best-of-N)
# is off, and eval never touches the critic at all.
[ "${HELDOUT:-0}" = "1" ] && EVAL_TASKS+=("${HELDOUT_TASKS[@]}")

export PATH="$HOME/.local/bin:$PATH"

PROJECT_DIR="${PROJECT_DIR:-/home/mananaga/VLA/ogpo/vla-post-training}"
STORE_ROOT="${STORE_ROOT:-/data/user_data/mananaga/vla-post-training}"
EXP_NAME="mt4priv_${ARM}_s${SEED}"
CKPT_BASE_DIR="${CKPT_BASE_DIR:-$STORE_ROOT/checkpoints/ogpo_privileged}"

# Same interpreter contract as ogpo_multitask_4task.sh: reuse the babel venv,
# never `uv run` (it would re-resolve against the branch lockfile and mutate the
# venv, breaking the other checkout too). This tree's code comes from PYTHONPATH.
PY="${PY:-/home/mananaga/VLA/manan_babel/vla-post-training/.venv/bin/python}"
[ -x "$PY" ] || { echo "[priv] no interpreter at $PY" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="$GPU"
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/openpi/packages/openpi-client/src:$PROJECT_DIR/openpi/src:$PROJECT_DIR/openpi/packages/openpi-client:$PROJECT_DIR/molmospaces"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$STORE_ROOT/cache/openpi}"
export HF_HOME="${HF_HOME:-$STORE_ROOT/cache/huggingface}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.libero}"
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
# --- stability recipe, byte-for-byte the mt4 block ---
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
# --- reference-alignment knobs (emitted unconditionally so the env var is
# authoritative for ANY config, same contract as the mt4 script) ---
NUM_QS="${NUM_QS:-2}"            # PER TASK: the run holds NUM_QS x len(TASKS) Q nets
EXTRA_FLAGS+=(--rl.critic.num_qs "$NUM_QS" --rl.critic.num_vs "$NUM_QS")
CRITIC_RED="${CRITIC_RED:-min}"
EXTRA_FLAGS+=(--rl.critic.reduction "$CRITIC_RED")
SUCC_BONUS="${SUCC_BONUS:-0}"
EXTRA_FLAGS+=(--collect.success_reward_bonus "$SUCC_BONUS")
if [ "${SB_Q:-0}" = "1" ]; then
  EXTRA_FLAGS+=(--rl.critic_success_oversample)
else
  EXTRA_FLAGS+=(--rl.no-critic_success_oversample)
fi
TD_W="${TD_W:-1}"
# --- privileged-critic knobs ---
BACKUP="${BACKUP:-value}"
PSTATE_DIM="${PSTATE_DIM:-512}"
CRITIC_HID="${CRITIC_HID:-1024}"
CONFIG_NAME="${CONFIG_NAME:-pi05_libero_online_ogpo_privileged}"

N_ROLLOUTS="${N_ROLLOUTS:-5}"
COLLECT_INT="${COLLECT_INT:-10000}"
EVAL_ROLLOUTS="${EVAL_ROLLOUTS:-32}"
NUM_STEPS="${NUM_STEPS:-100000}"
SAVE_INT="${SAVE_INT:-200000}"

mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME" "$LIBERO_CONFIG_PATH" "$CKPT_BASE_DIR" \
         "$UV_CACHE_DIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" \
         "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" \
         "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

if [ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]; then
  printf 'n\n' | "$PY" -c "import libero.libero" >/dev/null 2>&1 || true
fi

echo "[priv] node=$(hostname) gpu=$GPU arm=$ARM seed=$SEED tasks=${#TASKS[@]} (=critic heads) eval_tasks=${#EVAL_TASKS[@]} backup=$BACKUP pstate_dim=$PSTATE_DIM ckpt=$CKPT_BASE_DIR"
echo "[priv] extra=${EXTRA_FLAGS[*]:-none}"

RUN=("$PY" scripts/exp.py)
[ "${DRY:-0}" = "1" ] && RUN=(echo "$PY" scripts/exp.py)

# Differences from scripts/ogpo_multitask_4task.sh's command line, and ONLY these:
#   + --collect.store_privileged_state / --collect.privileged_state_dim
#   + --rl.privileged_backup
#   - --collect.store_prefix_rep      (the privileged critic reads no prefix)
#   = --rl.n_samples 1                (mt4's BON_N default; best-of-N scoring
#                                      builds a prefix-embedding critic obs the
#                                      privileged critic cannot read, so it is
#                                      pinned rather than exposed)
"${RUN[@]}" \
  "$CONFIG_NAME" \
  --project_name ogpo_multitask \
  --group_name mt4_libero_privileged \
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
  --collect.tasks "${TASKS[@]}" \
  --collect.eval_tasks "${EVAL_TASKS[@]}" \
  --collect.store_privileged_state \
  --collect.privileged_state_dim "$PSTATE_DIM" \
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
  --rl.n_samples 1 \
  --rl.privileged_backup "$BACKUP" \
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
  --rl.critic.bronet_hidden_dim "$CRITIC_HID" \
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
  --batch_size 128 \
  "${EXTRA_FLAGS[@]}" \
  "$@"
