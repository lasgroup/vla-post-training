#!/bin/bash
# ---------------------------------------------------------------------------
# Multi-task OGPO launcher: the "4-task LIBERO" benchmark set
# (libero_90_79/31/82/38 — same set as fsft_libero_babel.sh and the
# scripts/configs/multitask/*_tasks4-16 YAMLs), running the stabilized
# single-task recipe from the stability study:
#
#   warmstart (PG muted 0->PG_START, succBC-only actor) -> linear PG ramp-in
#   + grpo_conservative advantage (Q-only, sign-unanimous across heads)
#   + EMA-quantile advantage normalizer + symmetric clip
#   + post-collection critic digestion burst
#
# plus the two multi-task OGPO knobs (per-task advantage normalization,
# task-balanced success-buffer BC sampling) and the 500k buffer used by the
# multitask configs.
#
# PER-TASK SEMANTICS (why the defaults differ from the single-task numbers):
#   collect.num_rollouts, num_initial_rollouts, num_eval_rollouts are all
#   PER TASK and multiply by len(tasks). Defaults below keep the totals at
#   the single-task recipe's calibration:
#     N_ROLLOUTS=5      -> 20 episodes/round total (same flood size the
#                          BURST=1000 digestion was tuned against)
#     INIT_ROLLOUTS=10  -> (5+10)*4 = 60-episode step-0 collection (same
#                          spark insurance as the colleague's cfg)
#
# Env vars (defaults = colleague's recipe; set =0 / empty to disable):
#   GPU           - CUDA device index or comma list, e.g. GPU=0 or GPU=0,1 (required)
#   FSDP          - fsdp_devices (default 1; set to the number of GPUs in GPU)
#   ARM           - run name suffix (default v0)
#   SEED          - default 0
#   TASKS         - space-separated train task ids (default: the 4-task set).
#                   EVAL_TASKS, the banner and PER_TASK_CRITIC's slot count all
#                   follow it. Per-task rollout knobs do NOT rescale themselves.
#   BATCH         - global policy batch = states per policy update (default 32).
#                   Actor memory scales with BATCH x rl.group_num_samples.
#   EP_MULT       - multiplies the episode TimeLimit truncation (libero_90: 400 -> 400*EP_MULT). Default 1.
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
#   PER_TASK_CRITIC - 1 => one disjoint Q/V critic pair PER TASK, routed by the
#                   buffer's task_index (rl.critic.num_tasks=${#TASKS[@]}; default 0
#                   = shared critics). Emitted unconditionally, so it overrides the
#                   config's own value either way (incl. the _pertask config).
#                   Needs BON_N=1 with HELDOUT=1 (held-out tasks have no critic).
#                   docs/changes/2026-08-21-per-task-critics/
#   HELDOUT       - 1 => eval also on the 25-task held-out block (default 0:
#                   eval on the 4 train tasks only). With HELDOUT=1 consider
#                   EVAL_ROLLOUTS=8 — eval cost is EVAL_ROLLOUTS x 29 episodes.
#   EVAL_ROLLOUTS - eval episodes PER TASK (default 32, the study's EMA eval)
#   MAX_RUNTIME   - seconds before the loop saves and exits for requeue at the
#                   next collect/eval boundary (default 169200 = 47 h, sized for
#                   the 48 h sbatch). MUST be under the sbatch --time, else SLURM
#                   SIGKILLs mid-step and the progress since the last 10k save is
#                   lost. With --time=25:00:00 use MAX_RUNTIME=86400 (24 h).
#   NUM_STEPS     - train steps (default 100000). Set 100001 so the loop
#                   reaches step 100000 and the final checkpoint gets an
#                   in-run eval (the eval loop skips the last step otherwise).
#   CKPT_BASE_DIR - checkpoint root (default: your group_data dir, see below)
#   FRESH         - 1 => --overwrite (WIPES the checkpoint dir and starts at
#                   step 0). Default is --resume, so resubmitting an interrupted
#                   run continues it instead of destroying it.
#   DRY           - 1 => print the final command instead of running it
#
# Usage:  GPU=0 bash scripts/ogpo_multitask_4task.sh
#         GPU=1 ARM=noburst BURST=0 bash scripts/ogpo_multitask_4task.sh
# ---------------------------------------------------------------------------
set -euo pipefail

: "${GPU:?set GPU (cuda device index or comma list, e.g. 0 or 0,1)}"
ARM="${ARM:-v0}"
SEED="${SEED:-0}"
FSDP="${FSDP:-1}"

# The 4-task LIBERO train set (keep identical to the multitask configs so
# runs are comparable with the FSFT/AWR/BofN campaigns).
# Overridable by a space-separated TASKS env var for a different train set --
# EVAL_TASKS, the [mt4] banner and PER_TASK_CRITIC's slot count all derive from
# this array, so overriding here keeps every downstream count consistent (a
# trailing `--collect.tasks` on the CLI would not: tyro takes the last flag but
# the banner and ${#TASKS[@]} would still report the 4-task default).
# REMINDER: N_ROLLOUTS / INIT_ROLLOUTS / EVAL_ROLLOUTS are PER TASK, so a
# different task count changes the episode totals -- rescale them too.
read -ra TASKS <<< "${TASKS:-libero_90_79 libero_90_31 libero_90_82 libero_90_38}"
# Held-out eval block shared by the multitask YAMLs (generalization eval).
HELDOUT_TASKS=(
  libero_90_34 libero_90_37 libero_90_4 libero_90_71 libero_90_66
  libero_90_52 libero_90_75 libero_90_11 libero_90_18 libero_90_23
  libero_90_81 libero_90_10 libero_90_69 libero_90_48 libero_90_7
  libero_90_13 libero_90_0 libero_90_26 libero_90_77 libero_90_19
  libero_90_72 libero_90_22 libero_90_2 libero_90_68 libero_90_57
)
EVAL_TASKS=("${TASKS[@]}")
[ "${HELDOUT:-0}" = "1" ] && EVAL_TASKS+=("${HELDOUT_TASKS[@]}")

# Non-interactive shells (nohup over ssh) miss ~/.local/bin.
export PATH="$HOME/.local/bin:$PATH"

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
STORE_ROOT="${STORE_ROOT:-$PROJECT_DIR/run_store}"
EXP_NAME="mt4_${ARM}_s${SEED}"
# Checkpoints go to YOUR per-user group storage (never a colleague's dir).
CKPT_BASE_DIR="${CKPT_BASE_DIR:-/data/group_data/maxlab/common_datasets/${USER:-pchellap}/vla-post-training/checkpoints/ogpo_multitask_4task}"

export CUDA_VISIBLE_DEVICES="$GPU"
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/openpi/packages/openpi-client/src:$PROJECT_DIR/openpi/src:$PROJECT_DIR/openpi/packages/openpi-client:$PROJECT_DIR/molmospaces"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$STORE_ROOT/cache/openpi}"
export HF_HOME="${HF_HOME:-$STORE_ROOT/cache/huggingface}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$STORE_ROOT/libero}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$STORE_ROOT/cache/uv}"
export TORCH_HOME="${TORCH_HOME:-$STORE_ROOT/cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$STORE_ROOT/cache/triton}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$STORE_ROOT/cache/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$STORE_ROOT/cache/xdg}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$STORE_ROOT/config/xdg}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-$STORE_ROOT/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$STORE_ROOT/cache/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$STORE_ROOT/config/wandb}"
# 2026-08-24: wandb.init() spawns wandb-core and polls for its port file for
# 30 s by default. That file lands under WANDB_DIR (NFS home), so a slow mount
# fails the job before a single training step -- job 10208253 died exactly
# there after a 71-minute venv import. 300 s is still far under any real run.
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

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
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}"
export MUJOCO_EGL_DEVICE_ID="${GPU%%,*}"

EXTRA_FLAGS=()
# --- colleague's stability recipe (each individually disable-able) ---
PG_START="${PG_START:-20000}"
PG_RAMP="${PG_RAMP:-5000}"
[ "${PG_START}" != "0" ] && EXTRA_FLAGS+=(--rl.pg_start_step "$PG_START" --rl.pg_ramp_steps "$PG_RAMP")
# Authoritative for the same reason as the block below: CONS=0 previously emitted
# nothing, leaving the ref config's grpo_conservative default in place.
if [ "${CONS:-1}" = "1" ]; then
  EXTRA_FLAGS+=(--rl.advantage_combination grpo_conservative)
else
  EXTRA_FLAGS+=(--rl.advantage_combination reduced)
fi
[ "${NORM:-1}" = "1" ] && EXTRA_FLAGS+=(--rl.normalize_group_advantage)
# CLIP_SYM=0 and CLIP_SYM= both disable (adv_clip_sym 0 would zero every advantage).
CLIP_SYM="${CLIP_SYM-4.0}"
[ -n "$CLIP_SYM" ] && [ "$CLIP_SYM" != "0" ] && EXTRA_FLAGS+=(--rl.adv_clip_sym "$CLIP_SYM")
BURST="${BURST:-1000}"
[ "$BURST" != "0" ] && EXTRA_FLAGS+=(--rl.post_collection_critic_steps "$BURST")
INIT_ROLLOUTS="${INIT_ROLLOUTS-10}"
[ -n "$INIT_ROLLOUTS" ] && EXTRA_FLAGS+=(--collect.num_initial_rollouts "$INIT_ROLLOUTS")
# --- multi-task knobs ---
[ "${MT_ADV:-1}" = "1" ] && EXTRA_FLAGS+=(--rl.normalize_advantage_per_task)
[ "${MT_BAL:-1}" = "1" ] && EXTRA_FLAGS+=(--rl.balance_success_buffer_tasks)
# Per-task critics. Emitted UNCONDITIONALLY (same contract as the reference-
# alignment block below): the env var is authoritative for any CONFIG_NAME, so
# PER_TASK_CRITIC=0 against pi05_libero_online_ogpo_sft_pertask yields shared
# critics, and =1 against the default config yields per-task ones. The slot
# count is the distinct train-task count, which the learner asserts.
if [ "${PER_TASK_CRITIC:-0}" = "1" ]; then
  EXTRA_FLAGS+=(--rl.critic.num_tasks "${#TASKS[@]}")
else
  EXTRA_FLAGS+=(--rl.critic.num_tasks None)
fi
# --- backbone LoRA (docs/changes/2026-08-29-backbone-lora/) ---
# Emitted UNCONDITIONALLY, both branches, same contract as PER_TASK_CRITIC
# above: it is a real CLI flag (not a config-name swap), so the env var stays
# authoritative for ANY CONFIG_NAME and the "reproduce the baseline stack by
# env vars alone" contract holds. LORA=1 trains the 2B PaliGemma LLM backbone
# through rank-16/alpha-16 adapters on stack 0 (+27.87M params, +6.5%); SigLIP
# and the base weights stay frozen; the critic's prefix embeddings stay
# adapter-free (R1). NEW-RUN ONLY: the checkpoint tree gains 10 leaves, so
# --resume across a LORA flip fails loudly in orbax (intended). tyro renders
# the bool as a flag PAIR — the value form (`--backbone_lora True`) does not
# parse; these two spellings are the only valid ones.
# NOTE: --rl.dedup_group_prefix (below) is MEMORY-LOAD-BEARING when LORA=1 —
# dedup-off pays G backbone backwards per update instead of one.
if [ "${LORA:-0}" = "1" ]; then
  EXTRA_FLAGS+=(--backbone_lora)
else
  EXTRA_FLAGS+=(--no-backbone_lora)
fi
# --- reference-alignment knobs (docs/changes/2026-08-20-ogpo-reference-alignment) ---
# Every default below reproduces the pre-alignment recipe EXACTLY. These flags are
# emitted UNCONDITIONALLY, not only when set away from the default: the aligned
# config (pi05_libero_online_ogpo_ref) carries the aligned value as its dataclass
# default, so a conditional guard would silently ignore an env var set back to the
# baseline value -- NUM_QS=2 would emit nothing and leave num_qs at 10. Always
# emitting makes the env var authoritative for ANY config, which is what the
# "reproduce the baseline stack by env vars alone" contract requires.
# Consequence: the emitted command line gains these flags versus the pre-alignment
# script. The RESOLVED config is unchanged at the defaults; that equivalence is what
# tests/ogpo/test_verifier_alignment.py pins, not the command text.
NUM_QS="${NUM_QS:-2}"            # reference: 10 (num_vs follows num_qs)
EXTRA_FLAGS+=(--rl.critic.num_qs "$NUM_QS" --rl.critic.num_vs "$NUM_QS")
CRITIC_RED="${CRITIC_RED:-min}"  # reference: mean
EXTRA_FLAGS+=(--rl.critic.reduction "$CRITIC_RED")
BON_N="${BON_N:-1}"              # best-of-N collection; >1 enables it (AWR:313)
EXTRA_FLAGS+=(--rl.n_samples "$BON_N")
SUCC_BONUS="${SUCC_BONUS:-0}"    # reward on the terminating step; 0 = original behavior
EXTRA_FLAGS+=(--collect.success_reward_bonus "$SUCC_BONUS")
if [ "${SB_Q:-0}" = "1" ]; then
  EXTRA_FLAGS+=(--rl.critic_success_oversample)
else
  EXTRA_FLAGS+=(--rl.no-critic_success_oversample)
fi
TD_W="${TD_W:-1}"              # reference blends MC in via a separate loss; 0.95 = 95% TD / 5% MC
CONFIG_NAME="${CONFIG_NAME:-pi05_libero_online_ogpo_sft}"

N_ROLLOUTS="${N_ROLLOUTS:-5}"
# Global (not per-device) policy batch: the count of STATES per policy update.
# The actor jit expands it to BATCH x rl.group_num_samples SDE chains, so this
# is the dominant memory knob -- 32 -> 128 takes 256 -> 1024 chains. Must be
# divisible by jax.device_count() (filtered_sft_learner.py:232).
BATCH="${BATCH:-32}"
# Episode truncation multiplier (collect.episode_steps_multiplier). 2 doubles
# the TimeLimit; failed rollouts then take up to 2x wall time during
# collection and eval on low-SR tasks.
EP_MULT="${EP_MULT:-1}"
COLLECT_INT="${COLLECT_INT:-10000}"
EVAL_ROLLOUTS="${EVAL_ROLLOUTS:-32}"
NUM_STEPS="${NUM_STEPS:-100000}"
MAX_RUNTIME="${MAX_RUNTIME:-169200}"

mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME" "$LIBERO_CONFIG_PATH" "$CKPT_BASE_DIR" \
         "$UV_CACHE_DIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" \
         "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" \
         "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

# LIBERO prompts interactively on first import if its config file is missing —
# fatal in a batch job (EOFError). Seed the default config beforehand.
if [ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]; then
  printf 'n\n' | uv run python -c "import libero.libero" >/dev/null 2>&1 || true
fi

# Checkpoint mode. --resume by default so a resubmitted (requeued, preempted,
# hand-restarted) run continues rather than being destroyed -- --overwrite
# rmtrees the checkpoint dir (openpi checkpoints.py:26-29). Safe on a first
# launch: openpi downgrades resume to a fresh start when the dir is absent or
# holds no checkpoints (:22-33, :56-61). FRESH=1 opts back into the wipe. An
# explicit CKPT_MODE_FLAG (the read-only probes pass --resume) still wins. The
# two are mutually exclusive (openpi config.py:558 raises if both are set), so
# this is one flag, not two booleans.
if [ -n "${CKPT_MODE_FLAG:-}" ]; then
  :
elif [ "${FRESH:-0}" = "1" ]; then
  CKPT_MODE_FLAG=--overwrite
else
  CKPT_MODE_FLAG=--resume
fi

echo "[mt4] node=$(hostname) gpu=$GPU arm=$ARM seed=$SEED tasks=${#TASKS[@]} eval_tasks=${#EVAL_TASKS[@]} rollouts/task=$N_ROLLOUTS ckpt=$CKPT_BASE_DIR"
echo "[mt4] checkpoint mode=$CKPT_MODE_FLAG $([ "$CKPT_MODE_FLAG" = "--overwrite" ] && echo '(WIPES the checkpoint dir)' || echo '(continues an existing run; FRESH=1 to wipe)')"
echo "[mt4] extra=${EXTRA_FLAGS[*]:-none}"

# ENTRY: swap the entry point while keeping this exact flag block. Used by the
# read-only analysis probes (scripts/probe_counterfactual_rollouts.py), which
# take the same tyro CLI as exp.py, so there is no second copy of the config to
# drift. Default is unchanged.
ENTRY="${ENTRY:-scripts/exp.py}"
RUN=(uv run "$ENTRY")
[ "${DRY:-0}" = "1" ] && RUN=(echo uv run "$ENTRY")

"${RUN[@]}" \
  "$CONFIG_NAME" \
  --project_name ogpo_multitask \
  --group_name mt4_libero \
  --exp_name "$EXP_NAME" \
  --checkpoint_base_dir "$CKPT_BASE_DIR" \
  --seed "$SEED" \
  --fsdp_devices "$FSDP" \
  "$CKPT_MODE_FLAG" \
  --log_interval 25 \
  --save_interval 100000 \
  --keep_period 100000 \
  --num_train_steps "$NUM_STEPS" \
  --ema_decay 0.99 \
  --lr_schedule.value 2.5e-5 \
  --max_runtime "$MAX_RUNTIME" \
  --collect.episode_steps_multiplier "$EP_MULT" \
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
  --batch_size "$BATCH" \
  "${EXTRA_FLAGS[@]}" \
  "$@"
