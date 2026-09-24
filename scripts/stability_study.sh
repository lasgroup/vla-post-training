#!/bin/bash
# ---------------------------------------------------------------------------
# Single-task stability study launcher (one arm per invocation, single GPU).
#
# Base recipe = abl_viii_r2, the single-task winner (frozen backbone, GRPO
# G=8 group-mean baseline, eps=0.1, success-buffer BC), PLUS periodic
# 32-episode EMA evals every 10k steps (the study's low-noise measurement).
#
# Interventions are toggled per-arm via env vars (all default OFF = baseline):
#   ARM        - run name suffix (required), e.g. base_s1, E, N, NC, G, ENCG
#   SEED       - default 0
#   GPU        - CUDA device index (required on multi-GPU nodes)
#   EMA        - actor ema_decay (default 0.99; E arms use 0.999)
#   NORM       - 1 => --rl.normalize_group_advantage (EMA-quantile scale)
#   CLIP_SYM   - float => --rl.adv_clip_sym <v> (symmetric clip, post-norm)
#   ACCUM      - int  => --rl.policy_grad_accum <M> (micro-batch grad accum)
#   CONS       - 1 => --rl.advantage_combination grpo_conservative:
#                per-head Q_i - mean_G(Q_i), sign-unanimous combine across
#                the 2 Q heads. V(s) NEVER enters the advantage (it cancels
#                per-head under the group baseline; this mode makes that
#                explicit). Replaces the vanilla group-mean centering.
#   BURST      - int => --rl.post_collection_critic_steps <K>: K critic-only
#                updates right after each collection round (not step 0),
#                before the next policy update, so the actor never ranks
#                fresh actions with an uncalibrated critic. K=1000 ≈ 30
#                visits per new transition at mid-run buffer size.
#   QS         - int => --rl.critic.num_qs/num_vs <n> (default 2). With
#                CONS=1, sign-unanimity across n heads is a much stricter
#                gate (2 correlated heads share delusions — the CAB crash).
#   FRESH      - 1 => --overwrite (WIPES the checkpoint dir, starts at step 0).
#                Default is --resume, so re-running an arm continues it.
#   TASK       - ONE LIBERO task id for collect.tasks AND collect.eval_tasks
#                (default libero_90_44; a space-separated list renders as a
#                single bad token -- multi-task goes through the trailing
#                arguments below). The exp name stays stab_${ARM}, so put the
#                task in ARM as the single-task campaign did
#                (ARM=NCB_libero_90_38 TASK=libero_90_38 -> stab_NCB_libero_90_38).
#                Mirrors the sibling vla_single_task checkout's recipe, which
#                produced those checkpoints. TRAP (same as LORA, see
#                ws_bcbb_pipeline.sh:20-24): a TASK exported in the submitting
#                shell reaches every arm launched through this recipe,
#                ws_bcbb_pipeline.sh included, and silently retargets it --
#                the banner's task= field is the only tell.
#   CKPT_BASE_DIR - checkpoint root (default $STORE_ROOT/checkpoints/stability_study);
#                set it alone, not STORE_ROOT, to restore a run from another root.
#   TD_W       - float => --rl.critic.td_weight_schedule.{init,end}_value <v>
#                (default 1, i.e. 100% TD target -- reproduces prior behavior
#                exactly). Emitted UNCONDITIONALLY: this flag was previously a
#                fixed `1` that silently overrode any CONFIG_NAME's own
#                td_weight_schedule default (e.g. pi05_libero_online_ogpo_ref's
#                0.95), same contract as ogpo_multitask_4task.sh's TD_W.
#   SUCC_BONUS - float => --collect.success_reward_bonus <v> (default 0,
#                reproduces prior behavior exactly -- this flag was previously
#                never emitted). Not baked into any registered config's own
#                default (collect.success_reward_bonus is a global
#                CollectionConfig default of 0.0), so it needs setting
#                explicitly regardless of CONFIG_NAME -- mirrors
#                ogpo_multitask_4task.sh's SUCC_BONUS (reference default 90).
#   See scripts/stability_study_ref.sh for the reference-aligned single-task
#   wrapper (CONFIG_NAME=pi05_libero_online_ogpo_ref, TD_W=0.95, SUCC_BONUS=90).
#   DRY        - 1 => print the final command instead of running the entry
#                point. Still seeds $LIBERO_CONFIG_PATH/config.yaml if it is
#                missing (that block runs first, as in ogpo_multitask_4task.sh).
#   Trailing arguments are appended verbatim after the flag block (tyro takes
#   the last occurrence), e.g. `... bash scripts/stability_study.sh --rl.n_samples 8`
#   -- the read-only probes use this for knobs the recipe does not expose.
#
# Usage:  ARM=N NORM=1 GPU=3 bash scripts/stability_study.sh
#         ARM=N NORM=1 GPU=3 DRY=1 bash scripts/stability_study.sh   # print, don't run
# ---------------------------------------------------------------------------
set -euo pipefail

: "${ARM:?set ARM (run name suffix)}"
: "${GPU:?set GPU (cuda device index)}"
SEED="${SEED:-0}"
EMA="${EMA:-0.99}"
TASK="${TASK:-libero_90_44}"
TD_W="${TD_W:-1}"
SUCC_BONUS="${SUCC_BONUS:-0}"
# Global (not per-device) policy batch: states per policy update. The actor
# jit expands it to BATCH x rl.group_num_samples SDE chains -- the dominant
# memory knob (mirrors ogpo_multitask_4task.sh's BATCH). Must divide
# jax.device_count() (filtered_sft_learner.py:232).
BATCH="${BATCH:-32}"

# Non-interactive shells (nohup over ssh) miss ~/.local/bin.
export PATH="$HOME/.local/bin:$PATH"

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
STORE_ROOT="${STORE_ROOT:-$PROJECT_DIR/run_store}"
EXP_NAME="stab_${ARM}"
# Overridable so a probe can restore a run whose checkpoints live under
# another root while every cache (openpi assets, LIBERO config, wandb) stays
# in the local store -- how the single-task campaign itself was launched
# (sibling vla_single_task recipe :46). Redirecting STORE_ROOT instead moves
# the openpi asset cache too and the policy-twin load then fails
# (q_spread_stab job 10352338, _METADATA missing under .../cache/openpi).
CKPT_BASE_DIR="${CKPT_BASE_DIR:-$STORE_ROOT/checkpoints/stability_study}"

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
export MUJOCO_EGL_DEVICE_ID="$GPU"

EXTRA_FLAGS=()
# Reference-alignment: unconditional, mirrors ogpo_multitask_4task.sh:221-222
# (SUCC_BONUS is not baked into any config's own default -- see TD_W/SUCC_BONUS
# header docs above).
EXTRA_FLAGS+=(--collect.success_reward_bonus "$SUCC_BONUS")
[ "${NORM:-0}" = "1" ] && EXTRA_FLAGS+=(--rl.normalize_group_advantage)
[ -n "${CLIP_SYM:-}" ] && EXTRA_FLAGS+=(--rl.adv_clip_sym "$CLIP_SYM")
[ -n "${ACCUM:-}" ] && EXTRA_FLAGS+=(--rl.policy_grad_accum "$ACCUM")
[ "${CONS:-0}" = "1" ] && EXTRA_FLAGS+=(--rl.advantage_combination grpo_conservative)
[ -n "${BURST:-}" ] && EXTRA_FLAGS+=(--rl.post_collection_critic_steps "$BURST")
[ -n "${QS:-}" ] && EXTRA_FLAGS+=(--rl.critic.num_qs "$QS" --rl.critic.num_vs "$QS")
[ -n "${UTD:-}" ] && EXTRA_FLAGS+=(--rl.critic_utd "$UTD")
[ "${BURST_MC:-0}" = "1" ] && EXTRA_FLAGS+=(--rl.burst_use_mc_targets)
# Policy warmstart: PG term muted before this step (BC-only actor phase).
[ -n "${PG_START:-}" ] && EXTRA_FLAGS+=(--rl.pg_start_step "$PG_START")
# Linear PG ramp-in length after the handoff (0 = hard switch).
[ -n "${PG_RAMP:-}" ] && EXTRA_FLAGS+=(--rl.pg_ramp_steps "$PG_RAMP")
# Pipeline hooks (ws_bcbb_pipeline.sh): alternate entry script / config name /
# step budget / save interval / initial weights checkpoint.
ENTRY="${ENTRY:-scripts/exp.py}"
CONFIG_NAME="${CONFIG_NAME:-pi05_libero_online_ogpo_sft}"
N_STEPS="${N_STEPS:-100000}"
SAVE_INT="${SAVE_INT:-100000}"
[ -n "${WEIGHT_LOADER:-}" ] && EXTRA_FLAGS+=(--weight_loader.params_path "$WEIGHT_LOADER")
# Ralf-style filtered SFT BC (success-masked online batch instead of succ buffer)
[ "${FSFT:-0}" = "1" ] && EXTRA_FLAGS+=(--rl.bc_filtered_sft)
# FC (frequent collection): override interval/rollouts, e.g. FC_INT=1000 FC_ROLLOUTS=2
[ -n "${FC_INT:-}" ] && COLLECT_INT="$FC_INT" || COLLECT_INT=10000
[ -n "${FC_ROLLOUTS:-}" ] && N_ROLLOUTS="$FC_ROLLOUTS" || N_ROLLOUTS=20
# Extra episodes at the step-0 collection ONLY (spark insurance: kills the
# empty-success-buffer lottery; 100 extra => P(0 successes) ~ 0.2% at 5% SR).
[ -n "${INIT_ROLLOUTS:-}" ] && EXTRA_FLAGS+=(--collect.num_initial_rollouts "$INIT_ROLLOUTS")
# --- backbone LoRA (docs/changes/2026-08-29-backbone-lora/) ---
# Emitted UNCONDITIONALLY, both branches (mirrors ogpo_multitask_4task.sh —
# second copy, keep in sync): a real CLI flag, so the env var is authoritative
# for any CONFIG_NAME. LORA=1 trains the 2B PaliGemma LLM backbone through
# rank-16/alpha-16 adapters (+27.87M params); SigLIP and the base weights stay
# frozen; the critic's prefix embeddings stay adapter-free (R1). NEW-RUN ONLY
# (--resume across a LORA flip fails loudly in orbax). tyro renders the bool as
# a flag PAIR — `--backbone_lora True` does not parse.
# NOTE: --rl.dedup_group_prefix (below) is MEMORY-LOAD-BEARING when LORA=1.
if [ "${LORA:-0}" = "1" ]; then
  EXTRA_FLAGS+=(--backbone_lora)
else
  EXTRA_FLAGS+=(--no-backbone_lora)
fi

mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME" "$LIBERO_CONFIG_PATH" "$CKPT_BASE_DIR" \
         "$UV_CACHE_DIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" \
         "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" \
         "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

# LIBERO prompts interactively on first import if its config file is missing —
# fatal in a batch job (EOFError; q_spread_stab job 10352129 died there on a
# redirected STORE_ROOT). Seed the default config beforehand, exactly as
# ogpo_multitask_4task.sh:253-254 and the sibling vla_single_task recipe do.
if [ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]; then
  printf 'n\n' | uv run python -c "import libero.libero" >/dev/null 2>&1 || true
fi

# Checkpoint mode. --resume by default so re-running an interrupted arm
# continues it; --overwrite rmtrees the checkpoint dir (openpi
# checkpoints.py:26-29). Safe on a first launch: openpi downgrades resume to a
# fresh start when the dir is absent or holds no checkpoints. FRESH=1 opts back
# into the wipe; an explicit CKPT_MODE_FLAG wins over both. (Deliberately a
# second copy of ogpo_multitask_4task.sh's block: the two recipes' env preambles
# have already diverged and are not being unified here — OQ-2.)
if [ -n "${CKPT_MODE_FLAG:-}" ]; then
  :
elif [ "${FRESH:-0}" = "1" ]; then
  CKPT_MODE_FLAG=--overwrite
else
  CKPT_MODE_FLAG=--resume
fi

echo "[stability-study] node=$(hostname) gpu=$GPU arm=$ARM task=$TASK seed=$SEED ema=$EMA extra=${EXTRA_FLAGS[*]:-none}"
echo "[stability-study] checkpoint mode=$CKPT_MODE_FLAG $([ "$CKPT_MODE_FLAG" = "--overwrite" ] && echo '(WIPES the checkpoint dir)' || echo '(continues an existing run; FRESH=1 to wipe)')"

# ENTRY swaps the python entry point for the read-only probes (same contract as
# ogpo_multitask_4task.sh); DRY=1 prints the command instead of running it.
RUN=(uv run "$ENTRY")
[ "${DRY:-0}" = "1" ] && RUN=(echo uv run "$ENTRY")

"${RUN[@]}" \
  "$CONFIG_NAME" \
  --project_name ogpo_stability \
  --group_name stability_study \
  --exp_name "$EXP_NAME" \
  --checkpoint_base_dir "$CKPT_BASE_DIR" \
  --seed "$SEED" \
  --fsdp_devices 1 \
  "$CKPT_MODE_FLAG" \
  --log_interval 25 \
  --save_interval "$SAVE_INT" \
  --keep_period "$SAVE_INT" \
  --num_train_steps "$N_STEPS" \
  --ema_decay "$EMA" \
  --lr_schedule.value 2.5e-5 \
  --max_runtime 169200 \
  --collect.tasks "$TASK" \
  --collect.eval_tasks "$TASK" \
  --collect.store_prefix_rep \
  --collect.collect_interval "$COLLECT_INT" \
  --collect.num_rollouts "$N_ROLLOUTS" \
  --collect.env_num 8 \
  --collect.eval_env_num 8 \
  --collect.eval_interval 10000 \
  --rl.beta 0.05 \
  --rl.discount 0.995 \
  --rl.online_ratio 1.0 \
  --rl.buffer_capacity 250000 \
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
