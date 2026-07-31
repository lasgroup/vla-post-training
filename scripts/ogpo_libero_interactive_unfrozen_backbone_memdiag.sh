#!/bin/bash
# ---------------------------------------------------------------------------
# OPERATION-HISTORY DIAGNOSTIC launcher for the unfrozen-backbone OGPO run.
#
#     bash scripts/ogpo_libero_interactive_unfrozen_backbone_memdiag.sh
#
# Goal: produce a chronological history of every high-level operation with
# VRAM readings, so the OOM can be attributed to an exact phase AND (via the
# XLA dump) an exact op/source line.
#
# It is scripts/ogpo_libero_interactive_unfrozen_backbone_memprofile.sh with:
#   1. Entry point scripts/exp_ogpo_unfrozen_backbone_memdiag.py, which wraps
#      every phase (init_train_state, collect, critic_update, policy_update,
#      ema_update, buffer_sample, ...) with a JSONL memory/operation log
#      ($MEMDIAG_DIR/history.jsonl) plus a 1 Hz background VRAM sampler.
#      Wrapped jitted calls are block_until_ready'd, so the async OOM cannot
#      be mis-attributed to a later phase.
#   2. A short warm-up before online RL: policy.training_start_step=10
#      (steps 1-9 are critic-only updates; the OOM-suspect PPO policy update
#      first runs at step 10). Override with POLICY_START=<n>.
#   3. Fewer collection rollouts (NUM_ROLLOUTS=8) — enough to fill the buffer
#      past batch_size, faster to reach the failure.
#   4. JAX_LOG_COMPILES=1 so each jit compilation is visible in the history
#      (a buffer-assignment OOM during COMPILATION is then distinguishable
#      from a runtime allocation OOM).
#
# Inherited from the memprofile script (unchanged): no preallocation so the
# OOM lands at the real allocation and prints the "Peak buffers" report; HLO +
# buffer-assignment dump under $HLO_DUMP_DIR; unfiltered tracebacks; the
# NUM_SDE_STEPS / BATCH_SIZE sweep knobs (defaults = real values 10 / 32).
#
# OUTPUTS (all under $MEMDIAG_DIR):
#   history.jsonl   - the operation/memory history (see footer for how to read)
#   console.log     - full stdout+stderr incl. XLA's peak-buffer report
#   traceback.txt   - full Python traceback of the fatal error
#   live_buffers_before_first_policy_update.prof - pprof of live buffers
#   ../hlo_dump/... - XLA HLO + buffer assignment (op -> source line)
# ---------------------------------------------------------------------------

set -euo pipefail

# ===========================================================================
# CONFIG -- edit these for your node
# ===========================================================================
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
STORE_ROOT="${STORE_ROOT:-$PROJECT_DIR/run_store}"

EXP_NAME="${EXP_NAME:-pi05_ogpo_unfrozen_backbone_MEMDIAG}"
CKPT_BASE_DIR="${CKPT_BASE_DIR:-$STORE_ROOT/checkpoints/ogpo_memdiag}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Diagnostic knobs.
POLICY_START="${POLICY_START:-10}"                 # first PPO policy update step
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-$((POLICY_START + 5))}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-8}"                  # episodes collected at step 0
NUM_SDE_STEPS="${NUM_SDE_STEPS:-10}"               # sweep to confirm scaling
BATCH_SIZE="${BATCH_SIZE:-32}"                     # sweep to confirm scaling

# Where the operation/memory history goes.
MEMDIAG_DIR="${MEMDIAG_DIR:-$STORE_ROOT/memdiag/$EXP_NAME}"
export MEMDIAG_LOG="${MEMDIAG_LOG:-$MEMDIAG_DIR/history.jsonl}"
export MEMDIAG_SAMPLE_SEC="${MEMDIAG_SAMPLE_SEC:-1.0}"

# Where XLA writes the HLO + buffer-assignment dump for this run.
HLO_DUMP_DIR="${HLO_DUMP_DIR:-$STORE_ROOT/hlo_dump/$EXP_NAME}"

# ===========================================================================
# (Usually no edits needed below this line)
# ===========================================================================
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/openpi/packages/openpi-client/src:$PROJECT_DIR/openpi/src:$PROJECT_DIR/openpi/packages/openpi-client:$PROJECT_DIR/molmospaces"

# --- keep every cache/config INSIDE the repo (safe on a shared machine) -----
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$STORE_ROOT/cache/openpi}"
export HF_HOME="${HF_HOME:-$STORE_ROOT/cache/huggingface}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$STORE_ROOT/libero}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$STORE_ROOT/cache/uv}"
export TORCH_HOME="${TORCH_HOME:-$STORE_ROOT/cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$STORE_ROOT/cache/triton}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$STORE_ROOT/cache/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$STORE_ROOT/cache/xdg}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$STORE_ROOT/config/xdg}"
export WANDB_DIR="${WANDB_DIR:-$STORE_ROOT/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$STORE_ROOT/cache/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$STORE_ROOT/config/wandb}"

# MuJoCo rendering backend. Default: headless GPU rendering via EGL. If you
# don't have working EGL, launch with MUJOCO_GL=osmesa (CPU software rendering
# -- no EGL/display needed, just slower); PYOPENGL_PLATFORM follows it.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# EGL vendor ICD selection (only matters when MUJOCO_GL=egl). Same resolution
# order as the base launcher: explicit override > system NVIDIA JSON >
# generated vendor JSON under STORE_ROOT (no root needed).
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

# NCCL (single node / single GPU: no InfiniBand, no multi-GPU collectives).
export NCCL_CUMEM_ENABLE=0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

# ===========================================================================
# PROFILING env (same as the memprofile script)
# ===========================================================================
# Don't preallocate: allocate on demand so the OOM lands at the real allocation
# and the runtime prints its "Peak buffers" report naming the largest buffers.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Let JAX allocate near capacity so the peak reflects the true requirement.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
# Dump HLO + buffer-assignment so the giant buffer maps to an op + source line.
mkdir -p "$HLO_DUMP_DIR" "$MEMDIAG_DIR"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_dump_to=$HLO_DUMP_DIR --xla_dump_hlo_as_text"
# Full C++ logs (incl. the peak-buffer report) and un-filtered Python traceback.
export TF_CPP_MIN_LOG_LEVEL=0
export JAX_TRACEBACK_FILTERING=off
# Log every jit compilation: separates "OOM while compiling" from "OOM at run".
export JAX_LOG_COMPILES="${JAX_LOG_COMPILES:-1}"

export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES%%,*}"

NUM_GPUS="$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")"
FSDP_DEVICES="${FSDP_DEVICES:-$NUM_GPUS}"

# Diagnostic run: no wandb.
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME" "$LIBERO_CONFIG_PATH" "$CKPT_BASE_DIR" \
         "$UV_CACHE_DIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" \
         "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" \
         "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

echo "[ogpo-memdiag] node=$(hostname) gpus=$CUDA_VISIBLE_DEVICES fsdp=$FSDP_DEVICES"
echo "[ogpo-memdiag] policy_start=$POLICY_START num_train_steps=$NUM_TRAIN_STEPS num_rollouts=$NUM_ROLLOUTS num_sde_steps=$NUM_SDE_STEPS batch_size=$BATCH_SIZE"
echo "[ogpo-memdiag] history  -> $MEMDIAG_LOG"
echo "[ogpo-memdiag] console  -> $MEMDIAG_DIR/console.log"
echo "[ogpo-memdiag] HLO dump -> $HLO_DUMP_DIR"

set +e
uv run scripts/exp_ogpo_unfrozen_backbone_memdiag.py \
  pi05_libero_online_ogpo_sft_unfrozen_backbone \
  --project_name ogpo_sweep \
  --group_name ogpo_memdiag \
  --exp_name "$EXP_NAME" \
  --checkpoint_base_dir "$CKPT_BASE_DIR" \
  --seed 0 \
  --fsdp_devices 1 \
  --overwrite \
  --log_interval 1 \
  --save_interval 100000 \
  --num_train_steps "$NUM_TRAIN_STEPS" \
  --lr_schedule.value 2.5e-5 \
  --max_runtime 169200 \
  --collect.tasks libero_90_44 \
  --collect.eval_tasks libero_90_44 \
  --collect.store_prefix_rep \
  --collect.collect_interval 10000 \
  --collect.num_rollouts "$NUM_ROLLOUTS" \
  --collect.env_num 8 \
  --collect.eval_env_num 8 \
  --collect.eval_interval 99999 \
  --rl.beta 0.05 \
  --rl.discount 0.995 \
  --rl.online_ratio 1.0 \
  --rl.buffer_capacity 250000 \
  --rl.policy.update_interval 1 \
  --rl.policy.training_start_step "$POLICY_START" \
  --rl.critic.td_weight_schedule.init_value 1 \
  --rl.critic.td_weight_schedule.end_value 1 \
  --rl.critic.td_weight_schedule.switch_step 999999 \
  --rl.critic.no-use_distributional_critic \
  --rl.critic.num_value_bins 1 \
  --rl.critic.batch_size 1024 \
  --rl.critic.pre_training_steps 0 \
  --rl.critic.use_bronet \
  --rl.critic.bronet_hidden_dim 1024 \
  --rl.critic.inference_start_step 1 \
  --rl.group_num_samples 1 \
  --rl.clip_epsilon 0.01 \
  --rl.bc_coeff 1.0 \
  --rl.num_sde_steps "$NUM_SDE_STEPS" \
  --rl.noise_level 0.02 \
  --rl.adv_strategy subtract_v \
  --rl.critic.value_target_type one_hot \
  --batch_size "$BATCH_SIZE" \
  2>&1 | tee "$MEMDIAG_DIR/console.log"
STATUS=${PIPESTATUS[0]}
set -e

echo ""
echo "[ogpo-memdiag] exit status: $STATUS"
cat <<EOF

============================ HOW TO READ THE OUTPUT ============================
The run is EXPECTED to die at the first policy update (step $POLICY_START).

1. WHICH OPERATION OOM'd (the history you asked for):
     tail -50 "$MEMDIAG_LOG" | grep -v '"ev": "sample"'
   The last 'phase_start' without a matching 'phase_end' -- or the explicit
   'phase_ERROR' record -- is the failing operation. Expected sequence:
     run_start -> init_train_state(jit) -> base_init -> awsft critic init
     -> collect_data (start_data_collection, sample_action, end_...)
     -> steps 1..$((POLICY_START-1)): buffer_sample + critic_update(jit) + ema_update(jit)
     -> step $POLICY_START: buffer_sample -> policy_update(jit)  <-- phase_ERROR here
   Each record carries bytes_in_use/peak_bytes_in_use, so you also get the
   memory staircase across phases. The 'sample' records (1 Hz) give the VRAM
   curve INSIDE the long policy_update call.

2. WHICH TENSOR/OP inside that phase:
   In console.log, right after "RESOURCE_EXHAUSTED"/"ran out of memory": the
   Peak buffers report lists the largest live buffers with shapes.

3. WHICH SOURCE LINE allocates it:
     grep -rE "f32\[[0-9,]+\]" "$HLO_DUMP_DIR"/*buffer-assignment* | sort -t'[' -k2 -rn | head
   then open the matching module .txt and read
   metadata={op_name=..., source_file=..., source_line=...}
   (expect the gemma attention path reached via score_chain_under_model /
   the BC compute_loss in src/rl/ogpo/update_actor.py).

4. Baseline vs transient: live_buffers_before_first_policy_update.prof is a
   pprof snapshot of everything resident BEFORE the failing step
   (weights + optimizer state + EMA + batch). View with: pprof -http ... file.

5. Confirm scaling (peak should track num_sde_steps x batch):
     NUM_SDE_STEPS=2 bash scripts/ogpo_libero_interactive_unfrozen_backbone_memdiag.sh
     BATCH_SIZE=8    bash scripts/ogpo_libero_interactive_unfrozen_backbone_memdiag.sh
================================================================================
EOF

exit "$STATUS"
