#!/bin/bash
# ---------------------------------------------------------------------------
# Filtered-SFT runscript, shared by the three backbone-tuning arms.
# Do not sbatch this directly -- use scripts/fsft_none.sh / fsft_lora.sh /
# fsft_full.sh, which only set PG_TUNE and the job name.
#
# PG_TUNE selects how much of PaliGemma is trained. The action expert (plus the
# small action heads) is trained in ALL three arms; that is the only thing the
# arms have in common besides the recipe below.
#
#   none  pi05_libero_online_filtered_sft_frozen_backbone
#         PaliGemma LLM + SigLIP frozen and cast to bf16. ~430M trainable.
#   lora  pi05_libero_online_filtered_sft_lora_backbone
#         paligemma_variant=gemma_2b_lora: rank-16 attn+ffn adapters trainable
#         (~28M), base LLM weights and SigLIP frozen. ~458M trainable.
#         The pretrained checkpoint has no adapters -- they start from the
#         model's own normal(0.01) init, both in the train state (the weight
#         loader skips `.*lora.*`) and in the collection policy
#         (src/training/policy_utils.py).
#   full  pi05_libero_online_filtered_sft
#         No freeze filter: LLM, SigLIP and action expert all finetuned.
#         ~3.35G trainable. This is what fsft_libero_babel.sh ran.
#
# Everything else is fsft_libero_babel.sh's recipe verbatim (5k steps, collect
# every 500, lr 2.5e-5, batch 256), so the three arms differ from the existing
# FSFT baseline in exactly one field. The exception is the task set: training
# now defaults to 4 tasks (libero_90_82/79/31/38) rather than libero_90_38
# alone, so these runs are not comparable to the earlier single-task numbers.
#
# STATE=1 additionally gives the POLICY the proprioceptive state as pi0.5's
# discrete language tokens (default 0 = the state-blind pi05_libero recipe, which
# is what every existing arm ran). The critic already sees state either way.
#
# Env knobs: PG_TUNE (required), STATE, SEED, ARM (exp-name suffix), TASKS/EVAL_TASKS,
# NUM_STEPS, COLLECT_INT, N_ROLLOUTS, BATCH_SIZE, LR, FSDP, GPU, SAVE_INT,
# STORE_ROOT, CKPT_BASE_DIR, PROJECT_DIR, PY, DRY.
# ---------------------------------------------------------------------------
set -euo pipefail

PG_TUNE="${PG_TUNE:?set PG_TUNE=none|lora|full, or launch via scripts/fsft_none.sh etc.}"
case "$PG_TUNE" in
  none) CONFIG_NAME=pi05_libero_online_filtered_sft_frozen_backbone ;;
  lora) CONFIG_NAME=pi05_libero_online_filtered_sft_lora_backbone ;;
  full) CONFIG_NAME=pi05_libero_online_filtered_sft ;;
  *) echo "[fsft] PG_TUNE must be none|lora|full, got '$PG_TUNE'" >&2; exit 1 ;;
esac

SEED="${SEED:-0}"
ARM="${ARM:-v0}"
# STATE=1 feeds proprioception to the POLICY as pi0.5's discrete language tokens
# ("Task: <text>, State: <8 ints in 0..255>;\nAction: "). Off by default: the
# pi05_libero checkpoint was finetuned with discrete_state_input=False, so the
# existing arms -- and the fsft_pg_full baseline -- stay byte-identical at STATE=0.
STATE="${STATE:-0}"
EXTRA_FLAGS=()
STATE_SUFFIX=""
if [ "$STATE" = "1" ]; then
  EXTRA_FLAGS+=(--model.discrete-state-input)
  STATE_SUFFIX="_state"
fi
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
_NGPU=$(awk -F, '{print NF}' <<<"$GPU")
FSDP="${FSDP:-$_NGPU}"

# Non-interactive shells (nohup over ssh) miss ~/.local/bin.
export PATH="$HOME/.local/bin:$PATH"

# Absolute, not derived from $0: sbatch copies the batch script into its spool
# dir, so dirname "$0" does not resolve back to the checkout under SLURM.
PROJECT_DIR="${PROJECT_DIR:-/home/mananaga/VLA/ogpo/vla-post-training}"
# Everything on user_data, like ogpo_privileged.sh / ogpo_multitask_4task.sh:
# group_data IOs are slow and maxlab is unreachable. Note this is NOT where
# fsft_libero_babel.sh kept its caches (group_data), so the first run here
# re-downloads the pi05 checkpoint; export STORE_ROOT to reuse the old cache.
STORE_ROOT="${STORE_ROOT:-/data/user_data/mananaga/vla-post-training}"
EXP_NAME="${EXP_NAME:-fsft_pg_${PG_TUNE}${STATE_SUFFIX}_${ARM}_s${SEED}}"
CKPT_BASE_DIR="${CKPT_BASE_DIR:-$STORE_ROOT/checkpoints/fsft_backbone_ablation}"

# Reuse the babel venv rather than building one here: pyproject.toml and
# uv.lock are byte-identical between the checkouts, and that venv has no
# editable install of vla-post-training/openpi/molmospaces -- all three come
# from PYTHONPATH below, so the interpreter runs THIS tree's code. Call the
# interpreter directly, never `uv run`: uv re-resolves against the branch
# lockfile and mutates the venv, which would break the babel checkout too.
PY="${PY:-/home/mananaga/VLA/manan_babel/vla-post-training/.venv/bin/python}"
[ -x "$PY" ] || { echo "[fsft] no interpreter at $PY" >&2; exit 1; }

cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/openpi/packages/openpi-client/src:$PROJECT_DIR/openpi/src:$PROJECT_DIR/openpi/packages/openpi-client:$PROJECT_DIR/molmospaces"

export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$STORE_ROOT/cache/openpi}"
export HF_HOME="${HF_HOME:-$STORE_ROOT/cache/huggingface}"
# Your existing seeded config (~/.libero/config.yaml).
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.libero}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$STORE_ROOT/cache/uv}"
export TORCH_HOME="${TORCH_HOME:-$STORE_ROOT/cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$STORE_ROOT/cache/triton}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$STORE_ROOT/cache/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$STORE_ROOT/cache/xdg}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$STORE_ROOT/config/xdg}"
# Online, using the ~/.netrc credentials; WANDB_MODE=offline for local logging.
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
# Leave real VRAM headroom on the shared GPUs for MuJoCo/EGL offscreen framebuffers.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}"

export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_EGL_DEVICE_ID="${GPU%%,*}"

mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME" "$LIBERO_CONFIG_PATH" "$CKPT_BASE_DIR" \
         "$UV_CACHE_DIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" \
         "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" \
         "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

# LIBERO prompts interactively on first import if its config file is missing --
# fatal in a batch job (EOFError). Seed the default config beforehand.
if [ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]; then
  printf 'n\n' | "$PY" -c "import libero.libero" >/dev/null 2>&1 || true
fi

# Default training set: 4 LIBERO-90 tasks. Note this is NOT written as
# `("${TASKS[@]:-a b c}")` -- inside double quotes that default collapses into a
# single argv word ("a b c"), which openpi would reject as one bogus task name.
# Branch on whether the caller set anything instead. `${TASKS+x}` (not
# `${#TASKS[@]}`) is the test that survives `set -u` when TASKS is unset.
if [ -z "${TASKS+x}" ]; then
  TASKS=(libero_90_82 libero_90_79 libero_90_31 libero_90_38)
fi
if [ -z "${EVAL_TASKS+x}" ]; then
  EVAL_TASKS=("${TASKS[@]}")
fi

echo "[fsft] node=$(hostname) job=${SLURM_JOB_ID:-none} pg_tune=$PG_TUNE state=$STATE config=$CONFIG_NAME exp=$EXP_NAME gpu=$GPU fsdp=$FSDP"

# Filtered SFT has no critic, so none of the --rl.critic.* / --rl.beta knobs from
# awr_libero_babel.sh apply here. Schedule follows
# scripts/configs/multitask/fsft/libero/fsft_libero_tasks4-16_seed0_v0.yaml
# (5k steps, collect every 500).
RUN=("$PY" scripts/exp.py)
[ "${DRY:-0}" = "1" ] && RUN=(echo "$PY" scripts/exp.py)

exec "${RUN[@]}" \
  "$CONFIG_NAME" \
  --project_name openpi \
  --group_name fsft_backbone_ablation \
  --exp_name "$EXP_NAME" \
  --checkpoint_base_dir "$CKPT_BASE_DIR" \
  --seed "$SEED" \
  --fsdp_devices "$FSDP" \
  --overwrite \
  --log_interval 25 \
  --save_interval "${SAVE_INT:-100000}" \
  --num_train_steps "${NUM_STEPS:-5000}" \
  --lr_schedule.value "${LR:-2.5e-5}" \
  --max_runtime "${MAX_RUNTIME:-169200}" \
  --collect.tasks "${TASKS[@]}" \
  --collect.eval_tasks "${EVAL_TASKS[@]}" \
  --collect.collect_interval "${COLLECT_INT:-500}" \
  --collect.num_rollouts "${N_ROLLOUTS:-20}" \
  --collect.env_num 8 \
  --collect.eval_env_num 8 \
  --collect.eval_interval "${EVAL_INT:-4999}" \
  --rl.discount 0.995 \
  --rl.online_ratio 1.0 \
  --rl.buffer_capacity 500000 \
  --batch_size "${BATCH_SIZE:-256}" \
  "${EXTRA_FLAGS[@]}" \
  "$@"
