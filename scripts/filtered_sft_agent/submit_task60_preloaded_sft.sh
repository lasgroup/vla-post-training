#!/usr/bin/env bash
set -euo pipefail

# Submit two dataset-only filtered-SFT runs for LIBERO task 60:
#   1. random-200 selected rollout pickles
#   2. reward-filtered-200 selected rollout pickles
# Training uses only --rl.preload_episodes_from_path. No collection happens in
# scripts/filtered_sft_agent/preloaded_sft_exp.py.

REPO_ROOT="$(pwd -P)"
ACCOUNT="${ACCOUNT:-a143}"
PARTITION="${PARTITION:-normal}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
ENVIRONMENT="${ENVIRONMENT:-vla-post-training-marco-reward}"
GPUS="${GPUS:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
BUFFER_CAPACITY="${BUFFER_CAPACITY:-250000}"
EVAL_ROLLOUTS="${EVAL_ROLLOUTS:-32}"
EVAL_ENVS="${EVAL_ENVS:-1}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}"

DATASET_BASE="${DATASET_BASE:-/capstor/scratch/cscs/dsimoes/vlm-rm/filtered_sft_exports/final_task0_task60_200_20260701}"
RANDOM_EPISODES="${RANDOM_EPISODES:-${DATASET_BASE}/random_200_seed0/task60_random_n200_seed0/episodes}"
REWARD_EPISODES="${REWARD_EPISODES:-${DATASET_BASE}/reward_lcb_beta1_200_seed0/task60_reward_n200_seed0/episodes}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/capstor/scratch/cscs/${USER}/vla-post-training/preloaded_sft_task60_${RUN_TAG}}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${OUTPUT_ROOT}/checkpoints}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
SBATCH_DIR="${LOG_DIR}/sbatch"
CACHE_ROOT="/capstor/scratch/cscs/${USER}"
VENV_PATH="${VENV_PATH:-/.venv}"
mkdir -p "$LOG_DIR" "$SBATCH_DIR" "$CHECKPOINT_BASE_DIR"

validate_episode_dir() {
  local label="$1"
  local dir="$2"
  if [[ ! -d "$dir" ]]; then
    echo "[$label] missing episode directory: $dir" >&2
    exit 1
  fi
  local count
  count=$(find "$dir" -maxdepth 1 -type f -name 'episode_*.pkl' | wc -l)
  echo "[$label] episode_count=$count dir=$dir"
  if [[ "$count" -le 0 ]]; then
    echo "[$label] no episode_*.pkl files found in $dir" >&2
    exit 1
  fi
}

validate_episode_dir random "$RANDOM_EPISODES"
validate_episode_dir reward "$REWARD_EPISODES"

submit_one() {
  local label="$1"
  local episodes_dir="$2"
  local exp_name="task60_${label}_preloaded_sft_${RUN_TAG}"
  local script_path="${SBATCH_DIR}/${exp_name}.sbatch.sh"
  local stdout_path="${LOG_DIR}/${exp_name}-%j.out"
  local stderr_path="${LOG_DIR}/${exp_name}-%j.err"

  cat > "$script_path" <<SBATCH
#!/usr/bin/env bash
set -euo pipefail

cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/openpi/src:$REPO_ROOT/openpi/packages/openpi-client/src"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.45}"
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export EGL_PLATFORM="surfaceless"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export MUJOCO_EGL_NUM_DEVICES="${MUJOCO_EGL_NUM_DEVICES:-1}"
export XDG_CACHE_HOME="$CACHE_ROOT/.cache"
export UV_CACHE_DIR="$CACHE_ROOT/uv-cache"
export HF_HOME="$CACHE_ROOT/hf-home"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf-datasets"
export WANDB_DIR="$OUTPUT_ROOT/wandb"
export MPLCONFIGDIR="$CACHE_ROOT/mpl-cache"
export JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax-cache"
export PRELOADED_SFT_FINAL_EVAL="1"
mkdir -p "\$XDG_CACHE_HOME" "\$UV_CACHE_DIR" "\$HF_HOME" "\$HF_DATASETS_CACHE" "\$WANDB_DIR" "\$MPLCONFIGDIR" "\$JAX_COMPILATION_CACHE_DIR"
# Repair the known fsspec/OpenPI cache nesting where gs://.../params may land as params/params/*.
OPENPI_PARAMS_PARENT="$CACHE_ROOT/openpi_assets/openpi-assets/checkpoints/pi05_libero/params"
if [[ -d "\$OPENPI_PARAMS_PARENT/params" && ! -e "\$OPENPI_PARAMS_PARENT/_METADATA" ]]; then
  (cd "\$OPENPI_PARAMS_PARENT" && for x in _METADATA _sharding array_metadatas d manifest.ocdbt ocdbt.process_0; do [[ -e "params/\$x" && ! -e "\$x" ]] && ln -s "params/\$x" "\$x" || true; done)
fi

echo "[\$(date --iso-8601=seconds)] host=\$(hostname) repo=\$(pwd -P)"
echo "[\$(date --iso-8601=seconds)] label=$label episodes=$episodes_dir"
echo "[\$(date --iso-8601=seconds)] checkpoint_base=$CHECKPOINT_BASE_DIR exp_name=$exp_name"

srun --account="$ACCOUNT" --environment="$ENVIRONMENT" --ntasks=1 \
  "$VENV_PATH/bin/python" scripts/filtered_sft_agent/preloaded_sft_exp.py \
    pi05_libero_online_filtered_sft \
    --exp_name "$exp_name" \
    --checkpoint_base_dir "$CHECKPOINT_BASE_DIR" \
    --overwrite \
    --no-wandb_enabled \
    --seed 0 \
    --batch_size "$BATCH_SIZE" \
    --fsdp_devices "$GPUS" \
    --num_train_steps "$NUM_TRAIN_STEPS" \
    --save_interval "$SAVE_INTERVAL" \
    --log_interval "$LOG_INTERVAL" \
    --rl.online_ratio 1.0 \
    --rl.buffer_capacity "$BUFFER_CAPACITY" \
    --rl.preload_episodes_from_path "$episodes_dir" \
    --collect.tasks libero_90_60 \
    --collect.eval_tasks libero_90_60 \
    --collect.eval_env_num "$EVAL_ENVS" \
    --collect.num_eval_rollouts "$EVAL_ROLLOUTS"
SBATCH

  chmod +x "$script_path"

  local job_id
  local sbatch_args=(
    --account="$ACCOUNT"
    --partition="$PARTITION"
    --time="$TIME_LIMIT"
    --ntasks=1
    --cpus-per-task="$CPUS_PER_TASK"
    -G "$GPUS"
    --job-name="fsft60_${label}"
    --output="$stdout_path"
    --error="$stderr_path"
    "$script_path"
  )
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry-run] sbatch --test-only ${sbatch_args[*]}"
    sbatch --test-only "${sbatch_args[@]}"
    job_id="DRYRUN_${label}"
  else
    job_id=$(sbatch --parsable "${sbatch_args[@]}")
  fi

  echo "$label job_id=$job_id exp_name=$exp_name stdout=${stdout_path//%j/$job_id} stderr=${stderr_path//%j/$job_id} checkpoint_dir=${CHECKPOINT_BASE_DIR}/pi05_libero_online_filtered_sft/${exp_name}"
  printf '{"label":"%s","job_id":"%s","exp_name":"%s","episodes_dir":"%s","stdout":"%s","stderr":"%s","checkpoint_dir":"%s"}\n' \
    "$label" "$job_id" "$exp_name" "$episodes_dir" "${stdout_path//%j/$job_id}" "${stderr_path//%j/$job_id}" "${CHECKPOINT_BASE_DIR}/pi05_libero_online_filtered_sft/${exp_name}" >> "${OUTPUT_ROOT}/submitted_jobs.jsonl"
}

: > "${OUTPUT_ROOT}/submitted_jobs.jsonl"
submit_one random "$RANDOM_EPISODES"
submit_one reward "$REWARD_EPISODES"

cat > "${OUTPUT_ROOT}/submission_manifest.json" <<EOF
{
  "run_tag": "${RUN_TAG}",
  "repo_root": "${REPO_ROOT}",
  "output_root": "${OUTPUT_ROOT}",
  "checkpoint_base_dir": "${CHECKPOINT_BASE_DIR}",
  "venv_path": "${VENV_PATH}",
  "random_episodes": "${RANDOM_EPISODES}",
  "reward_episodes": "${REWARD_EPISODES}",
  "num_train_steps": ${NUM_TRAIN_STEPS},
  "batch_size": ${BATCH_SIZE},
  "gpus": ${GPUS},
  "eval_rollouts": ${EVAL_ROLLOUTS},
  "eval_envs": ${EVAL_ENVS},
  "environment": "${ENVIRONMENT}",
  "account": "${ACCOUNT}",
  "partition": "${PARTITION}",
  "time_limit": "${TIME_LIMIT}"
}
EOF

echo "submission_manifest=${OUTPUT_ROOT}/submission_manifest.json"
echo "submitted_jobs=${OUTPUT_ROOT}/submitted_jobs.jsonl"
