#!/usr/bin/env bash
set -euo pipefail

ACCOUNT=${ACCOUNT:-a0220}
PARTITION=${PARTITION:-normal}
ENVIRONMENT=${ENVIRONMENT:-vla-post-training-marco-reward}
REPO_ROOT=${REPO_ROOT:-/users/dsimoes/worktrees/vla-post-training/marco/reward}
RUN_ROOT=${RUN_ROOT:?Set RUN_ROOT to the output directory produced by submit_task0_successlike_sft.sh}
CACHE_ROOT=${CACHE_ROOT:-/capstor/scratch/cscs/dsimoes}
VENV_PATH=${VENV_PATH:-/.venv}
TASK_ID=${TASK_ID:-libero_90_0}
CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR:-$RUN_ROOT/checkpoints}
LOG_DIR=${RUN_ROOT}/logs
SBATCH_DIR=${LOG_DIR}/sbatch
RENDER_SCRIPT=${RENDER_SCRIPT:-$RUN_ROOT/scripts/render_task0_policy_gifs.py}
RENDER_EPISODES=${RENDER_EPISODES:-3}
RENDER_FRAME_STRIDE=${RENDER_FRAME_STRIDE:-4}
RENDER_SKIP_DEPENDENCY=${RENDER_SKIP_DEPENDENCY:-0}
SFT_RENDER_TIME_LIMIT=${SFT_RENDER_TIME_LIMIT:-03:00:00}
PRETRAINED_RENDER_TIME_LIMIT=${PRETRAINED_RENDER_TIME_LIMIT:-02:00:00}
mkdir -p "$SBATCH_DIR" "$RUN_ROOT/gifs" "$(dirname "$RENDER_SCRIPT")"
if [[ ! -f "$RENDER_SCRIPT" ]]; then
  cp "$REPO_ROOT/scripts/filtered_sft_agent/render_policy_gifs.py" "$RENDER_SCRIPT"
fi

sbatch_dependency_args() {
  local dep=$1
  if [[ "$RENDER_SKIP_DEPENDENCY" == "1" ]]; then
    return 0
  fi
  printf '%s\n' "--dependency=afterok:$dep"
}

test -f "$RUN_ROOT/submitted_jobs.jsonl" || { echo "Missing $RUN_ROOT/submitted_jobs.jsonl" >&2; exit 1; }
cd "$REPO_ROOT"

common_env_block() {
  cat <<ENV
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/openpi/src:$REPO_ROOT/openpi/packages/openpi-client/src"
export OPENPI_DATA_HOME=$CACHE_ROOT/openpi_assets
export XDG_CACHE_HOME=$CACHE_ROOT/.cache
export UV_CACHE_DIR=$CACHE_ROOT/uv-cache
export HF_HOME=$CACHE_ROOT/hf-home
export HF_DATASETS_CACHE=$CACHE_ROOT/hf-datasets
export WANDB_DIR=$RUN_ROOT/wandb
export MPLCONFIGDIR=$CACHE_ROOT/mpl-cache
export JAX_COMPILATION_CACHE_DIR=$CACHE_ROOT/jax-cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless
export MUJOCO_EGL_DEVICE_ID=0
export MUJOCO_EGL_NUM_DEVICES=1
mkdir -p "\$OPENPI_DATA_HOME" "\$XDG_CACHE_HOME" "\$UV_CACHE_DIR" "\$HF_HOME" "\$HF_DATASETS_CACHE" "\$WANDB_DIR" "\$MPLCONFIGDIR" "\$JAX_COMPILATION_CACHE_DIR"
OPENPI_PARAMS_PARENT="$CACHE_ROOT/openpi_assets/openpi-assets/checkpoints/pi05_libero/params"
if [[ -d "\$OPENPI_PARAMS_PARENT/params" && ! -e "\$OPENPI_PARAMS_PARENT/_METADATA" ]]; then
  (cd "\$OPENPI_PARAMS_PARENT" && for x in _METADATA _sharding array_metadatas d manifest.ocdbt ocdbt.process_0; do [[ -e "params/\$x" && ! -e "\$x" ]] && ln -s "params/\$x" "\$x" || true; done)
fi
ENV
}

submit_render_sft() {
  local label=$1
  local dep=$2
  local exp_name=$3
  local script_path="$SBATCH_DIR/render_${label}.sbatch.sh"
  local stdout_path="$LOG_DIR/render_${label}-%j.out"
  local stderr_path="$LOG_DIR/render_${label}-%j.err"
  cat > "$script_path" <<SBATCH
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO_ROOT"
$(common_env_block)
export RENDER_POLICY_KIND=sft
export RENDER_POLICY_LABEL="$label"
export RENDER_OUTPUT_DIR="$RUN_ROOT/gifs/$label"
export RENDER_EPISODES="$RENDER_EPISODES"
export RENDER_FRAME_STRIDE="$RENDER_FRAME_STRIDE"
echo "[\$(date --iso-8601=seconds)] render_sft label=$label dependency=$dep exp_name=$exp_name"
srun --account="$ACCOUNT" --environment="$ENVIRONMENT" --ntasks=1 \
  "$VENV_PATH/bin/python" "$RENDER_SCRIPT" \
    pi05_libero_online_filtered_sft \
    --exp_name "$exp_name" \
    --checkpoint_base_dir "$CHECKPOINT_BASE_DIR" \
    --resume \
    --no-wandb_enabled \
    --seed 100 \
    --batch_size 256 \
    --fsdp_devices 4 \
    --num_train_steps 4001 \
    --save_interval 1000 \
    --log_interval 50 \
    --rl.online_ratio 1.0 \
    --rl.buffer_capacity 1 \
    --collect.tasks "$TASK_ID" \
    --collect.eval_tasks "$TASK_ID" \
    --collect.eval_env_num 1 \
    --collect.num_eval_rollouts "$RENDER_EPISODES"
SBATCH
  chmod +x "$script_path"
  local job_id
  job_id=$(sbatch --parsable \
    $(sbatch_dependency_args "$dep") \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --time="$SFT_RENDER_TIME_LIMIT" \
    --ntasks=1 \
    --cpus-per-task=12 \
    -G 4 \
    --job-name="gif0_${label}" \
    --output="$stdout_path" \
    --error="$stderr_path" \
    "$script_path")
  printf '{"kind":"render","policy_kind":"sft","label":"%s","job_id":"%s","dependency":"%s","exp_name":"%s","stdout":"%s","stderr":"%s","output_dir":"%s"}\n' \
    "$label" "$job_id" "$dep" "$exp_name" "${stdout_path//%j/$job_id}" "${stderr_path//%j/$job_id}" "$RUN_ROOT/gifs/$label" >> "$RUN_ROOT/render_jobs.jsonl"
  echo "render_${label} job_id=$job_id dependency=$dep"
}

submit_render_pretrained() {
  local label=$1
  local dep=$2
  local exp_name=$3
  local script_path="$SBATCH_DIR/render_${label}.sbatch.sh"
  local stdout_path="$LOG_DIR/render_${label}-%j.out"
  local stderr_path="$LOG_DIR/render_${label}-%j.err"
  cat > "$script_path" <<SBATCH
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO_ROOT"
$(common_env_block)
export RENDER_POLICY_KIND=pretrained
export RENDER_POLICY_LABEL="$label"
export RENDER_OUTPUT_DIR="$RUN_ROOT/gifs/$label"
export RENDER_EPISODES="$RENDER_EPISODES"
export RENDER_FRAME_STRIDE="$RENDER_FRAME_STRIDE"
echo "[\$(date --iso-8601=seconds)] render_pretrained label=$label dependency=$dep"
srun --account="$ACCOUNT" --environment="$ENVIRONMENT" --ntasks=1 \
  "$VENV_PATH/bin/python" "$RENDER_SCRIPT" \
    pi05_libero_online_filtered_sft \
    --exp_name "$exp_name" \
    --checkpoint_base_dir "$CHECKPOINT_BASE_DIR" \
    --overwrite \
    --no-wandb_enabled \
    --seed 100 \
    --batch_size 1 \
    --fsdp_devices 1 \
    --num_train_steps 1 \
    --save_interval 1000 \
    --log_interval 50 \
    --rl.online_ratio 1.0 \
    --rl.buffer_capacity 1 \
    --collect.tasks "$TASK_ID" \
    --collect.eval_tasks "$TASK_ID" \
    --collect.eval_env_num 1 \
    --collect.num_eval_rollouts "$RENDER_EPISODES"
SBATCH
  chmod +x "$script_path"
  local job_id
  job_id=$(sbatch --parsable \
    $(sbatch_dependency_args "$dep") \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --time="$PRETRAINED_RENDER_TIME_LIMIT" \
    --ntasks=1 \
    --cpus-per-task=8 \
    -G 1 \
    --job-name="gif0_pretrained" \
    --output="$stdout_path" \
    --error="$stderr_path" \
    "$script_path")
  printf '{"kind":"render","policy_kind":"pretrained","label":"%s","job_id":"%s","dependency":"%s","exp_name":"%s","stdout":"%s","stderr":"%s","output_dir":"%s"}\n' \
    "$label" "$job_id" "$dep" "$exp_name" "${stdout_path//%j/$job_id}" "${stderr_path//%j/$job_id}" "$RUN_ROOT/gifs/$label" >> "$RUN_ROOT/render_jobs.jsonl"
  echo "render_${label} job_id=$job_id dependency=$dep"
}

: > "$RUN_ROOT/render_jobs.jsonl"
while IFS=$'\t' read -r kind label job_id exp_name; do
  if [[ "$kind" == "sft" ]]; then
    submit_render_sft "$label" "$job_id" "$exp_name"
  elif [[ "$kind" == "pretrained" ]]; then
    submit_render_pretrained "$label" "$job_id" "$exp_name"
  fi
done < <(python3 - "$RUN_ROOT/submitted_jobs.jsonl" <<'PY'
import json
import sys
from pathlib import Path
for line in Path(sys.argv[1]).read_text().splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    print(row["kind"], row["label"], row["job_id"], row.get("exp_name", ""), sep="\t")
PY
)

echo "render_jobs=$RUN_ROOT/render_jobs.jsonl"
cat "$RUN_ROOT/render_jobs.jsonl"
