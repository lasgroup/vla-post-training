#!/usr/bin/env bash
set -euo pipefail

ACCOUNT=${ACCOUNT:-a0220}
PARTITION=${PARTITION:-normal}
ENVIRONMENT=${ENVIRONMENT:-vla-post-training-marco-reward}
REPO_ROOT=${REPO_ROOT:-/users/dsimoes/worktrees/vla-post-training/marco/reward}
RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}
RUN_ROOT=${RUN_ROOT:-/capstor/scratch/cscs/dsimoes/vla-post-training/preloaded_sft_task0_successlike_${RUN_TAG}}
LOG_DIR=${RUN_ROOT}/logs
SBATCH_DIR=${LOG_DIR}/sbatch
CHECKPOINT_BASE_DIR=${RUN_ROOT}/checkpoints
CACHE_ROOT=/capstor/scratch/cscs/dsimoes
VENV_PATH=${VENV_PATH:-/.venv}

TASK_ID=${TASK_ID:-libero_90_0}
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-4001}
BATCH_SIZE=${BATCH_SIZE:-256}
GPUS=${GPUS:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-16}
TRAIN_TIME_LIMIT=${TRAIN_TIME_LIMIT:-06:00:00}
PRETRAINED_TIME_LIMIT=${PRETRAINED_TIME_LIMIT:-02:00:00}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
LOG_INTERVAL=${LOG_INTERVAL:-50}
BUFFER_CAPACITY=${BUFFER_CAPACITY:-200000}
EVAL_ROLLOUTS=${EVAL_ROLLOUTS:-100}
EVAL_ENVS=${EVAL_ENVS:-1}
EVAL_INTERVAL=${EVAL_INTERVAL:-500}
PERIODIC_EVAL=${PERIODIC_EVAL:-0}

DATASET_BASE=/capstor/scratch/cscs/dsimoes/vlm-rm/filtered_sft_exports/task0_validation150_reward_std_20260701/success_like_datasets
RANDOM_EPISODES=${RANDOM_EPISODES:-${DATASET_BASE}/random_val150_full/episodes}
REWARD_EPISODES=${REWARD_EPISODES:-${DATASET_BASE}/reward_top150_pred_success/episodes}
REWARD_STD_EPISODES=${REWARD_STD_EPISODES:-${DATASET_BASE}/reward_std_top150_pred_success/episodes}
RANDOM_MANIFEST=${RANDOM_MANIFEST:-${DATASET_BASE}/random_val150_full/selected_manifest.jsonl}
REWARD_MANIFEST=${REWARD_MANIFEST:-${DATASET_BASE}/reward_top150_pred_success/selected_manifest.jsonl}
REWARD_STD_MANIFEST=${REWARD_STD_MANIFEST:-${DATASET_BASE}/reward_std_top150_pred_success/selected_manifest.jsonl}

mkdir -p "$LOG_DIR" "$SBATCH_DIR" "$CHECKPOINT_BASE_DIR" "$RUN_ROOT/scripts"
cp "$REPO_ROOT/scripts/filtered_sft_agent/eval_pretrained_direct.py" "$RUN_ROOT/scripts/eval_pretrained_direct.py"
cp "$REPO_ROOT/scripts/filtered_sft_agent/render_policy_gifs.py" "$RUN_ROOT/scripts/render_policy_gifs.py"

cd "$REPO_ROOT"

echo "run_root=$RUN_ROOT"
echo "repo_root=$REPO_ROOT"
echo "repo_head=$(git rev-parse HEAD)"
echo "task_id=$TASK_ID"

python3 - "$RUN_ROOT/dataset_summary.json" "$RUN_ROOT/dataset_summary.csv" <<'PY'
import csv
import json
import sys
from pathlib import Path

out_json = Path(sys.argv[1])
out_csv = Path(sys.argv[2])
base = Path('/capstor/scratch/cscs/dsimoes/vlm-rm/filtered_sft_exports/task0_validation150_reward_std_20260701/success_like_datasets')
variants = [
    ('random_val150_full', 'random full 150', base/'random_val150_full'),
    ('reward_top150_pred_success', 'reward top150 predicted-success subset', base/'reward_top150_pred_success'),
    ('reward_std_top150_pred_success', 'reward+std top150 predicted-success subset', base/'reward_std_top150_pred_success'),
]
rows = []
for label, description, root in variants:
    ep_dir = root / 'episodes'
    manifest = root / 'selected_manifest.jsonl'
    data = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    labels = [int(float(r['label'])) for r in data]
    pairs = []
    for r in data:
        pr = r.get('prediction_row') or {}
        if 'pred' in pr:
            pairs.append((int(float(pr['pred'])), int(float(r['label']))))
    tp = sum(1 for p,y in pairs if p == 1 and y == 1) if pairs else None
    fp = sum(1 for p,y in pairs if p == 1 and y == 0) if pairs else None
    tn = sum(1 for p,y in pairs if p == 0 and y == 0) if pairs else None
    fn = sum(1 for p,y in pairs if p == 0 and y == 1) if pairs else None
    n = len(list(ep_dir.glob('episode_*.pkl')))
    successes = sum(labels)
    failures = len(labels) - successes
    rows.append({
        'dataset': label,
        'description': description,
        'episodes': n,
        'manifest_rows': len(data),
        'actual_successes': successes,
        'actual_failures': failures,
        'dataset_success_rate': successes / len(labels) if labels else 0.0,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'episodes_dir': str(ep_dir),
        'selected_manifest': str(manifest),
    })
out_json.write_text(json.dumps({'datasets': rows}, indent=2, sort_keys=True) + '\n')
with out_csv.open('w', newline='', encoding='utf-8') as f:
    fieldnames = ['dataset','description','episodes','manifest_rows','actual_successes','actual_failures','dataset_success_rate','tp','fp','tn','fn','episodes_dir','selected_manifest']
    w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
print(out_json)
print(out_csv)
PY

validate_episode_dir() {
  local label=$1
  local dir=$2
  if [[ ! -d "$dir" ]]; then
    echo "[$label] missing episode directory: $dir" >&2
    exit 1
  fi
  local count
  count=$(find "$dir" -maxdepth 1 -type f -name 'episode_*.pkl' | wc -l)
  echo "[$label] episode_count=$count dir=$dir"
  if [[ "$count" -le 0 ]]; then
    echo "[$label] no episode_*.pkl files found" >&2
    exit 1
  fi
}

validate_episode_dir random_val150_full "$RANDOM_EPISODES"
validate_episode_dir reward_top150_pred_success "$REWARD_EPISODES"
validate_episode_dir reward_std_top150_pred_success "$REWARD_STD_EPISODES"

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
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.45}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-0}
export MUJOCO_EGL_NUM_DEVICES=${MUJOCO_EGL_NUM_DEVICES:-1}
mkdir -p "\$OPENPI_DATA_HOME" "\$XDG_CACHE_HOME" "\$UV_CACHE_DIR" "\$HF_HOME" "\$HF_DATASETS_CACHE" "\$WANDB_DIR" "\$MPLCONFIGDIR" "\$JAX_COMPILATION_CACHE_DIR"
OPENPI_PARAMS_PARENT="$CACHE_ROOT/openpi_assets/openpi-assets/checkpoints/pi05_libero/params"
if [[ -d "\$OPENPI_PARAMS_PARENT/params" && ! -e "\$OPENPI_PARAMS_PARENT/_METADATA" ]]; then
  (cd "\$OPENPI_PARAMS_PARENT" && for x in _METADATA _sharding array_metadatas d manifest.ocdbt ocdbt.process_0; do [[ -e "params/\$x" && ! -e "\$x" ]] && ln -s "params/\$x" "\$x" || true; done)
fi
ENV
}

submit_sft() {
  local label=$1
  local episodes_dir=$2
  local exp_name="task0_${label}_preloaded_sft_${RUN_TAG}"
  local script_path="$SBATCH_DIR/${exp_name}.sbatch.sh"
  local stdout_path="$LOG_DIR/${exp_name}-%j.out"
  local stderr_path="$LOG_DIR/${exp_name}-%j.err"
  cat > "$script_path" <<SBATCH
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO_ROOT"
$(common_env_block)
export PRELOADED_SFT_FINAL_EVAL=1
export PRELOADED_SFT_PERIODIC_EVAL=$PERIODIC_EVAL

echo "[\$(date --iso-8601=seconds)] sft_label=$label task=$TASK_ID episodes=$episodes_dir"
echo "[\$(date --iso-8601=seconds)] repo=\$(pwd -P) head=\$(git rev-parse HEAD)"
echo "[\$(date --iso-8601=seconds)] train_steps=$NUM_TRAIN_STEPS eval_rollouts=$EVAL_ROLLOUTS eval_envs=$EVAL_ENVS periodic_eval=$PERIODIC_EVAL"

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
    --collect.tasks "$TASK_ID" \
    --collect.eval_tasks "$TASK_ID" \
    --collect.eval_interval "$EVAL_INTERVAL" \
    --collect.eval_env_num "$EVAL_ENVS" \
    --collect.num_eval_rollouts "$EVAL_ROLLOUTS"
SBATCH
  chmod +x "$script_path"
  local job_id
  job_id=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --time="$TRAIN_TIME_LIMIT" \
    --ntasks=1 \
    --cpus-per-task="$CPUS_PER_TASK" \
    -G "$GPUS" \
    --job-name="fsft0_${label}" \
    --output="$stdout_path" \
    --error="$stderr_path" \
    "$script_path")
  printf '{"kind":"sft","label":"%s","job_id":"%s","exp_name":"%s","episodes_dir":"%s","stdout":"%s","stderr":"%s","checkpoint_dir":"%s"}\n' \
    "$label" "$job_id" "$exp_name" "$episodes_dir" "${stdout_path//%j/$job_id}" "${stderr_path//%j/$job_id}" "${CHECKPOINT_BASE_DIR}/pi05_libero_online_filtered_sft/${exp_name}" >> "$RUN_ROOT/submitted_jobs.jsonl"
  echo "$label job_id=$job_id"
}

submit_pretrained() {
  local label=pretrained_reference
  local exp_name="task0_pretrained_reference_${RUN_TAG}"
  local script_path="$SBATCH_DIR/${exp_name}.sbatch.sh"
  local stdout_path="$LOG_DIR/${exp_name}-%j.out"
  local stderr_path="$LOG_DIR/${exp_name}-%j.err"
  cat > "$script_path" <<SBATCH
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO_ROOT"
$(common_env_block)
export PRETRAINED_EVAL_METRICS_PATH="$RUN_ROOT/pretrained_reference_metrics.json"

echo "[\$(date --iso-8601=seconds)] pretrained_reference task=$TASK_ID rollouts=$EVAL_ROLLOUTS"
echo "[\$(date --iso-8601=seconds)] repo=\$(pwd -P) head=\$(git rev-parse HEAD)"
srun --account="$ACCOUNT" --environment="$ENVIRONMENT" --ntasks=1 \
  "$VENV_PATH/bin/python" scripts/filtered_sft_agent/eval_pretrained_direct.py \
    pi05_libero_online_filtered_sft \
    --exp_name "$exp_name" \
    --checkpoint_base_dir "$CHECKPOINT_BASE_DIR" \
    --overwrite \
    --no-wandb_enabled \
    --seed 0 \
    --batch_size 1 \
    --fsdp_devices 1 \
    --num_train_steps 1 \
    --save_interval 1000 \
    --log_interval 50 \
    --rl.online_ratio 1.0 \
    --rl.buffer_capacity 1 \
    --collect.tasks "$TASK_ID" \
    --collect.eval_tasks "$TASK_ID" \
    --collect.eval_env_num "$EVAL_ENVS" \
    --collect.num_eval_rollouts "$EVAL_ROLLOUTS"
SBATCH
  chmod +x "$script_path"
  local job_id
  job_id=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --time="$PRETRAINED_TIME_LIMIT" \
    --ntasks=1 \
    --cpus-per-task=8 \
    -G 1 \
    --job-name="eval0_pretrained" \
    --output="$stdout_path" \
    --error="$stderr_path" \
    "$script_path")
  printf '{"kind":"pretrained","label":"%s","job_id":"%s","exp_name":"%s","stdout":"%s","stderr":"%s","metrics_path":"%s"}\n' \
    "$label" "$job_id" "$exp_name" "${stdout_path//%j/$job_id}" "${stderr_path//%j/$job_id}" "$RUN_ROOT/pretrained_reference_metrics.json" >> "$RUN_ROOT/submitted_jobs.jsonl"
  echo "$label job_id=$job_id"
}

: > "$RUN_ROOT/submitted_jobs.jsonl"
submit_sft random_val150_full "$RANDOM_EPISODES"
submit_sft reward_top150_pred_success "$REWARD_EPISODES"
submit_sft reward_std_top150_pred_success "$REWARD_STD_EPISODES"
submit_pretrained

cat > "$RUN_ROOT/submission_manifest.json" <<EOF
{
  "run_tag": "$RUN_TAG",
  "run_root": "$RUN_ROOT",
  "repo_root": "$REPO_ROOT",
  "repo_head": "$(git rev-parse HEAD)",
  "task_id": "$TASK_ID",
  "num_train_steps": $NUM_TRAIN_STEPS,
  "batch_size": $BATCH_SIZE,
  "gpus": $GPUS,
  "eval_rollouts": $EVAL_ROLLOUTS,
  "eval_envs": $EVAL_ENVS,
  "periodic_eval": $PERIODIC_EVAL,
  "eval_interval": $EVAL_INTERVAL,
  "account": "$ACCOUNT",
  "partition": "$PARTITION",
  "environment": "$ENVIRONMENT",
  "train_time_limit": "$TRAIN_TIME_LIMIT",
  "pretrained_time_limit": "$PRETRAINED_TIME_LIMIT"
}
EOF

echo "submission_manifest=$RUN_ROOT/submission_manifest.json"
echo "submitted_jobs=$RUN_ROOT/submitted_jobs.jsonl"
