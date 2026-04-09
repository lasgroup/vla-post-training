#!/usr/bin/env bash

set -euo pipefail

checkpoint_dir="${CHECKPOINT_DIR:-gs://openpi-assets/checkpoints/pi05_droid_jointpos}"
env_range="${ENV_RANGE:-0-29}"
episodes_per_env="${EPISODES_PER_ENV:-10}"
max_steps="${MAX_STEPS:-450}"
num_shards="${NUM_SHARDS:-4}"
seed="${SEED:-0}"
output_root="${OUTPUT_ROOT:-outputs/molmo_env_sweep}"
run_name="${RUN_NAME:-molmo_first30_sr_$(date -u +%Y%m%d_%H%M%S)}"
output_dir="${OUTPUT_DIR:-${output_root%/}/${run_name}}"
output_prefix="${OUTPUT_PREFIX:-results}"

mkdir -p "$output_dir"
mkdir -p logs

echo "[runner] checkpoint_dir=$checkpoint_dir"
echo "[runner] env_range=$env_range"
echo "[runner] episodes_per_env=$episodes_per_env"
echo "[runner] num_shards=$num_shards"
echo "[runner] output_dir=$output_dir"

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup INT TERM

for ((shard=0; shard<num_shards; shard++)); do
  gpu_index=$((shard % 4))
  (
    export PYTHONUNBUFFERED=1
    export CUDA_VISIBLE_DEVICES="$gpu_index"
    export MUJOCO_EGL_DEVICE_ID=0
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"

    uv run python scripts/eval_molmo_env_sweep.py \
      --checkpoint-dir "$checkpoint_dir" \
      --env-range "$env_range" \
      --episodes-per-env "$episodes_per_env" \
      --max-steps "$max_steps" \
      --seed "$seed" \
      --num-shards "$num_shards" \
      --shard-index "$shard" \
      --render-device 0 \
      --output-dir "$output_dir" \
      --output-prefix "$output_prefix"
  ) 2>&1 | sed "s/^/[worker $shard gpu $gpu_index] /" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

python3 - <<'PY' "$output_dir" "$output_prefix"
import csv
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
output_prefix = sys.argv[2]
worker_jsons = sorted(out_dir.glob(f'{output_prefix}_shard*.json'))
if not worker_jsons:
    raise SystemExit(f'No worker JSON files found in {out_dir}')

rows = []
for path in worker_jsons:
    rows.extend(json.loads(path.read_text()))

rows_by_env = sorted(rows, key=lambda row: row['env_id'])
rows_ranked = sorted(
    rows,
    key=lambda row: (-row['sr'], -row['successes'], row['mean_steps'], row['env_id']),
)
ranked_rows = [{'rank': i, **row} for i, row in enumerate(rows_ranked, start=1)]

(out_dir / 'combined.by_env.json').write_text(json.dumps(rows_by_env, indent=2) + '\n')
(out_dir / 'combined.ranked.json').write_text(json.dumps(ranked_rows, indent=2) + '\n')

for filename, payload in [
    ('combined.by_env.csv', rows_by_env),
    ('combined.ranked.csv', ranked_rows),
]:
    path = out_dir / filename
    fieldnames = sorted({key for row in payload for key in row.keys()}) if payload else ['env_id', 'env_name', 'sr']
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(payload)

print('[runner] wrote:')
print(f'[runner]   {out_dir / "combined.by_env.json"}')
print(f'[runner]   {out_dir / "combined.by_env.csv"}')
print(f'[runner]   {out_dir / "combined.ranked.json"}')
print(f'[runner]   {out_dir / "combined.ranked.csv"}')
print('[runner] top 10 by SR:')
for row in ranked_rows[:10]:
    print(
        '[runner] '
        f"#{row['rank']:02d} {row['env_name']} sr={row['sr']:.3f} "
        f"({row['successes']}/{row['episodes']}) | house={row['house_index']} | "
        f"{row['task_description']}"
    )
PY

echo "[runner] complete"
echo "[runner] result_dir=$output_dir"
