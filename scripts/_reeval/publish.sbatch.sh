#!/bin/bash
set -euo pipefail
cd /users/ralfroemer/vla-post-training
ulimit -c 0
srun --account=ab037 --environment=vla-post-training uv run scripts/_reeval/publish_reevals_to_wandb.py "$@"
