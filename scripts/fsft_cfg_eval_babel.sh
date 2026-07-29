#!/bin/bash
#SBATCH --partition=maxlab
#SBATCH --qos=maxlab_qos
#SBATCH --nodelist=babel-m9-16
#SBATCH --job-name=fsft_cfg_eval
#SBATCH --gres=gpu:2
#SBATCH --constraint=VRAM_96GB
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=/home/mananaga/logs/%j/.out
#SBATCH --error=/home/mananaga/logs/%j/.out

set -euo pipefail

PROJECT_DIR=/home/mananaga/VLA/manan_babel/vla-post-training
STORE_ROOT=/data/group_data/maxlab/common_datasets/mananaga/vla-post-training
EXP_NAME=pi05_libero_online_filtered_sft_multitask4_cfg_seed0
CKPT_BASE_DIR=$STORE_ROOT/checkpoints/fsft_multitask_cfg
# Scratch dir for this eval. Kept separate from CKPT_BASE_DIR because the run
# dir under it gets wiped on startup (see the --resume note below).
CKPT_OUT=$STORE_ROOT/checkpoints/fsft_multitask_cfg_eval

cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/openpi/packages/openpi-client/src:$PROJECT_DIR/openpi/src:$PROJECT_DIR/openpi/packages/openpi-client:$PROJECT_DIR/molmospaces"

export OPENPI_DATA_HOME="$STORE_ROOT/cache/openpi"
export HF_HOME="$STORE_ROOT/cache/huggingface"
export LIBERO_CONFIG_PATH="$HOME/.libero"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

export NCCL_CUMEM_ENABLE=0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES%%,*}"

mkdir -p "$CKPT_OUT"

echo "[fsft-cfg-eval] node=$(hostname) job=${SLURM_JOB_ID:-none} exp=${EXP_NAME}"

# Re-run the step-5000 guidance sweep of fsft_cfg_libero_babel.sh as a *paired*
# eval. The in-training sweep was unpaired: evaluate_policy did not seed the env,
# so each scale saw different initial states, and at 32 rollouts/task the
# per-scale spread (0.398-0.445 overall) sat inside one standard error.
#
# Two things changed in src/training/collect.py:
#   - evaluate_policy seeds the eval envs (config.seed + offset + step), so every
#     scale replays the same initial states.
#   - evaluate_policy_sweep rewinds the agent PRNG before each scale, so the
#     action noise is identical too.
# The scales now differ only in guidance, and episodes are matched one-to-one
# across scales -- pair them (McNemar) rather than comparing marginal rates.
#
# num_eval_rollouts 100 (vs 32 in training): 400 episodes/scale, per-task SE
# ~0.049 and overall SE ~0.025 even before pairing. Guided scales cost ~2x per
# denoising step; the 512-episode training sweep took ~27 min, so 1600 episodes
# here should land near ~1.5h. Raise this if you want to resolve <5 points.
#
# Weights come in through the weight loader, NOT --resume. Resuming is broken
# with this orbax version: restore_state hands it an abstract tree whose leaves
# are nnx.VariableState (`.value` attribute) while the checkpoint stores each
# param as a {'value': array} dict, so the treedefs don't match, orbax falls
# back to metadata-derived restore args, and deserialization dies on
# "sharding ... Got None". CheckpointWeightLoader goes through
# model.restore_params, which targets the checkpoint's own metadata tree and
# strips the trailing 'value' key -- the path the base pi05 weights already use.
#
# Consequences of not resuming, all fine for eval:
#   - only the inference params load (~13G, not the 42G train_state), and the
#     saved `params` item IS the EMA (checkpoints._split_params), which is what
#     _sample_action prefers anyway;
#   - no replay-shard restore, so rl.buffer_capacity no longer matters;
#   - training_steps is 0, so metrics land in eval_metrics_step0.json and the
#     env seed is config.seed + offset + 0 -- identical across scales, which is
#     all the pairing needs.
#
# CKPT_OUT is a scratch dir and MUST NOT be the training run's directory:
# without --resume the learner passes overwrite=not config.resume=True to
# initialize_checkpoint_dir, which wipes whatever it points at.
#
# W&B stays off so this does not overwrite the summary of the original run
# (3a4pmfrg) -- the same eval/cfg<scale>/... keys would be replaced in place.
# Metrics land in $CKPT_OUT/.../$EXP_NAME/eval_metrics_step0.json;
# plot them with `python scripts/plot_cfg_eval.py --json <that file>`.

exec uv run scripts/eval.py \
  pi05_libero_online_filtered_sft \
  --exp_name "$EXP_NAME" \
  --checkpoint_base_dir "$CKPT_OUT" \
  --weight-loader.params-path "$CKPT_BASE_DIR/pi05_libero_online_filtered_sft/$EXP_NAME/5000/params" \
  --no-wandb_enabled \
  --seed 0 \
  --fsdp_devices 2 \
  --collect.tasks libero_90_79 libero_90_31 libero_90_82 libero_90_38 \
  --collect.eval_tasks libero_90_79 libero_90_31 libero_90_82 libero_90_38 \
  --collect.num_eval_rollouts 100 \
  --collect.eval_env_num 8 \
  --rl.online_ratio 1.0 \
  --rl.discount 0.995 \
  --rl.cfg_dropout_prob 0.1 \
  --rl.cfg_scale 1.0 1.5 2.0 3.0
