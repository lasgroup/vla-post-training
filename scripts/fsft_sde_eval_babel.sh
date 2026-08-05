#!/bin/bash
#SBATCH --partition=maxlab
#SBATCH --qos=maxlab_qos
#SBATCH --nodelist=babel-m9-16
#SBATCH --job-name=fsft_sde_eval
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
# Scratch root for this eval. Each noise level gets its own subdir under it --
# see the per-level dir note below.
CKPT_OUT_ROOT=$STORE_ROOT/checkpoints/fsft_multitask_sde_eval

# Noise levels to evaluate, one eval.py run each. Log-spaced rather than linear
# because the two references sit an order of magnitude apart:
#   0.0  -- ODE baseline (Euler on the probability flow).
#   0.02 -- matches OGPO's per-step noise. Upstream runs tapered
#           sigma = 0.01*sqrt(1-t) (scripts/ogpo_fpo/*.sh) or constant 0.005
#           (configs/algos/ogpo.yaml), applied as the step std directly. Pi05
#           instead uses sigma_t*sqrt(|dt|) with sigma_t = level*sqrt(t/(1-t)),
#           so equal per-step std at mid-chain lands near 0.02. The schedules
#           have different shapes, so this matches magnitude, not the curve.
#   0.1  -- intermediate.
#   0.3  -- the default in this repo's OGPOSFTLearnerConfig / FlowGRPO configs,
#           i.e. what our own training runs at. ~13x OGPO at mid-chain.
NOISE_LEVELS=("$@")
if [ ${#NOISE_LEVELS[@]} -eq 0 ]; then
  NOISE_LEVELS=(0.0 0.02 0.1 0.3)
fi

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

echo "[fsft-sde-eval] node=$(hostname) job=${SLURM_JOB_ID:-none} exp=${EXP_NAME}"
echo "[fsft-sde-eval] noise levels: ${NOISE_LEVELS[*]}"

# Same checkpoint, same tasks, same episode budget as fsft_cfg_eval_babel.sh --
# the only thing that varies is the sampler. rl.sde_noise_level = 0.0 integrates
# the probability-flow ODE (Euler, x_{t+dt} = x_t + dt*v_t, the default path);
# > 0 switches to the distribution-preserving SDE whose per-step kernel is
# Pi0._get_sde_dist, sigma_t = noise_level * sqrt(t/(1-t)). Both run num_steps=10
# and, at cfg_scale 1.0, one velocity evaluation per step -- so NFE is matched
# across levels and the comparison is compute-neutral.
#
# CFG is pinned off (cfg_scale 1.0) so this isolates the sampler. The two are
# composable in the model (guidance is applied to v_t before the SDE kernel is
# built), but mixing them here would confound the two axes.
#
# Sequential runs instead of an in-process sweep: each `uv run` is a fresh
# process seeded from --seed, so the agent PRNG (jax.random.key(seed)), the
# policy PRNG (jax.random.key(0)) and the env seed (seed + offset + 0) are all
# identical across levels. Episodes are therefore matched one-to-one without any
# explicit PRNG rewind -- pair them (McNemar) rather than comparing marginals.
# Note the policy PRNG matters here in a way it does not for the ODE: under the
# SDE it draws the per-step transition noise, whereas on the deterministic path
# that key is inert and only the externally drawn initial noise is used.
#
# Per-level CKPT_OUT and exp_name are mandatory, not cosmetic: without --resume
# the learner passes overwrite=True to initialize_checkpoint_dir and wipes the
# dir it points at, and every run writes the same eval_metrics_step0.json
# filename. Sharing one dir would have each level delete its predecessor.
#
# Weights come in through the weight loader, NOT --resume -- see the long note
# in fsft_cfg_eval_babel.sh for why resuming is broken with this orbax version.
#
# `|| echo` deliberately overrides `set -e` for the eval calls so one bad level
# does not cost the remaining ones. Four env create/close cycles in one job is
# four chances to hit the MuJoCo/EGL teardown hang.

for SDE in "${NOISE_LEVELS[@]}"; do
  RUN_NAME="${EXP_NAME}_sde${SDE}"
  RUN_OUT="$CKPT_OUT_ROOT/sde${SDE}"
  mkdir -p "$RUN_OUT"

  echo "[fsft-sde-eval] === noise_level=$SDE -> $RUN_OUT ==="

  uv run scripts/eval.py \
    pi05_libero_online_filtered_sft \
    --exp_name "$RUN_NAME" \
    --checkpoint_base_dir "$RUN_OUT" \
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
    --rl.cfg_scale 1.0 \
    --rl.sde_noise_level "$SDE" \
    || echo "[fsft-sde-eval] FAILED at noise_level=$SDE"
done

echo "[fsft-sde-eval] done. Metrics:"
echo "  $CKPT_OUT_ROOT/sde<level>/pi05_libero_online_filtered_sft/${EXP_NAME}_sde<level>/eval_metrics_step0.json"
