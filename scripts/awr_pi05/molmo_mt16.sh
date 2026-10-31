#!/bin/bash
#SBATCH --job-name=awr_pi05_molmo_mt16
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
# Add --partition / --account / --constraint=VRAM_96GB for your cluster (the pool below is sized for 96 GB GPUs).
# Paper AWR (Peng et al. 2019) on pi0.5, MolmoSpaces FrankaPickDroidMiniBench (pi05_droid_jointpos, agent.pi05_robot=droid), 16 tasks, four each of box (267 / 366 / 138 / 143), spatula (176 / 373 / 93 / 394), cup (48 / 185 / 24 / 62) and tissue paper (4 / 260 / 221 / 348); a subset of the mt64 and mt144 task sets.
# Hyperparameters = the most recent molmo mt144 OGPO runs (seeds 1 / 2, 4 x 96 GB GPUs): `git show bf60f38:paper_expts_final/molmo_pi05_mt144_ogpo_s1.sh`
# (history of pranav/vla-ogpo). Only what scales with the task set differs: env_name, parallel_envs, inference_batch, buffer_size, cpus.
# One seed (0). Snapshot resume (training.resume_snapshot_dir) is not wired in: a --resume_snapshot_dir= is ignored.
# Host RAM (mt16): the replay images are ~0.6 of the ~0.62 MB per transition and live in RAM unless buffer_storage_root is set. A full
# 1.2M buffer is ~744 GB (the earlier mt16 recipe noted ~720 GB, on a 2 TB p5.48xlarge). A 10-round run fills less: ~360 GB at ~180 steps
# per episode (an estimate, not measured) plus ~2.5 GiB per worker (32 workers, measured on 3 tasks). The --mem=400G above is the earlier
# mt16 header's and may be too small: raise it, or pass --buffer_storage_root=<node-local dir with >= ~0.75 TB free> (images memmapped
# there, deleted on exit, as the mt144 runs did).
# Twin of scripts/ogpo_pi05/molmo_mt16.sh: same tasks, rounds, buffer, batch sizes, GPUs, pi05 optimiser and ODE eval; only the learning algorithm
# differs. Each round: collect with the ONLINE actor (one unclipped ODE sample, no best-of-N) -> R^lambda for every buffered transition
# (TD(lambda), bootstrapped with last round's V) -> awr_value_steps V steps on R^lambda (2 bronet_vla heads) -> updates_per_round policy
# steps, loss mean[min(exp((R^lambda - V) / beta), omega_max) * CFM] on buffer windows. No Q critic, no PPO / SDE chains, no BC anchor.
# Step axis = policy steps; eval is ODE only. NOT YET RUN on MolmoSpaces: it is the OGPO recipe plus the AWR delta that the LIBERO AWR
# recipe (scripts/ogpo/libero_pi05_mt4_awr.sh on pranav/pi05-awr) used.
# Setup, once: `bash scripts/ogpo/setup_molmo_venv.sh --molmospaces=<lasgroup/molmospaces checkout @ 2a828d5> --openpi_client=<openpi/packages/openpi-client>`
# builds .venv-molmo. Then, with OPENPI_CACHE, MLSPACES_ASSETS_DIR, MLSPACES_CACHE_DIR and NLTK_DATA set to directories to fill, run
# scripts/ogpo/download_molmo_assets.sbatch (pi05_droid_jointpos, the PaliGemma tokenizer, MolmoSpaces assets; its TASKS defaults to the
# mt16 task set, which contains this recipe's tasks).
# Required paths, from the environment or --VAR=...: SAVE_DIR, MLSPACES_ASSETS_DIR, MLSPACES_CACHE_DIR, NLTK_DATA, and OPENPI_CACHE
# unless --pi05_checkpoint= and --pi05_tokenizer= are both given (a launcher on a remote node can pass them). The script exits if one is
# empty. Overrides: --var=value (a shell variable below; an unknown name is silently ignored).
# Logging: log=true sends metrics to Weights & Biases (project OGPO_molmo): run `wandb login` first, or `export WANDB_MODE=offline` to keep
# the run files local. --log=false turns W&B off but also stops the CSV logs and checkpoint_latest.pkl from being written.
#   bash scripts/awr_pi05/molmo_mt16.sh                       # from the checkout root
#   sbatch scripts/awr_pi05/molmo_mt16.sh
# Molmo env notes (maintainer, 2026-10-14/15): discount 0.995, action chunk 15 (5 executed per replan), no action clip (model-space
# joint deltas, +-1 = DROID's 1st / 99th percentile), terminate on success (reward -1 per step, no +5 bonus), episode length 450,
# vector envs required (the sim must stay out of the trainer process). A cold worker needs ~113 s to import + 80-200 s for its first
# reset; steps take ~0.11 s, so collection is CPU-bound.
mkdir -p logs
set -e

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1         # [ROUND] / eval prints reach the .out log live (block-buffered otherwise)
# NCCL peer-to-peer all-reduce hangs on 8 x L40S nodes without NVLink; route collectives through host memory there. No-op on other GPUs.
if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q L40S; then
    export NCCL_P2P_DISABLE=1; echo "[L40S] NCCL_P2P_DISABLE=1"
fi

# === Config ===
seed=0
device="0,1,2,3"                # data parallel over these (training.gpu_mode=multi); 4 x 96 GB GPUs as the mt144 runs
gpu_mode=multi
zero_actor_state=false          # multi-GPU ZeRO-1: shards the actor optimiser state across GPUs; ~20x slower on 8 x L40S, so off
jax_frac=0.70                   # the pool (0.70 x 96 GB = 67 GB) holds the K=1 update (59.8 GiB measured at 128 rows / GPU); the rest is EGL workers
env_name="molmo_pi05-267,molmo_pi05-366,molmo_pi05-138,molmo_pi05-143,molmo_pi05-176,molmo_pi05-373,molmo_pi05-93,molmo_pi05-394,molmo_pi05-48,molmo_pi05-185,molmo_pi05-24,molmo_pi05-62,molmo_pi05-4,molmo_pi05-260,molmo_pi05-221,molmo_pi05-348"
eval_env_name=""                # default: the training tasks
task_balance=none               # none | bc | bc_ppo | all: stratify the BC anchor / PPO states / all batches by task
run_name=""                     # default: awr_molmo_mt16_pi05_bronet_vla (save dir and wandb name are keyed on it)
wandb_name=""                   # default: run_name
project="OGPO_molmo"
run_group="molmo_mt16_pi05_awr"

# Vector envs (required for molmo_pi05): T tasks x parallel_envs worker processes, one batched inference pass per replan
parallel_envs=2                 # per task -> 32 workers; keep 16 x parallel_envs <= cpus-per-task - 2
inference_batch=32              # rows per encode / policy call, = all 32 slots (16 tasks x 2 workers); rows are sharded over the GPUs (rows / n_gpus each)
worker_egl_devices=auto         # workers' EGL GPUs, round robin over CUDA_VISIBLE_DEVICES; "" = all on the first GPU

# Round schedule (training.collection_mode=rounds)
num_rounds=10                   # collection rounds, as vla-post-training (rounds at 0, 1k, ..., 9k updates)
rollouts_per_round=20           # full episodes PER TASK per round (320 total), policy frozen while collecting
updates_per_round=1000          # AWR policy steps per round -> num_rounds x 1000 = 10k total
buffer_size=1200000             # ~0.62 MB / transition (up to ~744 GB of host RAM when full, see the header); FIFO (a round is 320 eps x <=450 steps)
eval_episodes=10                # PER TASK (x 16 tasks, ODE only under awr)
eval_interval=2000                # in updates (step axis = update count)
log_interval=1000

# OGPO
horizon_length=5                # executed steps per pi05 chunk (vla-post-training molmo replan_steps)
post_success_steps=0            # molmo_pi05 terminates on success (make_env rejects anything else)
discount=0.995                  # vla-post-training molmo rl.discount
tau=0.05
actor_tau=0.005
best_of_n=1                    # AWR collects with the online actor: one unclipped ODE sample, no best-of-N
num_qs=10
q_agg=subsample
pi05_bon_agg=subsample          # BoN head reduction min | subsample | mean (pi05 agent; replaces subsample_bon)
flow_steps=10
ft_flow_steps=10
bc_coeff=0.0                   # AWR has no BC anchor
adv_strategy=conservative       # per-head advantages, zeroed unless all num_qs heads agree on the sign (pg_helper.compute_ogpo_advantages)
clip_epsilon=0.01
entropy_coeff=0.0
use_bc_regularization=false
use_tapered_noise=true
use_constant_noise=false
error_correct_sde_to_ode=true
constant_noise_std=0.1
use_success_buffer=true
use_success_buffer_q=false
# No action clipping (vla-post-training molmo): model-space actions are normalised joint deltas, and +-1 = the
# 1st / 99th percentile of DROID's deltas, so a clip caps per-step joint motion.
act_min=-inf
act_max=inf
clip_intermediate_actions=false

# Batch / group size, global across GPUs (each must divide by the number of GPUs).
#   batch_size: critic/TD batch (PPO states are drawn from it); success_buffer_batch_size: success-buffer critic update;
#   bc_batch_size: BC anchor; group_num_samples: G chains per PPO state; ppo_batch_size: PPO states per update
batch_size=512
ppo_batch_size=32
success_buffer_batch_size=512
bc_batch_size=512
group_num_samples=32
grad_accum_steps=1              # 4 GPUs: 128 rows per GPU micro-batch (fits with pi05_drop_empty_camera)

# Q-variance reduction
q_variance_reduction=false     # no Q critic under awr
q_vr_num_samples=16
q_vr_reduction=median

# Critic: vla-post-training BroNet Q, no two-tier encoder (online LR = ppo_lr, constant)
pi05_critic_arch=bronet_vla
pi05_critic_pooling=all         # required by bronet_vla
pi05_bronet_hidden_dim=1024
pi05_bronet_depth=2
ppo_lr=1e-4                     # critic LR, as vla-post-training
critic_scheduler=constant
critic_weight_decay=1e-6
clip_grad_norm=1000.0
layer_norm=true
value_hidden_dims="[1024,1024,1024]"   # unused by bronet_vla (no two-tier, no MLP heads)

# pi05 actor (vla-post-training optimiser)
pi05_lr=2.5e-5
pi05_clip_grad_norm=1.0
pi05_weight_decay=1e-10
pi05_action_horizon=15          # pi05_droid_jointpos (vla-post-training make_base_molmo_config)
pi05_robot=droid
pi05_drop_empty_camera=true     # KV-only prefix passes skip the masked right-wrist slot (pooled critic features unchanged; 1.4x faster prefix)


# AWR (Peng et al. 2019, Alg. 1): V(s) <- TD(lambda) returns, policy <- exp((R^lambda - V) / beta) weighted flow matching
pi05_actor_objective=awr
use_value_fn=true
v_training_mode=lambda_return   # V(s) <- R^lambda
v_ensemble_size=2               # >= 2 required by the V ensemble; heads averaged (v_agg=mean)
v_agg=mean
awr_beta=0.05                   # paper's temperature
awr_weight_max=20.0             # paper's omega_max, a cap on the weight itself
awr_lambda=0.95                 # paper's TD(lambda)
awr_value_steps=200             # paper: 200 V steps per iteration (= per target refresh)
awr_value_refreshes=1           # K x (recompute R^lambda -> awr_value_steps V steps) per round
awr_normalize_adv=false         # reference: A standardised over the buffer, then exp(A / temp) with temp=1.0

# Paths: no defaults; set them in the environment or pass --VAR=... (checked after the overrides are parsed)
SAVE_DIR="${SAVE_DIR:-}"
OPENPI_CACHE="${OPENPI_CACHE:-}"                  # holds openpi-assets/ and big_vision/; not needed when pi05_checkpoint and pi05_tokenizer are given
pi05_checkpoint=""              # default: $OPENPI_CACHE/openpi-assets/checkpoints/pi05_droid_jointpos
pi05_tokenizer=""               # default: $OPENPI_CACHE/big_vision/paligemma_tokenizer.model
MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-}"    # MolmoSpaces assets / cache / NLTK data: scripts/ogpo/download_molmo_assets.sbatch
MLSPACES_CACHE_DIR="${MLSPACES_CACHE_DIR:-}"
NLTK_DATA="${NLTK_DATA:-}"
buffer_storage_root=""          # node-local dir for the replay buffer's images / next_images memmaps (a job-specific subdir, deleted on exit); "" = host RAM

log=true
grad_diagnostics=false          # off: the 2 extra actor grads per log window OOMed on 8x80 GB

# === Parse CLI overrides ===
for arg in "$@"; do
  case $arg in
    --*=*)
      key="${arg%%=*}"
      value="${arg#*=}"
      key="${key#--}"
      eval "$key=\"$value\""
      ;;
    *)
      echo "Unknown argument: $arg"
      exit 1
      ;;
  esac
done


need="SAVE_DIR MLSPACES_ASSETS_DIR MLSPACES_CACHE_DIR NLTK_DATA"
[ -n "$pi05_checkpoint" ] && [ -n "$pi05_tokenizer" ] || need="$need OPENPI_CACHE"
for v in $need; do
    [ -n "${!v}" ] || { echo "ERROR: $v is not set (environment variable or --$v=...)"; exit 1; }
done
export MLSPACES_ASSETS_DIR MLSPACES_CACHE_DIR NLTK_DATA
pi05_checkpoint="${pi05_checkpoint:-$OPENPI_CACHE/openpi-assets/checkpoints/pi05_droid_jointpos}"
pi05_tokenizer="${pi05_tokenizer:-$OPENPI_CACHE/big_vision/paligemma_tokenizer.model}"

run_name="${run_name:-awr_molmo_mt16_pi05_bronet_vla}"
wandb_name="${wandb_name:-$run_name}"

rb_dir=""
if [ -n "$buffer_storage_root" ]; then
    rb_dir="$buffer_storage_root/ogpo_rb_${run_name}_s${seed}_${SLURM_JOB_ID:-$$}"
    mkdir -p "$rb_dir"
    trap 'rm -rf "$rb_dir"' EXIT
    trap 'exit 143' TERM          # scancel / time limit: run the EXIT trap (python gets the signal too)
    echo "[buffer] images / next_images memmapped in $rb_dir ($(df -h --output=avail "$buffer_storage_root" | tail -1 | tr -d ' ') free)"
fi

export CUDA_VISIBLE_DEVICES=$device
export XLA_PYTHON_CLIENT_MEM_FRACTION=$jax_frac

# sbatch runs a copy from /var/spool, so BASH_SOURCE does not point at the checkout;
# fall back to walking up from the submit dir.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
if [ ! -f "$PROJECT_ROOT/.venv-molmo/bin/activate" ]; then
    PROJECT_ROOT=""
    d="${SLURM_SUBMIT_DIR:-$PWD}"
    while [ "$d" != "/" ]; do
        if [ -f "$d/.venv-molmo/bin/activate" ] && [ -f "$d/ogpo/main_pi05.py" ]; then PROJECT_ROOT="$d"; break; fi
        d="$(dirname "$d")"
    done
    [ -n "$PROJECT_ROOT" ] || { echo "ERROR: OGPO checkout with .venv-molmo not found (scripts/ogpo/setup_molmo_venv.sh); submit from inside it"; exit 1; }
fi
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/.venv-molmo/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

for f in "$pi05_checkpoint/params" "$pi05_checkpoint/assets/droid/norm_stats.json" "$pi05_tokenizer" \
         "$MLSPACES_ASSETS_DIR/benchmarks/molmospaces-bench-v1" "$NLTK_DATA"; do
    [ -e "$f" ] || { echo "ERROR: missing $f (scripts/ogpo/download_molmo_assets.sbatch)"; exit 1; }
done
n_workers=$(( $(echo "$env_name" | tr ',' '\n' | wc -l) * parallel_envs ))
if [ -n "${SLURM_CPUS_PER_TASK:-}" ] && [ $(( n_workers + 2 )) -gt "$SLURM_CPUS_PER_TASK" ]; then
    echo "ERROR: $n_workers molmo worker processes need cpus-per-task >= $(( n_workers + 2 )), got $SLURM_CPUS_PER_TASK"; exit 1
fi

echo "=== AWR-pi05 MOLMO MULTI-TASK-16 rounds=$num_rounds x $rollouts_per_round eps/task, $awr_value_steps V + $updates_per_round policy steps/round | tasks=$env_name | seed=$seed | batch=$batch_size (accum $grad_accum_steps) | critic=$pi05_critic_arch ${pi05_bronet_hidden_dim}x$pi05_bronet_depth pool=$pi05_critic_pooling | awr beta=$awr_beta lambda=$awr_lambda | $n_workers workers ==="

python ogpo/main_pi05.py \
    --algo=ogpo \
    --experiment.project=$project \
    --experiment.run_group=$run_group \
    --experiment.run_name=$run_name \
    --experiment.wandb_name=$wandb_name \
    --experiment.seed=$seed \
    --experiment.log=$log \
    --experiment.save_dir=$SAVE_DIR \
    --env.name=$env_name \
    ${eval_env_name:+--env.eval_name=$eval_env_name} \
    --training.task_balance=$task_balance \
    --env.vector_envs=true \
    --env.parallel_envs=$parallel_envs \
    --env.inference_batch=$inference_batch \
    --env.worker_egl_devices=$worker_egl_devices \
    --env.post_success_steps=$post_success_steps \
    --training.collection_mode=rounds \
    --training.num_rounds=$num_rounds \
    --training.rollouts_per_round=$rollouts_per_round \
    --training.updates_per_round=$updates_per_round \
    --training.awr_value_steps=$awr_value_steps \
    --training.awr_value_refreshes=$awr_value_refreshes \
    --training.gpu_mode=$gpu_mode \
    --training.zero_actor_state=$zero_actor_state \
    --buffer.size=$buffer_size \
    ${rb_dir:+--buffer.storage_dir=$rb_dir} \
    --eval.eval_episodes=$eval_episodes \
    --intervals.eval_interval=$eval_interval \
    --intervals.log_interval=$log_interval \
    --success_buffer.use_success_buffer=$use_success_buffer \
    --success_buffer.batch_size=$success_buffer_batch_size \
    --success_buffer.bc_batch_size=$bc_batch_size \
    --agent.horizon_length=$horizon_length \
    --agent.discount=$discount \
    --agent.tau=$tau \
    --agent.actor_tau=$actor_tau \
    --agent.best_of_n=$best_of_n \
    --agent.num_qs=$num_qs \
    --agent.q_agg=$q_agg \
    --agent.pi05_bon_agg=$pi05_bon_agg \
    --agent.flow_steps=$flow_steps \
    --agent.ft_flow_steps=$ft_flow_steps \
    --agent.bc_coeff=$bc_coeff \
    --agent.adv_strategy=$adv_strategy \
    --agent.clip_epsilon=$clip_epsilon \
    --agent.entropy_coeff=$entropy_coeff \
    --agent.use_bc_regularization=$use_bc_regularization \
    --agent.use_tapered_noise=$use_tapered_noise \
    --agent.use_constant_noise=$use_constant_noise \
    --agent.error_correct_sde_to_ode=$error_correct_sde_to_ode \
    --agent.constant_noise_std=$constant_noise_std \
    --agent.use_success_buffer_q=$use_success_buffer_q \
    --agent.act_min=$act_min \
    --agent.act_max=$act_max \
    --agent.clip_intermediate_actions=$clip_intermediate_actions \
    --agent.batch_size=$batch_size \
    --agent.ppo_batch_size=$ppo_batch_size \
    --agent.group_num_samples=$group_num_samples \
    --agent.grad_accum_steps=$grad_accum_steps \
    --agent.q_variance_reduction=$q_variance_reduction \
    --agent.q_vr_num_samples=$q_vr_num_samples \
    --agent.q_vr_reduction=$q_vr_reduction \
    --agent.pi05_critic_arch=$pi05_critic_arch \
    --agent.pi05_critic_pooling=$pi05_critic_pooling \
    --agent.pi05_bronet_hidden_dim=$pi05_bronet_hidden_dim \
    --agent.pi05_bronet_depth=$pi05_bronet_depth \
    --agent.ppo_lr=$ppo_lr \
    --agent.critic_scheduler=$critic_scheduler \
    --agent.critic_weight_decay=$critic_weight_decay \
    --agent.clip_grad_norm=$clip_grad_norm \
    --agent.layer_norm=$layer_norm \
    --agent.value_hidden_dims=$value_hidden_dims \
    --agent.pi05_checkpoint=$pi05_checkpoint \
    --agent.pi05_tokenizer=$pi05_tokenizer \
    --agent.pi05_action_horizon=$pi05_action_horizon \
    --agent.pi05_robot=$pi05_robot \
    --agent.pi05_drop_empty_camera=$pi05_drop_empty_camera \
    --agent.pi05_lr=$pi05_lr \
    --agent.pi05_clip_grad_norm=$pi05_clip_grad_norm \
    --agent.pi05_weight_decay=$pi05_weight_decay \
    --agent.grad_diagnostics=$grad_diagnostics \
    --agent.pi05_actor_objective=$pi05_actor_objective \
    --agent.use_value_fn=$use_value_fn \
    --agent.v_training_mode=$v_training_mode \
    --agent.v_ensemble_size=$v_ensemble_size \
    --agent.v_agg=$v_agg \
    --agent.awr_beta=$awr_beta \
    --agent.awr_weight_max=$awr_weight_max \
    --agent.awr_lambda=$awr_lambda \
    --agent.awr_normalize_adv=$awr_normalize_adv
