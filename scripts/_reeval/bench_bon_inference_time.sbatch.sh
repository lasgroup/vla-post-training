#!/bin/bash
#SBATCH --job-name=bon_latency
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --output=/capstor/scratch/cscs/ralfroemer/bon_latency/slurm-%j.out
set -euo pipefail
cd /users/ralfroemer/vla-post-training
ulimit -c 0
# Experiment #1: deployment latency of the BoN block vs N, env_num=1.
# Checkpoint pair comes from the existing reeval_final staging tree of
# multitask_tasks1_bofn_molmo_v0_seed0 (policy@100000 + rl_state@99999).
srun --account=ab037 --environment=vla-post-training env CUDA_VISIBLE_DEVICES=0 RESTORE_EXPLICIT_SHARDING=1 uv run scripts/_reeval/bench_bon_inference_time.py pi05_molmo_online_best_of_n --log_interval 25 --num_train_steps 100000 --max_runtime 39600 --collect.env_num 1 --collect.eval_env_num 1 --collect.eval_interval 99999 --collect.collect_interval 10000 --collect.num_rollouts 20 --collect.domain molmo --collect.tasks molmo_138 --collect.eval_tasks molmo_138 molmo_8 molmo_273 molmo_222 molmo_73 molmo_346 molmo_49 molmo_21 molmo_379 molmo_140 molmo_356 molmo_345 molmo_155 molmo_357 molmo_163 molmo_208 molmo_226 --collect.store_prefix_rep --rl.critic.batch_size 1024 --rl.critic.inference_start_step 1 --rl.critic.pre_training_steps 0 --rl.critic.td_weight_schedule.switch_step 999999 --rl.critic.td_weight_schedule.init_value 1 --rl.critic.td_weight_schedule.end_value 1 --rl.critic.no-use_distributional_critic --rl.critic.num_value_bins 1 --rl.critic.use_bronet --rl.critic.bronet_hidden_dim 1024 --rl.buffer_capacity 250000 --rl.online_ratio 1.0 --seed 0 --resume --no-overwrite --checkpoint_base_dir /capstor/store/cscs/swissai/a0220/ralfroemer/results/vla-post-training_multitask_tasks1_bofn_molmo_v0_seed0/0/reeval_final --project_name vla-post-training --group_name multitask_tasks1_bofn_molmo_v0_seed0 --exp_name vla-post-training_20260625-083520_519e39_seed0
