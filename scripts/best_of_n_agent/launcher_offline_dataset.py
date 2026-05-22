#!/usr/bin/env python3
"""Launcher for Best-of-N agent experiments.

Usage:
    uv run scripts/best_of_n_agent/launcher_offline_dataset.py \
    --project_name swiss_ai --duration 04:00:00 --requeue \
    --checkpoint_base_dir /capstor/scratch/cscs/ralfroemer/checkpoints \
    --print_commands     # inspect; drop to submit
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Union

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from launcher_util import (
    DEFAULT_LOG_DIR,
    apply_requeue_flags,
    dict_permutations,
    generate_run_commands,
    generate_srun_command,
    validate_unique_exp_names,
)

SCRIPT = "scripts/best_of_n_agent/exp.py"
CONFIG_NAME = "pi05_libero_online_best_of_n"
PROJECT_NAME = "value_learning"
DEFAULT_LOG_INTERVAL = 25
DEFAULT_SAVE_INTERVAL = 100_000
DEFAULT_SEED = 0
DEFAULT_BUFFER_CAPACITY = 50_000  # per task ~5k transitions; lazily allocated
DEFAULT_NUM_ROLLOUTS = 1
DEFAULT_COLLECT_INTERVAL = 6_000
DEFAULT_NUM_CRITIC_UPDATES_PER_BATCH = 10
DEFAULT_USE_TIME_TO_SUCCESS_AS_REWARD = True
DEFAULT_BATCH_SIZE = 256
DEFAULT_N_SAMPLES = 32
# Use 4 envs for sharding
DEFAULT_TRAIN_ENV_NUM = 4
DEFAULT_TASKS = ["libero_90_59-62"]
DEFAULT_EVAL_TASKS = ["libero_90_59-62"]
DEFAULT_EVAL_ENV_NUM = 4
DEFAULT_EVAL_INTERVAL = 10_000
DEFAULT_NUM_EVAL_ROLLOUTS = 1  # eval still fires at step 0; keep it cheap
NUM_TRAIN_STEPS = 1  # one loop iter: collect at step 0, save shard, exit
DEFAULT_CRITIC_TRAINING_START_STEP = 1_000_000_000  # never train critics (pure collection)
DEFAULT_CRITIC_INFERENCE_START_STEP = 1_000_000_000  # always base policy
DEFAULT_NUM_CPUS = 16

# ---------- Hyperparameter grid ----------
# Keys can be any `_config.cli()` override.
# If this dict is empty, one run is launched with config defaults.
applicable_configs: Dict[Union[str, tuple], List[Any]] = {
    # General -- single seed so we get exactly 50 rollouts per task (not 50 x n_seeds).
    "seed": [0],
    # Data collection/eval: 50 base-policy rollouts per task.
    "collect.num_rollouts": [50],
    # Critic training (unused here -- collection only -- but kept so downstream
    # flags like td_weight_schedule.switch_step remain defined).
    "collect.use_time_to_success_as_reward": [True],
    "rl.num_critic_updates_per_batch": [1],
    "rl.td_weight_schedule.init_value": [0.5],
    "rl.td_weight_schedule.end_value": [0.5],
    "rl.td_weight_schedule.switch_step": [500_000],
    "rl.num_value_bins": [1],
    # Best-of-N specific. Cache the (base-policy) pi0 prefix embedding in the buffer/shards.
    "collect.store_prefix_rep": [True],
    "rl.train_on_policy_value_function": [False],
    "rl.n_samples": [32],
    # Base policy only: critic_inference_start_step far beyond num_train_steps means
    # sample_actions always falls back to the base pi0.5 policy (no best-of-N selection).
    "rl.critic_inference_start_step": [1_000_000_000],
    # 100 LIBERO tasks = LIBERO-90 (90) + LIBERO-10 (10). One SLURM job per task.
    ("collect.tasks", "collect.eval_tasks"): [
        *[(f"libero_90_{i}", f"libero_90_{i}") for i in range(90)],
        *[(f"libero_10_{i}", f"libero_10_{i}") for i in range(10)],
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true", help="Print commands without submitting")
    parser.add_argument(
        "--mode",
        default="swiss-ai",
        choices=["swiss-ai", "local"],
        help="Execution mode",
    )
    parser.add_argument("--duration", default="03:30:00", help="SLURM time limit")
    parser.add_argument("--partition", default="normal", help="SLURM partition")
    parser.add_argument("--project_name", default=PROJECT_NAME, help="W&B project name")
    parser.add_argument("--group", default=None, help="W&B group name")
    parser.add_argument("--print_commands", action="store_true", help="Print full uv run commands without submitting")
    parser.add_argument("--config_name", default=CONFIG_NAME, help="Training config name")
    parser.add_argument("--log_interval", type=int, default=DEFAULT_LOG_INTERVAL)
    parser.add_argument(
        "--checkpoint_base_dir",
        default="/capstor/scratch/cscs/ralfroemer/checkpoints",
        help="Checkpoint base directory",
    )

    # Best-of-N defaults matching submit_train.sh
    parser.add_argument("--buffer_capacity", type=int, default=DEFAULT_BUFFER_CAPACITY)
    parser.add_argument("--num_rollouts", type=int, default=DEFAULT_NUM_ROLLOUTS)
    parser.add_argument("--collect_interval", type=int, default=DEFAULT_COLLECT_INTERVAL)
    parser.add_argument("--n_samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--train_env_num", type=int, default=DEFAULT_TRAIN_ENV_NUM)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--eval_tasks", nargs="+", default=DEFAULT_EVAL_TASKS)
    parser.add_argument("--eval_env_num", type=int, default=DEFAULT_EVAL_ENV_NUM)
    parser.add_argument("--eval_interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    parser.add_argument("--num_eval_rollouts", type=int, default=DEFAULT_NUM_EVAL_ROLLOUTS)
    parser.add_argument("--num_train_steps", type=int, default=NUM_TRAIN_STEPS)

    parser.add_argument("--store_prefix_rep", action="store_true", help="Store pi0 prefix embeddings in the replay buffer")
    parser.add_argument("--critic_training_start_step", type=int, default=DEFAULT_CRITIC_TRAINING_START_STEP)
    parser.add_argument("--critic_inference_start_step", type=int, default=DEFAULT_CRITIC_INFERENCE_START_STEP)
    parser.add_argument("--num_value_bins", type=int, default=1, help="1=regression, >1=categorical over return bins")
    parser.add_argument("--value_target_type", default="one_hot", choices=["one_hot", "two_hot"])
    parser.add_argument("--num_cpus", type=int, default=DEFAULT_NUM_CPUS)
    parser.add_argument("--log_dir", default=DEFAULT_LOG_DIR, help="Directory for SLURM .out log files")
    parser.add_argument("--requeue", action="store_true", help="Submit requeue-safe resumable jobs")

    args = parser.parse_args()

    combos = dict_permutations(applicable_configs)
    command_list = []
    for combo in combos:
        flags: Dict[str, Any] = {
            "overwrite": True,
            "project_name": args.project_name,
            "group": args.group,
            "seed": DEFAULT_SEED,
            "log_interval": args.log_interval,
            "checkpoint_base_dir": args.checkpoint_base_dir,
            "collect.num_rollouts": args.num_rollouts,
            "collect.collect_interval": args.collect_interval,
            "rl.buffer_capacity": args.buffer_capacity,
            "rl.num_critic_updates_per_batch": DEFAULT_NUM_CRITIC_UPDATES_PER_BATCH,
            "rl.n_samples": args.n_samples,
            "rl.critic_training_start_step": args.critic_training_start_step,
            "rl.critic_inference_start_step": args.critic_inference_start_step,
            "collect.use_time_to_success_as_reward": DEFAULT_USE_TIME_TO_SUCCESS_AS_REWARD,
            "batch_size": DEFAULT_BATCH_SIZE,
            "collect.env_num": args.train_env_num,
            "collect.eval_env_num": args.eval_env_num,
            "collect.tasks": args.tasks,
            "collect.eval_tasks": args.eval_tasks,
            "collect.eval_interval": args.eval_interval,
            "collect.num_eval_rollouts": args.num_eval_rollouts,
            "num_train_steps": args.num_train_steps,
            "rl.num_value_bins": args.num_value_bins,
            "rl.value_target_type": args.value_target_type,
            "save_interval": DEFAULT_SAVE_INTERVAL,
            "collect.store_prefix_rep": args.store_prefix_rep,
        }
        flags.update(combo)
        flags = apply_requeue_flags(flags, enabled=args.requeue)

        # Keep these in sync with policy_training_start_step
        policy_start = flags["rl.critic_inference_start_step"]
        if flags["rl.td_weight_schedule.switch_step"] == -1:
            flags["rl.td_weight_schedule.switch_step"] = policy_start
            flags["rl.critic_pre_training_steps"] = policy_start
        else:
            flags["rl.critic_pre_training_steps"] = 0

        # Store the base-rollout dataset in a dedicated folder, separate from training
        # checkpoints, with one identifiable dir per task. checkpoint_dir is
        # (checkpoint_base_dir / name / exp_name).resolve(); the leading ".." cancels the
        # fixed config `name` (pi05_libero_online_best_of_n, suppressed/non-overridable), so
        # shards land at <checkpoint_base_dir>/pi05_libero_base_rollouts/offline_<task>/.
        flags["exp_name"] = f"../pi05_libero_base_rollouts/offline_{flags['collect.tasks']}"
        command_list.append(flags)

    if args.requeue:
        validate_unique_exp_names(command_list)

    rendered_commands = [
        generate_srun_command(SCRIPT, args.config_name, flags=flags)
        for flags in command_list
    ]

    if args.print_commands:
        for cmd in rendered_commands:
            print(cmd)
        return

    generate_run_commands(
        rendered_commands,
        mode=args.mode,
        duration=args.duration,
        partition=args.partition,
        num_cpus=args.num_cpus,
        dry=args.dry,
        log_dir=args.log_dir,
        requeue=args.requeue,
    )


if __name__ == "__main__":
    main()
