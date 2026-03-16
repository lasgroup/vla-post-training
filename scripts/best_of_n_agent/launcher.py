#!/usr/bin/env python3
"""Launcher for Best-of-N agent experiments.

Usage:
    ./scripts/best_of_n_agent/launcher.py --project_name my_project
    ./scripts/best_of_n_agent/launcher.py --project_name my_project --dry
    ./scripts/best_of_n_agent/launcher.py --project_name my_project --mode local
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Union

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from launcher_util import (
    DEFAULT_CHECKPOINT_BASE_DIR,
    DEFAULT_LOG_DIR,
    auto_exp_name,
    dict_permutations,
    generate_run_commands,
    generate_srun_command,
)

SCRIPT = "scripts/best_of_n_agent/exp.py"
CONFIG_NAME = "pi05_libero_online_best_of_n"
PROJECT_NAME = "value_learning"
DEFAULT_LOG_INTERVAL = 50
DEFAULT_SEED = 0
DEFAULT_BUFFER_CAPACITY = 250000
DEFAULT_NUM_ROLLOUTS = 1
DEFAULT_COLLECT_INTERVAL = 300
DEFAULT_NUM_CRITIC_UPDATES_PER_BATCH = 10
DEFAULT_USE_TIME_TO_SUCCESS_AS_REWARD = True
DEFAULT_BATCH_SIZE = 256
DEFAULT_N_SAMPLES = 8
# Use 4 envs for sharding
DEFAULT_TRAIN_ENV_NUM = 4
DEFAULT_TASKS = ["libero_90_59-62"]
DEFAULT_EVAL_TASKS = ["libero_90_59-62"]
DEFAULT_EVAL_ENV_NUM = 4
DEFAULT_EVAL_INTERVAL = 300
DEFAULT_NUM_EVAL_ROLLOUTS = 32
NUM_TRAIN_STEPS = 5_000
DEFAULT_CRITIC_TRAINING_START_STEP = 0
DEFAULT_CRITIC_INFERENCE_START_STEP = 900
DEFAULT_NUM_CPUS = 16

# ---------- Hyperparameter grid ----------
# Keys can be any `_config.cli()` override.
# If this dict is empty, one run is launched with config defaults.
applicable_configs: Dict[Union[str, tuple], List[Any]] = {
    "seed": [0, 1, 2],
    "rl.n_samples": [8],
    "collect.use_time_to_success_as_reward": [True],
    "rl.train_on_policy_value_function": [True, False],
    "log_interval": [25],
    "rl.num_value_bins": [1, 20, 50],
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
    parser.add_argument("--config_name", default=CONFIG_NAME, help="Training config name")
    parser.add_argument("--log_interval", type=int, default=DEFAULT_LOG_INTERVAL)
    parser.add_argument(
        "--checkpoint_base_dir",
        default=DEFAULT_CHECKPOINT_BASE_DIR,
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

    parser.add_argument("--critic_training_start_step", type=int, default=DEFAULT_CRITIC_TRAINING_START_STEP)
    parser.add_argument("--critic_inference_start_step", type=int, default=DEFAULT_CRITIC_INFERENCE_START_STEP)
    parser.add_argument("--num_value_bins", type=int, default=1, help="1=regression, >1=categorical over return bins")
    parser.add_argument("--value_target_type", default="one_hot", choices=["one_hot", "two_hot"])
    parser.add_argument("--num_cpus", type=int, default=DEFAULT_NUM_CPUS)
    parser.add_argument("--log_dir", default=DEFAULT_LOG_DIR, help="Directory for SLURM .out log files")

    args = parser.parse_args()

    combos = dict_permutations(applicable_configs)
    command_list = []
    for idx, combo in enumerate(combos):
        flags: Dict[str, Any] = {
            "overwrite": True,
            "project_name": args.project_name,
            "algorithm": "best_of_n",
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
        }
        flags.update(combo)

        # Keep these in sync with policy_training_start_step
        policy_start = flags["rl.critic_inference_start_step"]
        flags["rl.td_weight_schedule.switch_step"] = policy_start
        flags["rl.critic_pre_training_steps"] = policy_start

        flags.setdefault("exp_name", auto_exp_name(args.project_name, flags, idx))

        cmd = generate_srun_command(SCRIPT, args.config_name, flags=flags)
        command_list.append(cmd)

    generate_run_commands(
        command_list,
        mode=args.mode,
        duration=args.duration,
        partition=args.partition,
        num_cpus=args.num_cpus,
        dry=args.dry,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
