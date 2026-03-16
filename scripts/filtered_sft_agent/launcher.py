#!/usr/bin/env python3
"""Launcher for Filtered-SFT agent experiments.

Usage:
    ./scripts/filtered_sft_agent/launcher.py --project_name my_project
    ./scripts/filtered_sft_agent/launcher.py --project_name my_project --dry
    ./scripts/filtered_sft_agent/launcher.py --project_name my_project --mode local
"""

import argparse
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from launcher_util import (
    DEFAULT_CHECKPOINT_BASE_DIR,
    auto_exp_name,
    dict_permutations,
    generate_run_commands,
    generate_srun_command,
)

SCRIPT = "scripts/filtered_sft_agent/exp.py"
CONFIG_NAME = "pi05_libero_online_filtered_sft"
PROJECT_NAME = "filtered_sft_agent_sweep"
DEFAULT_SEED = 0
DEFAULT_BUFFER_CAPACITY = 250000
DEFAULT_LOG_INTERVAL = 50
DEFAULT_NUM_ROLLOUTS = 1
DEFAULT_COLLECT_INTERVAL = 300
DEFAULT_BATCH_SIZE = 256
DEFAULT_TRAIN_ENV_NUM = 1
DEFAULT_TASKS = ["libero_90_59x1"]
DEFAULT_EVAL_ENV_NUM = 4
DEFAULT_EVAL_INTERVAL = 300
DEFAULT_NUM_EVAL_ROLLOUTS = 32
NUM_TRAIN_STEPS = 5_000
USE_SAME_EVAL_AND_TRAIN_TASK = True

# ---------- Hyperparameter grid ----------
# Keys can be any `_config.cli()` override.
# If this dict is empty, one run is launched with config defaults.
applicable_configs: Dict[str, List[Any]] = {
    "seed": [0, 2],
    "log_interval": [25],
    "batch_size": [256],
    "rl.policy_training_start_step": [900],
    "rl.online_ratio": [1.0],
    "rl.reset_policy_params_to_ema_period": [500, 1000],
    "collect.num_initial_rollouts": [5],
    "lr_schedule.value": [2.5e-6, 5.5e-6, 2.5e-5],
    "collect.tasks": [
        # "libero_90_2",
        # "libero_90_7",
        # "libero_90_9",
        # "libero_90_11",
        "libero_90_14",
        # "libero_90_26",
        # "libero_90_28",
        # "libero_90_30",
        # "libero_90_31",
        # "libero_90_35",
        "libero_90_38",
        # "libero_90_41",
        # "libero_90_53",
        "libero_90_59",
        # "libero_90_60",
        # "libero_90_61",
        # "libero_90_62",
        "libero_90_64",
        # "libero_90_74",
        # "libero_90_77",
        # "libero_90_79",
        "libero_90_82",
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
    parser.add_argument("--project_name", default=PROJECT_NAME, help="W&B project name")
    parser.add_argument("--config_name", default=CONFIG_NAME, help="Training config name")
    parser.add_argument("--log_interval", type=int, default=DEFAULT_LOG_INTERVAL)
    parser.add_argument(
        "--checkpoint_base_dir",
        default=DEFAULT_CHECKPOINT_BASE_DIR,
        help="Checkpoint base directory",
    )
    parser.add_argument("--buffer_capacity", type=int, default=DEFAULT_BUFFER_CAPACITY)

    parser.add_argument("--num_rollouts", type=int, default=DEFAULT_NUM_ROLLOUTS)
    parser.add_argument(
        "--collect_interval", type=int, default=DEFAULT_COLLECT_INTERVAL
    )
    parser.add_argument("--train_env_num", type=int, default=DEFAULT_TRAIN_ENV_NUM)
    parser.add_argument("--eval_env_num", type=int, default=DEFAULT_EVAL_ENV_NUM)
    parser.add_argument("--eval_interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    parser.add_argument("--num_eval_rollouts", type=int, default=DEFAULT_NUM_EVAL_ROLLOUTS)
    parser.add_argument("--num_train_steps", type=int, default=NUM_TRAIN_STEPS)

    args = parser.parse_args()

    combos = dict_permutations(applicable_configs)
    command_list = []
    for idx, combo in enumerate(combos):
        flags: Dict[str, Any] = {
            "overwrite": True,
            "project_name": args.project_name,
            "seed": DEFAULT_SEED,
            "log_interval": args.log_interval,
            "checkpoint_base_dir": args.checkpoint_base_dir,
            "collect.num_rollouts": args.num_rollouts,
            "collect.collect_interval": args.collect_interval,
            "rl.buffer_capacity": args.buffer_capacity,
            "batch_size": DEFAULT_BATCH_SIZE,
            "collect.env_num": args.train_env_num,
            "collect.eval_env_num": args.eval_env_num,
            "collect.tasks": DEFAULT_TASKS,
            "collect.eval_interval": args.eval_interval,
            "collect.num_eval_rollouts": args.num_eval_rollouts,
            "num_train_steps": args.num_train_steps,
        }
        flags.update(combo)

        task = flags["collect.tasks"]
        train_envs = flags["collect.env_num"]
        flags["collect.tasks"] = [f'{task}x{train_envs}']
        eval_envs = flags["collect.eval_env_num"]
        flags["collect.eval_tasks"] = [f'{task}x{eval_envs}']

        flags.setdefault("exp_name", auto_exp_name(args.project_name, flags, idx))

        cmd = generate_srun_command(SCRIPT, args.config_name, flags=flags)
        command_list.append(cmd)

    generate_run_commands(
        command_list,
        mode=args.mode,
        duration=args.duration,
        dry=args.dry,
    )


if __name__ == "__main__":
    main()
