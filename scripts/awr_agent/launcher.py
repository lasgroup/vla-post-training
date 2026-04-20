#!/usr/bin/env python3
"""Launcher for AWR-agent experiments.

Usage:
    ./scripts/awr_agent/launcher.py --project_name my_project
    ./scripts/awr_agent/launcher.py --project_name my_project --dry
    ./scripts/awr_agent/launcher.py --project_name my_project --mode local
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Union

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from launcher_util import (
    DEFAULT_CHECKPOINT_BASE_DIR,
    DEFAULT_LOG_DIR,
    apply_requeue_flags,
    auto_exp_name,
    dict_permutations,
    generate_run_commands,
    generate_srun_command,
    validate_unique_exp_names,
)

SCRIPT = "scripts/awr_agent/exp.py"
CONFIG_NAME = "pi05_libero_online_aw_sft"
PROJECT_NAME = "awr_agent_sweep"
DEFAULT_LOG_INTERVAL = 50
DEFAULT_SEED = 0
DEFAULT_BUFFER_CAPACITY = 250000
DEFAULT_POLICY_START_TRAINING = 1000
DEFAULT_POLICY_UPDATE_INTERVAL = 1
DEFAULT_NUM_ROLLOUTS = 1
DEFAULT_COLLECT_INTERVAL = 300
DEFAULT_NUM_CRITIC_UPDATES_PER_BATCH = 10
DEFAULT_USE_TIME_TO_SUCCESS_AS_REWARD = True
DEFAULT_BATCH_SIZE = 256
DEFAULT_TRAIN_ENV_NUM = 4
DEFAULT_TASKS = ["libero_90_59"]
DEFAULT_EVAL_ENV_NUM = 4
DEFAULT_EVAL_INTERVAL = 2000
DEFAULT_NUM_EVAL_ROLLOUTS = 8
NUM_TRAIN_STEPS = 10_000

# ---------- Hyperparameter grid ----------
# Keys can be any `_config.cli()` override.
# If this dict is empty, one run is launched with config defaults.
applicable_configs: Dict[Union[str, tuple], List[Any]] = {
    "seed": [0, 1, 2],
    "log_interval": [25],
    "rl.num_critic_updates_per_batch": [
        10
    ],
    "collect.use_time_to_success_as_reward": [True],
    "batch_size": [256],
    "rl.policy_training_start_step": [900],
    "rl.online_ratio": [1.0],
    "rl.reset_policy_params_to_ema_period": [500],
    "rl.use_mc_returns": [False],
    "collect.num_initial_rollouts": [5],
    "lr_schedule.value": [2.5e-5],
    "rl.td_weight_schedule.switch_step": [-1],
    "rl.store_success_episodes_only": [True],
    "rl.normalizer_config.ema_weight": [0.99, 1.0],
    ("collect.tasks", "collect.eval_tasks"): [
        # "libero_90_2",
        # "libero_90_7",
        # "libero_90_9",
        # "libero_90_11",
        (["libero_90_1-14", "libero_90_16-89"], ["libero_90_1-14", "libero_90_16-89"]),
        # "libero_90_26",
        # "libero_90_28",
        # "libero_90_30",
        # "libero_90_31",
        # "libero_90_35",
        # ("libero_90_38", "libero_90_38"),
        # "libero_90_41",
        # "libero_90_53",
        #("libero_90_59", "libero_90_59"),
        # "libero_90_60",
        # "libero_90_61",
        # "libero_90_62",
        # ("libero_90_64", "libero_90_64"),
        # "libero_90_74",
        # "libero_90_77",
        # "libero_90_79",
        # ("libero_90_82", "libero_90_82"),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry", action="store_true", help="Print commands without submitting"
    )
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
    parser.add_argument(
        "--config_name", default=CONFIG_NAME, help="Training config name"
    )
    parser.add_argument("--log_interval", type=int, default=DEFAULT_LOG_INTERVAL)
    parser.add_argument(
        "--checkpoint_base_dir",
        default=DEFAULT_CHECKPOINT_BASE_DIR,
        help="Checkpoint base directory",
    )
    parser.add_argument("--log_dir", default=DEFAULT_LOG_DIR, help="Directory for SLURM .out log files")

    # AWR defaults matching submit_train.sh
    parser.add_argument("--buffer_capacity", type=int, default=DEFAULT_BUFFER_CAPACITY)
    parser.add_argument(
        "--policy_start_training", type=int, default=DEFAULT_POLICY_START_TRAINING
    )
    parser.add_argument(
        "--policy_update_interval", type=int, default=DEFAULT_POLICY_UPDATE_INTERVAL
    )
    parser.add_argument("--num_rollouts", type=int, default=DEFAULT_NUM_ROLLOUTS)
    parser.add_argument(
        "--collect_interval", type=int, default=DEFAULT_COLLECT_INTERVAL
    )
    parser.add_argument("--train_env_num", type=int, default=DEFAULT_TRAIN_ENV_NUM)
    parser.add_argument("--eval_env_num", type=int, default=DEFAULT_EVAL_ENV_NUM)
    parser.add_argument("--eval_interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    parser.add_argument("--num_eval_rollouts", type=int, default=DEFAULT_NUM_EVAL_ROLLOUTS)
    parser.add_argument("--num_train_steps", type=int, default=NUM_TRAIN_STEPS)
    parser.add_argument("--requeue", action="store_true", help="Submit requeue-safe resumable jobs")

    args = parser.parse_args()

    combos = dict_permutations(applicable_configs)
    command_list = []
    for idx, combo in enumerate(combos):
        flags: Dict[str, Any] = {
            "overwrite": True,
            "project_name": args.project_name,
            "group": args.group,
            "seed": DEFAULT_SEED,
            "log_interval": args.log_interval,
            "checkpoint_base_dir": args.checkpoint_base_dir,
            "collect.num_rollouts": args.num_rollouts,
            "collect.collect_interval": args.collect_interval,
            "rl.policy_training_start_step": args.policy_start_training,
            "rl.policy_update_interval": args.policy_update_interval,
            "rl.buffer_capacity": args.buffer_capacity,
            "rl.num_critic_updates_per_batch": DEFAULT_NUM_CRITIC_UPDATES_PER_BATCH,
            "collect.use_time_to_success_as_reward": DEFAULT_USE_TIME_TO_SUCCESS_AS_REWARD,
            "batch_size": DEFAULT_BATCH_SIZE,
            "collect.env_num": args.train_env_num,
            "collect.eval_env_num": args.eval_env_num,
            "collect.tasks": DEFAULT_TASKS,
            "collect.eval_interval": args.eval_interval,
            "collect.num_eval_rollouts": args.num_eval_rollouts,
            "num_train_steps": args.num_train_steps,
        }
        flags.update(combo)
        flags = apply_requeue_flags(flags, enabled=args.requeue)

        # Keep these in sync with policy_training_start_step
        policy_start = flags["rl.policy_training_start_step"]
        flags["rl.critic_pre_training_steps"] = policy_start
        if flags["rl.td_weight_schedule.switch_step"] == -1:
            flags["rl.td_weight_schedule.switch_step"] = policy_start
        # If we only train on the mc returns, we do not need a target critic for policy updates.
        elif flags["rl.td_weight_schedule.switch_step"] >= NUM_TRAIN_STEPS:
            flags["rl.use_ema_critic"] = False

        flags.setdefault("exp_name", auto_exp_name(args.project_name, flags, idx))
        command_list.append(flags)

    if args.requeue:
        validate_unique_exp_names(command_list)

    rendered_commands = [
        generate_srun_command(SCRIPT, args.config_name, flags=flags)
        for flags in command_list
    ]

    generate_run_commands(
        rendered_commands,
        mode=args.mode,
        duration=args.duration,
        partition=args.partition,
        dry=args.dry,
        log_dir=args.log_dir,
        requeue=args.requeue,
    )


if __name__ == "__main__":
    main()
