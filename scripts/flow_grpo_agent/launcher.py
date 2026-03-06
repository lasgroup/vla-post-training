#!/usr/bin/env python3
"""Launcher for Flow-GRPO experiments.

Usage:
    ./scripts/flow_grpo_agent/launcher.py --project_name my_project
    ./scripts/flow_grpo_agent/launcher.py --project_name my_project --dry
    ./scripts/flow_grpo_agent/launcher.py --project_name my_project --mode local
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

SCRIPT = "scripts/flow_grpo_agent/exp.py"
CONFIG_NAME = "pi05_libero_online_flow_grpo_sft"
PROJECT_NAME = "flow_grpo_sweep"
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
DEFAULT_EVAL_ENV_NUM = 4
DEFAULT_EVAL_INTERVAL = 300
DEFAULT_NUM_EVAL_ROLLOUTS = 32

# ---------- Hyperparameter grid ----------
# Keys can be any `_config.cli()` override.
# If this dict is empty, one run is launched with config defaults.
applicable_configs: Dict[str, List[Any]] = {
    "seed": [0, 1, 2],
    "log_interval": [25],
    "rl.num_critic_updates_per_batch": [10],
    "collect.use_time_to_success_as_reward": [True],
    "batch_size": [256],
    "rl.policy_training_start_step": [300, 600, 900, 1200, 1500],
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
    parser.add_argument("--project_name", default=PROJECT_NAME, help="W&B project name")
    parser.add_argument(
        "--config_name", default=CONFIG_NAME, help="Training config name"
    )
    parser.add_argument("--log_interval", type=int, default=DEFAULT_LOG_INTERVAL)
    parser.add_argument(
        "--checkpoint_base_dir",
        default=DEFAULT_CHECKPOINT_BASE_DIR,
        help="Checkpoint base directory",
    )
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
    parser.add_argument("--eval_env_num", type=int, default=DEFAULT_EVAL_ENV_NUM)
    parser.add_argument("--eval_interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    parser.add_argument("--num_eval_rollouts", type=int, default=DEFAULT_NUM_EVAL_ROLLOUTS)

    args = parser.parse_args()

    combos = dict_permutations(applicable_configs)
    command_list = []
    for idx, combo in enumerate(combos):
        flags: Dict[str, Any] = {
            "overwrite": True,
            "project_name": args.project_name,
            "seed": DEFAULT_SEED,
            "collect.seed": DEFAULT_SEED,
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
            "collect.eval_env_num": args.eval_env_num,
            "collect.eval_interval": args.eval_interval,
            "collect.num_eval_rollouts": args.num_eval_rollouts,
        }
        flags.update(combo)

        # Keep these in sync with policy_training_start_step
        policy_start = flags["rl.policy_training_start_step"]
        flags["rl.td_weight_schedule.switch_step"] = policy_start
        flags["rl.critic_pre_training_steps"] = policy_start

        if "seed" in flags and "collect.seed" not in combo:
            flags["collect.seed"] = flags["seed"]

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
