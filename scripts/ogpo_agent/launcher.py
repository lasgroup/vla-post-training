#!/usr/bin/env python3
"""Launcher for OGPO-agent experiments.

Usage:
    ./scripts/ogpo_agent/launcher.py --project_name my_project
    ./scripts/ogpo_agent/launcher.py --project_name my_project --dry
    ./scripts/ogpo_agent/launcher.py --project_name my_project --mode local
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

SCRIPT = "scripts/ogpo_agent/exp.py"
CONFIG_NAME = "pi05_libero_online_ogpo_sft"
PROJECT_NAME = "libero_59"
WANDB_ENTITY = "RL-experiments"
os.environ.setdefault("WANDB_ENTITY", WANDB_ENTITY)
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
DEFAULT_TASKS = ["libero_90_59"]
DEFAULT_EVAL_TASKS = ["libero_90_59"]
DEFAULT_TRAIN_ENV_NUM = 1
DEFAULT_EVAL_ENV_NUM = 1
DEFAULT_EVAL_INTERVAL = 300
DEFAULT_NUM_EVAL_ROLLOUTS = 8
NUM_TRAIN_STEPS = 5_000

# ---------- Hyperparameter grid ----------
# Keys can be any `_config.cli()` override.
# If this dict is empty, one run is launched with config defaults.
applicable_configs: Dict[Union[str, tuple], List[Any]] = {
    "seed": [1],
    "log_interval": [25],
    "rl.num_critic_updates_per_batch": [10],
    "collect.use_time_to_success_as_reward": [True],
    "batch_size": [DEFAULT_BATCH_SIZE],
    "rl.policy_training_start_step": [900],
    # PPO defaults. v1 uses G=1 with a V-function baseline (advantage =
    # Q - V), not OGPO's group-relative baseline — much cheaper per update.
    # Switch to ``vanilla`` and bump G if you want OGPO-square parity.
    "rl.group_num_samples": [1],
    "rl.clip_epsilon": [0.2],
    "rl.bc_coeff": [1.0],
    "rl.num_sde_steps": [10],
    "rl.noise_level": [0.3],
    "rl.adv_strategy": ["subtract_v"],
    "collect.num_initial_rollouts": [5],
    "lr_schedule.value": [2.5e-5],
    "rl.store_success_episodes_only": [True],
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
    parser.add_argument("--duration", default="12:00:00", help="SLURM time limit")
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
    parser.add_argument(
        "--log_dir",
        default=DEFAULT_LOG_DIR,
        help="Directory for SLURM .out log files",
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
            "collect.eval_tasks": DEFAULT_EVAL_TASKS,
            "collect.eval_interval": args.eval_interval,
            "collect.num_eval_rollouts": args.num_eval_rollouts,
            "num_train_steps": args.num_train_steps,
        }
        flags.update(combo)
        flags = apply_requeue_flags(flags, enabled=args.requeue)

        # Keep critic warmup aligned with policy start.
        policy_start = flags["rl.policy_training_start_step"]
        flags["rl.critic_pre_training_steps"] = policy_start

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
