#!/usr/bin/env python3
"""Launcher for MPO experiments (Abdolmaleki et al. 2018, adapted for flow policies).

Usage:
    ./scripts/mpo/launcher.py --project_name mpo_sweep
    ./scripts/mpo/launcher.py --project_name mpo_sweep --dry
    ./scripts/mpo/launcher.py --project_name mpo_sweep --mode local
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Union

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from launcher_util import (
    DEFAULT_CHECKPOINT_BASE_DIR,
    DEFAULT_LOG_DIR,
    algo_exp_name,
    dict_permutations,
    generate_run_commands,
    generate_srun_command,
)

SCRIPT = "scripts/mpo/exp.py"
CONFIG_NAME = "pi05_libero_online_mpo"
PROJECT_NAME = "mpo_sweep"
DEFAULT_LOG_INTERVAL = 50
DEFAULT_SEED = 0
DEFAULT_BUFFER_CAPACITY = 250000
DEFAULT_POLICY_START_TRAINING = 200
DEFAULT_NUM_ROLLOUTS = 1
DEFAULT_COLLECT_INTERVAL = 300
DEFAULT_NUM_CRITIC_UPDATES_PER_BATCH = 2
DEFAULT_USE_TIME_TO_SUCCESS_AS_REWARD = True
DEFAULT_BATCH_SIZE = 256
DEFAULT_TRAIN_ENV_NUM = 1
DEFAULT_TASKS = ["libero_90_59"]
DEFAULT_EVAL_ENV_NUM = 4
DEFAULT_EVAL_INTERVAL = 300
DEFAULT_NUM_EVAL_ROLLOUTS = 32
NUM_TRAIN_STEPS = 5_000
DEFAULT_GROUP_SIZE = 8

# Short aliases for experiment names
NAME_KEYS = [
    ("collect.tasks", "t"),
    ("rl.policy_training_start_step", "ps"),
    ("rl.online_ratio", "or"),
    ("lr_schedule.peak_lr", "lr"),
    ("rl.group_size", "g"),
    ("rl.epsilon_e", "ee"),
    ("rl.epsilon_m", "em"),
    ("rl.num_steps", "ns"),
    ("rl.noise_level", "nl"),
    ("seed", "s"),
]

# ---------- Hyperparameter grid ----------
# The MPO-specific knobs worth sweeping are:
#   - epsilon_e: E-step KL budget (controls how aggressive the reweighting is)
#   - epsilon_m: M-step KL budget (controls how far the policy can move per step)
#   - group_size: number of action samples per state
#   - noise_level: flow sampling stochasticity
#   - lr: policy learning rate
applicable_configs: Dict[Union[str, tuple], List[Any]] = {
    "seed": [0],
    "log_interval": [25],
    "rl.num_critic_updates_per_batch": [2],
    "rl.policy_training_start_step": [200],
    "rl.online_ratio": [1.0],
    "collect.use_time_to_success_as_reward": [True],
    "rl.use_mc_returns": [False],
    "collect.num_initial_rollouts": [10],
    "rl.td_weight_schedule.switch_step": [-1],
    "rl.num_offline_pretraining_steps": [200],
    "rl.warm_start_critic_update_interval": [1],

    # --- MPO-specific knobs ---
    "lr_schedule.peak_lr": [1e-5, 5e-6],
    "batch_size": [256],
    "rl.group_size": [4, 8],
    "rl.epsilon_e": [0.05, 0.1],
    "rl.epsilon_m": [0.005, 0.01],
    "rl.noise_level": [0.2, 0.3],
    ("collect.tasks", "collect.eval_tasks"): [
        ("libero_90_14", "libero_90_14"),
        ("libero_90_59", "libero_90_59"),
        ("libero_90_64", "libero_90_64"),
        ("libero_90_82", "libero_90_82"),
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
    parser.add_argument("--duration", default="10:00:00", help="SLURM time limit")
    parser.add_argument("--partition", default="normal", help="SLURM partition")
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
    parser.add_argument("--num_rollouts", type=int, default=DEFAULT_NUM_ROLLOUTS)
    parser.add_argument(
        "--collect_interval", type=int, default=DEFAULT_COLLECT_INTERVAL
    )
    parser.add_argument("--train_env_num", type=int, default=DEFAULT_TRAIN_ENV_NUM)
    parser.add_argument("--eval_env_num", type=int, default=DEFAULT_EVAL_ENV_NUM)
    parser.add_argument("--eval_interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    parser.add_argument(
        "--num_eval_rollouts", type=int, default=DEFAULT_NUM_EVAL_ROLLOUTS
    )
    parser.add_argument("--num_train_steps", type=int, default=NUM_TRAIN_STEPS)
    parser.add_argument("--group_size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument(
        "--log_dir", default=DEFAULT_LOG_DIR, help="Directory for SLURM .out log files"
    )

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
            "rl.policy_training_start_step": args.policy_start_training,
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
            "rl.group_size": args.group_size,
        }
        flags.update(combo)

        # Ensure batch_size is divisible by group_size
        group_size = flags.get("rl.group_size", args.group_size)
        batch_size = flags.get("batch_size", DEFAULT_BATCH_SIZE)
        assert batch_size % group_size == 0, (
            f"batch_size ({batch_size}) must be divisible by group_size ({group_size})"
        )

        # Keep critic warmup aligned with policy start
        policy_start = flags["rl.policy_training_start_step"]
        flags["rl.critic_pre_training_steps"] = policy_start
        if flags.get("rl.td_weight_schedule.switch_step") == -1:
            flags["rl.td_weight_schedule.switch_step"] = policy_start

        flags["exp_name"] = algo_exp_name("mpo", flags, NAME_KEYS)

        cmd = generate_srun_command(SCRIPT, args.config_name, flags=flags)
        command_list.append(cmd)

    generate_run_commands(
        command_list,
        mode=args.mode,
        duration=args.duration,
        partition=args.partition,
        dry=args.dry,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
