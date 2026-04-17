#!/usr/bin/env python3
"""Launcher for Flow-MPO experiments.

Usage:
    ./scripts/flow_mpo_agent/launcher.py --project_name flow_mpo_sweep_nosoftmax
    ./scripts/flow_mpo_agent/launcher.py --project_name flow_mpo_sweep --dry
    ./scripts/flow_mpo_agent/launcher.py --project_name flow_mpo_sweep --mode local
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

SCRIPT = "scripts/flow_mpo_agent/exp.py"
CONFIG_NAME = "pi05_libero_online_flow_mpo_sft"
PROJECT_NAME = "flow_mpo_sweep"
DEFAULT_LOG_INTERVAL = 50
DEFAULT_SEED = 0
DEFAULT_BUFFER_CAPACITY = 250000
DEFAULT_POLICY_START_TRAINING = 900
DEFAULT_POLICY_UPDATE_INTERVAL = 1
DEFAULT_NUM_ROLLOUTS = 1
DEFAULT_COLLECT_INTERVAL = 300
DEFAULT_NUM_CRITIC_UPDATES_PER_BATCH = 10
DEFAULT_USE_TIME_TO_SUCCESS_AS_REWARD = True
DEFAULT_BATCH_SIZE = 64
DEFAULT_TRAIN_ENV_NUM = 1
DEFAULT_TASKS = ["libero_90_59"]
DEFAULT_EVAL_ENV_NUM = 4
DEFAULT_EVAL_INTERVAL = 300
DEFAULT_NUM_EVAL_ROLLOUTS = 32
NUM_TRAIN_STEPS = 5_000

# Short aliases for experiment names
NAME_KEYS = [
    ("collect.tasks", "t"),
    ("rl.policy_training_start_step", "ps"),
    ("rl.online_ratio", "or"),
    ("lr_schedule.peak_lr", "lr"),
    ("rl.beta", "b"),
    ("rl.weight_clip", "wc"),
    ("rl.clip_epsilon", "ce"),
    ("rl.num_steps", "ns"),
    ("rl.noise_level", "nl"),
    ("rl.use_ema_for_sampling", "ema"),
    ("rl.use_adaptive_advantage_scale", "aas"),
    ("seed", "s"),
]

# ---------- Hyperparameter grid ----------
# Stage 1: algorithm-specific knobs, one task, one seed.
# After finding the best setting, add shared knobs (lr, online_ratio, etc.)
# and multiple seeds/tasks in subsequent stages.
applicable_configs: Dict[Union[str, tuple], List[Any]] = {
    "seed": [0,1,2],
    "log_interval": [25],
    "rl.num_critic_updates_per_batch": [10],
    "rl.policy_training_start_step": [900],
    "rl.online_ratio": [1.0],
    "collect.use_time_to_success_as_reward": [True],
    "batch_size": [64],
    "rl.use_mc_returns": [False],
    "collect.num_initial_rollouts": [10],
    #"rl.td_weight_schedule.switch_step": [-1],
    "rl.store_success_episodes_only": [False],
    "rl.num_offline_pretraining_steps": [0],
    "rl.warm_start_critic_update_interval": [1],

    # --- flow-MPO-specific knobs ---
    "rl.reset_policy_params_to_ema_period": [None],
    #"rl.reset_optimizer_on_ema_reset": [True],
    #"rl.beta": [0.02, 0.05, 0.1],
    "rl.beta": [0.05],
    "rl.clip_epsilon": [0.2],
    "rl.num_steps": [5],
    "rl.noise_level": [0.3],
    "rl.use_adaptive_advantage_scale": [False, True],
    ("collect.tasks", "collect.eval_tasks"): [
        ("libero_90_14", "libero_90_14"),
        ("libero_90_59", "libero_90_59"),
        #("libero_90_64", "libero_90_64"),
        #("libero_90_82", "libero_90_82"),
    ],
    ("rl.td_weight_schedule.switch_step", "rl.td_weight_schedule.init_value", "rl.td_weight_schedule.end_value"): [
        (0, 0.5, 0.5),
        (900, 0.0, 1.0),
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
    parser.add_argument("--log_dir", default=DEFAULT_LOG_DIR, help="Directory for SLURM .out log files")
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

    args = parser.parse_args()

    combos = dict_permutations(applicable_configs)
    command_list = []
    job_names = []
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

        # Keep these in sync with policy_training_start_step
        policy_start = flags["rl.policy_training_start_step"]
        flags["rl.critic_pre_training_steps"] = policy_start
        if flags.get("rl.td_weight_schedule.switch_step") == -1:
            flags["rl.td_weight_schedule.switch_step"] = policy_start

        flags["exp_name"] = algo_exp_name("fmpo", flags, NAME_KEYS)

        cmd = generate_srun_command(SCRIPT, args.config_name, flags=flags)
        command_list.append(cmd)
        job_names.append(flags["exp_name"])

    generate_run_commands(
        command_list,
        mode=args.mode,
        duration=args.duration,
        partition=args.partition,
        dry=args.dry,
        log_dir=args.log_dir,
        job_names=job_names,
    )


if __name__ == "__main__":
    main()
