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
DEFAULT_SAVE_INTERVAL = 100_000
DEFAULT_SEED = 0
DEFAULT_BUFFER_CAPACITY = 250000
DEFAULT_POLICY_START_TRAINING = 1000
DEFAULT_POLICY_UPDATE_INTERVAL = 10
DEFAULT_NUM_ROLLOUTS = 4
DEFAULT_COLLECT_INTERVAL = 300
DEFAULT_NUM_CRITIC_UPDATES_PER_BATCH = 10
DEFAULT_USE_TIME_TO_SUCCESS_AS_REWARD = True
DEFAULT_BATCH_SIZE = 256
DEFAULT_N_SAMPLES = 1
DEFAULT_TRAIN_ENV_NUM = 4
DEFAULT_TASKS = ["libero_90_59"]
DEFAULT_EVAL_TASKS = ["libero_90_59"]
DEFAULT_EVAL_ENV_NUM = 4
DEFAULT_EVAL_INTERVAL = 10000
DEFAULT_NUM_EVAL_ROLLOUTS = 16
NUM_TRAIN_STEPS = 1_000_000
DEFAULT_CRITIC_TRAINING_START_STEP = 0
DEFAULT_CRITIC_INFERENCE_START_STEP = 900
DEFAULT_NUM_CPUS = 16

# ---------- Hyperparameter grid ----------
# Keys can be any `_config.cli()` override.
# If this dict is empty, one run is launched with config defaults.
applicable_configs: Dict[Union[str, tuple], List[Any]] = {
    "seed": [0, 1, 2],
    "log_interval": [25],
    "batch_size": [256],
    # Critic training schedule
    "collect.use_time_to_success_as_reward": [True],
    "rl.critic.num_updates_per_batch": [1],
    "rl.critic.batch_size": [1024],
    "rl.critic.td_weight_schedule.init_value": [1.0],
    "rl.critic.td_weight_schedule.end_value": [1.0],
    "rl.critic.td_weight_schedule.switch_step": [500_000],
    "rl.critic.num_value_bins": [1],
    "collect.fix_mc_returns": [True],
    # Policy training
    "rl.policy.training_start_step": [900],
    "rl.online_ratio": [1.0],
    "rl.policy.reset_params_to_ema_period": [500],
    # Best-of-N collection
    "collect.store_prefix_rep": [True],
    "rl.n_samples": [1],
    "rl.critic_inference_start_step": [0],
    # AWR-specific
    "rl.use_mc_returns": [False],
    "lr_schedule.value": [2.5e-5],
    "rl.store_success_episodes_only": [False],
    ("collect.tasks", "collect.eval_tasks"): [
        (["libero_90_59", "libero_90_59"]),
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

    # AWR defaults
    parser.add_argument("--buffer_capacity", type=int, default=DEFAULT_BUFFER_CAPACITY)
    parser.add_argument(
        "--policy_start_training", type=int, default=DEFAULT_POLICY_START_TRAINING
    )
    parser.add_argument(
        "--policy_update_interval", type=int, default=DEFAULT_POLICY_UPDATE_INTERVAL
    )
    parser.add_argument("--num_rollouts", type=int, default=DEFAULT_NUM_ROLLOUTS)
    parser.add_argument("--collect_interval", type=int, default=DEFAULT_COLLECT_INTERVAL)
    parser.add_argument("--n_samples", type=int, default=DEFAULT_N_SAMPLES,
                        help="Best-of-N samples during collection (1=disabled)")
    parser.add_argument("--filtered_sft_weight", type=float, default=0.0,
                        help="Weight for the filtered SFT loss term on successful transitions (0=disabled)")
    parser.add_argument("--train_env_num", type=int, default=DEFAULT_TRAIN_ENV_NUM)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--eval_tasks", nargs="+", default=DEFAULT_EVAL_TASKS)
    parser.add_argument("--eval_env_num", type=int, default=DEFAULT_EVAL_ENV_NUM)
    parser.add_argument("--eval_interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    parser.add_argument("--num_eval_rollouts", type=int, default=DEFAULT_NUM_EVAL_ROLLOUTS)
    parser.add_argument("--num_train_steps", type=int, default=NUM_TRAIN_STEPS)
    parser.add_argument("--save_interval", type=int, default=DEFAULT_SAVE_INTERVAL)
    parser.add_argument("--store_prefix_rep", action="store_true",
                        help="Store pi0 prefix embeddings in the replay buffer")
    parser.add_argument("--critic_training_start_step", type=int, default=DEFAULT_CRITIC_TRAINING_START_STEP)
    parser.add_argument("--critic_inference_start_step", type=int, default=DEFAULT_CRITIC_INFERENCE_START_STEP,
                        help="Step at which critic-guided best-of-N sampling begins")
    parser.add_argument("--num_value_bins", type=int, default=1,
                        help="1=regression, >1=categorical over return bins")
    parser.add_argument("--value_target_type", default="one_hot", choices=["one_hot", "two_hot"])
    parser.add_argument("--critic_batch_size", type=int, default=None,
                        help="Batch size for critic updates (None = same as policy batch_size)")
    parser.add_argument("--num_cpus", type=int, default=DEFAULT_NUM_CPUS)
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
            "rl.policy.training_start_step": args.policy_start_training,
            "rl.policy.update_interval": args.policy_update_interval,
            "rl.buffer_capacity": args.buffer_capacity,
            "rl.critic.num_updates_per_batch": DEFAULT_NUM_CRITIC_UPDATES_PER_BATCH,
            "rl.n_samples": args.n_samples,
            "rl.critic.training_start_step": args.critic_training_start_step,
            "rl.critic_inference_start_step": args.critic_inference_start_step,
            "rl.filtered_sft_weight": args.filtered_sft_weight,
            "collect.use_time_to_success_as_reward": DEFAULT_USE_TIME_TO_SUCCESS_AS_REWARD,
            "batch_size": DEFAULT_BATCH_SIZE,
            "collect.env_num": args.train_env_num,
            "collect.eval_env_num": args.eval_env_num,
            "collect.tasks": args.tasks,
            "collect.eval_tasks": args.eval_tasks,
            "collect.eval_interval": args.eval_interval,
            "collect.num_eval_rollouts": args.num_eval_rollouts,
            "num_train_steps": args.num_train_steps,
            "rl.critic.num_value_bins": args.num_value_bins,
            "rl.critic.value_target_type": args.value_target_type,
            "rl.critic.batch_size": args.critic_batch_size,
            "save_interval": args.save_interval,
            "collect.store_prefix_rep": args.store_prefix_rep,
        }
        flags.update(combo)
        flags = apply_requeue_flags(flags, enabled=args.requeue)

        # Keep these in sync with policy_training_start_step
        policy_start = flags["rl.policy.training_start_step"]
        flags["rl.critic.pre_training_steps"] = policy_start
        if flags["rl.critic.td_weight_schedule.switch_step"] == -1:
            flags["rl.critic.td_weight_schedule.switch_step"] = policy_start
        # If we only train on mc returns, no target critic needed for policy updates.
        elif flags["rl.critic.td_weight_schedule.switch_step"] >= NUM_TRAIN_STEPS:
            flags["rl.critic.use_ema"] = False

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
        num_cpus=args.num_cpus,
        dry=args.dry,
        log_dir=args.log_dir,
        requeue=args.requeue,
    )


if __name__ == "__main__":
    main()
