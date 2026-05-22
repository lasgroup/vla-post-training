#!/usr/bin/env python3
"""Launcher for PARL agent experiments (AWR base + periodic SFT anchoring).

Usage:
    ./scripts/parl_agent/launcher.py --project_name my_project
    ./scripts/parl_agent/launcher.py --project_name my_project --dry
    ./scripts/parl_agent/launcher.py --project_name my_project --mode local
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

SCRIPT = "scripts/parl_agent/exp.py"
CONFIG_NAME = "pi05_libero_online_parl"
PROJECT_NAME = "parl_agent_sweep"
DEFAULT_LOG_INTERVAL = 50
DEFAULT_SAVE_INTERVAL = 100_000
DEFAULT_SEED = 0
DEFAULT_BUFFER_CAPACITY = 250_000
DEFAULT_PARL_BUFFER_CAPACITY = 50_000
DEFAULT_POLICY_START_TRAINING = 1_000
DEFAULT_POLICY_UPDATE_INTERVAL = 1
DEFAULT_NUM_ROLLOUTS = 1
DEFAULT_COLLECT_INTERVAL = 300
DEFAULT_NUM_CRITIC_UPDATES_PER_BATCH = 10
DEFAULT_USE_TIME_TO_SUCCESS_AS_REWARD = True
DEFAULT_BATCH_SIZE = 256
DEFAULT_N_SAMPLES = 1
DEFAULT_TRAIN_ENV_NUM = 4
DEFAULT_TASKS = ["libero_90_59"]
DEFAULT_EVAL_TASKS = ["libero_90_59"]
DEFAULT_EVAL_ENV_NUM = 4
DEFAULT_EVAL_INTERVAL = 300
DEFAULT_NUM_EVAL_ROLLOUTS = 8
NUM_TRAIN_STEPS = 5_000
DEFAULT_CRITIC_TRAINING_START_STEP = 0
DEFAULT_CRITIC_INFERENCE_START_STEP = 0
DEFAULT_PARL_FREQUENCY = 100
DEFAULT_PARL_NUM_UPDATES = 1
DEFAULT_NUM_CPUS = 16

# ---------- Hyperparameter grid ----------
applicable_configs: Dict[Union[str, tuple], List[Any]] = {
    "seed": [0, 1, 2],
    "log_interval": [25],
    "batch_size": [256],
    # Critic training schedule
    "collect.use_time_to_success_as_reward": [True],
    "rl.critic.num_updates_per_batch": [1],
    "rl.critic.td_weight_schedule.init_value": [0.5],
    "rl.critic.td_weight_schedule.end_value": [0.5],
    "rl.critic.td_weight_schedule.switch_step": [500_000],
    "rl.critic.num_value_bins": [1],
    # Policy training
    "rl.policy.training_start_step": [900],
    "rl.online_ratio": [1.0],
    "rl.policy.reset_params_to_ema_period": [500],
    # AWR-specific
    "rl.use_mc_returns": [False],
    "lr_schedule.value": [2.5e-5],
    "rl.store_success_episodes_only": [False],
    # PARL-specific
    "parl.frequency": [25],
    "parl.store_success_only": [True],
    "parl.num_updates": [1],
    ("collect.tasks", "collect.eval_tasks"): [
        (["libero_90_1-14", "libero_90_16-89"], ["libero_90_1-14", "libero_90_16-89"]),
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
    parser.add_argument("--config_name", default=CONFIG_NAME, help="Training config name")
    parser.add_argument("--log_interval", type=int, default=DEFAULT_LOG_INTERVAL)
    parser.add_argument(
        "--checkpoint_base_dir",
        default=DEFAULT_CHECKPOINT_BASE_DIR,
        help="Checkpoint base directory",
    )
    parser.add_argument("--log_dir", default=DEFAULT_LOG_DIR, help="Directory for SLURM .out log files")

    # AWR / RL args
    parser.add_argument("--buffer_capacity", type=int, default=DEFAULT_BUFFER_CAPACITY)
    parser.add_argument("--policy_start_training", type=int, default=DEFAULT_POLICY_START_TRAINING)
    parser.add_argument("--policy_update_interval", type=int, default=DEFAULT_POLICY_UPDATE_INTERVAL)
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
    parser.add_argument("--save_interval", type=int, default=DEFAULT_SAVE_INTERVAL)
    parser.add_argument("--critic_training_start_step", type=int, default=DEFAULT_CRITIC_TRAINING_START_STEP)
    parser.add_argument("--critic_inference_start_step", type=int, default=DEFAULT_CRITIC_INFERENCE_START_STEP)
    parser.add_argument("--num_value_bins", type=int, default=1)
    parser.add_argument("--value_target_type", default="one_hot", choices=["one_hot", "two_hot"])
    parser.add_argument("--critic_batch_size", type=int, default=None)
    parser.add_argument("--filtered_sft_weight", type=float, default=0.0)
    parser.add_argument("--num_cpus", type=int, default=DEFAULT_NUM_CPUS)

    # PARL-specific args
    parser.add_argument("--parl_frequency", type=int, default=DEFAULT_PARL_FREQUENCY,
                        help="Run SFT every N training steps instead of base RL")
    parser.add_argument("--parl_buffer_capacity", type=int, default=DEFAULT_PARL_BUFFER_CAPACITY,
                        help="Capacity of PARL's dedicated SFT buffer")
    parser.add_argument("--parl_num_updates", type=int, default=DEFAULT_PARL_NUM_UPDATES,
                        help="Number of SFT gradient steps per PARL step")
    parser.add_argument("--parl_batch_size", type=int, default=None,
                        help="Batch size for PARL SFT updates (None = same as policy batch_size)")
    parser.add_argument("--parl_store_success_only", action="store_true", default=True,
                        help="Only store successful episodes in the PARL buffer")
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
            # PARL
            "parl.frequency": args.parl_frequency,
            "parl.buffer_capacity": args.parl_buffer_capacity,
            "parl.num_updates": args.parl_num_updates,
            "parl.batch_size": args.parl_batch_size,
            "parl.store_success_only": args.parl_store_success_only,
        }
        flags.update(combo)
        flags = apply_requeue_flags(flags, enabled=args.requeue)

        # Keep critic pre_training_steps in sync with policy start
        policy_start = flags["rl.policy.training_start_step"]
        flags["rl.critic.pre_training_steps"] = policy_start
        if flags["rl.critic.td_weight_schedule.switch_step"] == -1:
            flags["rl.critic.td_weight_schedule.switch_step"] = policy_start
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
