#!/usr/bin/env python3
"""Launcher for grouped-MPO experiments

Usage:
    ./scripts/grouped_mpo_agent/launcher.py --project_name vla_debug_april3
    ./scripts/grouped_mpo_agent/launcher.py --project_name my_project --dry
    ./scripts/grouped_mpo_agent/launcher.py --project_name my_project --mode local
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

SCRIPT = "scripts/mpo_agent/exp.py"
CONFIG_NAME = "pi05_libero_online_grouped_mpo_sft"
PROJECT_NAME = "grouped_mpo_agent_sweep"
DEFAULT_LOG_INTERVAL = 50
DEFAULT_SEED = 0
DEFAULT_BUFFER_CAPACITY = 250000
DEFAULT_POLICY_START_TRAINING = 1000
DEFAULT_POLICY_UPDATE_INTERVAL = 1
DEFAULT_NUM_ROLLOUTS = 1
DEFAULT_COLLECT_INTERVAL = 300
DEFAULT_NUM_CRITIC_UPDATES_PER_BATCH = 10
DEFAULT_USE_TIME_TO_SUCCESS_AS_REWARD = True
DEFAULT_BATCH_SIZE = 128
DEFAULT_TRAIN_ENV_NUM = 1
DEFAULT_TASKS = ["libero_90_59x1"]
DEFAULT_EVAL_TASKS = ["libero_90_59x4"]
DEFAULT_EVAL_ENV_NUM = 4
DEFAULT_EVAL_INTERVAL = 300
DEFAULT_NUM_EVAL_ROLLOUTS = 32
DEFAULT_GROUP_SIZE = 8
DEFAULT_NUM_STEPS = 5
DEFAULT_TD_WEIGHT_SWITCH_STEP = 1200
DEFAULT_TD_WEIGHT_RAMP_STEPS = 300
NUM_TRAIN_STEPS = 5_000
TASK_SWEEP: List[Dict[str, Any]] = [
    {
        "collect.tasks": ["libero_90_14x1"],
        "collect.eval_tasks": ["libero_90_14x4"],
    },
    {
        "collect.tasks": ["libero_90_59x1"],
        "collect.eval_tasks": ["libero_90_59x4"],
    },
    {
        "collect.tasks": ["libero_90_64x1"],
        "collect.eval_tasks": ["libero_90_64x4"],
    },
    {
        "collect.tasks": ["libero_90_82x1"],
        "collect.eval_tasks": ["libero_90_82x4"],
    },
]

applicable_configs: Dict[str, List[Any]] = {
    "seed": [0, 1],
    "log_interval": [25],
    "rl.num_critic_updates_per_batch": [10],
    "collect.use_time_to_success_as_reward": [True],
    "batch_size": [128],
    "collect.eval_interval": [100],
    "rl.policy_training_start_step": [900],
    "rl.online_ratio": [0.5, 1.0],
    "collect.num_initial_rollouts": [10],
    "rl.td_weight_schedule.switch_step": [1200],
    #"rl.save_all_episodes": [False, True],
    "rl.policy_only_successful": [False],
    "rl.use_deterministic_anchor": [True],
    "rl.drop_low_diversity_groups": [True],
    #"rl.align_critic_sampling": [True],
    "rl.normalize_adv": [False, True],
    "rl.freeze_critic_at_step": [1200],
    "rl.policy_update_interval": [1],
    "collect.collect_interval": [300],
    "lr_schedule.peak_lr": [5e-6, 3e-5],
    "rl.sft_anchor_coef": [0.0],
    "rl.min_advantage_std": [0.05],
    #"rl.weight_clip": [5.0, 20.0],
    #"rl.use_buffer_actions_for_loss": [False, True],
    "rl.kl_coef": [0.0, 0.01],
    "rl.reset_policy_params_to_ema_period": [100],
    "rl.reset_optimizer_on_ema_reset": [True],
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
    parser.add_argument("--duration", default="10:00:00", help="SLURM time limit")
    parser.add_argument("--project_name", default=PROJECT_NAME, help="W&B project name")
    parser.add_argument("--config_name", default=CONFIG_NAME, help="Training config name")
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
    parser.add_argument("--collect_interval", type=int, default=DEFAULT_COLLECT_INTERVAL)
    parser.add_argument("--train_env_num", type=int, default=DEFAULT_TRAIN_ENV_NUM)
    parser.add_argument("--eval_env_num", type=int, default=DEFAULT_EVAL_ENV_NUM)
    parser.add_argument("--eval_interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    parser.add_argument("--num_eval_rollouts", type=int, default=DEFAULT_NUM_EVAL_ROLLOUTS)
    parser.add_argument("--num_train_steps", type=int, default=NUM_TRAIN_STEPS)
    parser.add_argument("--group_size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--num_steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--td_weight_switch_step", type=int, default=DEFAULT_TD_WEIGHT_SWITCH_STEP)
    parser.add_argument("--td_weight_ramp_steps", type=int, default=DEFAULT_TD_WEIGHT_RAMP_STEPS)

    args = parser.parse_args()

    combos = dict_permutations(applicable_configs)
    command_list = []
    job_names = []
    tracked_name_keys = [
        "collect.tasks",
        *[k for k, v in applicable_configs.items() if len(v) > 1],
        "collect.eval_tasks",
    ]
    for idx, combo in enumerate(combos):
        for task_idx, task_flags in enumerate(TASK_SWEEP):
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
                "collect.eval_interval": args.eval_interval,
                "collect.num_eval_rollouts": args.num_eval_rollouts,
                "num_train_steps": args.num_train_steps,
                "rl.group_size": args.group_size,
                "rl.num_steps": args.num_steps,
            }
            default_name_flags = dict(flags)
            default_name_flags.update(TASK_SWEEP[0])
            flags.update(combo)
            flags.update(task_flags)

            policy_start = flags["rl.policy_training_start_step"]
            flags["rl.td_weight_schedule.ramp_steps"] = args.td_weight_ramp_steps
            flags["rl.critic_pre_training_steps"] = policy_start

            # Halve batch size when SFT anchor is active (extra memory for anchor forward pass).
            if flags.get("rl.sft_anchor_coef", 0.0) > 0.0:
                flags["batch_size"] = flags["batch_size"] // 2
            
            # Tie decay_lr to peak_lr:
            # - 5e-6 → constant LR (decay_lr=5e-6, default decay_steps)
            # - 3e-5 → cosine decay to 3e-7 in 500 steps
            if "lr_schedule.peak_lr" in flags:
                plr = flags["lr_schedule.peak_lr"]
                if plr == 3e-5:
                    flags["lr_schedule.decay_lr"] = 3e-7
                    flags["lr_schedule.decay_steps"] = 500
                else:
                    flags["lr_schedule.decay_lr"] = plr

            flags.setdefault(
                "exp_name",
                auto_exp_name(
                    args.project_name,
                    flags,
                    idx * len(TASK_SWEEP) + task_idx,
                    defaults=default_name_flags,
                    tracked_keys=tracked_name_keys,
                    algorithm_name="grouped_mpo",
                ),
            )

            cmd = generate_srun_command(SCRIPT, args.config_name, flags=flags)
            command_list.append(cmd)
            job_names.append(flags["exp_name"])

    generate_run_commands(
        command_list,
        mode=args.mode,
        duration=args.duration,
        dry=args.dry,
        job_names=job_names,
    )


if __name__ == "__main__":
    main()
