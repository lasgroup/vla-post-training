#!/usr/bin/env python3
"""Launcher for Flow-PG experiments.

Holds AWR's tuned defaults constant and sweeps Flow-PG's distinctive knobs:
kl_coef (KL penalty against EMA), clip_epsilon (PPO trust region — newly
added after the bug fix that introduced the importance ratio), num_steps,
noise_level.

Usage:
    ./scripts/flow_pg_agent/launcher.py --project_name flow_pg_sweep
    ./scripts/flow_pg_agent/launcher.py --project_name flow_pg_sweep --dry
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

SCRIPT = "scripts/flow_pg_agent/exp.py"
CONFIG_NAME = "pi05_libero_online_flow_pg_sft"
PROJECT_NAME = "flow_pg_sweep"
NUM_TRAIN_STEPS = 10_000

applicable_configs: Dict[Union[str, tuple], List[Any]] = {
    "seed": [0, 1, 2],
    "log_interval": [25],
    "batch_size": [256],

    # AWR defaults — frozen.
    "rl.beta": [0.2],
    "rl.normalizer_config.ema_weight": [0.99],
    "rl.use_mc_returns": [False],
    "rl.policy_training_start_step": [900],
    "rl.online_ratio": [1.0],
    "rl.reset_policy_params_to_ema_period": [500],
    "rl.store_success_episodes_only": [False],
    "rl.num_critic_updates_per_batch": [10],
    "lr_schedule.value": [2.5e-5],
    "collect.num_initial_rollouts": [5],
    "collect.num_rollouts": [8],
    "collect.use_time_to_success_as_reward": [True],
    "rl.td_weight_schedule.init_value": [0.5],
    "rl.td_weight_schedule.end_value": [0.5],
    "rl.td_weight_schedule.switch_step": [500_000],

    # Flow-PG knobs
    "rl.kl_coef": [0.0, 0.01, 0.1],
    "rl.clip_epsilon": [0.2],
    "rl.num_steps": [10],
    "rl.noise_level": [0.3],
    "rl.use_ema_for_sampling": [True],
    "rl.store_buffer_actions_in_batch": [False],

    ("collect.tasks", "collect.eval_tasks"): [
        ("libero_90_43", "libero_90_43"),
        ("libero_90_47", "libero_90_47"),
        ("libero_90_59", "libero_90_59"),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--mode", default="swiss-ai", choices=["swiss-ai", "local"])
    parser.add_argument("--duration", default="03:30:00")
    parser.add_argument("--partition", default="normal")
    parser.add_argument("--project_name", default=PROJECT_NAME)
    parser.add_argument("--group", default=None)
    parser.add_argument("--config_name", default=CONFIG_NAME)
    parser.add_argument("--checkpoint_base_dir", default=DEFAULT_CHECKPOINT_BASE_DIR)
    parser.add_argument("--log_dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--num_train_steps", type=int, default=NUM_TRAIN_STEPS)
    parser.add_argument("--requeue", action="store_true")
    args = parser.parse_args()

    combos = dict_permutations(applicable_configs)
    command_list: List[Dict[str, Any]] = []
    for idx, combo in enumerate(combos):
        flags: Dict[str, Any] = {
            "overwrite": True,
            "project_name": args.project_name,
            "group": args.group,
            "checkpoint_base_dir": args.checkpoint_base_dir,
            "num_train_steps": args.num_train_steps,
            "collect.env_num": 4,
            "collect.eval_env_num": 4,
            "collect.eval_interval": 2000,
            "collect.num_eval_rollouts": 8,
            "collect.collect_interval": 300,
            "rl.policy_update_interval": 1,
            "rl.buffer_capacity": 250_000,
        }
        flags.update(combo)
        flags = apply_requeue_flags(flags, enabled=args.requeue)

        policy_start = flags["rl.policy_training_start_step"]
        flags["rl.critic_pre_training_steps"] = policy_start

        flags.setdefault("exp_name", auto_exp_name(args.project_name, flags, idx))
        command_list.append(flags)

    if args.requeue:
        validate_unique_exp_names(command_list)

    rendered = [
        generate_srun_command(SCRIPT, args.config_name, flags=flags)
        for flags in command_list
    ]
    generate_run_commands(
        rendered,
        mode=args.mode,
        duration=args.duration,
        partition=args.partition,
        dry=args.dry,
        log_dir=args.log_dir,
        requeue=args.requeue,
    )


if __name__ == "__main__":
    main()
