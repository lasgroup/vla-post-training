#!/usr/bin/env python3
"""Launcher for AWR (Advantage-Weighted SFT) experiments.

Usage:
    ./scripts/awr/launcher.py                    # submit to SLURM with confirmation
    uv run scripts/awr/launcher.py --dry         # print commands without submitting
    ./scripts/awr/launcher.py --mode local       # run locally instead of SLURM
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from launcher_util import dict_permutations, generate_run_commands, generate_srun_command

SCRIPT = "scripts/awr/exp.py"
CONFIG_NAME = "pi05_libero_online_aw_sft"
PROJECT_NAME = "awr_sweep"

# ---------- Hyperparameter grid ----------
# Edit this dictionary to define your sweep.
# Each key maps to a list of values; all combinations are launched.
applicable_configs = {
    "seed": [0, 1, 2, 3, 4],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true", help="Print commands without submitting")
    parser.add_argument("--mode", default="swiss-ai", choices=["swiss-ai", "local"], help="Execution mode")
    parser.add_argument("--duration", default="03:30:00", help="SLURM time limit")
    parser.add_argument("--project_name", default=PROJECT_NAME, help="W&B project name")
    parser.add_argument("--config_name", default=CONFIG_NAME, help="Training config name")
    parser.add_argument("--log_interval", type=int, default=25, help="Logging interval")
    parser.add_argument("--checkpoint_base_dir", default=None, help="Checkpoint base directory")
    # AWR-specific defaults
    parser.add_argument("--buffer_capacity", type=int, default=250000)
    parser.add_argument("--policy_start_training", type=int, default=1000)
    parser.add_argument("--policy_update_interval", type=int, default=1)
    parser.add_argument("--num_rollouts", type=int, default=100)
    parser.add_argument("--collect_interval", type=int, default=400)
    args = parser.parse_args()

    command_list = []
    for combo in dict_permutations(applicable_configs):
        seed = combo["seed"]
        exp_name = f"{args.project_name}_seed{seed}"

        flags = {
            "overwrite": True,
            "exp_name": exp_name,
            "project_name": args.project_name,
            "seed": seed,
            "collect.seed": seed,
            "log_interval": args.log_interval,
            "collect.num_rollouts": args.num_rollouts,
            "collect.collect_interval": args.collect_interval,
            "rl.policy_training_start_step": args.policy_start_training,
            "rl.policy_update_interval": args.policy_update_interval,
            "rl.buffer_capacity": args.buffer_capacity,
        }
        if args.checkpoint_base_dir is not None:
            flags["checkpoint_base_dir"] = args.checkpoint_base_dir

        # Merge any extra hyperparameters from the grid (beyond seed)
        for k, v in combo.items():
            if k != "seed" and k not in flags:
                flags[k] = v

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
