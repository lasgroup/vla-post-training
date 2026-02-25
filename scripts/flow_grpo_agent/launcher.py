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

# ---------- Hyperparameter grid ----------
# Keys can be any `_config.cli()` override.
# If this dict is empty, one run is launched with config defaults.
applicable_configs: Dict[str, List[Any]] = {
    # "seed": [0, 1, 2, 3, 4],
    # "collect.seed": [0, 1, 2, 3, 4],
    # "log_interval": [25],
    # "rl.normalize_adv": [True, False],
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
    parser.add_argument("--project_name", default=PROJECT_NAME, help="W&B project name")
    parser.add_argument("--config_name", default=CONFIG_NAME, help="Training config name")
    parser.add_argument("--log_interval", type=int, default=DEFAULT_LOG_INTERVAL)
    parser.add_argument(
        "--checkpoint_base_dir",
        default=DEFAULT_CHECKPOINT_BASE_DIR,
        help="Checkpoint base directory",
    )
    args = parser.parse_args()

    combos = dict_permutations(applicable_configs)
    command_list = []
    for idx, combo in enumerate(combos):
        # Keep defaults aligned with submit_train_flow_grpo.sh.
        flags: Dict[str, Any] = {
            "overwrite": True,
            "project_name": args.project_name,
            "seed": DEFAULT_SEED,
            "collect.seed": DEFAULT_SEED,
            "log_interval": args.log_interval,
            "checkpoint_base_dir": args.checkpoint_base_dir,
        }
        flags.update(combo)
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
