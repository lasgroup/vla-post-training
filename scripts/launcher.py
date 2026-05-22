#!/usr/bin/env python3
"""Launcher for experiments.

Usage:
    python scripts/launcher.py --config scripts/configs/filtered_sft.yaml
"""

import argparse
import datetime as dt
import itertools
import logging
import os
import secrets
import shlex
import shutil
import sys
from typing import Any, Dict, List, Optional
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Default SLURM settings matching existing bash scripts
DEFAULT_ACCOUNT = "a143"
DEFAULT_ENVIRONMENT = "vla-post-training"
DEFAULT_DURATION = "03:30:00"
DEFAULT_PARTITION = "normal"
DEFAULT_REQUEUE_SIGNAL_LEAD_SECONDS = 120
RESULTS_DIR = f"/capstor/scratch/cscs/{os.environ.get('USER', 'unknown')}/results"


def generate_srun_command(
    script: str,
    config_name: str,
    mode: str = 'swissai',
    flags: Optional[Dict[str, Any]] = None,
    account: str = DEFAULT_ACCOUNT,
    environment: str = DEFAULT_ENVIRONMENT,
) -> str:
    """Generate an srun command for a training script.

    Args:
        script: Path to the Python training script (relative to project root).
        config_name: Positional config name (e.g. "pi05_libero_online_flow_grpo_sft").
        flags: Dictionary of CLI flags and their values.
        account: SLURM account.
        environment: SLURM environment name.

    Returns:
        Full srun command string.
    """
    tokens = [
        "srun",
        f"--account={account}",
        f"--environment={environment}",
    ] if mode == 'swissai' else []
    tokens += [
        "uv",
        "run",
        script,
        config_name,
    ]
    tokens.extend(flags_to_cli_tokens(flags))
    return " ".join(shlex.quote(str(tok)) for tok in tokens)


def _write_sbatch_script(path: str, *, command: str, requeue: bool) -> None:
    cwd = os.getcwd()
    if requeue:
        script = f"""#!/bin/bash
set -euo pipefail

cd {shlex.quote(cwd)}

requeue_requested=0
requeue_submitted=0
child_pid=""

handle_term() {{
  if [[ "$requeue_requested" -eq 1 ]]; then
    return
  fi
  requeue_requested=1
  echo "[$(date --iso-8601=seconds)] Received SIGTERM in batch shell for job ${{SLURM_JOB_ID}}; requesting requeue." >&2
  if scontrol requeue "${{SLURM_JOB_ID}}"; then
    requeue_submitted=1
    echo "[$(date --iso-8601=seconds)] Requeue submitted for job ${{SLURM_JOB_ID}}." >&2
  else
    echo "[$(date --iso-8601=seconds)] Failed to requeue job ${{SLURM_JOB_ID}}." >&2
  fi
  if [[ -n "$child_pid" ]]; then
    kill -TERM "$child_pid" 2>/dev/null || true
  fi
}}

trap handle_term TERM

{command} &
child_pid=$!
child_status=0
wait "$child_pid" || child_status=$?

if [[ "$requeue_requested" -eq 1 ]]; then
  if [[ "$requeue_submitted" -eq 1 ]]; then
    exit 0
  fi
  exit "$child_status"
fi

exit "$child_status"
"""
    else:
        script = f"""#!/bin/bash
set -euo pipefail

cd {shlex.quote(cwd)}

exec {command}
"""

    with open(path, "w", encoding="ascii") as f:
        f.write(script)
    os.chmod(path, 0o755)

def auto_exp_name(project_name: str, combo: Dict[str, Any], run_idx: int) -> str:
    """Generate a unique experiment name suitable for checkpoint directories."""
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = secrets.token_hex(3)
    if "seed" in combo:
        return f"{project_name}_{timestamp}_{suffix}_seed{combo['seed']}"
    return f"{project_name}_{timestamp}_{suffix}_run{run_idx}"


def _bool_flag_name(flag: str, value: bool) -> str:
    """Build tyro-compatible bool flag names for flat and nested fields.

    Examples:
        rl.normalize_adv=True  -> --rl.normalize_adv
        rl.normalize_adv=False -> --rl.no-normalize_adv
        overwrite=False        -> --no-overwrite
    """
    if value:
        return flag
    if "." in flag:
        prefix, leaf = flag.rsplit(".", 1)
        return f"{prefix}.no-{leaf}"
    return f"no-{flag}"


def flag_to_cli_tokens(flag: str, value: Any) -> List[str]:
    """Convert a single override into CLI token(s) for tyro-compatible parsers."""
    if value is None:
        return []
    if isinstance(value, bool):
        return [f"--{_bool_flag_name(flag, value)}"]
    if isinstance(value, (list, tuple)):
        return [f"--{flag}", *[str(v) for v in value]]
    return [f"--{flag}", str(value)]


def flags_to_cli_tokens(flags: Optional[Dict[str, Any]]) -> List[str]:
    """Convert an overrides dict to CLI tokens, preserving key order."""
    if not flags:
        return []
    tokens: List[str] = []
    for flag, value in flags.items():
        tokens.extend(flag_to_cli_tokens(flag, value))
    return tokens


def generate_run_commands(
    combos: List[str],
    script: str,
    config_name: str,
    result_dir: str,
    project_name: str,
    group_name: str,
    duration: str = DEFAULT_DURATION,
    partition: str = DEFAULT_PARTITION,
    account: str = DEFAULT_ACCOUNT,
    mode: str = "swiss-ai",
    dry: bool = False,
    prompt: bool = True,
    requeue: bool = False,
) -> None:
    """Submit or run a list of commands.

    Args:
        combos: List of dictionaries containing command-line arguments.
        script: Path to the Python training script (relative to project root).
        config_name: Positional config name.
        result_dir: Base directory for results (checkpoints, logs, etc.).
        project_name: Name of the project.
        group_name: Name of the experiment group.
        duration: Time limit for SLURM jobs.
        account: SLURM account.
        mode: "swiss-ai" for sbatch submission, "local" for sequential local execution.
        dry: If True, only print commands without executing.
        prompt: If True, ask for confirmation before submitting.
    """

    assert mode in ["swiss-ai", "local"], f"Unknown mode: {mode}"

    # update combos with checkpoint_base_dir
    results_dir = os.path.join(result_dir, '_'.join([project_name, group_name]))
    for i, combo in enumerate(combos):
        combo["checkpoint_base_dir"] = os.path.join(results_dir, str(i))
        combo["project_name"] = project_name
        combo["group_name"] = group_name
        combo["exp_name"] = auto_exp_name(project_name, combo, i)
    
    if not dry:
        # create results directory, handling existing directory
        try:
            os.mkdir(results_dir)
        except FileExistsError:
            # ask what to do if results_dir exists
            print(f"Directory {results_dir} exists. Delete?")
            if input().lower() in ["y", "yes"]:
                print("Deleting result directory.")
                shutil.rmtree(results_dir, ignore_errors=True)
                os.mkdir(results_dir)
            else:
                print("Exiting.")
                exit(0)
        print(f"Results directory: {results_dir}")
        for combo in combos:
            os.mkdir(combo["checkpoint_base_dir"])

        # dump git info for reproducibility
        commit_hash = os.popen("git rev-parse HEAD").read().strip()
        with open(os.path.join(results_dir, "commit_hash.txt"), "w") as f:
            f.write(commit_hash)
        git_diff = os.popen("git diff").read()
        with open(os.path.join(results_dir, "git_diff.txt"), "w") as f:
            f.write(git_diff)

    # create command list
    command_list = [generate_srun_command(script, config_name, flags=combo, mode=mode) for combo in combos]
    if mode == "swiss-ai":

        bsub_cmd = f"sbatch --account={account} --time={duration} --partition={partition} "
        if requeue:
            bsub_cmd += f"--requeue --signal=B:TERM@{DEFAULT_REQUEUE_SIGNAL_LEAD_SECONDS} "

        cluster_cmds = []
        for i, (cmd, combo) in enumerate(zip(command_list, combos)):
            script_path = os.path.join(combo["checkpoint_base_dir"], f"job_{i:04d}.sbatch.sh")
            if not dry:
                _write_sbatch_script(script_path, command=cmd, requeue=requeue)
            cluster_cmds.append(bsub_cmd + f"--output={combo['checkpoint_base_dir']}/slurm-%j.out " + shlex.quote(script_path))
        command_list = cluster_cmds

    if dry:
        [print(cmd) for cmd in command_list]
        return

    if mode == "local":
        print("Running the first job locally.")
        os.system(command_list[0])
        return

    # submit jobs
    if prompt:
        answer = input(
            f"About to run {len(command_list)} jobs. Proceed? [yes/no] "
        )
        if answer != "yes":
            print("Aborting.")
            return

    for cmd in command_list:
        os.system(cmd)


def dict_permutations(d: dict) -> List[dict]:
    """Generate all combinations from a dict of lists (cartesian product).

    Keys can be str (single param) or tuple of str (grouped params swept together).
    For tuple keys, each value must be a list of lists: one inner list per combo,
    with one element per key in the tuple.  E.g.::

        {("collect.tasks", "collect.eval_tasks"): [[["t1"], ["t1"]], [["t2"], ["t2"]]]}

    Raises ValueError if the same parameter appears under more than one key.
    """
    seen: set = set()
    for k in d:
        for key in ((k,) if isinstance(k, str) else k):
            if key in seen:
                raise ValueError(f"Conflicting key in grid: '{key}'")
            seen.add(key)

    groups = [(([k], [[v] for v in vals]) if isinstance(k, str) else (list(k), vals))
              for k, vals in d.items()]
    result = []
    for combo in itertools.product(*[g[1] for g in groups]):
        flat = {}
        for (keys, _), vals in zip(groups, combo):
            flat.update(zip(keys, vals))
        result.append(flat)
    return result


def apply_requeue_flags(flags: Dict[str, Any]) -> Dict[str, Any]:
    updated = dict(flags)
    overrides = {}
    desired_values = {
        "resume": True,
        "overwrite": False,
        "requeue": True,
    }
    for key, value in desired_values.items():
        if updated.get(key) != value:
            overrides[key] = (updated.get(key), value)
        updated[key] = value
    if overrides:
        logging.warning(
            "Requeue enabled: overriding flags for resumable execution: %s",
            ", ".join(
                f"{key}={old!r}->{new!r}" for key, (old, new) in overrides.items()
            ),
        )
    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/filtered_sft.yaml")
    parser.add_argument("--dry", action="store_true", help="Print commands without submitting")
    parser.add_argument("--mode", default="swiss-ai", choices=["swiss-ai", "local"], help="Execution mode")
    parser.add_argument("--duration", default="03:30:00", help="SLURM time limit")
    parser.add_argument("--partition", default="normal", help="SLURM partition")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--requeue", action="store_true", help="Submit requeue-safe resumable jobs")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = config = yaml.load(f, Loader=yaml.FullLoader)

    combos = dict_permutations(config["params"])
    if args.requeue:
        combos = [apply_requeue_flags(combo) for combo in combos]

    generate_run_commands(
        combos,
        script=config["script"],
        config_name=config["config_name"],
        result_dir=RESULTS_DIR,
        project_name=config["project_name"],
        group_name=config["group_name"],
        mode=args.mode,
        duration=args.duration,
        partition=args.partition,
        dry=args.dry,
        prompt=not args.force,
        requeue=args.requeue
    )


if __name__ == "__main__":
    main()