#!/usr/bin/env python3
"""Launcher for experiments.

Usage:
    ./scripts/launcher.py --config scripts/configs/filtered_sft.yaml
"""

import argparse
import datetime as dt
import itertools
import os
import secrets
import shlex
import shutil
import sys
from typing import Any, Dict, List, Optional
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Default SLURM settings matching existing bash scripts
DEFAULT_ACCOUNT = "ab037"
DEFAULT_ENVIRONMENT = "vla-post-training"
DEFAULT_DURATION = "11:59:00"
DEFAULT_PARTITION = "normal"
# GPU memory fraction reserved for JAX. 0.85 leaves headroom for the EGL/MuJoCo
# renderer during collection; raise (e.g. 0.9) when policy training needs more.
DEFAULT_MEM_FRACTION = 0.85
REQUEUE_EXIT_CODE = 42
# How many times in a row a job may crash (non-zero exit that is not the
# graceful requeue code) before the launcher stops retrying it.
MAX_CRASH_RETRIES = 3
RESULTS_DIR = f"/capstor/store/cscs/swissai/a0220/{os.environ.get('USER', 'unknown')}/results"


def generate_srun_command(
    script: str,
    config_name: str,
    mode: str = 'swiss-ai',
    flags: Optional[Dict[str, Any]] = None,
    account: str = DEFAULT_ACCOUNT,
    environment: str = DEFAULT_ENVIRONMENT,
    mem_fraction: float = DEFAULT_MEM_FRACTION,
) -> str:
    """Generate an srun command for a training script.

    Args:
        script: Path to the Python training script (relative to project root).
        config_name: Positional config name (e.g. "pi05_libero_online_flow_grpo_sft").
        flags: Dictionary of CLI flags and their values.
        account: SLURM account.
        environment: SLURM environment name.
        mem_fraction: XLA_PYTHON_CLIENT_MEM_FRACTION. Lower (e.g. 0.85) leaves more
            GPU for the EGL/MuJoCo renderer; higher (e.g. 0.9) gives JAX more room
            for policy training at the cost of render headroom.

    Returns:
        Full srun command string.
    """
    tokens = [
        "srun",
        f"--account={account}",
        f"--environment={environment}",
    ] if mode == 'swiss-ai' else []
    tokens += [
        # XLA_PYTHON_CLIENT_MEM_FRACTION trades GPU between JAX and the EGL/MuJoCo
        # renderer: lower leaves headroom for rendering (fewer read_pixels worker
        # crashes during collection), higher gives JAX room for policy training.
        "env",
        f"XLA_PYTHON_CLIENT_MEM_FRACTION={mem_fraction}",
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
        crash_state = f"{path}.crash_retries"
        script = f"""#!/bin/bash
set -euo pipefail

cd {shlex.quote(cwd)}

# Disable core dumps: the Molmo OpenGL renderer occasionally SIGABRTs and dumps
# multi-GB core files into cwd, which fills the home-dir quota and then breaks
# job startup (uv cache writes fail with EDQUOT).
ulimit -c 0

# Mount the host's EGL libraries into the container. Without the graphics
# capability the image's bundled libEGL_nvidia is used, which breaks headless
# rendering on GPUs 1-3 whenever it lags the host driver (since 2026-10-02).
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Consecutive-crash counter. Reset on any clean exit or planned requeue, so a
# long run that requeues many times at walltime keeps its full crash budget.
CRASH_STATE={shlex.quote(crash_state)}
MAX_CRASH_RETRIES={MAX_CRASH_RETRIES}

child_status=0
{command} || child_status=$?

if [[ "$child_status" -eq 0 ]]; then
  rm -f "$CRASH_STATE"
  exit 0
fi

if [[ "$child_status" -eq {REQUEUE_EXIT_CODE} ]]; then
  echo "[$(date --iso-8601=seconds)] Job ${{SLURM_JOB_ID}} requested requeue." >&2
  rm -f "$CRASH_STATE"
  if scontrol requeue "${{SLURM_JOB_ID}}"; then
    echo "[$(date --iso-8601=seconds)] Requeue submitted for job ${{SLURM_JOB_ID}}." >&2
    exit 0
  fi
  echo "[$(date --iso-8601=seconds)] Failed to requeue job ${{SLURM_JOB_ID}}." >&2
  exit "$child_status"
fi

# Unplanned crash (e.g. the Molmo env worker dying with EOFError, or the
# renderer SIGABRTing). These exit non-zero and are NOT the graceful requeue
# code, so without this branch SLURM marks the job FAILED and training stops.
# The run resumes from its last checkpoint, so retry a bounded number of times.
crashes=0
if [[ -f "$CRASH_STATE" ]]; then
  crashes=$(cat "$CRASH_STATE" 2>/dev/null || echo 0)
fi
if [[ "$crashes" -lt "$MAX_CRASH_RETRIES" ]]; then
  echo $((crashes + 1)) > "$CRASH_STATE"
  echo "[$(date --iso-8601=seconds)] Job ${{SLURM_JOB_ID}} crashed (status $child_status); requeue attempt $((crashes + 1))/$MAX_CRASH_RETRIES." >&2
  if scontrol requeue "${{SLURM_JOB_ID}}"; then
    exit 0
  fi
  echo "[$(date --iso-8601=seconds)] Failed to requeue job ${{SLURM_JOB_ID}} after crash." >&2
else
  echo "[$(date --iso-8601=seconds)] Job ${{SLURM_JOB_ID}} crashed $crashes times in a row; giving up." >&2
fi

exit "$child_status"
"""
    else:
        script = f"""#!/bin/bash
set -euo pipefail

cd {shlex.quote(cwd)}

# Disable core dumps (see requeue branch comment above).
ulimit -c 0

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
    mem_fraction: float = DEFAULT_MEM_FRACTION,
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
            os.makedirs(results_dir)
        except FileExistsError:
            # ask what to do if results_dir exists
            print(f"Directory {results_dir} exists. Delete? [yes/no]")
            if input().lower() in ["yes"]:
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
    command_list = [generate_srun_command(script, config_name, flags=combo, mode=mode, account=account, mem_fraction=mem_fraction) for combo in combos]
    if mode == "swiss-ai":

        bsub_cmd = f"sbatch --account={account} --time={duration} --partition={partition} "
        if requeue:
            bsub_cmd += "--requeue --open-mode=append "

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
    }
    for key, value in desired_values.items():
        if updated.get(key) != value:
            overrides[key] = (updated.get(key), value)
        updated[key] = value
    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/filtered_sft.yaml")
    parser.add_argument("--dry", action="store_true", help="Print commands without submitting")
    parser.add_argument("--mode", default="swiss-ai", choices=["swiss-ai", "local"], help="Execution mode")
    parser.add_argument("--duration", default="11:59:00", help="SLURM time limit")
    parser.add_argument("--partition", default="normal", help="SLURM partition")
    parser.add_argument("--account", default=DEFAULT_ACCOUNT, help=f"SLURM account / project to charge (default {DEFAULT_ACCOUNT}). Applies to both the outer sbatch and the inner srun.")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--skip_requeue", action="store_true", help="Submit requeue-safe resumable jobs")
    parser.add_argument("--mem_fraction", type=float, default=DEFAULT_MEM_FRACTION, help=f"XLA_PYTHON_CLIENT_MEM_FRACTION (default {DEFAULT_MEM_FRACTION}). Raise (e.g. 0.9) to give JAX more room for policy training; lowers renderer headroom.")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = config = yaml.load(f, Loader=yaml.FullLoader)

    combos = dict_permutations(config["params"])
    if not args.skip_requeue:
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
        account=args.account,
        dry=args.dry,
        prompt=not args.force,
        requeue=not args.skip_requeue,
        mem_fraction=args.mem_fraction,
    )


if __name__ == "__main__":
    main()