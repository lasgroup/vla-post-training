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
import subprocess
import sys
from typing import Any, Dict, List, Optional
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Default SLURM settings matching existing bash scripts
DEFAULT_ACCOUNT = "a0220"
DEFAULT_ENVIRONMENT = "vla-post-training"
DEFAULT_DURATION = "11:59:00"
DEFAULT_PARTITION = "normal"
REQUEUE_EXIT_CODE = 42
RESULTS_DIR = f"/capstor/store/cscs/swissai/a0220/{os.environ.get('USER', 'unknown')}/results"

# Euler (ETH) cluster settings
_EULER_USER = os.environ.get("USER", "unknown")
EULER_RESULTS_DIR = f"/cluster/scratch/{_EULER_USER}/results"
EULER_CACHE_DIR = f"/cluster/scratch/{_EULER_USER}/openpi_cache"
DEFAULT_EULER_GPU = "a100_80gb"
DEFAULT_EULER_MEM_PER_CPU = "8G"
DEFAULT_EULER_CPUS = 8
DEFAULT_EULER_NUM_GPUS = 1

# GCS assets required for training: (url, path relative to cache dir)
_GCS_ASSETS = [
    ("gs://big_vision/paligemma_tokenizer.model", "big_vision/paligemma_tokenizer.model"),
    # LIBERO checkpoint (used by all pi05_libero_* configs)
    ("gs://openpi-assets/checkpoints/pi05_libero/params", "openpi-assets/checkpoints/pi05_libero/params"),
    ("gs://openpi-assets/checkpoints/pi05_libero/assets", "openpi-assets/checkpoints/pi05_libero/assets"),
    # Molmo / DROID checkpoint (used by all pi05_molmo_* configs)
    ("gs://openpi-assets/checkpoints/pi05_droid_jointpos/params", "openpi-assets/checkpoints/pi05_droid_jointpos/params"),
    ("gs://openpi-assets/checkpoints/pi05_droid_jointpos/assets", "openpi-assets/checkpoints/pi05_droid_jointpos/assets"),
]


def _libero_assets_present() -> bool:
    """Return True if LIBERO scene assets are installed in the hf-libero package dir."""
    import importlib.util
    import pathlib

    spec = importlib.util.find_spec("libero.libero")
    if spec is None or spec.origin is None:
        return True  # libero not installed; nothing to check
    scenes_dir = pathlib.Path(spec.origin).parent / "assets" / "scenes"
    return scenes_dir.exists() and any(scenes_dir.iterdir())


def ensure_assets(cache_dir: str) -> None:
    """Check if all required assets are cached/installed; download any missing ones.

    Checks both GCS assets (openpi checkpoints) and LIBERO scene assets (hf-libero
    ships without them and downloads from lerobot/libero-assets on first use).
    Runs on the calling machine (login node), not inside the submitted job.
    """
    missing_gcs = [url for url, rel in _GCS_ASSETS if not os.path.exists(os.path.join(cache_dir, rel))]
    missing_libero = not _libero_assets_present()

    if not missing_gcs and not missing_libero:
        print(f"All assets present in {cache_dir}.")
        return

    if missing_gcs:
        print(f"{len(missing_gcs)} GCS asset(s) missing from {cache_dir}, downloading now...")
    if missing_libero:
        print("LIBERO scene assets missing from hf-libero package, downloading now...")

    download_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_assets.py")
    subprocess.run([sys.executable, download_script, "--cache_dir", cache_dir], check=True)


def generate_srun_command(
    script: str,
    config_name: str,
    mode: str = 'swiss-ai',
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
    ] if mode == 'swiss-ai' else []
    tokens += [
        "uv",
        "run",
        script,
        config_name,
    ]
    tokens.extend(flags_to_cli_tokens(flags))
    return " ".join(shlex.quote(str(tok)) for tok in tokens)


def _write_sbatch_script(
    path: str,
    *,
    command: str,
    requeue: bool,
    mode: str = "swiss-ai",
    openpi_data_home: Optional[str] = None,
) -> None:
    cwd = os.getcwd()
    data_home_export = (
        f"export OPENPI_DATA_HOME={shlex.quote(openpi_data_home)}\n" if openpi_data_home else ""
    )
    # Euler compute nodes have no persistent PYTHONPATH/venv preallocation setup like the
    # swiss-ai container image does; these exports are only needed (and only tested) there.
    euler_exports = (
        f"export PYTHONPATH={shlex.quote(cwd)}:${{PYTHONPATH:-}}\n"
        "export XLA_PYTHON_CLIENT_PREALLOCATE=false\n"
    ) if mode == "euler" else ""
    if requeue:
        script = f"""#!/bin/bash
set -euo pipefail

cd {shlex.quote(cwd)}
{euler_exports}{data_home_export}child_status=0
{command} || child_status=$?

if [[ "$child_status" -eq {REQUEUE_EXIT_CODE} ]]; then
  echo "[$(date --iso-8601=seconds)] Job ${{SLURM_JOB_ID}} requested requeue." >&2
  if scontrol requeue "${{SLURM_JOB_ID}}"; then
    echo "[$(date --iso-8601=seconds)] Requeue submitted for job ${{SLURM_JOB_ID}}." >&2
    exit 0
  fi
  echo "[$(date --iso-8601=seconds)] Failed to requeue job ${{SLURM_JOB_ID}}." >&2
  exit "$child_status"
fi

exit "$child_status"
"""
    else:
        script = f"""#!/bin/bash
set -euo pipefail

cd {shlex.quote(cwd)}
{euler_exports}{data_home_export}exec {command}
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
    partition: Optional[str] = DEFAULT_PARTITION,
    account: str = DEFAULT_ACCOUNT,
    mode: str = "swiss-ai",
    dry: bool = False,
    prompt: bool = True,
    requeue: bool = False,
    gpu_type: Optional[str] = None,
    num_gpus: int = 1,
    mem: Optional[str] = None,
    openpi_data_home: Optional[str] = None,
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

    assert mode in ["swiss-ai", "local", "euler"], f"Unknown mode: {mode}"

    # update combos with checkpoint_base_dir
    results_dir = os.path.join(result_dir, '_'.join([project_name, group_name]))
    for i, combo in enumerate(combos):
        combo["checkpoint_base_dir"] = os.path.join(results_dir, str(i))
        combo["project_name"] = project_name
        combo["group_name"] = group_name
        combo["exp_name"] = auto_exp_name(project_name, combo, i)
        # fsdp_devices defaults to 1 (replication); shard across all GPUs unless overridden in YAML.
        if num_gpus > 1 and "fsdp_devices" not in combo:
            combo["fsdp_devices"] = num_gpus
    
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
    command_list = [generate_srun_command(script, config_name, flags=combo, mode=mode) for combo in combos]
    if mode in ("swiss-ai", "euler"):

        bsub_cmd = f"sbatch --time={duration} "
        if mode != "euler":
            bsub_cmd += f"--account={account} "
        if partition:
            bsub_cmd += f"--partition={partition} "
        if mode == "euler" and gpu_type:
            bsub_cmd += f"--gpus={gpu_type}:{num_gpus} "
        if mode == "euler" and mem:
            bsub_cmd += f"--mem-per-cpu={mem} --cpus-per-task={DEFAULT_EULER_CPUS * num_gpus} "
        if requeue:
            bsub_cmd += "--requeue --open-mode=append "

        cluster_cmds = []
        for i, (cmd, combo) in enumerate(zip(command_list, combos)):
            script_path = os.path.join(combo["checkpoint_base_dir"], f"job_{i:04d}.sbatch.sh")
            if not dry:
                _write_sbatch_script(
                    script_path,
                    command=cmd,
                    requeue=requeue,
                    mode=mode,
                    openpi_data_home=openpi_data_home,
                )
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
    for key, value in {"resume": True, "overwrite": False}.items():
        updated[key] = value
    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/filtered_sft.yaml")
    parser.add_argument("--dry", action="store_true", help="Print commands without submitting")
    parser.add_argument("--mode", default="swiss-ai", choices=["swiss-ai", "local", "euler"], help="Execution mode")
    parser.add_argument("--duration", default="11:59:00", help="SLURM time limit")
    parser.add_argument("--partition", default=None, help="SLURM partition (default: 'normal' for swiss-ai, none for euler)")
    parser.add_argument("--gpu_type", default=DEFAULT_EULER_GPU, help="GPU type for Euler mode (e.g. rtx_3090, a100_80gb)")
    parser.add_argument("--num_gpus", type=int, default=DEFAULT_EULER_NUM_GPUS, help="Number of GPUs per job (Euler mode)")
    parser.add_argument("--mem", default=None, help="Memory per CPU for Euler sbatch jobs (e.g. 8G, 16G); total = mem * cpus-per-task")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--skip_requeue", action="store_true", help="Submit requeue-safe resumable jobs")

    args = parser.parse_args()

    # Resolve cluster-specific defaults based on mode
    if args.mode == "euler":
        result_dir = EULER_RESULTS_DIR
        partition = args.partition  # None means no --partition flag; Euler doesn't need one
        mem = args.mem or DEFAULT_EULER_MEM_PER_CPU
        openpi_data_home = EULER_CACHE_DIR
        if not args.dry:
            ensure_assets(EULER_CACHE_DIR)
    else:
        result_dir = RESULTS_DIR
        partition = args.partition or DEFAULT_PARTITION
        mem = args.mem  # not used for swiss-ai
        openpi_data_home = None

    with open(args.config, "r") as f:
        config = config = yaml.load(f, Loader=yaml.FullLoader)

    combos = dict_permutations(config["params"])
    if not args.skip_requeue:
        combos = [apply_requeue_flags(combo) for combo in combos]

    generate_run_commands(
        combos,
        script=config["script"],
        config_name=config["config_name"],
        result_dir=result_dir,
        project_name=config["project_name"],
        group_name=config["group_name"],
        mode=args.mode,
        duration=args.duration,
        partition=partition,
        account=DEFAULT_ACCOUNT,
        dry=args.dry,
        prompt=not args.force,
        requeue=not args.skip_requeue,
        gpu_type=args.gpu_type if args.mode == "euler" else None,
        num_gpus=args.num_gpus,
        mem=mem if args.mode == "euler" else None,
        openpi_data_home=openpi_data_home,
    )


if __name__ == "__main__":
    main()