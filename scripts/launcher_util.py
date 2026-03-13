import datetime as dt
import itertools
import os
import secrets
import shlex
from typing import Any, Dict, List, Optional

# Default SLURM settings matching existing bash scripts
DEFAULT_ACCOUNT = "a143"
DEFAULT_ENVIRONMENT = "vla-post-training"
DEFAULT_DURATION = "03:30:00"
DEFAULT_PARTITION = "normal"
# Online configs default to num_workers=4; keep at least that many CPUs per task.
DEFAULT_CPUS_PER_TASK = 4
DEFAULT_CHECKPOINT_BASE_DIR = f"/capstor/scratch/cscs/{os.environ.get('USER', 'unknown')}/checkpoints"


def generate_srun_command(
    script: str,
    config_name: str,
    flags: Optional[Dict[str, Any]] = None,
    account: str = DEFAULT_ACCOUNT,
    environment: str = DEFAULT_ENVIRONMENT,
    ntasks: int = 1,
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
        f"--ntasks={ntasks}",
        "uv",
        "run",
        script,
        config_name,
    ]
    tokens.extend(flags_to_cli_tokens(flags))
    return " ".join(shlex.quote(str(tok)) for tok in tokens)


def auto_exp_name(project_name: str, combo: Dict[str, Any], run_idx: int) -> str:
    """Generate a unique experiment name suitable for checkpoint directories."""
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = secrets.token_hex(3)
    if "seed" in combo:
        return f"{project_name}_{timestamp}_{suffix}_seed{combo['seed']}"
    return f"{project_name}_{timestamp}_{suffix}_run{run_idx}"


def _normalize_flag_name(flag: str) -> str:
    return flag[2:] if flag.startswith("--") else flag


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
    normalized_flag = _normalize_flag_name(flag)
    if value is None:
        return []
    if isinstance(value, bool):
        return [f"--{_bool_flag_name(normalized_flag, value)}"]
    if isinstance(value, (list, tuple)):
        return [f"--{normalized_flag}", *[str(v) for v in value]]
    return [f"--{normalized_flag}", str(value)]


def flags_to_cli_tokens(flags: Optional[Dict[str, Any]]) -> List[str]:
    """Convert an overrides dict to CLI tokens, preserving key order."""
    if not flags:
        return []
    tokens: List[str] = []
    for flag, value in flags.items():
        tokens.extend(flag_to_cli_tokens(flag, value))
    return tokens


def generate_run_commands(
    command_list: List[str],
    num_tasks: int = 1,
    num_cpus: int = DEFAULT_CPUS_PER_TASK,
    num_gpus: int = 0,
    mem: int = 0,
    duration: str = DEFAULT_DURATION,
    partition: str = DEFAULT_PARTITION,
    account: str = DEFAULT_ACCOUNT,
    mode: str = "swiss-ai",
    dry: bool = False,
    prompt: bool = True,
) -> None:
    """Submit or run a list of commands.

    Args:
        command_list: List of srun command strings.
        num_tasks: Number of tasks per job (only used if > 0).
        num_cpus: CPUs per task (only used if > 0).
        num_gpus: GPUs per task (only used if > 0).
        mem: Memory per CPU in MB (only used if > 0).
        duration: Time limit for SLURM jobs.
        account: SLURM account.
        mode: "swiss-ai" for sbatch submission, "local" for sequential local execution.
        dry: If True, only print commands without executing.
        prompt: If True, ask for confirmation before submitting.
    """
    if mode == "swiss-ai":
        cluster_cmds = []
        bsub_cmd = f"sbatch --account={account} --time={duration} --partition={partition} "

        if num_tasks > 0:
            bsub_cmd += f"--ntasks={num_tasks} "
        if num_cpus > 0:
            bsub_cmd += f"--cpus-per-task={num_cpus} "
        if num_gpus > 0:
            bsub_cmd += f"-G {num_gpus} "
        if mem > 0:
            bsub_cmd += f"--mem-per-cpu={mem} "

        for cmd in command_list:
            cluster_cmds.append(bsub_cmd + f'--wrap="{cmd}"')

        if dry:
            for cmd in cluster_cmds:
                print(cmd)
        else:
            if prompt:
                answer = input(
                    f"About to launch {len(command_list)} jobs. Proceed? [yes/no] "
                )
            else:
                answer = "yes"
            if answer == "yes":
                for cmd in cluster_cmds:
                    os.system(cmd)

    elif mode == "local":
        if dry:
            for cmd in command_list:
                print(cmd)
        else:
            if prompt:
                answer = input(
                    f"About to run {len(command_list)} jobs locally. Proceed? [yes/no] "
                )
            else:
                answer = "yes"
            if answer == "yes":
                for cmd in command_list:
                    os.system(cmd)
    else:
        raise NotImplementedError(f"Unknown mode: {mode}")


def dict_permutations(d: dict) -> List[dict]:
    """Generate all combinations from a dict of lists (cartesian product)."""
    keys = d.keys()
    values = d.values()
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]
