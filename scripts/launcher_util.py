import itertools
import os
from typing import Any, Dict, List, Optional


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Default SLURM settings matching existing bash scripts
DEFAULT_ACCOUNT = "a143"
DEFAULT_ENVIRONMENT = "vla-post-training"
DEFAULT_DURATION = "03:30:00"
DEFAULT_CHECKPOINT_BASE_DIR = f"/capstor/scratch/cscs/{os.environ.get('USER', 'unknown')}/checkpoints"


def generate_srun_command(
    script: str,
    config_name: str,
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
    cmd = f"srun --account={account} --environment={environment} uv run {script} {config_name}"
    if flags is not None:
        for flag, value in flags.items():
            if isinstance(value, bool):
                if value:
                    cmd += f" --{flag}"
            else:
                cmd += f" --{flag} {value}"
    return cmd


def generate_run_commands(
    command_list: List[str],
    num_cpus: int = 1,
    num_gpus: int = 0,
    mem: int = 0,
    duration: str = DEFAULT_DURATION,
    account: str = DEFAULT_ACCOUNT,
    mode: str = "swiss-ai",
    dry: bool = False,
    prompt: bool = True,
) -> None:
    """Submit or run a list of commands.

    Args:
        command_list: List of srun command strings.
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
        bsub_cmd = f"sbatch --account={account} --time={duration} "

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
