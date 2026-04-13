import datetime as dt
import hashlib
import itertools
import json
import os
import secrets
import shlex
from typing import Any, Dict, List, Optional, Tuple

# Default SLURM settings matching existing bash scripts
DEFAULT_ACCOUNT = "a143"
DEFAULT_ENVIRONMENT = "vla-post-training"
DEFAULT_DURATION = "03:30:00"
DEFAULT_PARTITION = "normal"
# Online configs default to num_workers=4; keep at least that many CPUs per task.
DEFAULT_CPUS_PER_TASK = 4
DEFAULT_CHECKPOINT_BASE_DIR = f"/capstor/scratch/cscs/{os.environ.get('USER', 'unknown')}/checkpoints"
DEFAULT_LOG_DIR = "logs"


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


def _fmt_exp_value(value: Any) -> str:
    """Format a hyperparameter value for use in experiment names."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        if value == 0:
            return "0"
        s = f"{value:.0e}" if abs(value) < 1e-3 or abs(value) >= 1e3 else str(value)
        return s.replace("+", "").replace(".", "p")
    if isinstance(value, (list, Tuple)):
        if len(value) == 1:
            return _fmt_exp_value(value[0])
        return "-".join(_fmt_exp_value(v) for v in value)
    s = str(value)
    if s.startswith("libero_90_"):
        s = s.split("_")[-1]
    return s.replace("/", "-").replace(".", "p")


def algo_exp_name(
    prefix: str, flags: Dict[str, Any], keys: List[Tuple[str, str]]
) -> str:
    """Generate a descriptive experiment name from swept hyperparameters.

    Args:
        prefix: Short algorithm prefix (e.g. "fgrpo", "fmpo", "fpg").
        flags: Full flags dict for this run.
        keys: List of (flag_key, short_alias) pairs to include in the name.

    Returns:
        Name like ``fmpo_t59_lr1e-5_b0p05_ns10_s0_ab12cd``.
    """
    parts = [prefix]
    for key, alias in keys:
        if key in flags:
            parts.append(f"{alias}{_fmt_exp_value(flags[key])}")
    digest = hashlib.sha1(
        json.dumps(flags, sort_keys=True, default=str).encode()
    ).hexdigest()[:6]
    return "_".join(parts + [digest])


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
    log_dir: str = DEFAULT_LOG_DIR,
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
        if not dry:
            os.makedirs(log_dir, exist_ok=True)
        cluster_cmds = []
        bsub_cmd = f"sbatch --account={account} --time={duration} --partition={partition} --output={log_dir}/slurm-%j.out "

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
