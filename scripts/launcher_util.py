import datetime as dt
import itertools
import os
import re
import secrets
import shlex
from typing import Any, Dict, Iterable, List, Optional

# Default SLURM settings matching existing bash scripts
DEFAULT_ACCOUNT = "a143"
DEFAULT_ENVIRONMENT = "vla-post-training"
DEFAULT_DURATION = "03:30:00"
# Online configs default to num_workers=4; keep at least that many CPUs per task.
DEFAULT_CPUS_PER_TASK = 4
DEFAULT_CHECKPOINT_BASE_DIR = f"/capstor/scratch/cscs/{os.environ.get('USER', 'unknown')}/checkpoints"
DEFAULT_SLURM_LOG_DIR = "/users/mertalbaba/vla-post-training/logs"


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


def _sanitize_name_value(value: Any) -> str:
    if isinstance(value, bool):
        value = "true" if value else "false"
    elif isinstance(value, (list, tuple)):
        value = "-".join(str(v) for v in value)
    else:
        value = str(value)
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-_.")
    return value or "none"


_SHORT_LABELS: Dict[str, str] = {
    "collect.tasks": "task",
    "collect.eval_tasks": "evaltask",
    "collect.num_initial_rollouts": "initroll",
    "collect.use_time_to_success_as_reward": "tts",
    "rl.use_deterministic_anchor": "anchor",
    "rl.drop_low_diversity_groups": "dropdiv",
    "rl.align_critic_sampling": "aligncrit",
    "rl.reset_policy_params_to_ema_period": "emareset",
    "rl.policy_training_start_step": "polstart",
    "rl.td_weight_schedule.switch_step": "tdswitch",
    "rl.td_weight_schedule.ramp_steps": "tdramp",
    "rl.num_critic_updates_per_batch": "ncriticup",
    "rl.policy_only_successful": "polsucc",
    "rl.save_all_episodes": "saveall",
    "rl.normalize_adv": "normadv",
    "rl.use_mpo_advantage_weight": "mpoadv",
    "rl.use_mc_returns": "mcret",
    "rl.advantage_scale": "advscale",
    "rl.noise_level": "noise",
    "rl.online_ratio": "onratio",
    "rl.kl_coef": "kl",
    "rl.sft_anchor_coef": "sftanch",
    "rl.min_advantage_std": "minadv",
    "rl.use_offline_for_critic": "offcritic",
    "rl.offline_critic_reward": "offrew",
    "rl.prefill_buffer_from_disk": "prefill",
}


def _name_label(flag: str) -> str:
    if flag in _SHORT_LABELS:
        return _SHORT_LABELS[flag]
    return flag.rsplit(".", 1)[-1]


def auto_exp_name(
    project_name: str,
    combo: Dict[str, Any],
    run_idx: int,
    *,
    defaults: Optional[Dict[str, Any]] = None,
    tracked_keys: Optional[Iterable[str]] = None,
    algorithm_name: Optional[str] = None,
) -> str:
    """Generate a unique experiment name suitable for checkpoint directories."""
    prefix = algorithm_name if algorithm_name is not None else project_name
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = secrets.token_hex(3)
    if defaults is None and tracked_keys is None:
        if "seed" in combo:
            return f"{prefix}_{timestamp}_{suffix}_seed{combo['seed']}"
        return f"{prefix}_{timestamp}_{suffix}_run{run_idx}"

    parts = [prefix, timestamp, suffix]
    seen = set()
    name_keys = list(tracked_keys or [])
    if "collect.tasks" in combo and "collect.tasks" not in name_keys:
        name_keys.insert(0, "collect.tasks")
    if "seed" in combo and "seed" not in name_keys:
        name_keys.append("seed")

    for key in name_keys:
        if key in seen or key not in combo:
            continue
        seen.add(key)
        if (
            key not in {"collect.tasks", "seed"}
            and defaults is not None
            and key in defaults
            and combo[key] == defaults[key]
        ):
            continue
        if key == "seed":
            parts.append(f"seed{_sanitize_name_value(combo[key])}")
        else:
            label = _name_label(key)
            #parts.append(f"{label}-{_sanitize_name_value(combo[key])}")

    return "_".join(parts)


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
    account: str = DEFAULT_ACCOUNT,
    log_dir: str = DEFAULT_SLURM_LOG_DIR,
    mode: str = "swiss-ai",
    dry: bool = False,
    prompt: bool = True,
    job_names: Optional[List[str]] = None,
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
        job_names: Optional list of job names (one per command). Used for SLURM
            --job-name and log file names instead of the default 'slurm-%j'.
    """
    if mode == "swiss-ai":
        os.makedirs(log_dir, exist_ok=True)
        cluster_cmds = []

        base_cmd = f"sbatch --account={account} --time={duration} "
        if num_tasks > 0:
            base_cmd += f"--ntasks={num_tasks} "
        if num_cpus > 0:
            base_cmd += f"--cpus-per-task={num_cpus} "
        if num_gpus > 0:
            base_cmd += f"-G {num_gpus} "
        if mem > 0:
            base_cmd += f"--mem-per-cpu={mem} "

        for i, cmd in enumerate(command_list):
            if job_names is not None and i < len(job_names):
                name = job_names[i]
                log_prefix = os.path.join(log_dir, f"{name}-%j")
            else:
                log_prefix = os.path.join(log_dir, "slurm-%j")
            sbatch_cmd = (
                base_cmd
                + f"--output={shlex.quote(log_prefix + '.out')} "
                + f"--error={shlex.quote(log_prefix + '.err')} "
            )
            if job_names is not None and i < len(job_names):
                sbatch_cmd += f"--job-name={shlex.quote(job_names[i])} "
            cluster_cmds.append(sbatch_cmd + f'--wrap="{cmd}"')

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
