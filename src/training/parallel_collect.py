"""Parallel data collection: split each collection round over K collector jobs.

Enabled by `collect.num_collectors = K`. At a collection step S the training job
(the "main" job) does not collect. Instead it:
  1. saves its state, stages one checkpoint dir per collector and writes their job
     scripts plus `<checkpoint_base_dir>/parallel_launch.sh`, then exits with
     PARALLEL_EXIT_CODE;
  2. its job wrapper runs parallel_launch.sh, which submits the K collectors and a
     fresh copy of the main job that waits for them (`--dependency=afterany`).

Collector k runs the same exp.py on a contiguous slice of `collect.tasks` (task
order preserved), with the critic and policy of step S symlinked read-only from the
main run. It keeps the usual pause/resume and periodic saves. When its slice is
done it copies its replay shards into the main run as `step_<S>_p<k>_<i>.h5` and
writes a completion marker.

The restarted main job restores its buffer from all shards in sorted order:
earlier rounds, then any in-process partial shards `step_<S>_c<NN>.h5`, then
p01, p02, ... So data enters the FIFO buffer in the same task-major order as
in-process collection, and the first tasks of the list are still evicted first.
It then merges the collectors' per-task counts into the round metrics and trains.
Slices whose collector did not finish are relaunched (up to MAX_ATTEMPTS).
"""
import collections
import json
import logging
import os
from pathlib import Path
import re
import shlex
import shutil

from src.training.runtime_state import (
    replay_shard_dir,
    resume_state_path,
    save_epoch_state,
    write_resume_manifest,
)

PARALLEL_EXIT_CODE = 43
MAX_ATTEMPTS = 3
LAUNCH_FILE = "parallel_launch.sh"


def task_slices(tasks: list[str], k: int) -> list[list[str]]:
    """Contiguous, near-equal slices of the task list, in order."""
    if len(set(tasks)) != len(tasks):
        raise ValueError("parallel collection needs a task list without repeats")
    n = len(tasks)
    bounds = [round(i * n / k) for i in range(k + 1)]
    return [tasks[bounds[i]:bounds[i + 1]] for i in range(k)]


def _parallel_dir(main_dir: Path) -> Path:
    return main_dir / "runtime_state" / "parallel"


def _marker_path(main_dir: Path, step: int, k: int) -> Path:
    return _parallel_dir(main_dir) / f"step{step:08d}_p{k + 1:02d}.json"


def _stage_base(config, step: int, k: int) -> Path:
    return Path(config.checkpoint_base_dir).resolve() / "pcollect" / f"s{step:08d}_p{k + 1:02d}"


def _rollouts_per_task(config, step: int) -> int:
    n = config.collect.num_rollouts
    if step == 0 and config.collect.num_initial_rollouts is not None:
        n += config.collect.num_initial_rollouts
    return n


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- main job

def run_round(agent, config, step: int, progress: dict | None):
    """Main-job entry point at a collection step.

    Returns `(metrics, n_successful_episodes)` once every slice is done. Otherwise
    stages and writes the launch file for the missing slices and returns None; the
    caller then exits with PARALLEL_EXIT_CODE.
    """
    main_dir = Path(config.checkpoint_dir)
    # A round keeps the split it was launched with, so num_collectors can be changed
    # between rounds (it then applies from the next round on).
    plan_path = _parallel_dir(main_dir) / f"plan_step{step:08d}.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else {"attempts": 0}
    k_total = plan.get("num_collectors", config.collect.num_collectors)
    slices = plan.get("slices") or task_slices(list(config.collect.tasks), k_total)
    n_target = _rollouts_per_task(config, step)
    done_eps = dict((progress or {}).get("episodes_per_task", {}))
    done_succ = dict((progress or {}).get("successes_per_task", {}))

    markers = {}
    for k, tasks in enumerate(slices):
        m = _marker_path(main_dir, step, k)
        if m.exists():
            markers[k] = json.loads(m.read_text())
        elif all(done_eps.get(t, 0) >= n_target for t in tasks):
            # Fully collected in-process before switching to parallel collection.
            markers[k] = {
                "step": step, "part": k + 1, "tasks": tasks, "new_episodes": 0, "successful_episodes": 0,
                "episodes_per_task": {t: done_eps[t] for t in tasks},
                "successes_per_task": {t: done_succ.get(t, 0) for t in tasks},
            }
            _atomic_write(m, json.dumps(markers[k], indent=1))

    if len(markers) == k_total:
        return _merge(agent, config, step, progress, markers)

    if plan["attempts"] >= MAX_ATTEMPTS:
        raise RuntimeError(
            f"parallel collection at step {step}: slices {sorted(set(range(k_total)) - set(markers))} "
            f"still missing after {MAX_ATTEMPTS} launches; see {_stage_base(config, step, 0).parent}"
        )
    plan["attempts"] += 1

    # Save the current state (checkpoint only if none exists at this step) so the
    # collectors can load it and this job can resume at the same step.
    if progress is None:
        progress = {"step": step, "part": 0, "episodes_per_task": {}, "successes_per_task": {}}
    save_epoch_state(agent, config, prepare_for_resume=True, collection_progress=progress)

    job_script = Path(config.checkpoint_base_dir).resolve() / "job_0000.sbatch.sh"
    script_text = job_script.read_text()
    opts = _sbatch_opts(script_text)
    # Submit each collector separately and always resubmit the main job, waiting on
    # whichever collectors were accepted: a failed submission is then just a missing
    # slice that the next attempt relaunches, and the run never stalls.
    launch = ["#!/bin/bash", "set -uo pipefail", 'ids=""', "failed=0"]
    missing = [k for k in range(k_total) if k not in markers]
    for k in missing:
        stage_job = _stage_collector(agent, config, step, k, slices[k], progress, script_text)
        launch.append(
            f"if id=$(sbatch --parsable {opts} --output={shlex.quote(str(stage_job.parent))}/slurm-%j.out "
            f"{shlex.quote(str(stage_job))}); then id=${{id%%;*}}; ids=\"$ids:$id\"; "
            f'echo "collector {k + 1}/{k_total} (step {step}): job $id"; '
            f'else failed=1; echo "collector {k + 1}/{k_total} (step {step}): submission failed" >&2; fi'
        )
    launch.append('if [[ -n "$ids" ]]; then dep="--dependency=afterany$ids --kill-on-invalid-dep=no"; '
                  'else dep="--begin=now+600"; fi')
    launch.append(
        f"main=$(sbatch --parsable $dep {opts} --output={shlex.quote(str(job_script.parent))}/slurm-%j.out "
        f"{shlex.quote(str(job_script))}) || exit 1"
    )
    launch.append(f'echo "main job (resumes step {step} after collectors$ids): ${{main%%;*}}"')
    plan.update({"step": step, "num_collectors": k_total, "missing": [k + 1 for k in missing],
                 "slices": slices})
    _atomic_write(plan_path, json.dumps(plan, indent=1))
    _atomic_write(job_script.parent / LAUNCH_FILE, "\n".join(launch) + "\n")
    logging.info(
        "Parallel collection at step %d: launching collectors %s (attempt %d/%d).",
        step, [k + 1 for k in missing], plan["attempts"], MAX_ATTEMPTS,
    )
    return None


def _sbatch_opts(script_text: str) -> str:
    account = re.search(r"--account=(\S+)", script_text).group(1)
    partition = os.environ.get("SLURM_JOB_PARTITION", "normal")
    duration = os.environ.get("PARALLEL_COLLECT_TIME") or ("01:30:00" if partition == "debug" else "11:59:00")
    return f"--account={account} --time={duration} --partition={partition} --requeue --open-mode=append"


def _stage_collector(agent, config, step, k, tasks, progress, script_text) -> Path:
    """Create collector k's checkpoint dir (if new) and job script; return the script path."""
    main_dir = Path(config.checkpoint_dir)
    main_base = Path(config.checkpoint_base_dir).resolve()
    stage_base = _stage_base(config, step, k)
    stage_dir = stage_base / main_dir.parent.name / main_dir.name

    if not (stage_dir / "runtime_state" / "resume_state.json").exists():
        (stage_dir / "rl_state").mkdir(parents=True, exist_ok=True)
        (stage_dir / "runtime_state" / "replay_shards").mkdir(parents=True, exist_ok=True)
        for link, target in ((stage_dir / str(step), main_dir / str(step)),
                             (stage_dir / "rl_state" / str(step), main_dir / "rl_state" / str(step))):
            if not target.exists():
                raise FileNotFoundError(f"missing checkpoint for collectors: {target}")
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(target)
        manifest = json.loads(resume_state_path(config).read_text())
        manifest["step"] = step
        manifest["replay_shard_dir"] = str(stage_dir / "runtime_state" / "replay_shards")
        manifest["collection_progress"] = {
            "step": step,
            "part": progress.get("part", 0),
            "episodes_per_task": {t: progress["episodes_per_task"].get(t, 0) for t in tasks},
            "successes_per_task": {t: progress["successes_per_task"].get(t, 0) for t in tasks},
        }
        _atomic_write(stage_dir / "runtime_state" / "resume_state.json", json.dumps(manifest, indent=2))
        # Episodes of this slice already collected before staging, so the collector can
        # report how many it added even if it resumes across several processes.
        baseline = sum(manifest["collection_progress"]["episodes_per_task"].values())
        _atomic_write(stage_dir / "runtime_state" / "parallel_baseline.json", json.dumps({"episodes": baseline}))
    # else: an earlier attempt of this collector saved progress; it resumes from there.

    # Collector job script: the main job's script, pointed at the stage dir, with this
    # slice as the task list.
    text = script_text.replace(str(main_base), str(stage_base))
    lines = text.splitlines()
    i = next(n for n, l in enumerate(lines) if l.startswith("srun "))
    toks = shlex.split(lines[i].replace(" || child_status=$?", ""))
    toks = _replace_flag(toks, "--collect.tasks", tasks)
    toks = _replace_flag(toks, "--collect.num_collectors", None)
    toks += ["--collect.collector_index", str(k), "--collect.parallel_main_dir", str(main_dir), "--no-wandb_enabled"]
    line = " ".join(shlex.quote(t) for t in toks) + " || child_status=$?"
    check = shlex.split(line.replace(" || child_status=$?", ""))
    j = check.index("--collect.tasks")
    if (check[j + 1:j + 1 + len(tasks)] != tasks or not check[j + 1 + len(tasks)].startswith("--")
            or check[check.index("--checkpoint_base_dir") + 1] != str(stage_base)
            or "--collect.num_collectors" in check):
        raise RuntimeError(f"collector {k + 1} job line failed checks: {line[:300]}")
    lines[i] = line
    job = stage_base / "job_0000.sbatch.sh"
    _atomic_write(job, "\n".join(lines) + "\n")
    job.chmod(0o755)
    return job


def _replace_flag(toks: list[str], flag: str, values: list[str] | None) -> list[str]:
    """Drop `flag` and its values from a token list; re-add it with `values` if given."""
    if flag in toks:
        i = toks.index(flag)
        j = i + 1
        while j < len(toks) and not toks[j].startswith("--"):
            j += 1
        toks = toks[:i] + toks[j:]
    if values is not None:
        toks = toks + [flag] + list(values)
    return toks


def _merge(agent, config, step, progress, markers):
    eps, succ = {}, {}
    new_eps = n_success = 0
    for k in sorted(markers):
        m = markers[k]
        eps.update(m["episodes_per_task"])
        succ.update(m["successes_per_task"])
        new_eps += m["new_episodes"]
        n_success += m["successful_episodes"]
    n_target = _rollouts_per_task(config, step)
    short = {t: eps.get(t, 0) for t in config.collect.tasks if eps.get(t, 0) != n_target}
    if short:
        raise RuntimeError(f"parallel collection at step {step}: wrong episode counts {short}")
    merged_flag = _parallel_dir(Path(config.checkpoint_dir)) / f"step{step:08d}_merged.json"
    already_merged = merged_flag.exists()
    if not already_merged:  # a restart after the merge must not count the round twice
        agent.total_collected_episodes += new_eps
    total, total_succ = sum(eps.values()), sum(succ.values())
    metrics = {"success_rate": float(total_succ) / float(total)}
    metrics.update({f"success_rate/{t}": succ[t] / eps[t] for t in set(config.collect.tasks)})
    metrics["collect/num_parts"] = (progress or {}).get("part", 0) + len(markers)
    metrics["collect/num_collectors"] = len(markers)
    # Record that this round is merged, so a restart before the next save does not
    # merge (and count) it twice.
    if not already_merged:
        write_resume_manifest(agent, config, None)
        _atomic_write(merged_flag, json.dumps(metrics, indent=1))
    logging.info("Merged parallel collection at step %d: %d episodes, success %.3f.",
                 step, total, metrics["success_rate"])
    return metrics, n_success


# --------------------------------------------------------------------- collector job

def start_collector(agent, config) -> None:
    """Give this collector its own action-sampling RNG stream."""
    import jax

    agent._rng = jax.random.fold_in(agent._rng, 1000 + config.collect.collector_index)


def finish_collector(agent, config, step: int, metrics: dict, n_success: int) -> None:
    """Deliver this collector's slice to the main run and mark it complete."""
    k = config.collect.collector_index
    main_dir = Path(config.collect.parallel_main_dir)
    stage_shards = replay_shard_dir(config)
    agent._online_data_buffer.save_shard(stage_shards / f"step_{step:08d}_z.h5")

    names = []
    dest_dir = main_dir / "runtime_state" / "replay_shards"
    for i, src in enumerate(sorted(stage_shards.glob(f"step_{step:08d}*.h5"))):
        name = f"step_{step:08d}_p{k + 1:02d}_{i:03d}.h5"
        tmp = dest_dir / f".{name}.tmp"
        shutil.copy2(src, tmp)
        os.replace(tmp, dest_dir / name)
        names.append(name)

    tasks = list(config.collect.tasks)
    n = _rollouts_per_task(config, step)
    marker = {
        "step": step, "part": k + 1, "tasks": tasks, "shards": names,
        "episodes_per_task": {t: n for t in tasks},
        "successes_per_task": {t: int(round(metrics[f"success_rate/{t}"] * n)) for t in tasks},
        "new_episodes": len(tasks) * n - json.loads(
            (Path(config.checkpoint_dir) / "runtime_state" / "parallel_baseline.json").read_text())["episodes"],
        "successful_episodes": int(n_success),
    }
    _atomic_write(_marker_path(main_dir, step, k), json.dumps(marker, indent=1))
    logging.info("Collector %d finished step %d: %d shards delivered to %s.", k + 1, step, len(names), dest_dir)
