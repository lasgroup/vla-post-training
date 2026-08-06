"""Bar-plot the per-noise-level eval from a fsft_sde_eval_babel.sh job log.

That sweep runs eval.py once per rl.sde_noise_level with wandb disabled, so the
numbers live in the slurm .out (and in one eval_metrics_step0.json per level).
One subplot per task plus overall, one bar per noise level.

  python scripts/plot_sde_eval.py /home/mananaga/logs/9788879/.out
"""

import argparse
import json
import math
import re

import matplotlib.pyplot as plt
import numpy as np

LEVEL_RE = re.compile(r"noise_level=([0-9.eE+-]+)")
OVERALL_RE = re.compile(r"overall success_rate: ([0-9.]+)")
TASK_RE = re.compile(r"^\s+(\S+): ([0-9.]+)\s*\(")
ROLLOUTS_RE = re.compile(r"--collect\.num_eval_rollouts\s+(\d+)")


def collect_levels(log_text):
    """{level: {"overall": sr, "tasks": {task: sr}}} from the job log."""
    levels = {}
    entry = None
    for line in log_text.splitlines():
        m = LEVEL_RE.search(line)
        if m and "FAILED" not in line:
            entry = levels.setdefault(float(m.group(1)), {"tasks": {}})
            continue
        if entry is None:
            continue
        m = OVERALL_RE.search(line)
        if m:
            entry["overall"] = float(m.group(1))
            continue
        m = TASK_RE.match(line.split("[I]")[-1] if "[I]" in line else "")
        if m:
            entry["tasks"][m.group(1)] = float(m.group(2))
    # A level whose eval died leaves an empty entry -- drop it rather than
    # plotting it as a zero.
    return {k: v for k, v in levels.items() if v["tasks"]}


def wilson(p, n, z=1.96):
    """95% Wilson score interval as (lower_err, upper_err) around p."""
    if not n:
        return 0.0, 0.0
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p - max(0.0, center - half), min(1.0, center + half) - p


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log", help="slurm .out from fsft_sde_eval_babel.sh")
    p.add_argument(
        "--n-per-task",
        type=int,
        help="episodes per task per level. Read from the eval.py invocation in "
        "the log when not given.",
    )
    p.add_argument("--out", default="sde_eval.png")
    p.add_argument("--no-ci", action="store_true", help="drop the error bars")
    p.add_argument("--json", help="also dump the parsed numbers here")
    args = p.parse_args()

    with open(args.log) as f:
        log_text = f.read()

    levels = collect_levels(log_text)
    if not levels:
        raise SystemExit(f"{args.log} has no per-noise-level success rates")

    n_per_task = args.n_per_task
    if n_per_task is None:
        # The launcher does not echo its eval.py argv, so this usually falls
        # through to the script's default of 100.
        m = ROLLOUTS_RE.search(log_text)
        n_per_task = int(m.group(1)) if m else 100
        print(f"assuming num_eval_rollouts={n_per_task} per task")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({f"{k:g}": v for k, v in sorted(levels.items())}, f, indent=2)

    tasks = sorted({t for v in levels.values() for t in v["tasks"]})
    order = sorted(levels)
    # One subplot per env, plus a final one for the across-env average.
    panels = [*tasks, None]

    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 4.2), sharey=True)
    axes = [axes] if len(panels) == 1 else list(axes)

    for ax, task in zip(axes, panels):
        labels = [f"{s:g}" for s in order]
        if task is None:
            values = [levels[s].get("overall", 0.0) for s in order]
            n = n_per_task * len(tasks)
        else:
            values = [levels[s]["tasks"].get(task, 0.0) for s in order]
            n = n_per_task
        errs = None if args.no_ci else np.array([wilson(v, n) for v in values]).T
        bars = ax.bar(
            labels,
            values,
            yerr=errs,
            capsize=4,
            ecolor="0.3",
            color="tab:orange" if task is None else "tab:blue",
        )
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=12)
        ax.set_title(f"{task or 'overall'} (n={n})")
        ax.set_xlabel("sde noise level")
        ax.set_ylim(0, 1.05)

    axes[0].set_ylabel("success rate")
    suffix = "" if args.no_ci else " (95% Wilson CI, unpaired)"
    fig.suptitle(
        f"FSFT + SDE sampler — eval success rate per env across noise levels{suffix}"
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"{args.log} -> {args.out}")


if __name__ == "__main__":
    main()
