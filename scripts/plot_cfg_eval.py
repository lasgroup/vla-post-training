"""Bar-plot the per-cfg-scale eval from the latest FSFT+CFG wandb run.

The sweep in fsft_cfg_libero_babel.sh logs `eval/cfg<scale>/success_rate[/<task>]`
once per guidance scale. One subplot per scale, one bar per task plus overall.

  python scripts/plot_cfg_eval.py [--run-name "FSFT + CFG"] [--out cfg_eval.png]
"""

import argparse
import json
import math
import os
import sys

# The repo has a local `wandb/` run dir that shadows the package when cwd is root.
sys.path = [p for p in sys.path if p not in ("", ".", os.getcwd())]

import matplotlib.pyplot as plt
import numpy as np
import wandb


def find_run(api, entity, project, run_name):
    runs = api.runs(f"{entity}/{project}", order="-created_at")
    for run in runs:
        if run.name == run_name:
            return run
    raise SystemExit(f"no run named {run_name!r} in {entity}/{project}")


def collect_scales(summary):
    """{scale: {"overall": sr, "tasks": {task: sr}}} from the eval/cfg* keys."""
    scales = {}
    for key, value in summary.items():
        if not key.startswith("eval/cfg"):
            continue
        tag, _, rest = key.removeprefix("eval/").partition("/")
        entry = scales.setdefault(float(tag.removeprefix("cfg")), {"tasks": {}})
        if rest == "success_rate":
            entry["overall"] = float(value)
        elif rest.startswith("success_rate/"):
            entry["tasks"][rest.removeprefix("success_rate/")] = float(value)
    return scales


def wilson(p, n, z=1.96):
    """95% Wilson score interval as (lower_err, upper_err) around p.

    Wilson rather than the normal approximation: at n=32 the latter is badly
    off near 0 and 1, and can put the bar outside [0, 1].
    """
    if not n:
        return 0.0, 0.0
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p - max(0.0, center - half), min(1.0, center + half) - p


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--entity", default="diverse-data-synthesis")
    p.add_argument("--project", default="openpi")
    p.add_argument("--run-name", default="FSFT + CFG")
    p.add_argument("--json", help="read an eval_metrics_step*.json instead of W&B")
    p.add_argument(
        "--n-per-task",
        type=int,
        help="episodes per task per scale (collect.num_eval_rollouts). Read from "
        "the W&B run config when available, required with --json.",
    )
    p.add_argument("--out", default="cfg_eval.png")
    args = p.parse_args()

    n_per_task = args.n_per_task
    if args.json:
        with open(args.json) as f:
            summary, title, source = json.load(f), args.json, args.json
    else:
        run = find_run(wandb.Api(), args.entity, args.project, args.run_name)
        summary, title, source = dict(run.summary), run.name, f"{run.name} ({run.id})"
        if n_per_task is None:
            n_per_task = (run.config.get("collect") or {}).get("num_eval_rollouts")
    if n_per_task is None:
        raise SystemExit("episode count unknown: pass --n-per-task")

    scales = collect_scales(summary)
    if not scales:
        raise SystemExit(f"{source} has no eval/cfg* metrics")

    tasks = sorted({t for v in scales.values() for t in v["tasks"]})
    order = sorted(scales)
    # One subplot per env, plus a final one for the across-env average.
    panels = [*tasks, None]

    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 4.2), sharey=True)
    axes = [axes] if len(panels) == 1 else list(axes)

    for ax, task in zip(axes, panels):
        labels = [f"{s:g}" for s in order]
        if task is None:
            values = [scales[s].get("overall", 0.0) for s in order]
            n = n_per_task * len(tasks)
        else:
            values = [scales[s]["tasks"].get(task, 0.0) for s in order]
            n = n_per_task
        errs = np.array([wilson(v, n) for v in values]).T
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
        ax.set_xlabel("cfg scale")
        ax.set_ylim(0, 1.05)

    axes[0].set_ylabel("success rate")
    fig.suptitle(
        f"{title} — eval success rate per env across guidance scales "
        "(95% Wilson CI, unpaired)"
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"{source} -> {args.out}")


if __name__ == "__main__":
    main()
