#!/usr/bin/env python3
"""Mean +- std of an eval metric across OGPO-pi05 runs, on a shared update axis.

Usage (OGPO's .venv has no pandas; use the babel venv's interpreter):
    PY=/home/mananaga/VLA/manan_babel/vla-post-training/.venv/bin/python
    $PY scripts/ogpo/plot_seeds.py
    $PY scripts/ogpo/plot_seeds.py --metric eval_sde/success --out plots/libero82_seeds_success

Runs logged on the env-step axis (scripts/ogpo/libero_82_pi05.sh) are shifted by their
collection-only prefix (`start_training`) so x = number of updates for every run; their
step-0 eval (before any update) maps to update 0. Runs logged on the update axis
(libero_82.sh) are used as is. Each run is linearly interpolated onto a
common grid for the mean / std.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import wandb

ENTITY, PROJECT = "diverse-data-synthesis", "OGPO_libero_pixel"
# display name -> collection-only env steps to subtract (0 for update-axis runs)
DEFAULT_RUNS = {
    "ogpo_libero82_pi05seed82723x": 20000,
    "ogpo_libero82_pi05_matchedseed0": 0,
}

COLOR = "C0"  # matplotlib default blue


def fetch(api, name, offset, metric):
    runs = [r for r in api.runs(f"{ENTITY}/{PROJECT}", filters={"display_name": name})]
    if not runs:
        raise ValueError(f"no run named {name!r} in {ENTITY}/{PROJECT}")
    run = max(runs, key=lambda r: r.summary.get("_step") or -1)  # skip crashed duplicates
    h = run.history(keys=[metric], samples=10_000, pandas=True)
    steps = h["_step"].to_numpy(dtype=float)
    vals = h[metric].astype(float).to_numpy()
    x = np.where(steps == 0, 0.0, steps - offset)  # step-0 eval precedes any update
    keep = x >= 0
    print(f"{name} ({run.id}): " + ", ".join(f"{int(a)}:{b:.3g}" for a, b in zip(x[keep], vals[keep])))
    return run.id, x[keep], vals[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="eval_sde/return")
    ap.add_argument("--runs", nargs="+", default=None,
                    help="name:offset pairs (default: the two task-82 runs)")
    ap.add_argument("--grid-step", type=int, default=500)
    ap.add_argument("--out", default="/home/mananaga/OGPO/plots/libero82_seeds_eval_sde_return")
    args = ap.parse_args()
    runs = (dict((s.rsplit(":", 1)[0], int(s.rsplit(":", 1)[1])) for s in args.runs)
            if args.runs else DEFAULT_RUNS)

    api = wandb.Api(timeout=60)
    data = [(name, *fetch(api, name, off, args.metric)) for name, off in runs.items()]

    x_max = min(x.max() for _, _, x, _ in data)  # mean only where every run has data
    grid = np.arange(0, x_max + 1, args.grid_step)
    curves = np.stack([np.interp(grid, x, y) for _, _, x, y in data])
    mean, std = curves.mean(0), curves.std(0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.fill_between(grid, mean - std, mean + std, color=COLOR, alpha=0.2, lw=0)
    ax.plot(grid, mean, color=COLOR, lw=2)
    ax.set_xlabel("updates")
    ax.set_ylabel(args.metric)
    ax.set_title(args.metric)
    ax.set_xlim(0, x_max)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {args.out}.png / .pdf")
    print("grid mean:", ", ".join(f"{int(g)}:{m:.1f}±{s:.1f}" for g, m, s in zip(grid[::4], mean[::4], std[::4])))


if __name__ == "__main__":
    main()
