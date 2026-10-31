#!/usr/bin/env python3
"""Mean +- std over seeds for the LIBERO-1 (task 82) and LIBERO-4 (mt4) OGPO-pi05 runs.

Usage (needs wandb + matplotlib + numpy + pandas; OGPO's .venv has no pandas):
    PY=/home/mananaga/VLA/manan_babel/vla-post-training/.venv/bin/python          # babel
    PY=/home/sarveshp/Research/envs/openpi-jax-latest/bin/python              # maxlab-host-002 (or: uv run --no-project --with wandb --with matplotlib --with pandas ...)
    WANDB_API_KEY=<key> $PY scripts/ogpo/plot_libero_seeds.py                   # LIBERO-4 = latest (v2) runs
    $PY scripts/ogpo/plot_libero_seeds.py --libero4 v1                          # the first matched runs
    $PY scripts/ogpo/plot_libero_seeds.py --metric eval/success
Output: <repo>/plots/libero_seeds_*.{png,pdf} (override with --out).

Writes two figures:
  <out>_libero1_vs_libero4.{png,pdf}  left: LIBERO-1, right: LIBERO-4 (task-average)
  <out>_libero4_per_task.{png,pdf}    one panel per LIBERO-4 task
  <out>_vs_baselines.{png,pdf}        final OGPO (--bar-metric) next to BASELINES and OLD_OGPO
All runs are on the update axis (round schedule). At each eval step the mean / std
use the seeds that logged it, so a crashed seed drops out from its last step on.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import wandb

ENTITY, PROJECT = "diverse-data-synthesis", "OGPO_libero_pixel"
SEEDS = (0, 1, 2)
LIBERO1 = "ogpo_libero82_pi05_matched_consseed{}"
# LIBERO-4 run families (seed appended): v1 = matched settings, v2 = BoN32 + critic 512x2 + lr 1e-4 (2026-10-05).
LIBERO4_RUNS = {
    "v1": "ogpo_libero_mt4_pi05_matched_consseed{}",
    "v2": "ogpo_libero_mt4_pi05_bon32_c512x2_lr1e4seed{}",
}
LIBERO4 = LIBERO4_RUNS["v2"]
TASKS = (82, 79, 38, 31)
COLOR = "C0"  # matplotlib default blue

# Baselines read off actor_critic_ablation_variant_3.pdf (bar height, error-bar bottom, top).
# Approximate (eyeballed from the figure); their OGPO bar is left out on purpose.
BASELINES = {
    "LIBERO-1": {
        "FSFT+BoN": (0.83, 0.80, 0.85),
        "AWR": (0.37, 0.20, 0.60),
        "PARL": (0.45, 0.40, 0.50),
        "PARL+filter": (0.77, 0.65, 0.90),
    },
    "LIBERO-4": {
        "FSFT+BoN": (0.63, 0.58, 0.69),
        "AWR": (0.21, 0.12, 0.30),
        "PARL": (0.20, 0.15, 0.25),
        "PARL+filter": (0.70, 0.58, 0.84),
    },
}
BASE_POLICY = {"LIBERO-1": 0.33, "LIBERO-4": 0.235}  # dashed line in the same figure
# The PDF's own OGPO bar (vla-post-training implementation), plotted as "OGPO (old)".
OLD_OGPO = {"LIBERO-1": (0.47, 0.19, 0.84), "LIBERO-4": (0.05, 0.015, 0.095)}


def fetch(api, name, keys):
    runs = list(api.runs(f"{ENTITY}/{PROJECT}", filters={"display_name": name}))
    if not runs:
        print(f"[warn] no run named {name!r}")
        return {}
    run = max(runs, key=lambda r: r.summary.get("_step") or -1)
    out = {}
    for k in keys:
        h = run.history(keys=[k], samples=10_000, pandas=True)
        if k in h.columns:
            out[k] = dict(zip(h["_step"].astype(int), h[k].astype(float)))
    print(f"{name}: last step {run.summary.get('_step')} ({run.state})")
    return out


def mean_std(series):
    """series: list of {step: value} (one per seed) -> steps, mean, std, n."""
    steps = sorted(set().union(*series))
    vals = [[s[t] for s in series if t in s] for t in steps]
    return (np.array(steps), np.array([np.mean(v) for v in vals]),
            np.array([np.std(v) for v in vals]), np.array([len(v) for v in vals]))


def panel(ax, series, title, ylabel):
    steps, mean, std, n = mean_std(series)
    ax.fill_between(steps, mean - std, mean + std, color=COLOR, alpha=0.2, lw=0)
    ax.plot(steps, mean, color=COLOR, lw=2)
    ax.set_title(title)
    ax.set_xlabel("updates")
    ax.set_ylabel(ylabel)
    ax.set_xlim(steps.min(), steps.max())
    ax.tick_params(labelleft=True)  # sharey hides them on all but the first panel
    print(f"  {title}: " + ", ".join(f"{t}:{m:.2f}±{s:.2f}(n={k})" for t, m, s, k in zip(steps, mean, std, n)))


def final_mean_std(series):
    """Mean / std over seeds at the last eval step reached by any seed (seeds missing it drop out)."""
    steps, mean, std, n = mean_std(series)
    return mean[-1], std[-1], n[-1]


def bars(ax, suite, ogpo):
    names = list(BASELINES[suite]) + ["OGPO (old)", "OGPO (new)"]
    vals = [BASELINES[suite][k] for k in BASELINES[suite]] + [OLD_OGPO[suite]]
    heights = [v[0] for v in vals] + [ogpo[0]]
    lo = [v[0] - v[1] for v in vals] + [ogpo[1]]
    hi = [v[2] - v[0] for v in vals] + [ogpo[1]]
    x = np.arange(len(names))
    ax.bar(x, heights, color=COLOR, alpha=0.35, width=0.7)
    ax.bar(x[-1], heights[-1], color=COLOR, width=0.7)  # OGPO (new) in full colour
    ax.errorbar(x, heights, yerr=[lo, hi], fmt="none", ecolor="black", capsize=4, lw=1)
    ax.axhline(BASE_POLICY[suite], color="gray", ls="--", lw=1)
    ax.text(x[0] - 0.35, BASE_POLICY[suite] + 0.01, "base policy", color="gray", fontsize=8, va="bottom")
    # dashed level line at the current OGPO mean, so each baseline can be read against it directly
    ax.axhline(ogpo[0], color=COLOR, ls="--", lw=1.2, zorder=0)
    ax.text(x[0] - 0.35, ogpo[0] + 0.01, f"OGPO (new) {ogpo[0]:.2f}", color=COLOR, fontsize=8, va="bottom")
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_title(suite)
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1)
    ax.tick_params(labelleft=True)


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {path}.png / .pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="eval_sde/success")
    ap.add_argument("--bar-metric", default="eval/success",
                    help="OGPO metric in the baseline bar chart (ODE eval, like vla-post-training's evaluate_policy)")
    ap.add_argument("--libero4", default="v2", help="LIBERO-4 run family: " + " | ".join(LIBERO4_RUNS) + ", or a run-name pattern with {}")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                                   "plots", "libero_seeds"))
    args = ap.parse_args()
    libero4 = LIBERO4_RUNS.get(args.libero4, args.libero4)
    out = args.out if args.libero4 not in LIBERO4_RUNS or args.libero4 == "v2" else f"{args.out}_{args.libero4}"
    print(f"LIBERO-4 runs: {libero4.format('<seed>')}")
    task_key = lambda t: f"{args.metric}/libero_pi05-libero_90-{t}"

    api = wandb.Api(timeout=60)
    l1 = [fetch(api, LIBERO1.format(s), [args.metric, args.bar_metric]) for s in SEEDS]
    l4 = [fetch(api, libero4.format(s), [args.metric, args.bar_metric] + [task_key(t) for t in TASKS])
          for s in SEEDS]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    panel(axes[0], [r[args.metric] for r in l1 if args.metric in r], f"LIBERO-1: {args.metric}", args.metric)
    panel(axes[1], [r[args.metric] for r in l4 if args.metric in r], f"LIBERO-4: {args.metric}", args.metric)
    save(fig, f"{out}_libero1_vs_libero4")

    fig, axes = plt.subplots(1, len(TASKS), figsize=(4.2 * len(TASKS), 3.8), sharey=True)
    for ax, t in zip(axes, TASKS):
        panel(ax, [r[task_key(t)] for r in l4 if task_key(t) in r], f"LIBERO-4 task {t}: {args.metric}", args.metric)
    save(fig, f"{out}_libero4_per_task")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (suite, runs) in zip(axes, (("LIBERO-1", l1), ("LIBERO-4", l4))):
        m, s, n = final_mean_std([r[args.bar_metric] for r in runs if args.bar_metric in r])
        print(f"  OGPO {suite} final {args.bar_metric}: {m:.3f} ± {s:.3f} (n={n})")
        bars(ax, suite, (m, s))
    save(fig, f"{out}_vs_baselines")


if __name__ == "__main__":
    main()