"""Bar-plot the per-cfg-scale eval from hard-coded numbers (no W&B).

Same layout as plot_cfg_eval.py: one subplot per env plus an overall panel,
one bar per guidance scale.

  python scripts/plot_cfg_eval_hardcoded.py [--out cfg_eval.png]
"""

import argparse

import matplotlib.pyplot as plt

# pi05_libero_online_filtered_sft_multitask4_cfg_seed0, eval at step 0.
N_PER_TASK = 100
SCALES = {
    1.0: {"libero_90_31": 0.72, "libero_90_38": 0.32, "libero_90_79": 0.43, "libero_90_82": 0.46},
    1.5: {"libero_90_31": 0.77, "libero_90_38": 0.32, "libero_90_79": 0.43, "libero_90_82": 0.52},
    2.0: {"libero_90_31": 0.79, "libero_90_38": 0.35, "libero_90_79": 0.39, "libero_90_82": 0.50},
    3.0: {"libero_90_31": 0.74, "libero_90_38": 0.22, "libero_90_79": 0.44, "libero_90_82": 0.52},
}
OVERALL = {1.0: 0.4825, 1.5: 0.5100, 2.0: 0.5075, 3.0: 0.4800}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="cfg_eval.png")
    p.add_argument("--title", default="FSFT + CFG (multitask4, seed0, step 0)")
    args = p.parse_args()

    tasks = sorted({t for v in SCALES.values() for t in v})
    order = sorted(SCALES)
    panels = [*tasks, None]

    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 4.2), sharey=True)
    axes = [axes] if len(panels) == 1 else list(axes)

    for ax, task in zip(axes, panels):
        labels = [f"{s:g}" for s in order]
        if task is None:
            values = [OVERALL[s] for s in order]
            n = N_PER_TASK * len(tasks)
        else:
            values = [SCALES[s][task] for s in order]
            n = N_PER_TASK
        bars = ax.bar(
            labels,
            values,
            color="tab:orange" if task is None else "tab:blue",
        )
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
        ax.set_title(f"{task or 'overall'} (n={n})")
        ax.set_xlabel("cfg scale")
        ax.set_ylim(0, 1.05)

    axes[0].set_ylabel("success rate")
    fig.suptitle(f"{args.title} — eval success rate per env across guidance scales")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
