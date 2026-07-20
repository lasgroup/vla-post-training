#!/usr/bin/env python3
"""Print step-bucketed quantiles of OGPO debug metrics from a wandb run.

Usage:
    python tests/ogpo/wandb_quantiles.py
    python tests/ogpo/wandb_quantiles.py --run-name <other-run>
    python tests/ogpo/wandb_quantiles.py --entity <ent> --project <proj> --run-name <run>

By default the script targets the OGPO debug run hardcoded below
(``diverse-data-synthesis/libero_50``,
``libero_59_20260523-003059_b32efa_seed1``); override any of
``--entity / --project / --run-name`` on the CLI.
Buckets default to 500 training steps; pass ``--bucket 1000`` to change.

What it prints
--------------
For each metric of interest, a small table indexed by ``[step_start,
step_end)`` and showing ``count, mean, std, p05, p25, p50, p75, p95, min,
max``. The metrics are the OGPO-specific diagnostics that the actor /
critic / evaluator already log to W&B:

    actor/pg_loss, actor/bc_loss, actor/ratio_mean, actor/ratio_std,
    actor/approx_kl, actor/clipfrac, actor/advantage_mean,
    actor/advantage_std, actor/advantage_q_up, actor/advantage_q_low,
    actor/q_mean, actor/v_mean, actor/grad_norm, actor/param_norm,
    actor/loss, critic/q_*, critic/value_*, eval/* metrics, and
    ``online_buffer_size``.

Useful for figuring out *when* an OGPO run goes off the rails (e.g. clipfrac
saturating to 1, advantages collapsing to zero, BC loss exploding, eval
oscillating between 30-50%).
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import wandb


# Defaults for the current OGPO debug run. Override any of these from CLI.
DEFAULT_ENTITY = "diverse-data-synthesis"
DEFAULT_PROJECT = "ogpo_sweep"
DEFAULT_RUN_NAME = "pi05_libero_online_ogpo_sft_libero_90_44_seed0"


# Metrics the OGPO actor/critic emit. Wildcards are matched against the
# full W&B metric name. Anything not present in the run is silently
# skipped.
DEFAULT_METRICS: tuple[str, ...] = (
    # actor diagnostics (see src/rl/ogpo/update_actor.py)
    "actor/loss",
    "actor/pg_loss",
    "actor/bc_loss",
    "actor/ratio_mean",
    "actor/ratio_std",
    "actor/approx_kl",
    "actor/clipfrac",
    "actor/new_log_prob_mean",
    "actor/old_log_prob_mean",
    "actor/q_mean",
    "actor/v_mean",
    "actor/advantage_mean",
    "actor/advantage_std",
    "actor/advantage_min",
    "actor/advantage_max",
    "actor/advantage_q_up",
    "actor/advantage_q_low",
    "actor/advantage_median",
    "actor/grad_norm",
    "actor/param_norm",
    # critic diagnostics (advantage_weighted_sft/update_critic.py)
    "critic/q_*",
    "critic/value_*",
    # buffer + collect/eval
    "online_buffer_size",
    "eval/*",
    "collect/*",
)

DEFAULT_QUANTILES: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)


def _resolve_run(api: wandb.Api, entity: str, project: str,
                 run_name: str) -> "wandb.apis.public.Run":
    matches = list(api.runs(f"{entity}/{project}",
                            filters={"display_name": run_name}))
    if not matches:
        raise ValueError(
            f"No run with display name {run_name!r} in {entity}/{project}."
        )
    if len(matches) > 1:
        print(
            f"[warn] {len(matches)} runs match name {run_name!r}; "
            f"using most recently created: {matches[0].id}",
            file=sys.stderr,
        )
    return matches[0]


def _select_metric_columns(columns: Iterable[str],
                           patterns: Iterable[str]) -> list[str]:
    cols = list(columns)
    selected: list[str] = []
    seen: set[str] = set()
    for pat in patterns:
        for col in cols:
            if col in seen:
                continue
            if fnmatch.fnmatchcase(col, pat):
                selected.append(col)
                seen.add(col)
    return selected


def _fetch_history(run: "wandb.apis.public.Run",
                   metrics: list[str],
                   max_samples: int) -> pd.DataFrame:
    """Pull history for the requested metrics, one metric per request.

    ``run.history(keys=[a, b])`` inner-joins: it only returns steps where
    *every* requested key is non-null. Metrics logged on different cadences
    (``actor/*`` every log_interval vs ``eval/*`` every eval_interval) have
    an empty intersection, so a single multi-key call returns zero rows.
    Fetch each metric separately and outer-merge on ``_step`` instead.
    """
    frames: list[pd.DataFrame] = []
    for metric in metrics:
        try:
            d = run.history(keys=[metric], samples=max_samples,
                            pandas=True, x_axis="_step")
        except Exception as exc:  # pragma: no cover - tolerate API quirks
            print(f"[warn] history for {metric!r} failed ({exc}); skipping.",
                  file=sys.stderr)
            continue
        if "_step" not in d.columns or metric not in d.columns:
            continue
        d = d[["_step", metric]].dropna(subset=["_step"])
        if d.empty:
            continue
        d["_step"] = d["_step"].astype(int)
        frames.append(d.drop_duplicates(subset=["_step"]))

    if not frames:
        raise RuntimeError(
            "No metric returned any history rows. Was W&B logging skipped?"
        )

    df = frames[0]
    for other in frames[1:]:
        df = df.merge(other, on="_step", how="outer")
    return df.sort_values("_step").reset_index(drop=True)


def _bucketize(df: pd.DataFrame, bucket: int) -> pd.DataFrame:
    df = df.copy()
    df["_bucket_start"] = (df["_step"] // bucket) * bucket
    df["_bucket_end"] = df["_bucket_start"] + bucket
    return df


def _summary_for_metric(df: pd.DataFrame, metric: str,
                        quantiles: tuple[float, ...]) -> pd.DataFrame:
    series = df[metric]
    mask = series.notna()
    if not mask.any():
        return pd.DataFrame()
    sub = df.loc[mask, ["_bucket_start", "_bucket_end", metric]]
    grouped = sub.groupby(["_bucket_start", "_bucket_end"], sort=True)[metric]

    rows = []
    for (start, end), g in grouped:
        row: dict[str, float | int] = {
            "step_start": int(start),
            "step_end": int(end),
            "count": int(g.shape[0]),
            "mean": float(g.mean()),
            "std": float(g.std(ddof=0)) if g.shape[0] > 1 else 0.0,
            "min": float(g.min()),
            "max": float(g.max()),
        }
        qs = g.quantile(list(quantiles)).to_dict()
        for q, v in qs.items():
            label = f"p{int(round(100 * q)):02d}"
            row[label] = float(v)
        rows.append(row)
    out = pd.DataFrame(rows)
    ordered = (
        ["step_start", "step_end", "count", "mean", "std"]
        + [f"p{int(round(100 * q)):02d}" for q in quantiles]
        + ["min", "max"]
    )
    return out[ordered]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Print step-bucketed quantiles of W&B metrics for an OGPO run."
    )
    ap.add_argument("--entity", default=DEFAULT_ENTITY,
                    help=f"W&B entity (default: {DEFAULT_ENTITY})")
    ap.add_argument("--project", default=DEFAULT_PROJECT,
                    help=f"W&B project name (default: {DEFAULT_PROJECT})")
    ap.add_argument("--run-name", default=DEFAULT_RUN_NAME,
                    help=f"W&B run display name (default: {DEFAULT_RUN_NAME})")
    ap.add_argument("--bucket", type=int, default=500,
                    help="Step bucket width (default: 500)")
    ap.add_argument(
        "--metrics", nargs="+", default=list(DEFAULT_METRICS),
        help=(
            "Metric names or fnmatch patterns to summarize. "
            "Defaults to the OGPO actor/critic/eval diagnostics."
        ),
    )
    ap.add_argument(
        "--quantiles", nargs="+", type=float, default=list(DEFAULT_QUANTILES),
        help="Quantiles to report (default: 0.05 0.25 0.5 0.75 0.95)",
    )
    ap.add_argument(
        "--max-samples", type=int, default=100_000,
        help="Fallback cap for history() if scan_history is unavailable.",
    )
    ap.add_argument(
        "--csv", default=None,
        help="If set, also write the long-form summary to this CSV path.",
    )
    args = ap.parse_args()

    api = wandb.Api(timeout=60)
    run = _resolve_run(api, args.entity, args.project, args.run_name)
    print(f"Run: {run.entity}/{run.project}/{run.id}  ({run.name})")
    print(f"State: {run.state}  Steps: {run.summary.get('_step', '?')}")

    # Discover which metric names exist on the run. ``run.summary`` reflects
    # the *last* logged value per key, which is enough for *finished* runs
    # but lags for running ones — newly-introduced keys may not be in the
    # summary cache yet. Combine summary with ``run.history(samples=1)``
    # column names as a more authoritative discovery source.
    available_set = {k for k in run.summary.keys() if not k.startswith("_")}
    try:
        sample = run.history(samples=1, pandas=True)
        available_set.update(
            c for c in sample.columns if not c.startswith("_")
        )
    except Exception as exc:  # pragma: no cover - tolerate API quirks
        print(f"[warn] history(samples=1) failed ({exc}); "
              "falling back to summary keys only.", file=sys.stderr)
    available = sorted(available_set)
    selected = _select_metric_columns(available, args.metrics)
    if not selected:
        print("[err] No metrics matched. Available keys (first 60):",
              file=sys.stderr)
        for k in available[:60]:
            print(f"    {k}", file=sys.stderr)
        sys.exit(2)
    print(f"Metrics ({len(selected)}): {', '.join(selected)}")
    print(f"Bucket width: {args.bucket} steps")
    print()

    df = _fetch_history(run, selected, args.max_samples)
    df = _bucketize(df, args.bucket)

    qs = tuple(sorted(set(args.quantiles)))

    long_rows: list[pd.DataFrame] = []
    for metric in selected:
        summary = _summary_for_metric(df, metric, qs)
        if summary.empty:
            print(f"--- {metric}: no data ---")
            print()
            continue
        print(f"=== {metric} ===")
        # Use to_string so all rows print even if there are many.
        with pd.option_context("display.float_format", "{:.4g}".format,
                                "display.width", 200,
                                "display.max_rows", None):
            print(summary.to_string(index=False))
        print()
        if args.csv is not None:
            summary = summary.assign(metric=metric)
            long_rows.append(summary)

    if args.csv is not None and long_rows:
        long_df = pd.concat(long_rows, ignore_index=True)
        cols = ["metric"] + [c for c in long_df.columns if c != "metric"]
        long_df = long_df[cols]
        long_df.to_csv(args.csv, index=False)
        print(f"Wrote long-form CSV: {args.csv}")


if __name__ == "__main__":
    main()
