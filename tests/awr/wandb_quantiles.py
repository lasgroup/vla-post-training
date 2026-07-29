#!/usr/bin/env python3
"""Print step-bucketed quantiles of AWR debug metrics from a wandb run.

Usage:
    python tests/awr/wandb_quantiles.py
    python tests/awr/wandb_quantiles.py --run-name <other-run>
    python tests/awr/wandb_quantiles.py --list-metrics
    python tests/awr/wandb_quantiles.py --entity <ent> --project <proj> --run-name <run>

Defaults target the AWR run launched by scripts/awr_libero_babel.sh
(``diverse-data-synthesis/openpi``,
``pi05_libero_online_aw_sft_libero_90_44_seed0``); override any of
``--entity / --project / --run-name`` on the CLI. Buckets default to 500
training steps (= one collect_interval), pass ``--bucket 1000`` to change.

What it prints
--------------
1. Per-metric tables indexed by ``[step_start, step_end)`` with ``count,
   mean, std, p05, p25, p50, p75, p95, min, max`` for the AWR
   actor/critic/collect diagnostics:

       actor/loss, actor/awr_loss, actor/chunked_loss, actor/sft_loss,
       actor/advantage_{mean,std,min,max,median,q_up,q_low},
       actor/normalizer_{bias,scale}, actor/grad_norm, actor/param_norm,
       critic/q_*, critic/value_*, online_buffer_size, success_rate*,
       eval/*.

2. An AWR weight report (``--no-weights`` to skip). The actor logs the raw
   advantage but *not* the exponentiated weight it actually trains on, so
   this reconstructs it from the logged advantage quantiles and the run's
   own config::

       score = exp(min(A / (scale * beta), weight_clip)) / advantage_scale

   where ``scale`` is ``actor/normalizer_scale`` when
   ``rl.normalize_advantages`` is on, else 1. The two failure modes this
   is meant to catch:

   * ``A / (scale * beta)`` has near-zero spread -> every sample gets
     weight ~1 and AWR silently degenerates into vanilla BC over the whole
     buffer, failures included.
   * ``A / (scale * beta)`` routinely exceeds ``weight_clip`` -> the exp
     saturates and the loss becomes a hard argmax over a handful of
     transitions in the batch.
"""

from __future__ import annotations

import argparse
import fnmatch
import math
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
import wandb


# Defaults for the current AWR debug run. Override any of these from CLI.
DEFAULT_ENTITY = "diverse-data-synthesis"
DEFAULT_PROJECT = "openpi"
DEFAULT_RUN_NAME = "pi05_libero_online_aw_sft_libero_90_44_seed0"


# Metrics the AWR actor/critic emit (see
# src/rl/advantage_weighted_sft/{update_actor,update_critic}.py and
# advantage_weighted_sft_learner.py for where the prefixes come from).
# Wildcards are matched against the full W&B metric name. Anything not
# present in the run is silently skipped.
DEFAULT_METRICS: tuple[str, ...] = (
    # actor losses (update_actor.loss_fn aux_data)
    "actor/loss",
    "actor/awr_loss",
    "actor/chunked_loss",
    "actor/sft_loss",
    # advantage distribution the weights are built from
    "actor/advantage_mean",
    "actor/advantage_std",
    "actor/advantage_median",
    "actor/advantage_min",
    "actor/advantage_max",
    "actor/advantage_q_up",
    "actor/advantage_q_low",
    # running normalizer state applied as `scale` in update_actor
    "actor/normalizer_scale",
    "actor/normalizer_bias",
    # optimisation health
    "actor/grad_norm",
    "actor/param_norm",
    # critic diagnostics: loss/value_mean/td_loss/mc_loss/td_weight/
    # grad_norm/param_norm for both the Q and the V head
    "critic/q_*",
    "critic/value_*",
    # buffer + collect/eval. collect_data logs unprefixed `success_rate`,
    # evaluate_policy_sweep logs `eval/cfg<scale>/success_rate[/<task>]`.
    "online_buffer_size",
    "success_rate",
    "success_rate/*",
    "eval/*",
)

DEFAULT_QUANTILES: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)

# Advantage quantile metric -> label used in the weight report.
_ADV_POINTS: tuple[tuple[str, str], ...] = (
    ("actor/advantage_q_low", "q_low"),
    ("actor/advantage_median", "p50"),
    ("actor/advantage_q_up", "q_up"),
    ("actor/advantage_max", "max"),
)


def _resolve_run(api: wandb.Api, entity: str, project: str, run_name: str,
                 run_id: str | None, select: str) -> "wandb.apis.public.Run":
    """Resolve a run, disambiguating the many restarts sharing one name.

    ``exp_name`` is reused across relaunches, so this project has a dozen
    runs with the same display name — most of them crashed within a few
    steps. Picking blindly lands on an empty one, so list the candidates
    and select by ``--select`` (or pin with ``--run-id``).
    """
    if run_id is not None:
        return api.run(f"{entity}/{project}/{run_id}")

    matches = list(api.runs(f"{entity}/{project}",
                            filters={"display_name": run_name}))
    if not matches:
        raise ValueError(
            f"No run with display name {run_name!r} in {entity}/{project}."
        )
    if len(matches) == 1:
        return matches[0]

    def steps(r) -> int:
        value = r.summary.get("_step")
        return int(value) if value is not None else -1

    print(f"[info] {len(matches)} runs match name {run_name!r} "
          f"(--run-id pins one):", file=sys.stderr)
    for r in sorted(matches, key=lambda r: r.created_at):
        print(f"    {r.id}  {r.state:<8} steps={steps(r):<7} "
              f"created={r.created_at}", file=sys.stderr)

    if select == "longest":
        chosen = max(matches, key=steps)
    elif select == "running":
        live = [r for r in matches if r.state == "running"]
        if not live:
            raise ValueError("--select running: no matching run is running.")
        chosen = max(live, key=lambda r: r.created_at)
    else:  # latest: newest run that actually logged something
        with_data = [r for r in matches if steps(r) >= 0] or matches
        chosen = max(with_data, key=lambda r: r.created_at)
    print(f"[info] --select {select} -> {chosen.id}", file=sys.stderr)
    return chosen


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


def _config_get(config: dict[str, Any], path: str, default: Any) -> Any:
    """Look up a dotted config path in the W&B run config.

    ``init_wandb`` uploads ``dataclasses.asdict(config)``, which W&B may
    store either nested (``config['rl']['beta']``) or flattened
    (``config['rl.beta']``) depending on client version. Try both.
    """
    if path in config:
        return config[path]
    node: Any = config
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _awr_weight_report(df: pd.DataFrame, config: dict[str, Any],
                       bucket: int) -> None:
    """Reconstruct the AWR weights the actor trained on, per step bucket."""
    beta = float(_config_get(config, "rl.beta", float("nan")))
    weight_clip = float(_config_get(config, "rl.weight_clip", float("nan")))
    advantage_scale = float(_config_get(config, "rl.advantage_scale", 1.0))
    weight_type = str(_config_get(config, "rl.advantage_weight_type", "exp"))
    normalize = bool(_config_get(config, "rl.normalize_advantages", False))

    print("=== AWR weight report ===")
    print(f"beta={beta:g}  weight_clip={weight_clip:g}  "
          f"advantage_scale={advantage_scale:g}  "
          f"weight_type={weight_type}  normalize_advantages={normalize}")
    if weight_type == "relu":
        print("weight_type=relu: score = relu(A / scale), no exp/clip. "
              "Columns below show A/scale directly.")
    if math.isnan(beta) or math.isnan(weight_clip):
        print("[warn] beta/weight_clip missing from run config; "
              "falling back to the script defaults of the launcher "
              "(beta=10, weight_clip=3).")
        beta = 10.0 if math.isnan(beta) else beta
        weight_clip = 3.0 if math.isnan(weight_clip) else weight_clip

    needed = [m for m, _ in _ADV_POINTS] + ["actor/advantage_std",
                                            "actor/advantage_mean"]
    missing = [m for m in needed if m not in df.columns]
    if missing:
        print(f"[warn] missing advantage metrics {missing}; "
              "weight report skipped.")
        print()
        return

    scale_col = ("actor/normalizer_scale"
                 if normalize and "actor/normalizer_scale" in df.columns
                 else None)

    rows = []
    cols = needed + ([scale_col] if scale_col else [])
    sub = df[["_bucket_start", "_bucket_end", *cols]].dropna(
        subset=["actor/advantage_std"]
    )
    for (start, end), g in sub.groupby(["_bucket_start", "_bucket_end"],
                                       sort=True):
        scale = float(g[scale_col].mean()) if scale_col else 1.0
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        denom = scale * beta
        row: dict[str, float | int] = {
            "step_start": int(start),
            "step_end": int(end),
            "scale": scale,
            "A_mean": float(g["actor/advantage_mean"].mean()),
            "A_std": float(g["actor/advantage_std"].mean()),
        }
        # Spread of the exponent is what actually differentiates samples;
        # a constant offset only rescales every weight equally.
        row["z_std"] = row["A_std"] / denom
        for metric, label in _ADV_POINTS:
            a = float(g[metric].mean())
            if weight_type == "relu":
                row[f"w_{label}"] = max(a / scale, 0.0)
                continue
            z = a / denom
            row[f"z_{label}"] = z
            row[f"w_{label}"] = math.exp(min(z, weight_clip)) / advantage_scale
        if weight_type != "relu":
            lo = row["w_q_low"] if row["w_q_low"] > 0 else float("nan")
            row["w_up/w_low"] = row["w_q_up"] / lo
            # Effective sample size of the weights, as a fraction of batch
            # size. Treating A as roughly normal makes w lognormal with
            # log-std z_std, so ESS/N = 1/(1+CV^2) = exp(-z_std^2). This is
            # the number to compare against F-SFT, whose 0/1 weights give
            # ESS/N = success_rate.
            row["w_cv"] = math.sqrt(max(math.expm1(row["z_std"] ** 2), 0.0))
            row["ess_frac"] = math.exp(-row["z_std"] ** 2)
            # How far the top of the batch sits past the clip threshold.
            row["clip_hit_q_up"] = float(row["z_q_up"] >= weight_clip)
            row["clip_hit_max"] = float(row["z_max"] >= weight_clip)
        rows.append(row)

    if not rows:
        print("[warn] no rows with advantage data; weight report skipped.")
        print()
        return

    report = pd.DataFrame(rows)
    with pd.option_context("display.float_format", "{:.4g}".format,
                           "display.width", 220,
                           "display.max_rows", None,
                           "display.max_columns", None):
        print(report.to_string(index=False))
    print()

    if weight_type == "relu":
        print()
        return

    # Verdicts on the two ways AWR quietly stops being AWR.
    z_std = report["z_std"].to_numpy()
    clip_frac = float(report["clip_hit_q_up"].mean())
    print(f"z_std (advantage spread in exponent units): "
          f"first bucket {z_std[0]:.4g}, last bucket {z_std[-1]:.4g}, "
          f"median {np.median(z_std):.4g}")
    if np.median(z_std) < 0.05:
        print("  -> DEGENERATE: weights are effectively uniform. AWR is "
              "training as plain BC on the whole buffer, failures included. "
              "Lower rl.beta (or turn on rl.normalize_advantages) so the "
              "advantage spread actually reaches the exponent.")
    elif np.median(z_std) > weight_clip:
        print("  -> SATURATED: the advantage spread exceeds weight_clip, so "
              "the exp is pinned at the clip for much of the batch and the "
              "loss behaves like an argmax over a few transitions. Raise "
              "rl.beta or rl.weight_clip.")
    else:
        print("  -> spread is in a usable range; if AWR still underperforms, "
              "suspect the critic (check critic/q_* vs critic/value_* below) "
              "rather than the weighting.")
    print(f"buckets where the q_up advantage hits weight_clip: "
          f"{clip_frac:.0%}")

    # F-SFT is the same loss with 0/1 weights on a success-only buffer, so
    # its ESS/N is just the success rate. Report the beta that would give
    # AWR the same selectivity on the advantage spread it actually has.
    ess_last = float(report["ess_frac"].iloc[-1])
    a_std_last = float(report["A_std"].iloc[-1])
    scale_last = float(report["scale"].iloc[-1])
    print(f"ESS/N of the AWR weights (last bucket): {ess_last:.4f} "
          f"— 1.0 means every sample contributes equally, i.e. plain BC.")
    for target in (0.5, 0.25, 0.1):
        # ESS/N = exp(-(A_std/(scale*beta))^2)  ->  solve for beta.
        beta_star = a_std_last / (scale_last * math.sqrt(-math.log(target)))
        print(f"  beta for ESS/N={target:g} (F-SFT-like selectivity at "
              f"{target:.0%} success): {beta_star:.4g}")
    print()


def _value_bounds(config: dict[str, Any]) -> tuple[float, float] | None:
    """Mirror src.rl.value_distribution.get_value_bounds for the run config."""
    lower = _config_get(config, "rl.critic.value_lower_bound", None)
    upper = _config_get(config, "rl.critic.value_upper_bound", None)
    if lower is not None and upper is not None:
        return float(lower), float(upper)
    if not _config_get(config, "collect.use_time_to_success_as_reward", False):
        return 0.0, 1.0
    discount = float(_config_get(config, "rl.discount", float("nan")))
    horizon = _config_get(config, "collect.max_episode_steps", None)
    if horizon is None or not math.isfinite(discount):
        return None
    T = int(horizon)
    floor = (-(1.0 - discount ** T) / (1.0 - discount)
             if discount < 1.0 else -float(T))
    return floor, 0.0


def _failure_fixed_point(config: dict[str, Any]) -> float | None:
    """Value of a state whose episode never terminates.

    Rewards are -1 per env step, accumulated over an action chunk and
    bootstrapped with discount**action_horizon, so V = r_chunk / (1 - gamma_chunk).
    """
    if not _config_get(config, "collect.use_time_to_success_as_reward", False):
        return None
    discount = float(_config_get(config, "rl.discount", float("nan")))
    horizon = _config_get(config, "model.action_horizon", None)
    if horizon is None or not math.isfinite(discount) or discount >= 1.0:
        return None
    gamma_chunk = discount ** int(horizon)
    reward_chunk = -(1.0 - gamma_chunk) / (1.0 - discount)
    return reward_chunk / (1.0 - gamma_chunk)


def _critic_report(df: pd.DataFrame, config: dict[str, Any]) -> None:
    """Sanity-check the critic that produces A = Q - V."""
    q_col, v_col = "critic/q_value_mean", "critic/value_value_mean"
    if q_col not in df.columns or v_col not in df.columns:
        return
    sub = df[["_bucket_start", "_bucket_end", q_col, v_col,
              *[c for c in ("critic/q_td_loss", "critic/q_mc_loss",
                            "critic/q_td_weight", "critic/q_grad_norm",
                            "critic/value_td_loss", "critic/value_mc_loss",
                            "actor/advantage_std")
                if c in df.columns]]].dropna(subset=[q_col, v_col])
    if sub.empty:
        return
    print("=== critic report (A = Q - V) ===")
    agg = sub.groupby(["_bucket_start", "_bucket_end"], sort=True).mean()
    agg = agg.reset_index().rename(columns={"_bucket_start": "step_start",
                                            "_bucket_end": "step_end"})
    agg["Q-V"] = agg[q_col] - agg[v_col]
    if "actor/advantage_std" in agg.columns:
        # The batchwise advantage spread should not be dwarfed by the gap
        # between the mean Q and mean V; if it is, the critics disagree on
        # level rather than on which actions are good.
        agg["|Q-V|/A_std"] = agg["Q-V"].abs() / agg["actor/advantage_std"]
    with pd.option_context("display.float_format", "{:.4g}".format,
                           "display.width", 220,
                           "display.max_rows", None,
                           "display.max_columns", None):
        print(agg.to_string(index=False))

    # get_value_bounds assumes the episode ends at max_episode_steps, but
    # _save_episode_in_buffer bootstraps through truncation (discount is
    # only zeroed on true termination) and fix_mc_returns relabels a
    # constant-reward episode as reward/(1-discount). Both make the value
    # of a never-succeeding state the *infinite*-horizon fixed point, which
    # sits legitimately below the finite-horizon floor. Print both so a
    # critic parked on the failure fixed point is not misread as diverging.
    bounds = _value_bounds(config)
    if bounds is not None:
        lower, upper = bounds
        q_last = float(agg[q_col].iloc[-1])
        v_last = float(agg[v_col].iloc[-1])
        print(f"finite-horizon value bounds: [{lower:.4g}, {upper:.4g}]  "
              f"last bucket Q={q_last:.4g} V={v_last:.4g}")
        fixed_point = _failure_fixed_point(config)
        if fixed_point is not None:
            print(f"never-succeeds fixed point (bootstrapped truncation): "
                  f"{fixed_point:.4g}")
            if abs(v_last - fixed_point) < 0.05 * abs(fixed_point):
                print("  -> PARKED ON THE FAILURE FIXED POINT: the critic "
                      "predicts near-total failure everywhere. This is the "
                      "TD/MC target doing what it says, not divergence — "
                      "but it means the critic is close to a constant "
                      "function, so A = Q - V has almost no spread.")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Print step-bucketed quantiles of W&B metrics for an AWR run."
    )
    ap.add_argument("--entity", default=DEFAULT_ENTITY,
                    help=f"W&B entity (default: {DEFAULT_ENTITY})")
    ap.add_argument("--project", default=DEFAULT_PROJECT,
                    help=f"W&B project name (default: {DEFAULT_PROJECT})")
    ap.add_argument("--run-name", default=DEFAULT_RUN_NAME,
                    help=f"W&B run display name (default: {DEFAULT_RUN_NAME})")
    ap.add_argument("--run-id", default=None,
                    help="W&B run id; bypasses name lookup and --select.")
    ap.add_argument("--select", default="latest",
                    choices=("latest", "longest", "running"),
                    help=("Which run to use when several share the name: "
                          "newest with data (default), most steps, or the "
                          "one currently running."))
    ap.add_argument("--bucket", type=int, default=500,
                    help="Step bucket width (default: 500 = collect_interval)")
    ap.add_argument(
        "--metrics", nargs="+", default=list(DEFAULT_METRICS),
        help=(
            "Metric names or fnmatch patterns to summarize. "
            "Defaults to the AWR actor/critic/eval diagnostics."
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
    ap.add_argument("--list-metrics", action="store_true",
                    help="Print every metric key the run logged, then exit.")
    ap.add_argument("--no-weights", action="store_true",
                    help="Skip the reconstructed AWR weight report.")
    ap.add_argument(
        "--csv", default=None,
        help="If set, also write the long-form summary to this CSV path.",
    )
    args = ap.parse_args()

    api = wandb.Api(timeout=60)
    run = _resolve_run(api, args.entity, args.project, args.run_name,
                       args.run_id, args.select)
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

    if args.list_metrics:
        print(f"Available metrics ({len(available)}):")
        for k in available:
            print(f"    {k}")
        return

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

    if not args.no_weights:
        run_config = dict(run.config)
        _awr_weight_report(df, run_config, args.bucket)
        _critic_report(df, run_config)

    if args.csv is not None and long_rows:
        long_df = pd.concat(long_rows, ignore_index=True)
        cols = ["metric"] + [c for c in long_df.columns if c != "metric"]
        long_df = long_df[cols]
        long_df.to_csv(args.csv, index=False)
        print(f"Wrote long-form CSV: {args.csv}")


if __name__ == "__main__":
    main()
