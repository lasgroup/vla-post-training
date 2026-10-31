#!/usr/bin/env python3
"""Dump and diagnose EVERY W&B metric of an OGPO-pi05 run (scripts/ogpo/libero_<task>_pi05.sh).

Usage (OGPO's .venv has no pandas; use the babel venv's interpreter, never `uv run`):
    PY=/home/mananaga/VLA/manan_babel/vla-post-training/.venv/bin/python
    $PY scripts/ogpo/wandb_analysis.py
    $PY scripts/ogpo/wandb_analysis.py --list-runs
    $PY scripts/ogpo/wandb_analysis.py --list-metrics
    $PY scripts/ogpo/wandb_analysis.py --run-ids 8pl6bf0v --bucket 2000
    $PY scripts/ogpo/wandb_analysis.py --metrics 'online_agent/critic/*' --dump-csv /tmp/hist.csv
    $PY scripts/ogpo/wandb_analysis.py --run-ids a b     # adds a cross-run comparison

Same shape as vla-post-training/tests/wandb_ogpo_privileged.py (config audit,
logging health, step-bucketed quantile tables, reconstructed-quantity reports,
cross-run comparison), retargeted at what ogpo/runners/pi05_online_runner.py logs:

  online_agent/actor/*   info of ONE update, sampled every intervals.log_interval
  online_agent/critic/*  (the runner keeps the last update's info, it does not
                         average over the window, so each row is one batch)
  online_agent/{sec_per_update, rb_size, rb_success_frac}
  env/{success, length, episode}  one row per training episode (BoN rollouts)
  eval/*      ODE, online actor,       every intervals.eval_interval (+ step 0)
  eval_sde/*  best-of-N, target actor + target critic, same cadence

What it prints
--------------
1. Run + config audit (the pi05 launch flags as they landed in run.config).
2. Logging health: rows, first/last step, stride, NaN fraction per metric.
3. Expected-key coverage and unrecognised keys.
4. Per-metric quantile tables per step bucket, grouped by namespace.
5. Outcome report: eval ODE vs BoN with binomial std errors, training-episode
   success in rolling windows, replay-buffer success fraction.
6. Critic report: Q level against the reward-scale reference values
   (-1/step, +bonus on success, gamma), the success-vs-failure value gap the
   critic has to resolve, and the advantage signal-to-noise
   (critic TD RMSE vs within-group Q spread that PPO actually consumes).
7. Actor report: PPO ratio / clip saturation vs clip_epsilon, approx_kl,
   grad norm vs the actor clip, logprob drift, action clipping, BC loss.
8. Collapse report: splits the run at the best eval and ranks every metric by
   its standardized shift between the rising and falling phases.
9. Cross-run comparison when several runs are given.
"""

from __future__ import annotations

import argparse
import fnmatch
import math
import sys
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import wandb

DEFAULT_ENTITY = "diverse-data-synthesis"
DEFAULT_PROJECT = "OGPO_libero_pixel"
DEFAULT_RUN_NAMES = ("ogpo_libero82_pi05seed82723",)  # task 59 run: --run-ids 8pl6bf0v
DEFAULT_QUANTILES: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)

A = "online_agent/actor/"
C = "online_agent/critic/"
D = "online_agent/diag/"

METRIC_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("eval ODE (online actor)", ("eval/*",)),
    ("eval best-of-N (target actor + target critic)", ("eval_sde/*",)),
    ("training episodes (BoN rollouts)", ("env/*",)),
    ("actor: PPO / logprob", (A + "ratio*", A + "approx_kl", A + "lp*", A + "old_lp",
                              A + "pg_loss", A + "grad_norm")),
    ("actor: advantage / Q seen by the actor", (A + "adv*", A + "q_values", A + "grpo_*")),
    ("actor: BC / sampling / entropy", (A + "*",)),
    ("critic: accuracy vs Monte-Carlo return", (C + "q_mc_corr", C + "mc_*", C + "q_minus_mc",
                                                C + "q_success", C + "q_fail", C + "success_frac")),
    ("critic", (C + "*",)),
    ("gradient split / actor drift (log-time diagnostic)", (D + "*",)),
    ("runner", ("online_agent/*",)),
)

KNOWN_PATTERNS: tuple[str, ...] = (
    "eval/success", "eval/return", "eval/length", "eval/step",
    "eval_sde/success", "eval_sde/return", "eval_sde/length", "eval_sde/step",
    "env/success", "env/length", "env/episode", "env/step",
    A + "pg_loss", A + "bc_loss", A + "entropy", A + "lp", A + "lp_min", A + "lp_max",
    A + "old_lp", A + "adv", A + "adv_min", A + "adv_max", A + "adv_std", A + "adv_zero_frac", A + "q_values",
    A + "sigmas", A + "actions", A + "actions_std", A + "actions_clipped_frac",
    A + "grpo_adv_group_std", A + "grpo_adv_between_group_std", A + "grad_norm",
    A + "ratio", A + "ratio_std", A + "ratio_min", A + "ratio_max", A + "approx_kl",
    A + "ratio_clipped_upper", A + "ratio_clipped_lower", A + "ratio_clipped_total",
    A + "info_*",
    C + "td_loss", C + "q_mean", C + "q_max", C + "q_min", C + "next_q_mean",
    C + "next_actions_std", C + "grad/*",
    "online_agent/sec_per_update", "online_agent/rb_size", "online_agent/rb_success_frac",
    "online_agent/step",
    # added with window-averaged logging
    C + "target_q_mean", C + "q_head_std", C + "q_mc_corr", C + "mc_mean", C + "q_minus_mc",
    C + "q_success", C + "q_fail", C + "mc_success", C + "mc_fail", C + "success_frac",
    "online_agent/*_max", "online_agent/diag/*", "online_agent/success_flag",
    "online_agent/updates_in_window", "env/return", "env/mc_return_0",
    # round schedule (training.collection_mode=rounds): env/* are per-round means, step = update count
    "env/round", "env/episodes", "env/env_steps", "env/collect_minutes",
    "online_agent/episodes", "online_agent/env_steps", "online_agent/round",
)
_PRESENCE_ONLY: tuple[str, ...] = ("*/step", "env/episode", A + "info_logprob_steps")


# --------------------------------------------------------------------------- run resolution

def _steps(run) -> int:
    try:
        return int(run.summary.get("_step"))
    except (TypeError, ValueError):
        return -1


def _resolve_run(api: wandb.Api, entity: str, project: str, run_name: str, select: str):
    """Exact display name, then substring / glob, then dump the project."""
    matches = list(api.runs(f"{entity}/{project}", filters={"display_name": run_name}))
    if not matches:
        needle = run_name.strip().lower()
        pool = list(api.runs(f"{entity}/{project}"))
        matches = [r for r in pool if needle in (r.name or "").lower()
                   or fnmatch.fnmatch((r.name or "").lower(), needle)]
        if not matches:
            print(f"[err] no run matching {run_name!r} in {entity}/{project}:", file=sys.stderr)
            for r in sorted(pool, key=lambda r: r.created_at, reverse=True)[:60]:
                print(f"    {r.id}  {r.state:<9} steps={_steps(r):<8} {r.created_at}  {r.name}",
                      file=sys.stderr)
            raise ValueError(f"No run matching {run_name!r}.")
    if len(matches) == 1:
        return matches[0]
    print(f"[info] {len(matches)} runs match {run_name!r} (--run-ids pins one):", file=sys.stderr)
    for r in sorted(matches, key=lambda r: r.created_at):
        print(f"    {r.id}  {r.state:<9} steps={_steps(r):<8} created={r.created_at}", file=sys.stderr)
    if select == "longest":
        chosen = max(matches, key=_steps)
    elif select == "running":
        live = [r for r in matches if r.state == "running"]
        if not live:
            raise ValueError("--select running: no matching run is running.")
        chosen = max(live, key=lambda r: r.created_at)
    else:
        with_data = [r for r in matches if _steps(r) >= 0] or matches
        chosen = max(with_data, key=lambda r: r.created_at)
    print(f"[info] --select {select} -> {chosen.id}", file=sys.stderr)
    return chosen


# --------------------------------------------------------------------------- config

def _cfg(config: dict[str, Any], path: str, default: Any = None) -> Any:
    """Dotted lookup; W&B stores the config nested or flattened depending on client."""
    if path in config:
        return config[path]
    node: Any = config
    for part in path.split("."):
        if isinstance(node, dict) and "value" in node and part not in node:
            node = node["value"]
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    if isinstance(node, dict) and set(node) == {"value"}:
        node = node["value"]
    return node


# (path, expected value or None to just print). Expected = PLAN.md decisions.
_CONFIG_AUDIT: tuple[tuple[str, Any], ...] = (
    ("env.name", None),
    ("env.post_success_steps", None),
    ("training.online_steps", None),
    ("buffer.start_training", None),
    ("buffer.size", None),
    ("intervals.eval_interval", None),
    ("intervals.log_interval", None),
    ("eval.eval_episodes", None),
    ("success_buffer.use_success_buffer", True),
    ("success_buffer.batch_size", None),
    ("agent.horizon_length", 5),
    ("agent.pi05_action_horizon", 10),
    ("agent.pi05_lr", None),
    ("agent.pi05_clip_grad_norm", None),
    ("agent.pi05_weight_decay", None),
    ("agent.batch_size", None),
    ("agent.ppo_batch_size", None),
    ("agent.group_num_samples", None),
    ("agent.best_of_n", None),
    ("agent.clip_epsilon", None),
    ("agent.clip_min_epsilon_multiplier", None),
    ("agent.adv_strategy", None),
    ("agent.normalize_group", None),
    ("agent.entropy_coeff", None),
    ("agent.bc_coeff", None),
    ("agent.use_bc_regularization", None),
    ("agent.discount", None),
    ("agent.tau", None),
    ("agent.actor_tau", None),
    ("agent.num_qs", None),
    ("agent.q_agg", None),
    ("agent.subsample_bon", None),
    ("agent.use_success_buffer_q", None),
    ("agent.ppo_lr", None),
    ("agent.critic_scheduler", None),
    ("agent.clip_grad_norm", None),
    ("agent.value_hidden_dims", None),
    ("agent.flow_steps", None),
    ("agent.ft_flow_steps", None),
    ("agent.use_tapered_noise", True),
    ("agent.use_constant_noise", False),
    ("agent.error_correct_sde_to_ode", None),
    ("agent.constant_noise_std", None),
    ("agent.clip_intermediate_actions", None),
    ("agent.denoised_clip_value", None),
    ("agent.randn_clip_value", None),
    ("agent.normalize_denoising_horizon", None),
    ("agent.normalize_act_space_dimension", None),
)


def _config_report(config: dict[str, Any]) -> None:
    print("=== config audit (as landed in run.config) ===")
    rows = []
    for path, expected in _CONFIG_AUDIT:
        got = _cfg(config, path, "<MISSING>")
        verdict = "" if expected is None else ("OK" if got == expected else f"!! expected {expected!r}")
        rows.append({"key": path, "value": got, "check": verdict})
    with pd.option_context("display.max_rows", None, "display.width", 200, "display.max_colwidth", 70):
        print(pd.DataFrame(rows).to_string(index=False))
    b, g = _cfg(config, "agent.ppo_batch_size", 0), _cfg(config, "agent.group_num_samples", 0)
    h, gamma = _cfg(config, "agent.horizon_length", 0), _cfg(config, "agent.discount", float("nan"))
    print(f"\nPPO chains / update = ppo_batch {b} x G {g} = {int(b or 0) * int(g or 0)}; "
          f"executed chunk = {h} steps x 7 dims; TD bootstrap gamma^H = {gamma ** int(h or 0):.4g}")
    print()


def _expected_metrics(config: dict[str, Any]) -> dict[str, str]:
    exp = {k: "always logged by pi05_online_runner" for k in (
        "eval/success", "eval/return", "eval/length", "eval_sde/success", "eval_sde/return",
        "eval_sde/length", "env/success", "env/length", "env/episode",
        A + "pg_loss", A + "bc_loss", A + "ratio", A + "ratio_std", A + "approx_kl",
        A + "ratio_clipped_total", A + "lp", A + "old_lp", A + "adv_std", A + "q_values",
        A + "grpo_adv_group_std", A + "grpo_adv_between_group_std", A + "grad_norm",
        A + "actions_clipped_frac", A + "sigmas", A + "entropy",
        C + "td_loss", C + "q_mean", C + "q_max", C + "q_min", C + "next_q_mean",
        "online_agent/sec_per_update", "online_agent/rb_size", "online_agent/rb_success_frac")}
    # window-averaged logging + diagnostics (absent on runs from before they were added)
    for k in ("q_mc_corr", "q_minus_mc", "q_success", "q_fail", "mc_success", "mc_fail"):
        exp[C + k] = "critic accuracy vs MC return (mc_returns in the buffer)"
    for k in ("grad_pg_norm", "grad_bc_norm", "grad_cos_pg_bc", "drift_online_target_rel", "drift_online_init_rel"):
        exp[D + k] = "agent.grad_diagnostics at each log window"
    exp["env/return"] = "per-episode undiscounted return"
    return exp


# --------------------------------------------------------------------------- history

def _select(columns: Iterable[str], patterns: Iterable[str]) -> list[str]:
    cols, out, seen = list(columns), [], set()
    for pat in patterns:
        for col in cols:
            if col not in seen and fnmatch.fnmatchcase(col, pat):
                out.append(col)
                seen.add(col)
    return out


def _discover_metrics(run) -> list[str]:
    keys = {k for k in run.summary.keys() if not k.startswith("_")}
    try:
        keys.update(c for c in run.history(samples=1, pandas=True).columns if not c.startswith("_"))
    except Exception as exc:  # pragma: no cover
        print(f"[warn] history(samples=1) failed ({exc}); summary keys only.", file=sys.stderr)
    return sorted(keys)


def _fetch_history(run, metrics: list[str], max_samples: int) -> pd.DataFrame:
    """One request per metric, outer-merged on _step (multi-key history() inner-joins)."""
    frames, failed = [], []
    for metric in metrics:
        try:
            d = run.history(keys=[metric], samples=max_samples, pandas=True, x_axis="_step")
        except Exception as exc:  # pragma: no cover
            failed.append(f"{metric} ({exc})")
            continue
        if "_step" not in d.columns or metric not in d.columns:
            continue
        d = d[["_step", metric]].dropna(subset=["_step"])
        if d.empty:
            continue
        d["_step"] = d["_step"].astype(int)
        # W&B hands logged NaN / Infinity back as strings, which makes the column object dtype.
        d[metric] = pd.to_numeric(d[metric], errors="coerce")
        frames.append(d.drop_duplicates(subset=["_step"]))
    if failed:
        print(f"[warn] history failed for {len(failed)} metric(s): {', '.join(failed[:5])}", file=sys.stderr)
    if not frames:
        raise RuntimeError("No metric returned any history rows.")
    df = frames[0]
    for other in frames[1:]:
        df = df.merge(other, on="_step", how="outer")
    return df.sort_values("_step").reset_index(drop=True)


def _bucketize(df: pd.DataFrame, bucket: int) -> pd.DataFrame:
    df = df.copy()
    df["_bucket_start"] = (df["_step"] // bucket) * bucket
    df["_bucket_end"] = df["_bucket_start"] + bucket
    return df


def _show(df: pd.DataFrame, fmt: str = "{:.4g}", index: bool = False) -> None:
    with pd.option_context("display.float_format", fmt.format, "display.width", 250,
                           "display.max_rows", None, "display.max_columns", None):
        print(df.to_string(index=index))


def _bucket_mean(df: pd.DataFrame, cols: Sequence[str], dropna_on: str | None = None) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    sub = df[["_bucket_start", "_bucket_end", *cols]]
    sub = sub.dropna(subset=[dropna_on]) if dropna_on in cols else sub.dropna(subset=cols, how="all")
    if sub.empty:
        return pd.DataFrame()
    agg = sub.groupby(["_bucket_start", "_bucket_end"], sort=True).mean().reset_index()
    return agg.rename(columns={"_bucket_start": "step_start", "_bucket_end": "step_end"})


def _short(col: str) -> str:
    return col.replace(A, "a/").replace(C, "c/").replace("online_agent/", "")


# --------------------------------------------------------------------------- reports

def _logging_health(df: pd.DataFrame, config: dict[str, Any], expected: dict[str, str]) -> None:
    rows = []
    for metric in [c for c in df.columns if not c.startswith("_")]:
        col = df[metric]
        present = col.notna()
        steps = df.loc[present, "_step"].to_numpy()
        rows.append({
            "metric": metric, "n_nonnull": int(present.sum()),
            "first_step": int(steps.min()) if steps.size else -1,
            "last_step": int(steps.max()) if steps.size else -1,
            "stride": int(np.nanmedian(np.diff(steps))) if steps.size > 1 else -1,
            "nan_frac_of_own_rows": float(np.isnan(col[present].astype(float)).mean()) if present.any() else 1.0,
            "last_value": float(col[present].iloc[-1]) if present.any()
                          and pd.api.types.is_numeric_dtype(col) else np.nan,
        })
    health = pd.DataFrame(rows).sort_values("metric").reset_index(drop=True)
    print("=== logging health ===")
    print(f"configured: log_interval={_cfg(config, 'intervals.log_interval')}  "
          f"eval_interval={_cfg(config, 'intervals.eval_interval')}  "
          f"start_training={_cfg(config, 'buffer.start_training')}")
    print("online_agent/* should stride at log_interval and start at the first multiple after training "
          "starts; env/* strides at the episode length (<=400); eval*/ at eval_interval (plus step 0).")
    print("NOTE: online_agent/* is the info of the LAST update in each log window, not a window mean, "
          "so each row is one batch — expect high step-to-step noise.")
    _show(health)
    dead = health.loc[health["n_nonnull"] == 0, "metric"].tolist()
    if dead:
        print(f"\n!! {len(dead)} key(s) exist but carry no data: {dead}")
    # q/mc on success or failure transitions are NaN by design on windows without that kind
    nan_by_design = (C + "q_success", C + "mc_success", C + "q_fail", C + "mc_fail")
    nanny = health.loc[(health["nan_frac_of_own_rows"] > 0) & ~health["metric"].isin(nan_by_design),
                       "metric"].tolist()
    if nanny:
        print(f"\n!! NaN values inside logged rows (numerical blow-up?): {nanny}")
    last = int(health["last_step"].max())
    stalled = health[(health["n_nonnull"] > 0) & (health["last_step"] < 0.8 * last)
                     & ~health["metric"].str.startswith(("eval", "env"))]
    if not stalled.empty:
        print(f"\n!! stopped logging well before step {last}: {stalled['metric'].tolist()}")
    print()


def _coverage_report(available: Sequence[str], df: pd.DataFrame, expected: dict[str, str]) -> None:
    have = set(available)
    live = {c for c in df.columns if not c.startswith("_") and df[c].notna().any()}
    print("=== expected-key coverage ===")
    absent = sorted(k for k in expected if k not in have)
    nan_only = sorted(k for k in expected if k in have and k not in live)
    print(f"present with data: {sum(k in live for k in expected)}/{len(expected)}")
    if nan_only:
        print(f"logged but empty: {nan_only}")
    print(f"ABSENT: {absent}" if absent else "no expected key is missing.")
    unknown = sorted(k for k in have if not any(fnmatch.fnmatchcase(k, p) for p in KNOWN_PATTERNS))
    print("\n=== unrecognised keys ===")
    print("\n".join(f"    {k}" for k in unknown) if unknown else "    none")
    print()


def _summary_for_metric(df: pd.DataFrame, metric: str, quantiles: tuple[float, ...]) -> pd.DataFrame:
    series = df[metric]
    mask = series.notna()
    if not mask.any() or not pd.api.types.is_numeric_dtype(series):
        return pd.DataFrame()
    rows = []
    for (start, end), g in df.loc[mask, ["_bucket_start", "_bucket_end", metric]].groupby(
            ["_bucket_start", "_bucket_end"], sort=True)[metric]:
        row = {"step_start": int(start), "step_end": int(end), "count": int(g.shape[0]),
               "mean": float(g.mean()), "std": float(g.std(ddof=0)) if g.shape[0] > 1 else 0.0,
               "min": float(g.min()), "max": float(g.max())}
        for q, v in g.quantile(list(quantiles)).to_dict().items():
            row[f"p{int(round(100 * q)):02d}"] = float(v)
        rows.append(row)
    ordered = (["step_start", "step_end", "count", "mean", "std"]
               + [f"p{int(round(100 * q)):02d}" for q in quantiles] + ["min", "max"])
    return pd.DataFrame(rows)[ordered]


def _print_group_tables(df: pd.DataFrame, metrics: Sequence[str], quantiles: tuple[float, ...],
                        full: bool, csv_rows: list[pd.DataFrame] | None, run_name: str) -> None:
    remaining = list(metrics)
    groups = [*METRIC_GROUPS, ("other", ("*",))]
    for title, patterns in groups:
        group = _select(remaining, patterns)
        if not group:
            continue
        remaining = [m for m in remaining if m not in set(group)]
        print(f"########## {title} ##########\n")
        for metric in group:
            if not full and any(fnmatch.fnmatchcase(metric, p) for p in _PRESENCE_ONLY):
                continue
            summary = _summary_for_metric(df, metric, quantiles)
            if summary.empty:
                print(f"--- {metric}: no numeric data ---\n")
                continue
            print(f"=== {metric} ===")
            _show(summary)
            print()
            if csv_rows is not None:
                csv_rows.append(summary.assign(metric=metric, run=run_name))


def _binom_se(p: float, n: int) -> float:
    return math.sqrt(max(p * (1 - p), 1e-12) / max(n, 1))


def _outcome_report(df: pd.DataFrame, config: dict[str, Any], window: int) -> None:
    n_eval = int(_cfg(config, "eval.eval_episodes", 20) or 20)
    print("=== outcome report ===")
    cols = [c for c in ("eval/success", "eval_sde/success", "eval/length", "eval_sde/length",
                        "eval/return", "eval_sde/return") if c in df.columns]
    ev = df[["_step", *cols]].dropna(subset=[c for c in cols if c.endswith("success")], how="all")
    if not ev.empty:
        ev = ev.copy()
        for c in ("eval/success", "eval_sde/success"):
            if c in ev.columns:
                ev[c + "_se"] = [_binom_se(p, n_eval) for p in ev[c]]
        print(f"eval (n={n_eval} episodes each; se = binomial std error — differences under ~2 se are noise):")
        _show(ev.rename(columns={"_step": "step"}), "{:.3f}")
        for c in ("eval/success", "eval_sde/success"):
            if c in ev.columns:
                s = ev[["_step", c]].dropna()
                best = s.loc[s[c].idxmax()]
                print(f"  {c:<18} start {s[c].iloc[0]:.3f}  best {best[c]:.3f} @ step {int(best['_step'])}  "
                      f"last {s[c].iloc[-1]:.3f}")

    if "env/success" in df.columns:
        ep = df[["_step", "env/success", *[c for c in ("env/length",) if c in df.columns]]].dropna(
            subset=["env/success"]).reset_index(drop=True)
        ep["window"] = ep.index // window
        roll = ep.groupby("window").agg(step_first=("_step", "min"), step_last=("_step", "max"),
                                        episodes=("env/success", "size"), success=("env/success", "mean"))
        if "env/length" in ep.columns:
            succ_len = ep[ep["env/success"] > 0.5].groupby("window")["env/length"].mean()
            roll["succ_length"] = succ_len
        roll["se"] = [_binom_se(p, n) for p, n in zip(roll["success"], roll["episodes"])]
        print(f"\ntraining-episode success (BoN rollouts, target actor), windows of {window} episodes:")
        _show(roll.reset_index(drop=True), "{:.3f}")
        print(f"  total episodes {len(ep)}, successes {int(ep['env/success'].sum())} "
              f"({ep['env/success'].mean():.3f})")

    if "online_agent/rb_success_frac" in df.columns:
        rb = df[["_step", "online_agent/rb_size", "online_agent/rb_success_frac"]].dropna()
        print("\nreplay buffer (fraction of stored TRANSITIONS from successful episodes):")
        _show(rb.iloc[:: max(1, len(rb) // 12)].rename(columns={"_step": "step"}), "{:.4g}")
    print()


def _critic_report(df: pd.DataFrame, config: dict[str, Any], bonus: float) -> None:
    need = C + "q_mean"
    if need not in df.columns:
        return
    gamma = float(_cfg(config, "agent.discount", 0.99))
    H = int(_cfg(config, "agent.horizon_length", 5))
    print("=== critic report ===")
    # Reference values under reward -1/step, +bonus on the success step, termination on success,
    # truncation at 400 steps bootstrapping (so a never-succeeding state sits at -1/(1-gamma)).
    fail_fp = -1.0 / (1.0 - gamma)
    print(f"reward scale: -1/step, +{bonus} on success (terminates), truncation bootstraps; gamma={gamma}, "
          f"TD over {H} env steps (gamma^H={gamma ** H:.4g}).")
    print(f"  never-succeeds fixed point            Q = -1/(1-gamma) = {fail_fp:.2f}")
    for k in (25, 50, 100, 150, 250):
        q = -(1 - gamma ** (k - 1)) / (1 - gamma) + bonus * gamma ** (k - 1)
        print(f"  success in {k:>3} more steps           Q = {q:7.2f}   (gap to failure {q - fail_fp:6.2f})")
    print("  -> the critic must resolve these gaps; the policy only ever sees within-group Q differences.\n")

    cols = [C + "q_mean", C + "q_min", C + "q_max", C + "next_q_mean", C + "td_loss", C + "grad/norm",
            C + "next_actions_std", A + "q_values", A + "grpo_adv_group_std", A + "grpo_adv_between_group_std",
            A + "adv_std"]
    agg = _bucket_mean(df, cols, dropna_on=C + "q_mean")
    if agg.empty:
        return
    if C + "td_loss" in agg.columns:
        agg["td_rmse"] = np.sqrt(agg[C + "td_loss"])
    if C + "q_max" in agg.columns:
        agg["q_range"] = agg[C + "q_max"] - agg[C + "q_min"]
    if "td_rmse" in agg.columns and A + "grpo_adv_group_std" in agg.columns:
        # PPO consumes Q(s,a_i) - mean_i Q(s,a_i) over the G samples of ONE state. If that spread is
        # far below the critic's own TD error, the advantage ranking is dominated by critic noise.
        agg["td_rmse/within_group_std"] = agg["td_rmse"] / agg[A + "grpo_adv_group_std"]
    _show(agg.rename(columns=_short))
    if "td_rmse/within_group_std" in agg.columns:
        r = agg["td_rmse/within_group_std"].to_numpy()
        print(f"\ntd_rmse / within-group Q std: median {np.nanmedian(r):.3g}, last {r[-1]:.3g}")
        if np.nanmedian(r) > 10:
            print("  -> the within-group advantage is >10x smaller than the critic's TD error: PPO is "
                  "mostly following critic noise, not a real ranking of the G actions.")
    qm = agg[C + "q_mean"].to_numpy()
    if np.nanmax(qm) > bonus + 1 or np.nanmin(qm) < 1.5 * fail_fp:
        print(f"\n!! q_mean left the reachable range [{1.5 * fail_fp:.0f}, {bonus:.0f}] — divergence.")

    mc_cols = [C + c for c in ("q_mc_corr", "q_minus_mc", "q_success", "mc_success", "q_fail", "mc_fail",
                               "success_frac", "q_head_std")]
    mc = _bucket_mean(df, mc_cols, dropna_on=C + "q_mc_corr")
    if not mc.empty:
        mc["bias_success"] = mc[C + "q_success"] - mc[C + "mc_success"]
        mc["bias_fail"] = mc[C + "q_fail"] - mc[C + "mc_fail"]
        mc["q_gap(succ-fail)"] = mc[C + "q_success"] - mc[C + "q_fail"]
        mc["mc_gap(succ-fail)"] = mc[C + "mc_success"] - mc[C + "mc_fail"]
        print("\ncritic accuracy vs the observed discounted return-to-go (RB batches, window means).")
        print("  mc ignores the truncation bootstrap, so on truncated failures the TD fixed point sits BELOW mc "
              "(bias_fail < 0 is expected near the 400-step limit). Read q_mc_corr and the gaps, not raw bias.")
        _show(mc.rename(columns=_short))
        corr = mc[C + "q_mc_corr"].to_numpy()
        print(f"  q_mc_corr: first {corr[0]:.3g}  last {corr[-1]:.3g}  median {np.nanmedian(corr):.3g}")
        if np.nanmedian(corr) < 0.3:
            print("  -> the critic barely ranks states by their actual return.")
        if (mc["q_gap(succ-fail)"] < 0.5 * mc["mc_gap(succ-fail)"]).mean() > 0.5:
            print("  -> the critic compresses the success/failure value gap to under half of the real one.")
    print()


def _actor_report(df: pd.DataFrame, config: dict[str, Any]) -> None:
    if A + "ratio" not in df.columns:
        return
    eps = float(_cfg(config, "agent.clip_epsilon", float("nan")))
    mult = float(_cfg(config, "agent.clip_min_epsilon_multiplier", 1.0) or 1.0)
    gclip = float(_cfg(config, "agent.pi05_clip_grad_norm", float("nan")))
    cols = [A + c for c in ("pg_loss", "bc_loss", "ratio", "ratio_std", "ratio_min", "ratio_max",
                            "ratio_clipped_upper", "ratio_clipped_lower", "ratio_clipped_total",
                            "approx_kl", "lp", "old_lp", "grad_norm", "adv_std", "adv_zero_frac", "adv_min", "adv_max",
                            "sigmas", "entropy", "actions_std", "actions_clipped_frac")]
    agg = _bucket_mean(df, cols, dropna_on=A + "ratio")
    if agg.empty:
        return
    agg["lp-old_lp"] = agg[A + "lp"] - agg[A + "old_lp"]
    print("=== actor report (PPO health) ===")
    print(f"clip range = 1 +- {eps * mult:g}.  ratio_std comparable to that range means most samples sit "
          f"at the clip and carry no gradient. grad_norm above the actor clip ({gclip:g}) means every step "
          f"is rescaled to norm {gclip:g}.")
    _show(agg.rename(columns=_short))

    rs = df[A + "ratio_std"].dropna().to_numpy()
    ct = df[A + "ratio_clipped_total"].dropna().to_numpy()
    gn = df[A + "grad_norm"].dropna().to_numpy()
    kl = df[A + "approx_kl"].dropna().to_numpy()
    print(f"\nper-row: ratio_std median {np.median(rs):.3g} (clip half-width {eps * mult:.3g}); "
          f"clipped fraction median {np.median(ct):.3g}, max {ct.max():.3g}")
    print(f"         approx_kl median {np.median(kl):.3g}, max {kl.max():.3g}; "
          f"grad_norm median {np.median(gn):.3g}, frac > clip {np.mean(gn > gclip):.2f}")
    if np.median(ct) > 0.5:
        print("  -> SATURATED: over half the samples are outside the clip range after ONE update from the "
              "target actor. The PG gradient comes from the few inside it; consider the ratio reference "
              "(target actor with actor_tau) vs clip_epsilon.")
    if A + "adv_zero_frac" in df.columns and df[A + "adv_zero_frac"].notna().any():
        zf = df[A + "adv_zero_frac"].dropna().to_numpy()
        print(f"  zeroed advantages (conservative gate): first {zf[0]:.3g}, median {np.median(zf):.3g}, last {zf[-1]:.3g}")
        if np.median(zf) > 0.8:
            print("  -> the gate removes >80% of the PG signal; the actor is mostly BC.")
    if A + "actions_clipped_frac" in df.columns:
        acf = df[A + "actions_clipped_frac"].dropna().to_numpy()
        print(f"  actions at the [-1,1] normalised clip: median {np.median(acf):.3g}, max {acf.max():.3g} "
              f"(sampler vs logprob disagree on these; see PLAN.md)")
    if A + "bc_loss" in df.columns:
        bc = df[A + "bc_loss"].dropna().to_numpy()
        pg = np.abs(df[A + "pg_loss"].dropna().to_numpy())
        print(f"  |pg_loss| median {np.median(pg):.3g} vs bc_loss median {np.median(bc):.3g} "
              f"(bc_coeff {_cfg(config, 'agent.bc_coeff')}) — loss scale, not gradient share")

    diag = _bucket_mean(df, [D + c for c in ("grad_pg_norm", "grad_bc_norm", "grad_cos_pg_bc", "grad_pg_share",
                                             "drift_online_target_rel", "drift_online_init_rel",
                                             "drift_target_init_rel")], dropna_on=D + "grad_pg_norm")
    if not diag.empty:
        print("\ngradient split and actor drift (one extra batch per log window; pre-clip norms, BC scaled by bc_coeff):")
        print("  drift_* are ||a - b|| / ||init|| over the trainable action-expert params.")
        _show(diag.rename(columns=lambda c: c.replace(D, "")))
        share = diag[D + "grad_pg_share"].to_numpy()
        cos = diag[D + "grad_cos_pg_bc"].to_numpy()
        print(f"  PG share of the gradient: median {np.nanmedian(share):.3g}; cos(PG, BC): median {np.nanmedian(cos):.3g}")
        if np.nanmedian(share) < 0.1:
            print("  -> the update is almost pure BC; PPO barely steers the actor.")
        elif np.nanmedian(cos) < -0.2:
            print("  -> PG and BC pull in opposing directions.")
    print()


def _collapse_report(df: pd.DataFrame, top: int) -> None:
    """Split at the best eval; rank every metric by its standardized shift between phases."""
    key = next((c for c in ("eval_sde/success", "eval/success") if c in df.columns), None)
    if key is None:
        return
    ev = df[["_step", "eval/success", "eval_sde/success"]].dropna(how="any")
    if len(ev) < 3:
        return
    ev = ev.assign(mean_success=ev[["eval/success", "eval_sde/success"]].mean(axis=1))
    peak_row = ev.loc[ev["mean_success"].idxmax()]
    peak, last = int(peak_row["_step"]), int(ev["_step"].max())
    prev_steps = ev.loc[ev["_step"] < peak, "_step"]
    start = int(prev_steps.max()) if not prev_steps.empty else 0
    print("=== collapse report ===")
    print(f"peak mean eval success {peak_row['mean_success']:.3f} at step {peak}; "
          f"last {ev['mean_success'].iloc[-1]:.3f} at step {last}.")
    if peak >= last:
        print("  no decline after the peak.\n")
        return
    print(f"comparing RISING phase [{start}, {peak}) with FALLING phase [{peak}, {last}]; "
          f"shift = (mean_fall - mean_rise) / pooled std.")
    rows = []
    for m in [c for c in df.columns if c.startswith(("online_agent/", "env/")) and not c.endswith("/step")]:
        s = df[["_step", m]].dropna()
        a = s.loc[(s["_step"] >= start) & (s["_step"] < peak), m].astype(float)
        b = s.loc[(s["_step"] >= peak) & (s["_step"] <= last), m].astype(float)
        if len(a) < 2 or len(b) < 2:
            continue
        pooled = math.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1))) or 1e-12
        rows.append({"metric": _short(m), "rise_mean": a.mean(), "fall_mean": b.mean(),
                     "shift_sd": (b.mean() - a.mean()) / pooled, "n_rise": len(a), "n_fall": len(b)})
    if not rows:
        print()
        return
    table = pd.DataFrame(rows)
    table = table.reindex(table["shift_sd"].abs().sort_values(ascending=False).index)
    _show(table.head(top))
    print("  -> largest |shift| first. These are correlates of the decline, not causes; single-batch rows "
          "make |shift| < ~1 indistinguishable from noise.\n")


def _cross_run_compare(frames: dict[str, pd.DataFrame], metrics: Sequence[str], title: str) -> None:
    cols = {}
    for metric in metrics:
        per_run = {name: df.loc[df[metric].notna()].groupby("_bucket_start")[metric].mean()
                   for name, df in frames.items() if metric in df.columns and df[metric].notna().any()}
        if per_run:
            cols[metric] = per_run
    if not cols:
        return
    print(f"=== {title} ===")
    for metric, per_run in cols.items():
        table = pd.DataFrame(per_run)
        table.index.name = "step_start"
        print(f"--- {metric} ---")
        _show(table, index=True)
        print()


# --------------------------------------------------------------------------- driver

def _analyze_run(run, args: argparse.Namespace, csv_rows: list[pd.DataFrame]) -> pd.DataFrame | None:
    print("=" * 90)
    print(f"Run: {run.entity}/{run.project}/{run.id}  ({run.name})")
    print(f"State: {run.state}  Steps: {run.summary.get('_step', '?')}  Created: {run.created_at}  "
          f"Group: {run.group}")
    print("=" * 90 + "\n")
    config = dict(run.config)
    _config_report(config)

    available = _discover_metrics(run)
    selected = _select(available, args.metrics)
    if not selected:
        print("[warn] no metric matched; skipping run.\n")
        return None
    print(f"Selected {len(selected)}/{len(available)} logged metrics. Bucket width: {args.bucket} steps.\n")
    raw = _fetch_history(run, selected, args.max_samples)
    if args.dump_csv:
        path = args.dump_csv if len(args.run_ids or args.run_names) == 1 else f"{args.dump_csv}.{run.id}.csv"
        raw.to_csv(path, index=False)
        print(f"Wrote raw merged history ({len(raw)} rows x {raw.shape[1]} cols): {path}\n")
    df = _bucketize(raw, args.bucket)
    expected = _expected_metrics(config)

    _logging_health(df, config, expected)
    _coverage_report(available, df, expected)
    if not args.no_tables:
        _print_group_tables(df, selected, tuple(sorted(set(args.quantiles))), args.full,
                            csv_rows if args.csv else None, run.name)
    _outcome_report(df, config, args.episode_window)
    _critic_report(df, config, args.success_bonus)
    _actor_report(df, config)
    _collapse_report(df, args.top)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Dump and diagnose every W&B metric of an OGPO-pi05 run.")
    ap.add_argument("--entity", default=DEFAULT_ENTITY)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--run-names", nargs="+", default=list(DEFAULT_RUN_NAMES),
                    help=f"Display names, fuzzy-matched (default: {DEFAULT_RUN_NAMES[0]!r}).")
    ap.add_argument("--run-ids", nargs="+", default=None, help="Run ids; bypasses name lookup.")
    ap.add_argument("--select", default="latest", choices=("latest", "longest", "running"))
    ap.add_argument("--list-runs", action="store_true")
    ap.add_argument("--list-metrics", action="store_true")
    ap.add_argument("--bucket", type=int, default=2000, help="Step bucket width (default 2000).")
    ap.add_argument("--metrics", nargs="+", default=["*"], help="fnmatch patterns (default: every key).")
    ap.add_argument("--quantiles", nargs="+", type=float, default=list(DEFAULT_QUANTILES))
    ap.add_argument("--max-samples", type=int, default=100_000)
    ap.add_argument("--episode-window", type=int, default=5, help="Episodes per training-success window.")
    ap.add_argument("--success-bonus", type=float, default=5.0, help="Reward on the success step (env wrapper).")
    ap.add_argument("--top", type=int, default=25, help="Rows in the collapse report.")
    ap.add_argument("--full", action="store_true", help="Also tabulate presence-only keys.")
    ap.add_argument("--no-tables", action="store_true", help="Skip the per-metric quantile tables.")
    ap.add_argument("--no-compare", action="store_true")
    ap.add_argument("--csv", default=None, help="Write the long-form bucket summary here.")
    ap.add_argument("--dump-csv", default=None, help="Write the raw merged history here.")
    args = ap.parse_args()

    api = wandb.Api(timeout=60)
    if args.list_runs:
        for r in sorted(api.runs(f"{args.entity}/{args.project}"), key=lambda r: r.created_at, reverse=True):
            print(f"{r.id}  {r.state:<9} steps={_steps(r):<8} group={str(r.group):<22} {r.created_at}  {r.name}")
        return
    runs = ([api.run(f"{args.entity}/{args.project}/{rid}") for rid in args.run_ids] if args.run_ids
            else [_resolve_run(api, args.entity, args.project, n, args.select) for n in args.run_names])
    if args.list_metrics:
        for run in runs:
            keys = _discover_metrics(run)
            print(f"{run.name} ({run.id}): {len(keys)} metrics")
            for k in keys:
                new = "" if any(fnmatch.fnmatchcase(k, p) for p in KNOWN_PATTERNS) else "   <- NEW"
                print(f"    {k}{new}")
        return

    frames, csv_rows = {}, []
    for run in runs:
        df = _analyze_run(run, args, csv_rows)
        if df is not None:
            frames[f"{run.name}:{run.id}"] = df
    if not frames:
        print("[err] no run produced any history.", file=sys.stderr)
        sys.exit(2)
    if not args.no_compare and len(frames) > 1:
        for prefix, title in (("eval", "cross-run eval"), (C, "cross-run critic"), (A, "cross-run actor")):
            ms = sorted({c for df in frames.values() for c in df.columns if c.startswith(prefix)})
            _cross_run_compare(frames, ms, title)
    if args.csv and csv_rows:
        long_df = pd.concat(csv_rows, ignore_index=True)
        cols = ["run", "metric"] + [c for c in long_df.columns if c not in ("run", "metric")]
        long_df[cols].to_csv(args.csv, index=False)
        print(f"Wrote long-form CSV: {args.csv}")


if __name__ == "__main__":
    main()
