#!/usr/bin/env python3
"""Dump and diagnose EVERY W&B metric of a privileged-critic OGPO run.

Usage:
    python tests/wandb_ogpo_privileged.py
    python tests/wandb_ogpo_privileged.py --list-metrics
    python tests/wandb_ogpo_privileged.py --list-runs
    python tests/wandb_ogpo_privileged.py --run-names mt4priv_v0_s0 mt4_v0_s0
    python tests/wandb_ogpo_privileged.py --run-ids abc123 --bucket 2500
    python tests/wandb_ogpo_privileged.py --metrics 'critic/*' 'burst/*'

Same shape as ``tests/awr/wandb_quantiles.py`` in the manan_babel checkout
(step-bucketed quantile tables + reconstructed-quantity reports + a cross-run
comparison), retargeted at what scripts/ogpo_privileged.sh actually logs. The
difference in emphasis: that script knew its metric list up front, this one
does NOT. The privileged arm adds keys the baseline never had
(``critic/q_bootstrap_mean``, per-task ``success_rate/<task>``, ``burst/*``,
``critic_sb/*``, ``actor/adv_*``), and the point of this script is to find out
which of them ARE and ARE NOT reaching W&B. So the default metric selection is
``*`` — every key the run logged — grouped by namespace, with an explicit
report on keys that were expected-but-absent and keys that are present-but-
unrecognised.

Defaults target ``diverse-data-synthesis/ogpo_multitask``. NOTE the run name:
ogpo_privileged.sh sets ``--exp_name mt4priv_${ARM}_s${SEED}``, which is what
``wandb.init(name=...)`` receives, so the display name on the dashboard is
``mt4priv_v0_s0`` unless the run was renamed in the UI to something like
"OGPO- Privileged Critic". Both are tried: an exact display-name match first,
then a case-insensitive substring/glob match, and ``--list-runs`` prints every
run in the project when neither hits.

What it prints
--------------
1. **Run + config audit.** The launcher's flags as they landed in
   ``run.config``, with hard checks on the invariants the privileged learner
   asserts at construction (``collect.store_privileged_state`` on,
   ``collect.store_prefix_rep`` OFF, ``rl.n_samples == 1``,
   ``privileged_backup`` in {value, next_action_q}), plus the derived critic
   head count (len(collect.tasks) heads x rl.critic.num_qs nets).

2. **Logging-health report.** Per metric: how many rows W&B holds, the first
   and last step it appeared at, the observed step stride, and the fraction of
   rows that are NaN. This is the "what went wrong with wandb" panel:

   * exp.py stacks the per-step info dicts and reduces with ``jnp.nanmean``
     over keys unioned across the whole log_interval window, filling absent
     keys with NaN. A metric whose branch never fires is therefore logged as
     NaN, not omitted — an all-NaN column means the code path never ran, which
     looks identical to "missing" on the dashboard.
   * ``actor/*`` only exists on steps where the policy updated
     (``rl.policy.update_interval``, gated by ``policy.training_start_step``),
     ``critic/*`` on critic steps, ``burst/*`` only at collect boundaries,
     ``eval/*`` only at ``collect.eval_interval``. A stride that does not match
     the configured interval is the bug.
   * ``success_rate*`` and ``eval/*`` are logged with ``step=step`` from
     OUTSIDE the log_interval reduction (exp.py logs collect/eval info
     directly), so they can land on a step that carries no train metrics.

3. **Expected-key coverage.** The keys this config SHOULD emit, derived from
   the run config (see ``_expected_metrics``), split into present / all-NaN /
   absent. Absent is the actionable one.

4. **Unrecognised keys.** Anything logged that this script does not know
   about — i.e. genuinely new instrumentation, or a typo'd key name.

5. **Per-metric quantile tables** indexed by ``[step_start, step_end)`` with
   ``count, mean, std, p05, p25, p50, p75, p95, min, max``, grouped by
   namespace in the order actor -> critic -> critic_sb -> burst -> buffers ->
   collect -> eval -> other.

6. **Critic report.** Q vs V level, ``critic/q_mc_corr`` (THE number this
   experiment is about — the advantage only ever consumes the critic's
   ordering, so ranking correlation, not ``q_value_mean``, is the result), the
   TD/MC loss split, and the finite-horizon value bounds vs the
   never-succeeds fixed point so a critic parked on the failure fixed point is
   not misread as divergence.

7. **Actor report.** The PPO health panel: ratio spread, clipfrac,
   approx_kl, alive_fraction, grad_cos_pg_bc, and the advantage pipeline
   BEFORE vs AFTER post-processing. Note the ordering the learner uses --
   ``actor/advantage_*`` are measured inside the sampler jit, i.e. PRE
   normalizer / PRE pg-warmstart mute / PRE symmetric clip, while
   ``actor/advantage_std_final`` is measured after all three. The reported PG
   ramp weight (0 before ``rl.pg_start_step``, linear over
   ``rl.pg_ramp_steps``) tells you which regime each bucket was in, so a
   "dead" advantage during warmstart is not mistaken for a critic failure.

8. **Burst report** (``rl.post_collection_critic_steps``): the critic-only
   digestion steps run at each collect boundary, logged at the SAME step as
   the collection, so they alias onto one bucket per collect_interval.

9. **Per-task table.** ``success_rate/<task>`` (collection) and
   ``eval/success_rate/<task>`` (eval), one column per task, with the 4
   training tasks separated from the 25 held-out ones. Held-out tasks have no
   critic head (task id -1 -> zero one-hot), so their column is policy-only.

10. **Cross-run comparison** when several runs are given — e.g. the privileged
    arm against ``scripts/ogpo_multitask_4task.sh``'s baseline, which differs
    ONLY in the critic.
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
DEFAULT_PROJECT = "ogpo_multitask"
# The dashboard name the user sees; ogpo_privileged.sh's own exp_name is
# mt4priv_${ARM}_s${SEED}. _resolve_run tries both.
DEFAULT_RUN_NAMES = ("OGPO- Privileged Critic",)

DEFAULT_QUANTILES: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)

# Namespace groups, in print order. Every logged key is matched against these
# in order; whatever falls through lands in "other" and is also reported as
# unrecognised.
METRIC_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("actor (src/rl/ogpo/update_actor.py)", ("actor/*",)),
    ("critic (src/rl/ogpo/privileged/update_critic.py)", ("critic/q_*", "critic/value_*", "critic/*")),
    ("critic success-oversample burst (rl.critic_success_oversample)", ("critic_sb/*",)),
    ("post-collection digestion burst (rl.post_collection_critic_steps)", ("burst/*",)),
    ("buffers", ("online_buffer_size", "success_buffer_size", "*buffer*")),
    ("collect (src/training/collect.py::collect_data)", ("success_rate", "success_rate/*")),
    ("eval (src/training/collect.py::evaluate_policy)", ("eval/*",)),
)

# Keys this script recognises. Anything logged that matches none of these is
# printed in the "unrecognised" section — the whole point being to notice new
# instrumentation rather than silently drop it.
KNOWN_PATTERNS: tuple[str, ...] = (
    # --- actor: loss_and_grad_pg aux
    "actor/loss", "actor/pg_loss", "actor/pg_loss_unclipped", "actor/bc_loss",
    "actor/grad_norm", "actor/grad_norm_pg", "actor/grad_norm_bc",
    "actor/grad_cos_pg_bc", "actor/param_norm",
    "actor/ratio_mean", "actor/ratio_std", "actor/ratio_min", "actor/ratio_max",
    "actor/ratio_p05", "actor/ratio_p50", "actor/ratio_p95",
    "actor/log_ratio_mean", "actor/log_ratio_std", "actor/log_ratio_min",
    "actor/log_ratio_max",
    "actor/approx_kl", "actor/clipfrac", "actor/clipfrac_upper",
    "actor/clipfrac_lower", "actor/alive_fraction",
    "actor/new_log_prob_mean", "actor/old_log_prob_mean",
    # --- actor: sample_and_advantage aux (PRE post-processing)
    "actor/q_mean", "actor/v_mean",
    "actor/advantage_mean", "actor/advantage_std", "actor/advantage_min",
    "actor/advantage_max", "actor/advantage_median", "actor/advantage_q_up",
    "actor/advantage_q_low", "actor/cons_zero_frac",
    # --- actor: host-side advantage post-processing (ogpo_learner.update)
    "actor/adv_scale", "actor/adv_clip_sym_frac", "actor/advantage_std_final",
    # --- critic heads
    "critic/q_loss", "critic/q_value_mean", "critic/q_td_loss",
    "critic/q_mc_loss", "critic/q_td_weight", "critic/q_mc_corr",
    "critic/q_bootstrap_mean", "critic/q_grad_norm", "critic/q_param_norm",
    "critic/value_loss", "critic/value_value_mean", "critic/value_td_loss",
    "critic/value_mc_loss", "critic/value_td_weight", "critic/value_grad_norm",
    "critic/value_param_norm",
    "critic_sb/*", "burst/*",
    # --- buffers / collect / eval
    "online_buffer_size", "success_buffer_size",
    "success_rate", "success_rate/*",
    "eval/success_rate", "eval/success_rate/*",
    "eval/mean_success_episode_length", "eval/total_collected_episodes",
)

# Metrics whose *level* is meaningless but whose presence/absence is the
# diagnostic; skipped in the quantile tables unless --full.
_PRESENCE_ONLY: tuple[str, ...] = ("burst/steps", "eval/total_collected_episodes")


# --------------------------------------------------------------------------- run resolution

def _all_runs(api: wandb.Api, entity: str, project: str) -> list:
    return list(api.runs(f"{entity}/{project}"))


def _steps(run) -> int:
    value = run.summary.get("_step")
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _resolve_run(api: wandb.Api, entity: str, project: str, run_name: str,
                 select: str):
    """Resolve a run by display name, falling back to a fuzzy match.

    Two distinct reasons a name misses here. (1) ``exp_name`` is reused across
    relaunches, so a dozen runs share one display name and most of them died
    within a few steps -- picking blindly lands on an empty one. (2) The
    launcher's exp_name (``mt4priv_v0_s0``) and the dashboard name the user
    quotes ("OGPO- Privileged Critic") are different strings for the same run
    if it was renamed in the UI. So: exact match, then substring/glob, then
    dump the project.
    """
    matches = list(api.runs(f"{entity}/{project}",
                            filters={"display_name": run_name}))
    if not matches:
        needle = run_name.strip().lower()
        pool = _all_runs(api, entity, project)
        matches = [
            r for r in pool
            if needle in (r.name or "").lower()
            or fnmatch.fnmatch((r.name or "").lower(), needle)
            or needle.replace(" ", "") in (r.name or "").lower().replace(" ", "")
        ]
        if matches:
            print(f"[info] no exact display_name {run_name!r}; fuzzy-matched "
                  f"{len(matches)} run(s).", file=sys.stderr)
        else:
            print(f"[err] no run matching {run_name!r} in {entity}/{project}. "
                  f"Runs in this project:", file=sys.stderr)
            for r in sorted(pool, key=lambda r: r.created_at, reverse=True)[:60]:
                print(f"    {r.id}  {r.state:<9} steps={_steps(r):<8} "
                      f"{r.created_at}  {r.name}", file=sys.stderr)
            raise ValueError(f"No run matching {run_name!r}.")

    if len(matches) == 1:
        return matches[0]

    print(f"[info] {len(matches)} runs match {run_name!r} (--run-ids pins one):",
          file=sys.stderr)
    for r in sorted(matches, key=lambda r: r.created_at):
        print(f"    {r.id}  {r.state:<9} steps={_steps(r):<8} "
              f"created={r.created_at}  {r.name}", file=sys.stderr)

    if select == "longest":
        chosen = max(matches, key=_steps)
    elif select == "running":
        live = [r for r in matches if r.state == "running"]
        if not live:
            raise ValueError("--select running: no matching run is running.")
        chosen = max(live, key=lambda r: r.created_at)
    else:
        # latest: newest run that actually logged something. A crashed run can
        # transiently report no _step while its summary re-syncs, so probe the
        # history rather than trusting the summary.
        chosen = None
        for r in sorted(matches, key=lambda r: r.created_at, reverse=True):
            if _steps(r) >= 0:
                chosen = r
                break
            try:
                if not r.history(samples=1, pandas=True).empty:
                    chosen = r
                    break
            except Exception:  # pragma: no cover - tolerate API quirks
                continue
        if chosen is None:
            chosen = max(matches, key=lambda r: r.created_at)
    print(f"[info] --select {select} -> {chosen.id} ({chosen.name})",
          file=sys.stderr)
    return chosen


# --------------------------------------------------------------------------- config

def _config_get(config: dict[str, Any], path: str, default: Any = None) -> Any:
    """Look up a dotted path in the run config.

    ``init_wandb`` uploads ``dataclasses.asdict(config)``, which W&B stores
    either nested (``config['rl']['beta']``) or flattened (``config['rl.beta']``)
    depending on client version. Try both.
    """
    if path in config:
        return config[path]
    node: Any = config
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


_CONFIG_AUDIT: tuple[tuple[str, Any], ...] = (
    # (dotted path, expected value or None to just print)
    ("collect.store_privileged_state", True),
    ("collect.privileged_state_dim", None),
    ("collect.store_prefix_rep", False),
    ("rl.privileged_backup", None),
    ("rl.n_samples", 1),
    ("rl.group_num_samples", None),
    ("rl.critic.num_qs", None),
    ("rl.critic.num_vs", None),
    ("rl.critic.reduction", None),
    ("rl.critic.use_bronet", None),
    ("rl.critic.bronet_hidden_dim", None),
    ("rl.critic.batch_size", None),
    ("rl.critic.use_distributional_critic", None),
    ("rl.critic.num_value_bins", None),
    ("rl.critic.value_target_type", None),
    ("rl.critic.inference_start_step", None),
    ("rl.critic.td_weight_schedule.init_value", None),
    ("rl.advantage_combination", None),
    ("rl.normalize_group_advantage", None),
    ("rl.normalize_advantage_per_task", None),
    ("rl.balance_success_buffer_tasks", None),
    ("rl.adv_clip_sym", None),
    ("rl.adv_strategy", None),
    ("rl.clip_epsilon", None),
    ("rl.bc_coeff", None),
    ("rl.beta", None),
    ("rl.discount", None),
    ("rl.pg_start_step", None),
    ("rl.pg_ramp_steps", None),
    ("rl.post_collection_critic_steps", None),
    ("rl.critic_success_oversample", None),
    ("rl.use_success_buffer", None),
    ("rl.dedup_group_prefix", None),
    ("rl.online_ratio", None),
    ("rl.buffer_capacity", None),
    ("rl.policy.update_interval", None),
    ("rl.policy.training_start_step", None),
    ("collect.collect_interval", None),
    ("collect.eval_interval", None),
    ("collect.num_rollouts", None),
    ("collect.num_initial_rollouts", None),
    ("collect.num_eval_rollouts", None),
    ("log_interval", None),
    ("num_train_steps", None),
    ("batch_size", None),
)


def _config_report(config: dict[str, Any]) -> None:
    print("=== config audit (as landed in run.config) ===")
    rows = []
    for path, expected in _CONFIG_AUDIT:
        got = _config_get(config, path, "<MISSING>")
        verdict = ""
        if expected is not None:
            verdict = "OK" if got == expected else f"!! expected {expected!r}"
        rows.append({"key": path, "value": got, "check": verdict})
    table = pd.DataFrame(rows)
    with pd.option_context("display.max_rows", None, "display.width", 200,
                           "display.max_colwidth", 60):
        print(table.to_string(index=False))

    tasks = _config_get(config, "collect.tasks", []) or []
    eval_tasks = _config_get(config, "collect.eval_tasks", []) or []
    num_qs = _config_get(config, "rl.critic.num_qs", 0) or 0
    print(f"\ntrain tasks ({len(tasks)}): {list(tasks)}")
    print(f"eval tasks ({len(eval_tasks)}): "
          f"{len(set(eval_tasks) - set(tasks))} held out")
    # One INDEPENDENT critic per training task, num_qs nets per head.
    print(f"critic heads = len(collect.tasks) = {len(tasks)}; "
          f"Q nets = {len(tasks)} x num_qs({num_qs}) = {len(tasks) * int(num_qs or 0)}")

    problems = [f"{p}={_config_get(config, p, '<MISSING>')!r} (expected {e!r})"
                for p, e in _CONFIG_AUDIT
                if e is not None and _config_get(config, p, "<MISSING>") != e]
    backup = _config_get(config, "rl.privileged_backup", "<MISSING>")
    if backup not in ("value", "next_action_q"):
        problems.append(f"rl.privileged_backup={backup!r} is not a valid mode")
    if problems:
        print("\n!! CONFIG PROBLEMS — the privileged learner asserts these at "
              "construction, so a violation here means the run is NOT the "
              "privileged arm it claims to be:")
        for p in problems:
            print(f"    - {p}")
    print()


def _expected_metrics(config: dict[str, Any]) -> dict[str, str]:
    """Keys this config should emit -> why. Basis for the coverage report."""
    exp: dict[str, str] = {
        "actor/loss": "policy update (rl.policy.update_interval)",
        "actor/pg_loss": "PPO term",
        "actor/bc_loss": "BC anchor (rl.bc_coeff)",
        "actor/grad_norm": "optimisation health",
        "actor/grad_norm_pg": "PG-only grad norm",
        "actor/grad_norm_bc": "BC-only grad norm",
        "actor/grad_cos_pg_bc": "PG/BC gradient agreement",
        "actor/param_norm": "logged every policy.update_interval * N",
        "actor/ratio_mean": "PPO ratio",
        "actor/ratio_std": "PPO ratio spread",
        "actor/approx_kl": "PPO KL proxy",
        "actor/clipfrac": "PPO clip saturation",
        "actor/alive_fraction": "non-degenerate sample fraction",
        "actor/q_mean": "critic level seen by the actor",
        "actor/v_mean": "critic level seen by the actor",
        "actor/advantage_mean": "advantage, PRE post-processing",
        "actor/advantage_std": "advantage spread, PRE post-processing",
        "actor/advantage_q_up": "advantage p95, PRE post-processing",
        "actor/advantage_q_low": "advantage p05, PRE post-processing",
        "critic/q_loss": "Q head",
        "critic/q_value_mean": "Q level",
        "critic/q_td_loss": "Q TD term",
        "critic/q_mc_loss": "Q MC term",
        "critic/q_td_weight": "TD/MC blend (rl.critic.td_weight_schedule)",
        "critic/q_mc_corr": "RANKING QUALITY — the headline privileged metric",
        "critic/q_grad_norm": "Q optimisation health",
        "critic/value_loss": "V head",
        "critic/value_value_mean": "V level",
        "critic/value_td_loss": "V TD term",
        "critic/value_mc_loss": "V MC term",
        "online_buffer_size": "replay buffer occupancy",
        "success_rate": "collect_data, every collect.collect_interval",
        "eval/success_rate": "evaluate_policy, every collect.eval_interval",
        "eval/mean_success_episode_length": "evaluate_policy",
        "eval/total_collected_episodes": "evaluate_policy",
    }
    if _config_get(config, "rl.use_success_buffer", False):
        exp["success_buffer_size"] = "rl.use_success_buffer=True"
    if _config_get(config, "rl.advantage_combination") == "grpo_conservative":
        exp["actor/cons_zero_frac"] = ("grpo_conservative: fraction zeroed by "
                                       "cross-head sign disagreement")
    if _config_get(config, "rl.normalize_group_advantage", False):
        exp["actor/adv_scale"] = "rl.normalize_group_advantage=True (EMA scale)"
        exp["actor/advantage_std_final"] = "spread AFTER normalizer/clip"
    if _config_get(config, "rl.adv_clip_sym", None) not in (None, 0, "<MISSING>"):
        exp["actor/adv_clip_sym_frac"] = "rl.adv_clip_sym: fraction at the clip"
        exp["actor/advantage_std_final"] = "spread AFTER normalizer/clip"
    if int(_config_get(config, "rl.post_collection_critic_steps", 0) or 0) > 0:
        exp["burst/steps"] = "rl.post_collection_critic_steps>0"
        exp["burst/q_loss"] = "digestion burst, last Q step"
        exp["burst/q_mc_corr"] = "digestion burst ranking quality"
        exp["burst/value_loss"] = "digestion burst, last V step"
    if _config_get(config, "rl.critic_success_oversample", False):
        exp["critic_sb/q_loss"] = "rl.critic_success_oversample=True"
        exp["critic_sb/value_loss"] = "rl.critic_success_oversample=True"
    if _config_get(config, "rl.privileged_backup") == "next_action_q":
        exp["critic/q_bootstrap_mean"] = (
            "privileged_backup=next_action_q emits it; the 'value' default "
            "goes through the shared AWR Q step, which does NOT")
    for task in (_config_get(config, "collect.tasks", []) or []):
        exp[f"success_rate/{task}"] = "per-task collection success"
    for task in (_config_get(config, "collect.eval_tasks", []) or []):
        exp[f"eval/success_rate/{task}"] = "per-task eval success"
    return exp


# --------------------------------------------------------------------------- history

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


def _discover_metrics(run) -> list[str]:
    """Every metric key the run logged.

    ``run.summary`` holds the LAST value per key, which is complete for a
    finished run but lags for a live one -- a key introduced mid-run may not be
    in the summary cache yet. Union it with the history columns.
    """
    keys = {k for k in run.summary.keys() if not k.startswith("_")}
    try:
        sample = run.history(samples=1, pandas=True)
        keys.update(c for c in sample.columns if not c.startswith("_"))
    except Exception as exc:  # pragma: no cover - tolerate API quirks
        print(f"[warn] history(samples=1) failed ({exc}); summary keys only.",
              file=sys.stderr)
    return sorted(keys)


def _fetch_history(run, metrics: list[str], max_samples: int) -> pd.DataFrame:
    """Pull history one metric per request and outer-merge on ``_step``.

    ``run.history(keys=[a, b])`` INNER-joins: it returns only steps where every
    requested key is non-null. Metrics on different cadences (``actor/*`` every
    log_interval vs ``eval/*`` every eval_interval) have an empty intersection,
    so one multi-key call returns zero rows. Fetch separately instead.
    """
    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    for metric in metrics:
        try:
            d = run.history(keys=[metric], samples=max_samples, pandas=True,
                            x_axis="_step")
        except Exception as exc:  # pragma: no cover - tolerate API quirks
            failed.append(f"{metric} ({exc})")
            continue
        if "_step" not in d.columns or metric not in d.columns:
            continue
        d = d[["_step", metric]].dropna(subset=["_step"])
        if d.empty:
            continue
        d["_step"] = d["_step"].astype(int)
        frames.append(d.drop_duplicates(subset=["_step"]))
    if failed:
        print(f"[warn] history failed for {len(failed)} metric(s): "
              f"{', '.join(failed[:5])}{' ...' if len(failed) > 5 else ''}",
              file=sys.stderr)
    if not frames:
        raise RuntimeError("No metric returned any history rows. Was W&B "
                           "logging disabled (WANDB_MODE) or the run empty?")
    df = frames[0]
    for other in frames[1:]:
        df = df.merge(other, on="_step", how="outer")
    return df.sort_values("_step").reset_index(drop=True)


def _bucketize(df: pd.DataFrame, bucket: int) -> pd.DataFrame:
    df = df.copy()
    df["_bucket_start"] = (df["_step"] // bucket) * bucket
    df["_bucket_end"] = df["_bucket_start"] + bucket
    return df


# --------------------------------------------------------------------------- reports

def _logging_health(df: pd.DataFrame, config: dict[str, Any],
                    expected: dict[str, str]) -> pd.DataFrame:
    """Per-metric presence, cadence and NaN fraction.

    The NaN column is the load-bearing one. exp.py unions the keys of every
    per-step info dict inside a log_interval window and fills the gaps with
    NaN before ``jnp.nanmean``, so a key whose branch never ran is LOGGED as
    NaN rather than omitted -- indistinguishable from "missing" on the
    dashboard, but very distinguishable here.
    """
    metrics = [c for c in df.columns if not c.startswith("_")]
    rows = []
    for metric in metrics:
        col = df[metric]
        present = col.notna()
        steps = df.loc[present, "_step"].to_numpy()
        # n_nonnull counts the steps this key actually carried a value on.
        # null_frac is against the OUTER-MERGED step grid (the union of every
        # metric's steps), so a key on a slower cadence is legitimately near 1
        # — read it together with `stride`, not alone.
        rows.append({
            "metric": metric,
            "n_nonnull": int(present.sum()),
            "null_frac": float(1.0 - present.mean()),
            "first_step": int(steps.min()) if steps.size else -1,
            "last_step": int(steps.max()) if steps.size else -1,
            "stride": (int(np.nanmedian(np.diff(steps)))
                       if steps.size > 1 else -1),
            "last_value": (float(col[present].iloc[-1])
                           if present.any() and
                           pd.api.types.is_numeric_dtype(col) else np.nan),
        })
    health = pd.DataFrame(rows).sort_values("metric").reset_index(drop=True)

    log_interval = int(_config_get(config, "log_interval", 0) or 0)
    pol_int = int(_config_get(config, "rl.policy.update_interval", 0) or 0)
    collect_int = int(_config_get(config, "collect.collect_interval", 0) or 0)
    eval_int = int(_config_get(config, "collect.eval_interval", 0) or 0)

    print("=== logging health ===")
    print(f"configured cadences: log_interval={log_interval}  "
          f"policy.update_interval={pol_int}  "
          f"collect_interval={collect_int}  eval_interval={eval_int}")
    print("stride is the MEDIAN step gap between consecutive non-null rows. "
          "actor/critic keys should sit at log_interval (they are reduced "
          "inside exp.py's log_interval window), success_rate* at "
          "collect_interval, eval/* at eval_interval, burst/* at "
          "collect_interval.")
    with pd.option_context("display.float_format", "{:.4g}".format,
                           "display.width", 220, "display.max_rows", None,
                           "display.max_columns", None):
        print(health.to_string(index=False))

    dead = health.loc[health["n_nonnull"] == 0, "metric"].tolist()
    if dead:
        print(f"\n!! {len(dead)} metric(s) exist as a key but are NULL/NaN at "
              f"every step — their code path never ran:")
        for m in dead:
            print(f"    - {m}  ({expected.get(m, 'not in the expected set')})")
    stalled = health[(health["n_nonnull"] > 0) &
                     (health["last_step"] < 0.5 * health["last_step"].max())]
    if not stalled.empty:
        print(f"\n!! {len(stalled)} metric(s) stopped logging well before the "
              f"run's last step ({int(health['last_step'].max())}) — a crashed "
              f"branch, or a key renamed mid-run:")
        for _, r in stalled.iterrows():
            print(f"    - {r['metric']}  last_step={int(r['last_step'])}")
    print()
    return health


def _coverage_report(available: Sequence[str], df: pd.DataFrame,
                     expected: dict[str, str]) -> None:
    have = set(available)
    live = {c for c in df.columns
            if not c.startswith("_") and df[c].notna().any()}
    print("=== expected-key coverage ===")
    absent = {k: why for k, why in expected.items() if k not in have}
    nan_only = {k: why for k, why in expected.items()
                if k in have and k not in live}
    ok = [k for k in expected if k in live]
    print(f"present with data: {len(ok)}/{len(expected)}")
    if nan_only:
        print(f"\nlogged but all-NaN ({len(nan_only)}):")
        for k, why in sorted(nan_only.items()):
            print(f"    {k:<40} {why}")
    if absent:
        print(f"\nABSENT — never logged ({len(absent)}):")
        for k, why in sorted(absent.items()):
            print(f"    {k:<40} {why}")
    else:
        print("\nno expected key is missing.")
    print()

    unknown = sorted(k for k in have
                     if not any(fnmatch.fnmatchcase(k, p)
                                for p in KNOWN_PATTERNS))
    print("=== unrecognised keys (new instrumentation, or a typo) ===")
    if unknown:
        for k in unknown:
            print(f"    {k}")
    else:
        print("    none — every logged key is accounted for.")
    print()


def _summary_for_metric(df: pd.DataFrame, metric: str,
                        quantiles: tuple[float, ...]) -> pd.DataFrame:
    series = df[metric]
    mask = series.notna()
    if not mask.any() or not pd.api.types.is_numeric_dtype(series):
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
        for q, v in g.quantile(list(quantiles)).to_dict().items():
            row[f"p{int(round(100 * q)):02d}"] = float(v)
        rows.append(row)
    out = pd.DataFrame(rows)
    ordered = (["step_start", "step_end", "count", "mean", "std"]
               + [f"p{int(round(100 * q)):02d}" for q in quantiles]
               + ["min", "max"])
    return out[ordered]


def _print_group_tables(df: pd.DataFrame, metrics: Sequence[str],
                        quantiles: tuple[float, ...], full: bool,
                        csv_rows: list[pd.DataFrame] | None,
                        run_name: str) -> None:
    remaining = list(metrics)
    for title, patterns in METRIC_GROUPS:
        group = _select_metric_columns(remaining, patterns)
        if not group:
            continue
        remaining = [m for m in remaining if m not in set(group)]
        print(f"########## {title} ##########\n")
        for metric in group:
            if not full and metric in _PRESENCE_ONLY:
                continue
            summary = _summary_for_metric(df, metric, quantiles)
            if summary.empty:
                print(f"--- {metric}: no numeric data ---\n")
                continue
            print(f"=== {metric} ===")
            with pd.option_context("display.float_format", "{:.4g}".format,
                                   "display.width", 200,
                                   "display.max_rows", None):
                print(summary.to_string(index=False))
            print()
            if csv_rows is not None:
                csv_rows.append(summary.assign(metric=metric, run=run_name))
    if remaining:
        print("########## other ##########\n")
        for metric in remaining:
            summary = _summary_for_metric(df, metric, quantiles)
            if summary.empty:
                print(f"--- {metric}: no numeric data ---\n")
                continue
            print(f"=== {metric} ===")
            with pd.option_context("display.float_format", "{:.4g}".format,
                                   "display.width", 200,
                                   "display.max_rows", None):
                print(summary.to_string(index=False))
            print()
            if csv_rows is not None:
                csv_rows.append(summary.assign(metric=metric, run=run_name))


def _value_bounds(config: dict[str, Any]) -> tuple[float, float] | None:
    """Mirror src.rl.value_distribution.get_value_bounds for this config."""
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

    Rewards accumulate over an action chunk and bootstrap with
    discount**action_horizon, so V = r_chunk / (1 - gamma_chunk).
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
    """Level, ranking quality and TD/MC split of the privileged critic."""
    q_col, v_col = "critic/q_value_mean", "critic/value_value_mean"
    if q_col not in df.columns:
        return
    cols = [c for c in (q_col, v_col, "critic/q_mc_corr", "critic/q_td_loss",
                        "critic/q_mc_loss", "critic/q_td_weight",
                        "critic/q_bootstrap_mean", "critic/q_grad_norm",
                        "critic/value_td_loss", "critic/value_mc_loss",
                        "actor/advantage_std") if c in df.columns]
    sub = df[["_bucket_start", "_bucket_end", *cols]].dropna(subset=[q_col])
    if sub.empty:
        return
    print("=== critic report (privileged, A = Q - V) ===")
    agg = (sub.groupby(["_bucket_start", "_bucket_end"], sort=True).mean()
           .reset_index()
           .rename(columns={"_bucket_start": "step_start",
                            "_bucket_end": "step_end"}))
    if v_col in agg.columns:
        agg["Q-V"] = agg[q_col] - agg[v_col]
        if "actor/advantage_std" in agg.columns:
            # If the batchwise advantage spread is dwarfed by the gap between
            # mean Q and mean V, the two heads disagree on LEVEL rather than on
            # which actions are good — and only the ordering ever reaches the
            # actor.
            agg["|Q-V|/A_std"] = agg["Q-V"].abs() / agg["actor/advantage_std"]
    with pd.option_context("display.float_format", "{:.4g}".format,
                           "display.width", 220, "display.max_rows", None,
                           "display.max_columns", None):
        print(agg.to_string(index=False))

    if "critic/q_mc_corr" in agg.columns:
        corr = agg["critic/q_mc_corr"].to_numpy()
        print(f"\nq_mc_corr (Pearson corr of Q against the observed MC return "
              f"— THE privileged-critic result, since the advantage only ever "
              f"consumes the critic's ORDERING):")
        print(f"  first bucket {corr[0]:.4g}  last bucket {corr[-1]:.4g}  "
              f"median {np.nanmedian(corr):.4g}  max {np.nanmax(corr):.4g}")
        if np.nanmedian(corr) < 0.1:
            print("  -> the critic is not ranking: compare against the "
                  "baseline run's q_mc_corr before reading anything into "
                  "success rates.")

    bounds = _value_bounds(config)
    if bounds is not None:
        lower, upper = bounds
        print(f"\nfinite-horizon value bounds: [{lower:.4g}, {upper:.4g}]  "
              f"last bucket Q={float(agg[q_col].iloc[-1]):.4g}"
              + (f" V={float(agg[v_col].iloc[-1]):.4g}"
                 if v_col in agg.columns else ""))
        fixed_point = _failure_fixed_point(config)
        if fixed_point is not None and v_col in agg.columns:
            v_last = float(agg[v_col].iloc[-1])
            print(f"never-succeeds fixed point (bootstrapped truncation): "
                  f"{fixed_point:.4g}")
            if abs(v_last - fixed_point) < 0.05 * abs(fixed_point):
                print("  -> PARKED ON THE FAILURE FIXED POINT: the critic "
                      "predicts near-total failure everywhere. That is the "
                      "TD/MC target doing what it says, not divergence — but "
                      "it leaves A = Q - V with almost no spread.")
    if ("critic/q_bootstrap_mean" not in df.columns
            and _config_get(config, "rl.privileged_backup") == "next_action_q"):
        print("\n!! privileged_backup=next_action_q but critic/q_bootstrap_mean "
              "was never logged — the run is going through the shared AWR Q "
              "step, not privileged/update_critic.py.")
    print()


def _pg_weight(config: dict[str, Any], step: int) -> float:
    """The host-side PG multiplier applied to the advantage at ``step``."""
    start = int(_config_get(config, "rl.pg_start_step", 0) or 0)
    ramp = int(_config_get(config, "rl.pg_ramp_steps", 0) or 0)
    if step < start:
        return 0.0
    if ramp <= 0:
        return 1.0
    return min(1.0, max(0.0, (step - start) / float(ramp)))


def _actor_report(df: pd.DataFrame, config: dict[str, Any]) -> None:
    """PPO health plus the advantage pipeline before vs after post-processing."""
    if "actor/loss" not in df.columns:
        return
    cols = [c for c in ("actor/pg_loss", "actor/bc_loss", "actor/ratio_mean",
                        "actor/ratio_std", "actor/ratio_max", "actor/approx_kl",
                        "actor/clipfrac", "actor/clipfrac_upper",
                        "actor/clipfrac_lower", "actor/alive_fraction",
                        "actor/grad_norm", "actor/grad_norm_pg",
                        "actor/grad_norm_bc", "actor/grad_cos_pg_bc",
                        "actor/q_mean", "actor/v_mean",
                        "actor/advantage_mean", "actor/advantage_std",
                        "actor/advantage_q_up", "actor/advantage_q_low",
                        "actor/cons_zero_frac", "actor/adv_scale",
                        "actor/adv_clip_sym_frac",
                        "actor/advantage_std_final") if c in df.columns]
    sub = df[["_bucket_start", "_bucket_end", *cols]].dropna(
        subset=["_bucket_start"], how="any")
    if sub.empty or not cols:
        return
    agg = (sub.groupby(["_bucket_start", "_bucket_end"], sort=True).mean()
           .reset_index()
           .rename(columns={"_bucket_start": "step_start",
                            "_bucket_end": "step_end"}))
    agg.insert(2, "pg_w", [_pg_weight(config, int(s))
                           for s in agg["step_start"]])
    print("=== actor report (PPO + advantage pipeline) ===")
    print("pg_w is the host-side PG multiplier from rl.pg_start_step / "
          "rl.pg_ramp_steps. It ZEROES the advantage during warmstart, so "
          "buckets with pg_w=0 train on the BC anchor alone — a flat "
          "pg_loss/clipfrac there is the schedule, not a bug.")
    print("actor/advantage_* are measured INSIDE the sampler jit, i.e. BEFORE "
          "the normalizer, the pg_w mute and the symmetric clip. "
          "actor/advantage_std_final is measured after all three; compare the "
          "two rather than either alone.")
    with pd.option_context("display.float_format", "{:.4g}".format,
                           "display.width", 240, "display.max_rows", None,
                           "display.max_columns", None):
        print(agg.to_string(index=False))

    eps = _config_get(config, "rl.clip_epsilon", None)
    if "actor/clipfrac" in agg.columns and eps is not None:
        cf = agg["actor/clipfrac"].to_numpy()
        print(f"\nclipfrac vs rl.clip_epsilon={eps}: first {cf[0]:.4g}  "
              f"last {cf[-1]:.4g}  median {np.nanmedian(cf):.4g}")
        if np.nanmedian(cf) > 0.5:
            print("  -> SATURATED: over half of each batch is clipped, so the "
                  "PG term is mostly a constant. Raise clip_epsilon or shrink "
                  "the policy step.")
    if "actor/alive_fraction" in agg.columns:
        alive = float(agg["actor/alive_fraction"].iloc[-1])
        print(f"alive_fraction (last bucket): {alive:.4g}")
    if "actor/cons_zero_frac" in agg.columns:
        czf = agg["actor/cons_zero_frac"].to_numpy()
        print(f"cons_zero_frac (grpo_conservative gate: fraction of samples "
              f"zeroed by cross-head sign disagreement): median "
              f"{np.nanmedian(czf):.4g}, last {czf[-1]:.4g}")
        if np.nanmedian(czf) > 0.9:
            print("  -> the conservative gate is killing almost every sample; "
                  "the PG term is effectively off regardless of pg_w.")
    if "actor/grad_cos_pg_bc" in agg.columns:
        gc = agg["actor/grad_cos_pg_bc"].to_numpy()
        print(f"grad_cos_pg_bc (PG vs BC gradient agreement): median "
              f"{np.nanmedian(gc):.4g}, last {gc[-1]:.4g}")
    print()


def _pg_progress_report(df: pd.DataFrame, config: dict[str, Any]) -> None:
    """Reconstruct corr(ratio, advantage) — the thing PPO is actually moving.

    The run logs the surrogate but not whether the policy has reallocated any
    probability mass toward high-advantage actions, which is the only sense in
    which the PG term is "working". ``pg_loss_unclipped`` is exactly
    ``-E[r*A]``, so::

        Cov(r, A) = -pg_loss_unclipped - E[r] * E[A]
        corr(r, A) = Cov(r, A) / (ratio_std * advantage_std_final)

    ``pg_loss`` alone is uninformative here: the advantage is mean-zero by
    construction (group baseline) and the ratio is nearly constant across a
    batch, so ``E[rA] ~ Cov(r, A)`` and the surrogate sits at ~0 whether or
    not anything is being learned. The correlation is the scale-free version.
    A policy that is successfully exploiting its critic drives this UP; a
    value pinned near zero means the update is noise.

    E[A] here is the PRE-post-processing mean rescaled by the same host-side
    factors the loss saw (pg ramp / adv_scale); both are tiny, so the
    correction is small either way.
    """
    need = ("actor/pg_loss_unclipped", "actor/ratio_mean", "actor/ratio_std",
            "actor/advantage_mean")
    if any(c not in df.columns for c in need):
        return
    std_col = ("actor/advantage_std_final"
               if "actor/advantage_std_final" in df.columns
               else "actor/advantage_std")
    cols = [*need, std_col]
    sub = df[["_bucket_start", "_bucket_end", *cols]].dropna(subset=cols)
    if sub.empty:
        return
    agg = (sub.groupby(["_bucket_start", "_bucket_end"], sort=True).mean()
           .reset_index()
           .rename(columns={"_bucket_start": "step_start",
                            "_bucket_end": "step_end"}))
    rows = []
    for _, r in agg.iterrows():
        pg_w = _pg_weight(config, int(r["step_start"]))
        a_mean = float(r["actor/advantage_mean"]) * pg_w
        cov = -float(r["actor/pg_loss_unclipped"]) - float(r["actor/ratio_mean"]) * a_mean
        denom = float(r["actor/ratio_std"]) * float(r[std_col])
        rows.append({
            "step_start": int(r["step_start"]),
            "pg_w": pg_w,
            "ratio_mean": float(r["actor/ratio_mean"]),
            "ratio_std": float(r["actor/ratio_std"]),
            "A_std_used": float(r[std_col]),
            "-E[rA]": float(r["actor/pg_loss_unclipped"]),
            "cov(r,A)": cov,
            "corr(r,A)": cov / denom if denom > 0 else float("nan"),
        })
    report = pd.DataFrame(rows)
    live = report[report["pg_w"] > 0]
    if live.empty:
        return
    print("=== PG progress report (reconstructed corr(ratio, advantage)) ===")
    print("pg_loss is ~0 BY CONSTRUCTION (mean-zero advantage x near-constant "
          "ratio), so it is not evidence either way. corr(r, A) is: it is the "
          "fraction of the ratio's spread that lines up with the advantage, "
          "i.e. whether the live policy has actually moved toward the actions "
          "the critic ranks highly. Rows with pg_w=0 are the BC warmstart.")
    with pd.option_context("display.float_format", "{:.4g}".format,
                           "display.width", 220, "display.max_rows", None,
                           "display.max_columns", None):
        print(report.to_string(index=False))
    c = live["corr(r,A)"].to_numpy()
    print(f"\ncorr(r, A) over the PG-active phase: first {c[0]:.4g}  "
          f"last {c[-1]:.4g}  median {np.nanmedian(c):.4g}")
    if abs(np.nanmedian(c)) < 0.1:
        print("  -> THE PG TERM IS NOT MOVING THE POLICY. After every update "
              "the ratio is essentially uncorrelated with the advantage, so "
              "the actor is not exploiting the critic's ranking regardless of "
              "how good that ranking is. Look upstream of the critic: "
              "advantage magnitude after normalization, the conservative "
              "gate's zero fraction, and the PG/BC gradient balance.")
    print()


def _burst_report(df: pd.DataFrame, config: dict[str, Any]) -> None:
    burst_cols = [c for c in df.columns if c.startswith("burst/")]
    configured = int(_config_get(config, "rl.post_collection_critic_steps", 0) or 0)
    if configured <= 0 and not burst_cols:
        return
    print("=== post-collection digestion burst ===")
    if not burst_cols:
        print(f"!! rl.post_collection_critic_steps={configured} but NO burst/* "
              f"key was logged. exp.py only calls critic_digestion_burst() for "
              f"step>0 at a collect boundary and only if the agent exposes the "
              f"method — check that the run reached its second collection.")
        print()
        return
    sub = df[["_step", *burst_cols]].dropna(subset=burst_cols, how="all")
    print(f"configured rl.post_collection_critic_steps={configured}; "
          f"burst metrics are logged at the SAME step as the collection that "
          f"triggered them (one row per collect_interval).")
    with pd.option_context("display.float_format", "{:.4g}".format,
                           "display.width", 240, "display.max_rows", None,
                           "display.max_columns", None):
        print(sub.to_string(index=False))
    if "burst/steps" in sub.columns:
        got = sub["burst/steps"].dropna()
        if not got.empty and int(got.median()) != configured:
            print(f"\n!! burst/steps median {int(got.median())} != configured "
                  f"{configured}.")
    print()


def _per_task_report(df: pd.DataFrame, config: dict[str, Any]) -> None:
    """One column per task, train tasks separated from held-out ones."""
    train = list(_config_get(config, "collect.tasks", []) or [])
    eval_tasks = list(_config_get(config, "collect.eval_tasks", []) or [])
    heldout = [t for t in eval_tasks if t not in set(train)]

    for prefix, label in (("success_rate/", "collection success by task"),
                          ("eval/success_rate/", "eval success by task")):
        cols = [c for c in df.columns if c.startswith(prefix)]
        if not cols:
            continue
        sub = df[["_step", *cols]].dropna(subset=cols, how="all")
        if sub.empty:
            continue
        renamed = sub.rename(columns={c: c[len(prefix):] for c in cols})
        train_cols = [t for t in train if t in renamed.columns]
        held_cols = [t for t in heldout if t in renamed.columns]
        print(f"=== {label} ===")
        if train_cols:
            print(f"--- training tasks ({len(train_cols)}) — these have a "
                  f"critic head ---")
            with pd.option_context("display.float_format", "{:.3f}".format,
                                   "display.width", 240,
                                   "display.max_rows", None,
                                   "display.max_columns", None):
                print(renamed[["_step", *train_cols]].to_string(index=False))
        if held_cols:
            print(f"\n--- held-out tasks ({len(held_cols)}) — NO critic head "
                  f"(task id -1 one-hots to zero); policy-only generalisation "
                  f"---")
            with pd.option_context("display.float_format", "{:.3f}".format,
                                   "display.width", 240,
                                   "display.max_rows", None,
                                   "display.max_columns", None):
                print(renamed[["_step", *held_cols]].to_string(index=False))
        leftover = [c for c in renamed.columns
                    if c not in {"_step", *train_cols, *held_cols}]
        if leftover:
            print(f"\n!! task columns not in collect.tasks or collect.eval_tasks: "
                  f"{leftover}")
        print()


def _cross_run_compare(frames: dict[str, pd.DataFrame],
                       metrics: Sequence[str], title: str) -> None:
    """One table per metric with a column per run that logged it.

    Runs align on the step bucket, not wall clock, so a bucket row goes blank
    for a run that never reached it.
    """
    cols: dict[str, dict[str, pd.Series]] = {}
    for metric in metrics:
        per_run: dict[str, pd.Series] = {}
        for name, df in frames.items():
            if metric not in df.columns:
                continue
            sub = df.loc[df[metric].notna(), ["_bucket_start", metric]]
            if sub.empty:
                continue
            per_run[name] = sub.groupby("_bucket_start")[metric].mean()
        if per_run:
            cols[metric] = per_run
    if not cols:
        return
    print(f"=== {title} ===")
    for metric, per_run in cols.items():
        table = pd.DataFrame(per_run)
        table.index.name = "step_start"
        print(f"--- {metric} ({', '.join(per_run)}) ---")
        with pd.option_context("display.float_format", "{:.4g}".format,
                               "display.width", 220, "display.max_rows", None,
                               "display.max_columns", None):
            print(table.to_string())
        print()


# --------------------------------------------------------------------------- driver

def _analyze_run(run, args: argparse.Namespace,
                 csv_rows: list[pd.DataFrame]) -> pd.DataFrame | None:
    print("=" * 78)
    print(f"Run: {run.entity}/{run.project}/{run.id}  ({run.name})")
    print(f"State: {run.state}  Steps: {run.summary.get('_step', '?')}  "
          f"Created: {run.created_at}")
    try:
        print(f"Group: {run.group}  Tags: {list(run.tags)}")
    except Exception:  # pragma: no cover - older wandb clients
        pass
    print("=" * 78)
    print()

    config = dict(run.config)
    _config_report(config)

    available = _discover_metrics(run)
    selected = _select_metric_columns(available, args.metrics)
    if not selected:
        print("[warn] no metric matched; skipping run.\n")
        return None
    print(f"Selected {len(selected)}/{len(available)} logged metrics. "
          f"Bucket width: {args.bucket} steps.\n")

    df = _bucketize(_fetch_history(run, selected, args.max_samples), args.bucket)
    expected = _expected_metrics(config)

    _logging_health(df, config, expected)
    _coverage_report(available, df, expected)
    if not args.no_tables:
        qs = tuple(sorted(set(args.quantiles)))
        _print_group_tables(df, selected, qs, args.full,
                            csv_rows if args.csv else None, run.name)
    _critic_report(df, config)
    _actor_report(df, config)
    _pg_progress_report(df, config)
    _burst_report(df, config)
    _per_task_report(df, config)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Dump and diagnose every W&B metric of a privileged-critic "
                    "OGPO run.")
    ap.add_argument("--entity", default=DEFAULT_ENTITY,
                    help=f"W&B entity (default: {DEFAULT_ENTITY})")
    ap.add_argument("--project", default=DEFAULT_PROJECT,
                    help=f"W&B project (default: {DEFAULT_PROJECT})")
    ap.add_argument("--run-names", nargs="+", default=list(DEFAULT_RUN_NAMES),
                    help="Run display names; falls back to a fuzzy match. "
                         f"(default: {DEFAULT_RUN_NAMES[0]!r})")
    ap.add_argument("--run-ids", nargs="+", default=None,
                    help="Run ids; bypasses name lookup and --select.")
    ap.add_argument("--select", default="latest",
                    choices=("latest", "longest", "running"),
                    help="Which run when several share the name: newest with "
                         "data (default), most steps, or the live one.")
    ap.add_argument("--list-runs", action="store_true",
                    help="Print every run in the project, then exit.")
    ap.add_argument("--list-metrics", action="store_true",
                    help="Print every metric key each run logged, then exit.")
    ap.add_argument("--bucket", type=int, default=2500,
                    help="Step bucket width (default: 2500; the run's "
                         "collect_interval is 10000 and log_interval 25).")
    ap.add_argument("--metrics", nargs="+", default=["*"],
                    help="Metric names or fnmatch patterns (default: * = "
                         "every key the run logged).")
    ap.add_argument("--quantiles", nargs="+", type=float,
                    default=list(DEFAULT_QUANTILES),
                    help="Quantiles to report (default: .05 .25 .5 .75 .95)")
    ap.add_argument("--max-samples", type=int, default=100_000,
                    help="Cap on history() rows per metric.")
    ap.add_argument("--full", action="store_true",
                    help="Also tabulate presence-only metrics (burst/steps, "
                         "eval/total_collected_episodes).")
    ap.add_argument("--no-tables", action="store_true",
                    help="Skip the per-metric quantile tables; keep the "
                         "diagnostics. Fastest way to answer 'is wandb "
                         "logging what I think it is'.")
    ap.add_argument("--no-compare", action="store_true",
                    help="Skip the cross-run comparison.")
    ap.add_argument("--csv", default=None,
                    help="Also write the long-form summary to this CSV path.")
    args = ap.parse_args()

    api = wandb.Api(timeout=60)

    if args.list_runs:
        for r in sorted(_all_runs(api, args.entity, args.project),
                        key=lambda r: r.created_at, reverse=True):
            print(f"{r.id}  {r.state:<9} steps={_steps(r):<8} "
                  f"group={str(r.group):<28} {r.created_at}  {r.name}")
        return

    if args.run_ids:
        runs = [api.run(f"{args.entity}/{args.project}/{rid}")
                for rid in args.run_ids]
    else:
        runs = [_resolve_run(api, args.entity, args.project, name, args.select)
                for name in args.run_names]

    if args.list_metrics:
        for run in runs:
            keys = _discover_metrics(run)
            print(f"{run.name} ({run.id}): {len(keys)} metrics")
            for k in keys:
                mark = "" if any(fnmatch.fnmatchcase(k, p)
                                 for p in KNOWN_PATTERNS) else "   <- NEW"
                print(f"    {k}{mark}")
            print()
        return

    frames: dict[str, pd.DataFrame] = {}
    csv_rows: list[pd.DataFrame] = []
    for run in runs:
        df = _analyze_run(run, args, csv_rows)
        if df is not None:
            frames[f"{run.name}:{run.id}"] = df

    if not frames:
        print("[err] no run produced any history.", file=sys.stderr)
        sys.exit(2)

    if not args.no_compare and len(frames) > 1:
        # The privileged arm and scripts/ogpo_multitask_4task.sh differ ONLY in
        # the critic, so the critic panel is the apples-to-apples one.
        critic_metrics = sorted({c for df in frames.values() for c in df.columns
                                 if c.startswith("critic/")})
        _cross_run_compare(frames, critic_metrics,
                           "cross-run critic comparison (privileged vs baseline)")
        actor_metrics = sorted({c for df in frames.values() for c in df.columns
                                if c.startswith("actor/")})
        _cross_run_compare(frames, actor_metrics, "cross-run actor comparison")
        outcome_metrics = sorted({c for df in frames.values() for c in df.columns
                                  if c.startswith("eval/") or c == "success_rate"
                                  or c.startswith("success_rate/")
                                  or c.endswith("buffer_size")})
        _cross_run_compare(frames, outcome_metrics, "cross-run outcome comparison")

    if args.csv and csv_rows:
        long_df = pd.concat(csv_rows, ignore_index=True)
        cols = ["run", "metric"] + [c for c in long_df.columns
                                    if c not in ("run", "metric")]
        long_df[cols].to_csv(args.csv, index=False)
        print(f"Wrote long-form CSV: {args.csv}")


if __name__ == "__main__":
    main()
