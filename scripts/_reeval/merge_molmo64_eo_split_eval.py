#!/usr/bin/env python3
"""Recombine the two 40-task halves of the Molmo-64 EO split eval into one
80-task record, and append it to the source run's metrics.jsonl.

*** UNVERIFIED -- DRY-RUN BEFORE `--apply`. *** Written 2026-09-17 while
/capstor was unmounted, so it was never run against real sidecars.

Each half writes eval_metrics_m64eo_<tag>_<half>_step<S>.json into the shared
staging dir. Halves are disjoint 40-task subsets of the same 80 eval tasks and
were evaluated from the SAME staged checkpoint, so recombining is exact:

  per-task  : union of the two halves' `<p>/success_rate/<task>` keys (disjoint)
  aggregate : episode-weighted mean of the two half aggregates
              (SR_A*n_A + SR_B*n_B) / (n_A + n_B)
              With 40 tasks x 32 rollouts each, n_A == n_B == 1280, so this is
              just the mean -- but weighting by the reported episode counts is
              robust if a half ever collects a different number.
  episodes  : sum

That equals successes_total / 2560, i.e. exactly what a single-shot 80-task eval
would have reported. Output namespace follows the convention used by the other
EO cells: `eval/*` for the final (round 10), `eval_mid/*` for the mid (round 5).
Those runs had in-loop eval disabled, so there is no existing `eval/` record at
these steps to clobber.

Appends serially -- concurrent appends to one metrics.jsonl are not safe on
Lustre. Idempotent: a record already present at the same step with the same
success_rate is skipped.
"""
import argparse, json, re
from pathlib import Path

RES = Path("/capstor/store/cscs/swissai/a0220/ralfroemer/results")
OUT_PREFIX = {"mid": "eval_mid", "final": "eval"}
EXPECT_TASKS = 80


def load_half(sc):
    """-> (prefix, metrics dict) for one half's sidecar."""
    rec = json.loads(sc.read_text())
    # keys look like "<prefix>/success_rate" etc; recover the prefix
    pref = next((k.split("/", 1)[0] for k in rec if k.endswith("/success_rate")), None)
    if pref is None:
        raise ValueError("no <prefix>/success_rate key")
    return pref, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    # group sidecars by (staging ckpt dir, tag, step) -> {half: path}
    groups = {}
    for sc in sorted(RES.glob("*/*/speval_*/*/*/eval_metrics_m64eo_*_step*.json")):
        m = re.match(r"eval_metrics_m64eo_(mid|final)_([AB])_step(\d+)\.json$", sc.name)
        if not m:
            print(f"  SKIP unrecognised sidecar name: {sc.name}")
            continue
        tag, half, step = m.group(1), m.group(2), int(m.group(3))
        groups.setdefault((sc.parent, tag, step), {})[half] = sc

    if not groups:
        print("no split-eval sidecars found yet")
        return

    merged = skipped = failed = 0
    for (ckpt_dir, tag, step), halves in sorted(groups.items(), key=lambda kv: str(kv[0][0])):
        subrun = ckpt_dir.parent.parent.parent          # <subrun>/speval_<tag>/<cfg>/<exp>
        cfg_name, exp_name = ckpt_dir.parent.name, ckpt_dir.name
        target = subrun / cfg_name / exp_name / "metrics.jsonl"
        label = f"{subrun.parent.name}/{subrun.name} [{tag} step={step}]"

        if set(halves) != {"A", "B"}:
            print(f"  INCOMPLETE {label}: have halves {sorted(halves)} -- waiting for the other")
            failed += 1
            continue
        try:
            pA, recA = load_half(halves["A"])
            pB, recB = load_half(halves["B"])
        except Exception as e:
            print(f"  UNREADABLE {label}: {e}")
            failed += 1
            continue

        srA, srB = recA.get(pA + "/success_rate"), recB.get(pB + "/success_rate")
        nA = recA.get(pA + "/total_collected_episodes")
        nB = recB.get(pB + "/total_collected_episodes")
        if srA is None or srB is None or not nA or not nB:
            print(f"  INCOMPLETE METRICS {label}: eval(s) likely did not finish")
            failed += 1
            continue

        # per-task union; must be disjoint and cover all 80
        per = {}
        dupes = []
        for pref, rec in ((pA, recA), (pB, recB)):
            head = pref + "/success_rate/"
            for k, v in rec.items():
                if k.startswith(head):
                    t = k[len(head):]
                    if t in per:
                        dupes.append(t)
                    per[t] = v
        if dupes:
            print(f"  OVERLAP {label}: halves share tasks {sorted(set(dupes))[:5]} -- refusing")
            failed += 1
            continue
        if len(per) != EXPECT_TASKS:
            print(f"  TASK COUNT {label}: merged {len(per)} tasks, expected {EXPECT_TASKS} -- refusing")
            failed += 1
            continue

        op = OUT_PREFIX[tag]
        # Aggregate = unweighted mean of the per-task rates over all 80 tasks.
        # evaluate_policy gives every task exactly num_eval_rollouts episodes, so
        # successes/episodes == mean(per-task rates) exactly; verified against real
        # sidecars (each half's own `success_rate` == mean of its 40 per-task rates).
        # Do NOT weight by `total_collected_episodes`: that field is NOT the eval
        # rollout count (it reads 7680 for a 1280-rollout half, identical in both
        # halves because they resume from the same checkpoint), so weighting by it
        # is meaningless and summing it would double a value a single-shot eval
        # reports once.
        agg = sum(per.values()) / len(per)
        out = {op + "/success_rate": float(agg)}
        if abs(nA - nB) < 1e-9:
            # Both halves resume from the same checkpoint, so this counter agrees;
            # report it once, as a single-shot eval would.
            out[op + "/total_collected_episodes"] = float(nA)
        else:
            print(f"  NOTE {label}: halves disagree on total_collected_episodes "
                  f"({nA} vs {nB}); omitting the field rather than guessing")
        # Mean success-episode length is a mean over SUCCESSFUL episodes, so weight
        # by each half's success count (rate x equal rollouts), not by any counter.
        lA = recA.get(pA + "/mean_success_episode_length")
        lB = recB.get(pB + "/mean_success_episode_length")
        if lA is not None and lB is not None and (srA + srB) > 0:
            out[op + "/mean_success_episode_length"] = float((lA * srA + lB * srB) / (srA + srB))
        for t, v in per.items():
            out[op + "/success_rate/" + t] = float(v)
        out["step"] = step

        if not target.exists():
            print(f"  MISSING TARGET {target}")
            failed += 1
            continue
        key = op + "/success_rate"
        already = False
        with target.open() as f:
            for line in f:
                if key in line:
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    if int(d.get("step", -1)) == step and abs(d.get(key, -1) - agg) < 1e-12:
                        already = True
                        break
        if already:
            skipped += 1
            continue

        print(f"  {'APPEND' if args.apply else 'WOULD APPEND'} {label}: "
              f"{key}={agg:.4f} = mean over {len(per)} per-task rates "
              f"(half A={srA:.4f}, half B={srB:.4f})")
        if args.apply:
            with target.open("a") as f:
                f.write(json.dumps(out) + "\n")
        merged += 1

    print(f"\ngroups={len(groups)} merged={merged} already_present={skipped} problems={failed}")
    if not args.apply and merged:
        print("dry run -- re-run with --apply to write")


if __name__ == "__main__":
    main()
