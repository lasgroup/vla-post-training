#!/usr/bin/env python3
"""Merge new-task re-eval results into each source run's metrics.jsonl.

Each re-eval job writes `eval_metrics_eval_newtasks_step<S>.json` into its own
staging dir. This appends one record per result into the ORIGINAL run's
metrics.jsonl, serially -- concurrent appends from many jobs to one file are not
safe on Lustre, which is why the jobs do not write there themselves.

Idempotent: a record already present for the same step with the same
eval_newtasks/success_rate is skipped, so re-running is harmless.
"""
import argparse, json, glob, os, sys
from pathlib import Path

RES = Path("/capstor/store/cscs/swissai/a0220/ralfroemer/results")
KEY = "eval_newtasks/success_rate"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    sidecars = sorted(RES.glob("*/*/reeval_*/*/*/eval_metrics_eval_newtasks_step*.json"))
    if not sidecars:
        print("no re-eval sidecars found yet"); return

    merged = skipped = failed = 0
    for sc in sidecars:
        ck = sc.parent                                  # staging checkpoint dir
        stage_base = ck.parent.parent                   # <subrun>/reeval_<tag>
        subrun = stage_base.parent
        cfg_name, exp_name = ck.parent.name, ck.name
        target = subrun / cfg_name / exp_name / "metrics.jsonl"
        try:
            rec = json.loads(sc.read_text())
        except Exception as e:
            print(f"  UNREADABLE {sc}: {e}"); failed += 1; continue
        # NB: login node is Python 3.6 -- no str.removesuffix/removeprefix here.
        step = int(sc.name.split("step")[-1][: -len(".json")])
        if KEY not in rec:
            print(f"  NO {KEY} in {sc.name} ({subrun.parent.name}/{subrun.name}) -- eval likely did not finish")
            failed += 1; continue
        if not target.exists():
            print(f"  MISSING TARGET {target}"); failed += 1; continue

        already = False
        with target.open() as f:
            for line in f:
                if KEY in line:
                    try: d = json.loads(line)
                    except Exception: continue
                    if int(d.get("step", -1)) == step and abs(d.get(KEY, -1) - rec[KEY]) < 1e-12:
                        already = True; break
        if already:
            skipped += 1; continue

        out = {k: (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in rec.items()}
        out["step"] = step
        line = json.dumps(out) + "\n"
        print(f"  {'APPEND' if args.apply else 'WOULD APPEND'} step={step} {KEY}={rec[KEY]:.4f} "
              f"-> {subrun.parent.name}/{subrun.name}")
        if args.apply:
            with target.open("a") as f:
                f.write(line)
        merged += 1

    print(f"\nsidecars={len(sidecars)} merged={merged} already_present={skipped} problems={failed}")
    if not args.apply and merged:
        print("dry run -- re-run with --apply to write")


if __name__ == "__main__":
    main()
