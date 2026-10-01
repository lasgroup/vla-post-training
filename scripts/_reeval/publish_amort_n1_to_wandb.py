#!/usr/bin/env python3
"""Publish the experiment #3 (amortization) results into the ORIGINAL W&B runs.

Each staged eval wrote eval_metrics_eval_mid_n1_step50000.json next to its
staging checkpoint. The job itself also called wandb.log at step 50000, which
W&B DROPPED: those runs already hold history to ~99998 and history is
append-only and monotonic in _step. So the values never reached the cloud.

Summary is not step-based, so it can be written retroactively. That is what this
script does -- one scalar block per run, under the `eval_mid_n1/` prefix so
nothing collides with the existing `eval/*` (final) or `eval_mid/*` (round 5,
N=32) entries. It is also what this repo's analysis scripts read
(inspect_bofn_molmo_8_8_category.py does run.summary.get(...)), so the numbers
join to the originals with no extra key.

Guards before every write:
  - the run id must resolve, and its `group` must match the one implied by the
    subrun path (a mis-keyed wandb_id.txt would otherwise write into a
    stranger's run and there is no undo)
  - the sidecar must carry the aggregate plus exactly 17 per-task rates
  - if `eval_mid_n1/success_rate` is already present it must match, else the
    cell is reported and skipped

Ledger: .amort_n1_wandb_published.json (re-running is safe and idempotent).

    python scripts/_reeval/publish_amort_n1_to_wandb.py           # dry run
    python scripts/_reeval/publish_amort_n1_to_wandb.py --apply
"""
import argparse, json, os, re
from pathlib import Path

RES = Path("/capstor/store/cscs/swissai/a0220/ralfroemer/results")
REPO = Path("/users/ralfroemer/vla-post-training")
LEDGER = REPO / ".amort_n1_wandb_published.json"
ENTITY, PROJECT = "lsy-tum", "vla-post-training"
SIDECAR = "eval_metrics_eval_mid_n1_step50000.json"
PREFIX = "eval_mid_n1/"
STEP = 50000
EXPECT_TASKS = 17


def cells():
    for sc in sorted(RES.glob(f"*/*/amort_n1_mid/*/*/{SIDECAR}")):
        ckpt = sc.parent
        sub = ckpt.parent.parent.parent            # <subrun>/amort_n1_mid/<cfg>/<exp>
        group_dir = sub.parent.name                # vla-post-training_multitask_...
        expect_group = group_dir.replace("vla-post-training_", "", 1)
        yield dict(sidecar=sc, ckpt=ckpt, cell=f"{group_dir}/{sub.name}",
                   expect_group=expect_group,
                   run_id=(ckpt / "wandb_id.txt").read_text().strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import wandb
    api = wandb.Api(timeout=120)
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}

    todo = [c for c in cells() if c["cell"] not in ledger]
    if args.limit:
        todo = todo[: args.limit]
    print(f"sidecars={len(list(cells()))}  already_published={len(ledger)}  todo={len(todo)}")

    ok = skipped = problems = 0
    for c in todo:
        d = json.loads(c["sidecar"].read_text())
        agg = d.get(PREFIX + "success_rate")
        per = {k: v for k, v in d.items() if k.startswith(PREFIX + "success_rate/")}
        if agg is None or len(per) != EXPECT_TASKS:
            print(f"  BAD SIDECAR {c['cell']}: agg={agg} per_task={len(per)} -- skipping")
            problems += 1
            continue
        try:
            run = api.run(f"{ENTITY}/{PROJECT}/{c['run_id']}")
        except Exception as e:
            print(f"  NO RUN {c['cell']} id={c['run_id']}: {type(e).__name__} -- skipping")
            problems += 1
            continue
        if run.group != c["expect_group"]:
            print(f"  GROUP MISMATCH {c['cell']}: run {c['run_id']} is in "
                  f"'{run.group}', expected '{c['expect_group']}' -- REFUSING")
            problems += 1
            continue
        existing = run.summary.get(PREFIX + "success_rate")
        if existing is not None and abs(float(existing) - agg) > 1e-12:
            print(f"  CONFLICT {c['cell']}: run already has {existing}, sidecar says {agg} -- skipping")
            problems += 1
            continue

        payload = dict(d)
        payload[PREFIX + "step"] = STEP
        payload[PREFIX + "n_samples"] = 1
        print(f"  {'WRITE' if args.apply else 'would write'} {c['run_id']}  {c['cell']:<62} "
              f"sr={agg:.4f}  keys={len(payload)}")
        if not args.apply:
            continue

        run.summary.update(payload)
        check = api.run(f"{ENTITY}/{PROJECT}/{c['run_id']}").summary.get(PREFIX + "success_rate")
        if check is None or abs(float(check) - agg) > 1e-12:
            print(f"    VERIFY FAILED: read back {check}")
            problems += 1
            continue
        ledger[c["cell"]] = {"run_id": c["run_id"], "group": run.group,
                             "success_rate": agg, "keys": len(payload)}
        LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True))
        ok += 1

    print(f"\nwritten={ok} problems={problems} ledger={LEDGER} ({len(ledger)} cells)")
    if not args.apply:
        print("dry run -- pass --apply to write")


if __name__ == "__main__":
    main()
