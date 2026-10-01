#!/usr/bin/env python3
"""Republish the 9 broken LIBERO tasks1 BoN runs to W&B with their full history.

Problem (diagnosed 2026-09-20): in each group
multitask_tasks1_bofn_libero_v0_seed{0,1,2} only the libero_90_38 run is intact
on W&B (4001 history rows, _step=99999). The other three per seed are EMPTY
SHELLS -- config plus a summary block, zero history rows, _step=None. Their
original run ids (recorded in wandb_id.txt) return "not found": the originals
were deleted and someone recreated them carrying only the final summary. One is
still named `reupload_khyik48u` after the dead id; the rest carry bare ISO
timestamps. Hence "learning curves for only one task".

The local data is complete -- every one of the 12 runs has 4011 rows in
metrics.jsonl ending with the final eval at step 99999 -- so the history can
simply be replayed into fresh runs.

This script creates ONE NEW run per broken cell, in the same group, named with
the canonical exp_name from the local results dir, and replays every row of
metrics.jsonl at its recorded step. Config is templated from that seed's intact
libero_90_38 run (same sweep, same hyperparameters) with only collect.tasks,
exp_name and checkpoint_base_dir overridden.

libero_90_38 (index 3) is NEVER touched. The pre-existing empty shells are NOT
deleted -- that is a separate, destructive decision.

Ledger: .libero_tasks1_republished.json (skips cells already done, so re-running
is safe).
"""
import argparse, json, os
from pathlib import Path

RES = Path("/capstor/store/cscs/swissai/a0220/ralfroemer/results")
REPO = Path("/users/ralfroemer/vla-post-training")
LEDGER = REPO / ".libero_tasks1_republished.json"
ENTITY, PROJECT = "lsy-tum", "vla-post-training"
# intact reference run per seed (libero_90_38) -> config template
TEMPLATE = {0: "740f4dy7", 1: "0u40jv7b", 2: "q67zccbp"}
SKIP_TASK = "libero_90_38"          # index 3: intact, leave alone


def cell_paths(seed, idx):
    sub = RES / f"vla-post-training_multitask_tasks1_bofn_libero_v0_seed{seed}" / str(idx)
    exps = sorted(sub.glob("*/vla-post-training_*"))
    job = sorted(sub.glob("job_*.sbatch.sh"))
    if not exps or not job:
        return None
    exp = exps[0]
    task = None
    for line in job[0].read_text().splitlines():
        if "--collect.tasks " in line:
            task = line.split("--collect.tasks ", 1)[1].split()[0]
            break
    return dict(sub=sub, exp=exp, task=task, metrics=exp / "metrics.jsonl",
                exp_name=exp.name, group=f"multitask_tasks1_bofn_libero_v0_seed{seed}")


def load_history(mfile):
    """-> list of (step, {key: float|str}) in file order; numbers coerced to float."""
    rows = []
    for line in mfile.open():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        step = d.pop("step", None)
        if step is None:
            continue
        clean = {}
        for k, v in d.items():
            if isinstance(v, str):
                try:
                    v = float(v)
                except ValueError:
                    pass
            clean[k] = v
        if clean:
            rows.append((int(step), clean))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only do the first N cells")
    args = ap.parse_args()
    if args.apply:
        import wandb

    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
    todo = []
    for seed in (0, 1, 2):
        for idx in (0, 1, 2, 3):
            c = cell_paths(seed, idx)
            if c is None:
                print(f"  seed{seed}/{idx}: MISSING local dirs -- skipping"); continue
            if c["task"] == SKIP_TASK:
                continue                      # intact on wandb, leave untouched
            key = f"seed{seed}_idx{idx}_{c['task']}"
            if key in ledger:
                continue
            todo.append((key, seed, c))
    if args.limit:
        todo = todo[: args.limit]
    print(f"cells to republish: {len(todo)} (already done: {len(ledger)})")

    for key, seed, c in todo:
        rows = load_history(c["metrics"])
        steps = [s for s, _ in rows]
        mono = all(b >= a for a, b in zip(steps, steps[1:]))
        final = next((d for s, d in reversed(rows) if "eval/success_rate" in d), None)
        print(f"  {key}: rows={len(rows)} steps {steps[0]}..{steps[-1]} monotonic={mono} "
              f"final_eval={final.get('eval/success_rate') if final else None}")
        if not mono:
            print("    REFUSING: steps not monotonic (wandb would drop rows)"); continue
        if final is None:
            print("    REFUSING: no eval/success_rate row found"); continue
        if not args.apply:
            continue

        import wandb
        tpl = wandb.Api().run(f"{ENTITY}/{PROJECT}/{TEMPLATE[seed]}").config
        cfg = dict(tpl)
        cfg["collect"] = dict(cfg.get("collect") or {})
        cfg["collect"]["tasks"] = [c["task"]]
        cfg["exp_name"] = c["exp_name"]
        cfg["checkpoint_base_dir"] = str(c["sub"])
        cfg["republished_from_local_metrics"] = str(c["metrics"])

        run = wandb.init(entity=ENTITY, project=PROJECT, group=c["group"],
                         name=c["exp_name"], config=cfg, reinit=True,
                         settings=wandb.Settings(console="off"))
        for step, d in rows:
            wandb.log(d, step=step)
        run.summary.update(final)
        rid = run.id
        wandb.finish()
        ledger[key] = {"run_id": rid, "group": c["group"], "name": c["exp_name"],
                       "task": c["task"], "rows": len(rows),
                       "final_eval": final.get("eval/success_rate")}
        LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True))
        print(f"    -> published {rid} ({len(rows)} rows)")

    if not args.apply:
        print("\ndry run -- pass --apply to create the runs")
    else:
        print(f"\nledger -> {LEDGER} ({len(ledger)} republished)")


if __name__ == "__main__":
    main()
