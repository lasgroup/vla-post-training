#!/usr/bin/env python3
"""Add the Molmo-64 EO FINAL (round 10, step 500000) split evals to the existing
W&B `split_eval_*` runs, which so far carry only the mid eval (eval_mid/*).

Source: the merged record merge_molmo64_eo_split_eval.py appended to each run's
metrics.jsonl (eval/* at step 500000, 80 per-task rates). Written into the run
SUMMARY -- not history, which is append-only and already holds the mid row at
_step 0 -- under `eval/*`, the same namespace publish_reevals_to_wandb.py uses
for finals, so the run ends up with eval/* (final) + eval_mid/* (mid) exactly as
if both arms had been published together. Config gains the final checkpoint pair.

Guards: run id from .molmo64_eo_wandb_published.json must resolve to a run in
the expected group whose name ends in seed<N>; the record must have 80 per-task
rates; an existing eval/success_rate must match or the seed is skipped.
Idempotent: the ledger's `arms` gains "final" once written.

    python scripts/_reeval/publish_molmo64_eo_final_to_wandb.py --seeds 1 2          # dry run
    python scripts/_reeval/publish_molmo64_eo_final_to_wandb.py --seeds 1 2 --apply
"""
import argparse, json
from pathlib import Path

RES = Path("/capstor/store/cscs/swissai/a0220/ralfroemer/results")
LEDGER = Path("/users/ralfroemer/vla-post-training/.molmo64_eo_wandb_published.json")
ENTITY, PROJECT = "lsy-tum", "vla-post-training"
GROUP = "splitval_tasks64_bofn_plus_fsft_eo_molmo_v0"
STEP = 500000


def final_record(seed):
    sub = RES / f"vla-post-training_multitask_tasks64_bofn_plus_fsft_eo_molmo_v0_seed{seed}" / "0"
    exps = [p for p in sub.glob("pi05_molmo_online_aw_sft_expert_only/vla-post-training_*") if p.is_dir()]
    assert len(exps) == 1, exps
    exp = exps[0]
    rec = None
    with (exp / "metrics.jsonl").open() as f:
        for line in f:                      # grep-style: only parse candidate lines
            if '"eval/success_rate"' in line and f'"step": {STEP}' in line:
                rec = json.loads(line)
    return exp, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    import wandb
    api = wandb.Api(timeout=120)
    ledger = json.loads(LEDGER.read_text())

    for s in args.seeds:
        key = f"tasks64_eo_seed{s}"
        ent = ledger[key]
        if "final" in ent["arms"]:
            print(f"  seed{s}: final already published -- skip"); continue
        exp, rec = final_record(s)
        if rec is None:
            print(f"  seed{s}: no merged final record at step {STEP} -- skip"); continue
        payload = {k: v for k, v in rec.items() if k.startswith("eval/")}
        npt = sum(1 for k in payload if k.startswith("eval/success_rate/"))
        if npt != 80:
            print(f"  seed{s}: {npt} per-task rates (expected 80) -- skip"); continue
        run = api.run(f"{ENTITY}/{PROJECT}/{ent['run_id']}")
        if run.group != GROUP or not run.name.endswith(f"seed{s}"):
            print(f"  seed{s}: REFUSING, run {run.id} is '{run.name}' in '{run.group}'"); continue
        old = run.summary.get("eval/success_rate")
        if old is not None and abs(float(old) - payload["eval/success_rate"]) > 1e-12:
            print(f"  seed{s}: CONFLICT, run already has eval/success_rate={old}"); continue
        mid = run.summary.get("eval_mid/success_rate")
        print(f"  {'WRITE' if args.apply else 'would write'} seed{s} -> {run.id} ({run.name}): "
              f"eval/success_rate={payload['eval/success_rate']:.4f}  keys={len(payload)}  "
              f"(existing eval_mid/success_rate={mid})")
        if not args.apply:
            continue
        run.summary.update(payload)
        run.config["eval_step"] = STEP
        run.config["weight_loader"] = {"params_path": str(exp / str(STEP))}
        run.config["rl_state_path"] = str(exp / "rl_state" / str(STEP))
        run.update()
        back = api.run(f"{ENTITY}/{PROJECT}/{ent['run_id']}").summary
        ok = (abs(back.get("eval/success_rate", -1) - payload["eval/success_rate"]) < 1e-12
              and sum(1 for k in back.keys() if k.startswith("eval/success_rate/")) == 80
              and back.get("eval_mid/success_rate") == mid)
        print(f"    verify: {'OK' if ok else 'FAILED'}")
        if ok:
            ent["arms"] = sorted(set(ent["arms"]) | {"final"})
            ent["keys"] = ent.get("keys", 0) + len(payload)
            LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True))
    if not args.apply:
        print("dry run -- pass --apply to write")


if __name__ == "__main__":
    main()
