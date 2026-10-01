#!/usr/bin/env python3
"""Publish the new-split re-evals as fresh W&B runs (runs INSIDE the container).

Mirrors the FSFT `different_instances_eval` pattern exactly, verified against a
live FSFT run:
  - one NEW run per evaluated training run, project lsy-tum/vla-post-training
  - group   = bofn_molmo_tasks{N} / bofn_plus_fsft_eo_molmo_tasks{N}, seeds pooled
  - name    = different_instances_eval_<ts>_<6hex>_seed{S}   (== config exp_name)
  - config  = nested collect.* / weight_loader.*, flat name/seed/resume/...
  - metrics = ONE history row at _step=0, mirrored into the summary

New runs are used rather than writing into the training runs because a resumed
run silently drops history at an already-committed step -- that is why the first
attempt produced nothing.

Expert-only carries both checkpoints in the SAME run:
  eval/*      FINAL   (identical key names to FSFT -> directly comparable)
  eval_mid/*  round 5 (additive; a mid-unaware plot just ignores it)
"""
import argparse, hashlib, json, os, sys, time

MANIFEST = "/users/ralfroemer/vla-post-training/.reeval_wandb_manifest.json"
LEDGER = "/users/ralfroemer/vla-post-training/.reeval_wandb_published.json"
ENTITY, PROJECT = "lsy-tum", "vla-post-training"
CKPT_BASE = "/capstor/store/cscs/swissai/ab037/ralfroemer/results/different_instances_eval_{group}/{seed}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    # Optional overrides so the same publisher can serve other eval batches
    # (e.g. the Molmo-64 EO split evals) without touching the defaults above.
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--ledger", default=LEDGER)
    args = ap.parse_args()
    manifest_path, ledger_path = args.manifest, args.ledger

    # Import lazily so a dry run works on the login node (no wandb there).
    if args.apply:
        import wandb

    man = json.load(open(manifest_path))
    ledger = json.load(open(ledger_path)) if os.path.exists(ledger_path) else {}
    todo = [k for k in sorted(man) if k not in ledger]
    if args.limit:
        todo = todo[: args.limit]
    print("manifest=%d already_published=%d todo=%d" % (len(man), len(ledger), len(todo)))

    for subrun in todo:
        r = man[subrun]
        arms = r["arms"]
        ts = time.strftime("%Y%m%d-%H%M%S")
        h = hashlib.sha1(subrun.encode()).hexdigest()[:6]
        # A batch may override the name/project tag: the Molmo-64 EO split evals
        # are the STANDARD 80-task eval, not the new-8-task-split re-evals, and
        # must not be named as if they were.
        tag = r.get("name_prefix", "different_instances_eval")
        exp_name = "%s_%s_%s_seed%d" % (tag, ts, h, r["seed"])

        cfg = {
            "name": r["recipe"],
            "num_train_steps": 0,
            "resume": True,
            "seed": r["seed"],
            "exp_name": exp_name,
            "group_name": r["group_name"],
            "project_name": r.get("name_prefix", "different_instances_eval"),
            "checkpoint_base_dir": CKPT_BASE.format(group=r["group_name"], seed=r["seed"]),
            "collect": {
                "tasks": r["collect_tasks"],          # TRAINING tasks -> the join key
                "eval_tasks": r["eval_tasks"],
                "num_eval_rollouts": 32,
            },
        }
        row = {}
        if "final" in arms:
            a = arms["final"]
            cfg["weight_loader"] = {"params_path": a["params_path"]}
            cfg["rl_state_path"] = a["rl_state_path"]
            cfg["eval_step"] = a["step"]
            for k, v in a["metrics"].items():
                row["eval/" + k] = v
        if "mid" in arms:
            a = arms["mid"]
            cfg["params_path_mid"] = a["params_path"]
            cfg["rl_state_path_mid"] = a["rl_state_path"]
            cfg["eval_step_mid"] = a["step"]
            for k, v in a["metrics"].items():
                row["eval_mid/" + k] = v
            if "final" not in arms:
                cfg["weight_loader"] = {"params_path": a["params_path"]}

        print("  %-34s %-52s keys=%d arms=%s" % (r["group_name"], exp_name, len(row), sorted(arms)))
        if not args.apply:
            continue

        run = wandb.init(entity=ENTITY, project=PROJECT, group=r["group_name"],
                         name=exp_name, config=cfg, reinit=True,
                         settings=wandb.Settings(console="off"))
        wandb.log(row, step=0)
        run.summary.update(row)
        rid = run.id
        wandb.finish()
        ledger[subrun] = {"run_id": rid, "exp_name": exp_name, "group": r["group_name"],
                          "keys": len(row), "arms": sorted(arms)}
        with open(ledger_path, "w") as f:
            json.dump(ledger, f, indent=2, sort_keys=True)

    if not args.apply:
        print("\ndry run -- pass --apply to create the runs")
    else:
        print("\npublished. ledger -> %s (%d runs)" % (ledger_path, len(ledger)))


if __name__ == "__main__":
    main()
