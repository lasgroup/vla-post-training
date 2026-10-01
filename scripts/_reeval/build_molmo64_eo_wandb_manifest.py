#!/usr/bin/env python3
"""Build the W&B publish manifest for the Molmo-64 BoN+FSFT (EO) split evals.

These three runs had their in-loop eval DISABLED, so their mid (round 5) and
final (round 10) evals are produced by the separate split-eval jobs and merged
into each run's metrics.jsonl by merge_molmo64_eo_split_eval.py. Nothing about
them is on W&B: the split-eval jobs resume the ORIGINAL run and log at
step=250000, which is non-monotonic for a run already past 280k, so a resumed
run drops it -- the same silent failure that lost the first re-eval batch.

So they are published as NEW runs (a fresh run starts at step 0 and accepts the
row), consumed by publish_reevals_to_wandb.py, which expects:
    {<subrun key>: {recipe, seed, group_name, collect_tasks, eval_tasks,
                    arms: {mid|final: {step, params_path, rl_state_path,
                                       metrics: {<key without prefix>: value}}}}}

NOTE on grouping: these are the STANDARD 80-task eval (64 train + 16 held-out)
at a round boundary -- NOT the "new 8-task split" re-evals that live in the
different_instances_eval groups. They get their own group so the two eval sets
are never conflated in a plot.
"""
import json, re, sys
from pathlib import Path

RES = Path("/capstor/store/cscs/swissai/a0220/ralfroemer/results")
REPO = Path("/users/ralfroemer/vla-post-training")
OUT = REPO / ".molmo64_eo_wandb_manifest.json"
RECIPE = "pi05_molmo_online_aw_sft_expert_only"
# Distinct from the different_instances_eval groups on purpose (different eval set).
GROUP = "splitval_tasks64_bofn_plus_fsft_eo_molmo_v0"
NAME_PREFIX = "split_eval"   # NOT different_instances_eval: this is the standard 80-task eval
CFG = REPO / "scripts/configs/multitask/bofn_plus_fsft_eo/molmo/bofn_plus_fsft_molmo_tasks64_seed0_v0.yaml"
ARMS = {"mid": ("eval_mid/", 250000), "final": ("eval/", 500000)}


def task_lists():
    txt = CFG.read_text()
    start = txt.index("collect.eval_tasks]")
    region = txt[start:txt.index("collect.store_prefix_rep")]
    lists = re.findall(r"\[([^\[\]]*molmo_[^\[\]]*)\]", region, re.S)
    return re.findall(r"molmo_\d+", lists[0]), re.findall(r"molmo_\d+", lists[1])


def main():
    train_tasks, eval_tasks = task_lists()
    assert len(train_tasks) == 64 and len(eval_tasks) == 80, (len(train_tasks), len(eval_tasks))
    man = {}
    for seed in (0, 1, 2):
        sub = RES / f"vla-post-training_multitask_tasks64_bofn_plus_fsft_eo_molmo_v0_seed{seed}" / "0"
        exps = [p for p in sub.glob(f"{RECIPE}/vla-post-training_*") if p.is_dir()]
        if not exps:
            print(f"  seed{seed}: no exp dir"); continue
        exp = exps[0]
        mfile = exp / "metrics.jsonl"
        if not mfile.exists():
            print(f"  seed{seed}: no metrics.jsonl"); continue
        # last record for each arm prefix
        latest = {}
        for line in mfile.open():
            for arm, (pref, _) in ARMS.items():
                if f'"{pref}success_rate"' in line:
                    try:
                        latest[arm] = json.loads(line)
                    except Exception:
                        pass
        arms = {}
        for arm, (pref, step) in ARMS.items():
            rec = latest.get(arm)
            if rec is None:
                continue
            metrics = {k[len(pref):]: v for k, v in rec.items() if k.startswith(pref)}
            npt = sum(1 for k in metrics if k.startswith("success_rate/"))
            if npt != 80:
                print(f"  seed{seed} {arm}: {npt} per-task keys (expected 80) -- SKIPPING")
                continue
            arms[arm] = {
                "step": int(rec.get("step", step)),
                "params_path": str(exp / str(step)),
                "rl_state_path": str(exp / "rl_state" / str(step)),
                "metrics": metrics,
            }
        if not arms:
            print(f"  seed{seed}: no complete arm yet"); continue
        key = f"tasks64_eo_seed{seed}"
        man[key] = {
            "recipe": RECIPE,
            "seed": seed,
            "group_name": GROUP,
            "name_prefix": NAME_PREFIX,
            "collect_tasks": train_tasks,
            "eval_tasks": eval_tasks,
            "arms": arms,
        }
        print(f"  {key}: arms={sorted(arms)} "
              + " ".join(f"{a}={arms[a]['metrics']['success_rate']:.4f}@{arms[a]['step']}" for a in sorted(arms)))
    OUT.write_text(json.dumps(man, indent=2, sort_keys=True))
    print(f"\nwrote {OUT}  ({len(man)} entries, group={GROUP})")


if __name__ == "__main__":
    main()
