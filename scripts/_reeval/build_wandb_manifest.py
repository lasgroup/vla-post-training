#!/usr/bin/env python3
"""Assemble the publish manifest for the new-split re-evals.

Runs on the LOGIN NODE (python 3.6, no wandb). Gathers everything the publish
step needs into one JSON so the container job only has to call wandb.

Shape follows the FSFT `different_instances_eval` pattern: one NEW wandb run per
evaluated training run, group per (recipe, task-count) with seeds pooled, a
single history row at _step=0 mirrored into the summary.

Expert-only has two checkpoints per training run, so both live in one run:
  eval/*      -> the FINAL checkpoint   (same key names as FSFT -> directly comparable)
  eval_mid/*  -> the round-5 checkpoint (additive; a mid-unaware plot ignores it)
BoN has final only, so its runs carry exactly the 11 FSFT keys.
"""
import glob, json, os, re, sys

RES = "/capstor/store/cscs/swissai/a0220/ralfroemer/results"
NEW_TASKS = ["molmo_666", "molmo_602", "molmo_543", "molmo_273",
             "molmo_916", "molmo_505", "molmo_769", "molmo_675"]
RECIPE_GROUP = {
    "pi05_molmo_online_best_of_n": "bofn_molmo_tasks{}",
    "pi05_molmo_online_aw_sft_expert_only": "bofn_plus_fsft_eo_molmo_tasks{}",
}


def training_tasks(subrun):
    for js in sorted(glob.glob(os.path.join(subrun, "job_*.sbatch.sh"))):
        if "reeval" in js:
            continue
        line = next((l for l in open(js) if l.startswith("srun ")), "")
        m = re.search(r"--collect\.tasks ((?:molmo_\d+ )+)", line)
        if m:
            return m.group(1).split()
    return None


def main():
    out = {}
    problems = []
    for sc in sorted(glob.glob(RES + "/*/*/reeval_*/*/*/eval_metrics_eval_newtasks_step*.json")):
        ck = os.path.dirname(sc)
        exp_name = os.path.basename(ck)
        recipe = os.path.basename(os.path.dirname(ck))
        stage_base = os.path.dirname(os.path.dirname(ck))
        arm = os.path.basename(stage_base).replace("reeval_", "")
        subrun = os.path.dirname(stage_base)
        group_dir = os.path.basename(os.path.dirname(subrun))
        idx = os.path.basename(subrun)
        step = int(os.path.basename(sc).split("step")[-1][: -len(".json")])

        short = group_dir.replace("vla-post-training_multitask_", "")
        n = int(re.search(r"tasks(\d+)", short).group(1))
        seed = int(re.search(r"seed(\d+)", short).group(1))

        pol = os.path.realpath(os.path.join(ck, str(step)))
        rls = os.path.realpath(os.path.join(ck, "rl_state", str(step)))
        tasks = training_tasks(subrun)
        if tasks is None:
            problems.append((short + "/" + idx, "no training tasks in sbatch")); continue

        raw = json.load(open(sc))
        metrics = {}
        for k, v in raw.items():
            if not k.startswith("eval_newtasks/"):
                continue
            leaf = k[len("eval_newtasks/"):]
            metrics[leaf] = v
        # match the FSFT pattern exactly: this counter is a stale cumulative
        # total from the source run and is meaningless for an eval-only pass.
        metrics["total_collected_episodes"] = 0

        key = subrun  # one wandb run per evaluated training run
        rec = out.setdefault(key, {
            "recipe": recipe,
            "group_name": RECIPE_GROUP[recipe].format(n),
            "tasks": n, "seed": seed, "exp_name_src": exp_name,
            "collect_tasks": tasks, "eval_tasks": NEW_TASKS,
            "subrun": subrun, "arms": {},
        })
        rec["arms"][arm] = {"step": step, "params_path": os.path.join(pol, "params"),
                            "rl_state_path": rls, "metrics": metrics}

    for k, v in sorted(out.items()):
        if "final" not in v["arms"]:
            problems.append((os.path.basename(os.path.dirname(k)) + "/" + os.path.basename(k),
                             "no FINAL arm (mid only) -> run will carry eval_mid/* only"))

    path = "/users/ralfroemer/vla-post-training/.reeval_wandb_manifest.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("runs to publish: %d" % len(out))
    by_group = {}
    for v in out.values():
        by_group.setdefault(v["group_name"], []).append(v)
    for g in sorted(by_group):
        arms = {}
        for v in by_group[g]:
            for a in v["arms"]:
                arms[a] = arms.get(a, 0) + 1
        print("  %-34s %2d runs   arms: %s" % (g, len(by_group[g]), arms))
    print("total eval results covered: %d" % sum(len(v["arms"]) for v in out.values()))
    print("manifest -> " + path)
    if problems:
        print("\nNOTES:")
        for a, b in problems:
            print("  %s: %s" % (a, b))


if __name__ == "__main__":
    main()
