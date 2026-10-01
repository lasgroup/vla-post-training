"""
Re-upload the 9 deleted BoN libero:1 runs to lsy-tum/vla-post-training under
NEW wids, so main_figure.ipynb (in the plotting repo) picks them up again.

Reads each original run's local mirror (wandb-metadata.json + wandb-summary.json
under <repo>/wandb/run-*-<wid>/), calls wandb.init() with the same group / seed /
collect.tasks, copies the eval-summary keys, then finish()es. wandb.Api will
then see them as fresh finished runs.

Run once from a node with wandb login + lsy-tum team access:

    cd /users/ralfroemer/vla-post-training
    uv run scripts/reupload_deleted_bofn.py
"""
import json
import os
import re
from pathlib import Path

import wandb

REPO = Path(__file__).resolve().parent.parent
WANDB_ROOT = REPO / "wandb"
RESULTS_ROOT = Path("/capstor/store/cscs/swissai/a0220/ralfroemer/results")
ENTITY, PROJECT = "lsy-tum", "vla-post-training"

DELETED = [
    "khyik48u", "9miwb3qz", "au2w6r0y",       # seed 0
    "ct5x28us", "jy0ry8o9", "5icuf0m6",       # seed 1
    "a2kqxoxx", "jgc88jns", "sptms2n0",       # seed 2
]


def latest_mirror(wid):
    """Newest mirror dir for the wid, regardless of summary state."""
    dirs = sorted(WANDB_ROOT.glob(f"run-*-{wid}"))
    return dirs[-1] if dirs else None


def latest_mirror_with_summary(wid):
    """Newest mirror whose wandb-summary.json contains eval keys, else None."""
    for d in sorted(WANDB_ROOT.glob(f"run-*-{wid}"), reverse=True):
        summ_p = d / "files/wandb-summary.json"
        if not summ_p.exists() or summ_p.stat().st_size == 0:
            continue
        try:
            summ = json.loads(summ_p.read_text())
        except json.JSONDecodeError:
            continue
        if any(k.startswith("eval/success_rate/") for k in summ):
            return d
    return None


def find_results_subrun(wid):
    """Locate the results-tree subrun dir that logged this wid."""
    for wid_file in RESULTS_ROOT.glob("*/[0-9]*/**/wandb_id.txt"):
        try:
            if wid_file.read_text().strip() == wid:
                # The subrun dir is 3 levels up (…/results/<group>/<idx>/<cfg>/<exp>/wandb_id.txt).
                # slurm-*.out lives under <group>/<idx>/.
                return wid_file.parents[2]
        except OSError:
            continue
    return None


def summary_from_slurm(subrun_dir):
    """Reconstruct {eval/success_rate/<t>: v, ...} from the run's slurm logs."""
    best_key = None
    best = {}
    order = 0
    for log in sorted(subrun_dir.glob("slurm-*.out"), key=lambda p: p.stat().st_mtime):
        try:
            text = log.read_text(errors="ignore")
        except OSError:
            continue
        for line in text.splitlines():
            if "Eval at step" not in line:
                continue
            m = re.search(r"Eval at step (\d+)", line)
            if not m:
                continue
            step = int(m.group(1))
            sr = {
                f"eval/success_rate/{t}": float(v)
                for t, v in re.findall(r"eval/success_rate/(\S+?)=([0-9.]+)", line)
            }
            if not sr:
                continue
            order += 1
            key = (step, order)
            if best_key is None or key > best_key:
                best_key, best = key, sr
    return best


def parse_args(meta):
    args = meta.get("args", [])
    parsed = {"tasks": [], "seed": None, "group": None}
    if "--group_name" in args:
        parsed["group"] = args[args.index("--group_name") + 1]
    if "--seed" in args:
        try:
            parsed["seed"] = int(args[args.index("--seed") + 1])
        except (ValueError, IndexError):
            pass
    if "--collect.tasks" in args:
        i = args.index("--collect.tasks") + 1
        while i < len(args) and not args[i].startswith("--"):
            parsed["tasks"].append(args[i])
            i += 1
    return parsed


def main():
    os.environ["WANDB_ENTITY"] = ENTITY
    for wid in DELETED:
        # Metadata dir (for group / seed / collect.tasks) — use any mirror, they're identical.
        meta_dir = latest_mirror(wid)
        if meta_dir is None:
            print(f"{wid}: no local mirror under {WANDB_ROOT}, skipping")
            continue
        meta_p = meta_dir / "files/wandb-metadata.json"
        if not meta_p.exists():
            print(f"{wid}: no wandb-metadata.json, skipping")
            continue
        meta = json.loads(meta_p.read_text())
        info = parse_args(meta)
        if not info["group"] or info["seed"] is None or not info["tasks"]:
            print(f"{wid}: could not parse group/seed/tasks, skipping")
            continue

        # Summary — try wandb-summary.json first; else fall back to slurm log.
        summ_dir = latest_mirror_with_summary(wid)
        summ = {}
        if summ_dir is not None:
            summ = json.loads((summ_dir / "files/wandb-summary.json").read_text())
            src = f"wandb-summary at {summ_dir}"
        else:
            subrun = find_results_subrun(wid)
            if subrun is None:
                print(f"{wid}: no wandb-summary AND no results subrun on disk, skipping")
                continue
            summ = summary_from_slurm(subrun)
            if not summ:
                print(f"{wid}: no eval markers in slurm log at {subrun}, skipping")
                continue
            src = f"slurm log at {subrun}"

        print(f"{wid} -> new run in group={info['group']} seed={info['seed']} tasks={info['tasks']} (summary from {src})")
        run = wandb.init(
            entity=ENTITY,
            project=PROJECT,
            group=info["group"],
            job_type="reupload",
            config={
                "seed": info["seed"],
                "collect": {"tasks": info["tasks"]},
                "reuploaded_from_wid": wid,
            },
            name=meta.get("startedAt", "reupload_" + wid),
            reinit=True,
        )
        for k, v in summ.items():
            if k.startswith("_"):
                continue
            if not isinstance(v, (int, float, str, bool)):
                continue
            wandb.run.summary[k] = v
        wandb.finish()
        print(f"  uploaded as new wid={run.id}")


if __name__ == "__main__":
    main()
