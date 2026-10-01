"""
Overwrite the wandb-server summary of a run whose in-place summary went stale
after a rollback (wandb drops re-logged same-step points and the summary keeps
the older, wrong values). Reads the authoritative final-eval markers from the
run's slurm log and pushes them to the wandb summary.

The list below is (wid, subrun_dir_on_disk). Extend as needed.

Run:
    cd /users/ralfroemer/vla-post-training
    uv run scripts/fix_stale_wandb_summary.py
"""
import re
from pathlib import Path

import wandb

ENTITY, PROJECT = "lsy-tum", "vla-post-training"
RESULTS = Path("/capstor/store/cscs/swissai/a0220/ralfroemer/results")

# (wid, results-subrun-dir) pairs to repair.
TARGETS = [
    ("5h5uyjpr", RESULTS / "vla-post-training_multitask_tasks8_bofn_molmo_v0_seed2" / "1"),
]


def eval_from_logs(subdir: Path):
    """Return {task: sr} for the highest-step eval marker in this subrun's slurm logs."""
    best_key = None
    best = {}
    order = 0
    for log in sorted(subdir.glob("slurm-*.out"), key=lambda p: p.stat().st_mtime):
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
            sr = {t: float(v) for t, v in re.findall(r"eval/success_rate/(\S+?)=([0-9.]+)", line)}
            if not sr:
                continue
            order += 1
            key = (step, order)
            if best_key is None or key > best_key:
                best_key, best = key, sr
    return best_key, best


def main():
    api = wandb.Api(timeout=120)
    for wid, subdir in TARGETS:
        try:
            run = api.run(f"{ENTITY}/{PROJECT}/{wid}")
        except Exception as e:
            print(f"{wid}: cannot fetch from wandb ({e}); skipping")
            continue
        if not subdir.is_dir():
            print(f"{wid}: subrun dir {subdir} missing; skipping")
            continue
        key, sr = eval_from_logs(subdir)
        if not sr:
            print(f"{wid}: no eval markers in slurm log at {subdir}; skipping")
            continue
        print(f"{wid}: writing {len(sr)} eval keys (source: slurm log final step={key[0]})")
        # Overwrite each per-task key on the wandb summary.
        for t, v in sr.items():
            run.summary[f"eval/success_rate/{t}"] = float(v)
        # Also refresh the aggregate.
        run.summary["eval/success_rate"] = sum(sr.values()) / len(sr)
        run.summary.update()
        print(f"  ✔ pushed. Sample: {list(sorted(sr.items()))[:3]}")


if __name__ == "__main__":
    main()
