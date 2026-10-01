#!/usr/bin/env python3
"""Experiment #3: does BoN+FSFT amortize BoN?

Re-runs the MID eval (round 5, step 50000) of every single-task Molmo BoN+FSFT
run with `--rl.n_samples 1`, i.e. the policy alone. At N=1 the candidate set has
one element and `argmax` returns it unchanged (best_of_n_learner.py ~497-503),
so the actions are pure policy samples; the critic is still evaluated and its
score discarded, which costs time but cannot change behaviour.

Round 5, not round 10, on purpose: it is the point where BoN+FSFT has taken
500 actor steps over 5 collect rounds (100 episodes), which is exactly the pure
FSFT baseline's budget (num_train_steps 500, collect_interval 100). Comparing at
round 10 would confound amortization with twice the training budget.

Scope: 2 families (full fine-tuning, expert-only) x 4 single tasks x 3 seeds = 24.
The eval task set is left as the run's own 17 tasks (its training task + the 16
held-out ones) -- unlike build_reeval_jobs.py, nothing about the tasks changes.

Staging follows build_reeval_jobs.py exactly: symlinks to the real policy and
critic, a manifest at the target step, an EMPTY replay_shards dir, and a copy of
wandb_id.txt. The source run dirs are never modified.

    python3 scripts/_reeval/build_amortization_eval.py            # dry run
    python3 scripts/_reeval/build_amortization_eval.py --apply
"""
import argparse, json, re, shutil
from pathlib import Path

RES = Path("/capstor/store/cscs/swissai/a0220/ralfroemer/results")
PREFIX = "eval_mid_n1"
STAGE_TAG = "amort_n1"
S = 50000                      # round 5: policy@50000 and rl_state@50000 both exist
ACCOUNT = "ab037"


def scope():
    """(family, group, index) for all 24 cells."""
    out = []
    for fam, slug in (("FSFT", "bofn_plus_fsft"), ("EO", "bofn_plus_fsft_eo")):
        for s in (0, 1, 2):
            out.append((fam, f"vla-post-training_multitask_tasks1_{slug}_molmo_v0_seed{s}", 0))
            for i in (0, 1, 2):
                out.append((fam, f"vla-post-training_multitask_tasks1_{slug}_molmo_v0_seed{s}_rest", i))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_env_num", type=int, default=8)
    ap.add_argument("--num_eval_rollouts", type=int, default=32)
    ap.add_argument("--only", default=None)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    planned, problems = [], []
    for fam, grp, idx in scope():
        cell = f"{grp}/{idx}"
        if args.only and args.only not in cell:
            continue
        sub = RES / grp / str(idx)
        exps = [p for p in sorted(sub.glob("*/vla-post-training_*"))
                if p.is_dir() and STAGE_TAG not in str(p) and "reeval_" not in str(p)]
        if not exps:
            problems.append((cell, "no exp dir")); continue
        exp = exps[0]
        cfg_name, exp_name = exp.parent.name, exp.name
        pol = sorted(int(p.name) for p in exp.iterdir() if p.is_dir() and p.name.isdigit())
        rls = sorted(int(p.name) for p in (exp / "rl_state").iterdir() if p.name.isdigit())
        if S not in pol:
            problems.append((cell, f"no round-5 policy at {S} (have {pol})")); continue
        if S not in rls:
            problems.append((cell, f"no critic at {S} (have {rls})")); continue

        cands = sorted(q for q in sub.glob("job_*.sbatch.sh")
                       if "reeval" not in q.name and STAGE_TAG not in q.name)
        if not cands:
            problems.append((cell, "no job_*.sbatch.sh to template from")); continue
        srun_line = next(l for l in cands[0].read_text().splitlines() if l.startswith("srun "))
        n_tasks = len(re.findall(r"molmo_\d+", srun_line.split("--collect.eval_tasks")[1])) \
            if "--collect.eval_tasks" in srun_line else 0

        stage_base = sub / f"{STAGE_TAG}_mid"
        ckpt_dir = stage_base / cfg_name / exp_name
        planned.append(dict(fam=fam, cell=cell, exp=exp, n_tasks=n_tasks,
                            stage_base=stage_base, ckpt_dir=ckpt_dir))
        if not args.apply:
            continue

        (ckpt_dir / "rl_state").mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "runtime_state" / "replay_shards").mkdir(parents=True, exist_ok=True)
        for link, target in ((ckpt_dir / str(S), exp / str(S)),
                             (ckpt_dir / "rl_state" / str(S), exp / "rl_state" / str(S))):
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(target)
        manifest = json.loads((exp / "runtime_state" / "resume_state.json").read_text())
        manifest["step"] = S
        manifest["replay_shard_dir"] = str(ckpt_dir / "runtime_state" / "replay_shards")
        (ckpt_dir / "runtime_state" / "resume_state.json").write_text(json.dumps(manifest, indent=2))
        shutil.copy2(exp / "wandb_id.txt", ckpt_dir / "wandb_id.txt")
        (ckpt_dir / "AMORT_README.txt").write_text(
            f"Experiment #3 staging: round-5 mid eval re-run at n_samples=1.\n"
            f"Source run : {exp}\nStep       : {S} (policy and critic both exact)\n"
            f"Metrics    : {PREFIX}/*   -> eval_metrics_{PREFIX}_step{S}.json here\n"
            f"Eval tasks : unchanged from the source run.\n"
            f"replay_shards is intentionally EMPTY; eval never samples the buffer.\n")

        line = srun_line
        line = line.replace("scripts/exp.py", "scripts/eval.py")
        line = re.sub(r"--account=\S+", f"--account={ACCOUNT}", line)
        if "env XLA_PYTHON_CLIENT_MEM_FRACTION" in line:
            line = line.replace("env XLA_PYTHON_CLIENT_MEM_FRACTION",
                                f"env EVAL_METRIC_PREFIX={PREFIX} XLA_PYTHON_CLIENT_MEM_FRACTION")
        else:
            line = line.replace("uv run ", f"env EVAL_METRIC_PREFIX={PREFIX} uv run ", 1)
        line = re.sub(r"--checkpoint_base_dir \S+", f"--checkpoint_base_dir {stage_base}", line)
        line = re.sub(r"--collect\.eval_env_num \d+", f"--collect.eval_env_num {args.eval_env_num}", line)
        line = re.sub(r"--collect\.num_eval_rollouts \d+", "", line)
        line = re.sub(r"--collect\.eval_interval \d+", f"--collect.eval_interval {S}", line)
        line = re.sub(r"--rl\.n_samples \d+", "--rl.n_samples 1", line)   # THE experiment
        line = line.replace(" --requeue_before_eval", "")
        line = line.replace(" || child_status=$?", "")
        line += f" --collect.num_eval_rollouts {args.num_eval_rollouts}"

        checks = {
            f"EVAL_METRIC_PREFIX={PREFIX}": f"EVAL_METRIC_PREFIX={PREFIX}" in line,
            "eval.py": "scripts/eval.py" in line and "scripts/exp.py" not in line,
            "staging base": f"--checkpoint_base_dir {stage_base}" in line,
            "n_samples==1": len(re.findall(r"--rl\.n_samples 1(?!\d)", line)) == 1
                            and len(re.findall(r"--rl\.n_samples \d+", line)) == 1,
            "eval tasks kept (17)": n_tasks == 17,
            "no requeue_before_eval": "--requeue_before_eval" not in line,
            "resume": "--resume" in line,
            "single num_eval_rollouts": line.count("--collect.num_eval_rollouts") == 1,
            f"eval_interval == {S}": f"--collect.eval_interval {S}" in line,
            f"account {ACCOUNT}": f"--account={ACCOUNT}" in line,
        }
        missing = [k for k, ok in checks.items() if not ok]
        if missing:
            raise SystemExit(f"REFUSING to write {cell}: failed checks {missing}")

        job = sub / f"job_{STAGE_TAG}.sbatch.sh"
        job.write_text("#!/bin/bash\nset -euo pipefail\n\n"
                       "cd /users/ralfroemer/vla-post-training\n\nulimit -c 0\n\n"
                       f"# Experiment #3: round-5 mid eval at n_samples=1 (policy only).\n"
                       f"# Writes {PREFIX}/* ; source run dir untouched.\n{line}\n")
        job.chmod(0o755)

    for p in planned:
        print(f"  {p['fam']:4s} {p['cell'].replace('vla-post-training_multitask_tasks1_','')}  "
              f"eval_tasks={p['n_tasks']}")
    print(f"\nplanned: {len(planned)}   problems: {len(problems)}")
    for c, why in problems:
        print(f"  PROBLEM {c}: {why}")
    if not args.apply:
        print("\ndry run -- pass --apply to stage and write job scripts")


if __name__ == "__main__":
    main()
