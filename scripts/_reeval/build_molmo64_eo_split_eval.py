#!/usr/bin/env python3
"""Stage + generate the split (40+40) eval jobs for Molmo-64 BoN+FSFT (EO).

*** UNVERIFIED -- DRY-RUN AND EYEBALL BEFORE `--apply`. ***
Written 2026-09-17 while /capstor was unmounted (CSCS maintenance), so it could
not be tested against real checkpoints. Run it WITHOUT --apply first, read the
plan, then --apply and inspect one generated job script before submitting.

Why split: the EO-64 eval is 32 rollouts x 80 tasks = 2560 rollouts ~= 14.6 h at
eval_env_num 8 (Best-of-N scoring is compute-bound, so eval_env_num does not
speed it up), which does not fit the 11:59 wall. Splitting the 80 eval tasks into
two disjoint balanced 40-task halves (32 train + 8 held-out each, frozen in
scripts/_reeval/molmo64_eo_eval_half_{A,B}.txt) gives two ~7.3 h jobs that each
fit, and merge_molmo64_eo_split_eval.py recombines them into one 80-task record
identical to a single-shot eval.

These 3 EO-64 runs had their in-loop eval DISABLED (eval_interval 100000000) so
they train straight to 500k banking round-5 (~250000) and round-10 (~500000)
checkpoints; the evals are these separate jobs. Nothing here touches the source
run: it stages a checkpoint tree exactly like build_reeval_jobs.py --
    <subrun>/speval_<tag>/<config>/<exp>/
        <S>                     -> symlink to the policy checkpoint at step S
        rl_state/<S>            -> symlink to rl_state/<S>
        runtime_state/resume_state.json   {step: S, RNG copied from the run}
        runtime_state/replay_shards/      EMPTY
        wandb_id.txt            copied
`eval.py --resume` takes its step solely from resume_state.json. Both halves
resume from the SAME staged checkpoint and differ only in --collect.eval_tasks
and EVAL_METRIC_PREFIX, so their sidecars never collide.
"""
import argparse, json, re, shutil, sys
from pathlib import Path

RES = Path("/capstor/store/cscs/swissai/a0220/ralfroemer/results")
HERE = Path(__file__).resolve().parent
# Frozen balanced halves (32 train + 8 held-out each); a clean disjoint partition
# of the 80 eval tasks. Merge recombines them regardless of the split, but the
# balance keeps both jobs the same length so neither risks the wall.
HALVES = {
    "A": HERE.joinpath("molmo64_eo_eval_half_A.txt").read_text().split(),
    "B": HERE.joinpath("molmo64_eo_eval_half_B.txt").read_text().split(),
}
# Round-boundary targets. The runs save at collect boundaries; a boundary+1 rewind
# (e.g. 250001) is tolerated via TOL. mid == round 5, final == round 10.
TARGETS = {"mid": 250000, "final": 500000}
TOL = 5  # steps: nearest policy&critic to the boundary must be within this


def scope():
    for s in (0, 1, 2):
        yield (s, f"vla-post-training_multitask_tasks64_bofn_plus_fsft_eo_molmo_v0_seed{s}", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_env_num", type=int, default=8)
    ap.add_argument("--num_eval_rollouts", type=int, default=32)
    ap.add_argument("--only", default=None, help="substring filter on group/tag/half")
    ap.add_argument("--apply", action="store_true", help="actually create dirs + scripts")
    args = ap.parse_args()

    # sanity on the frozen halves (fail loudly rather than emit a wrong eval)
    A, B = HALVES["A"], HALVES["B"]
    if not (len(A) == 40 and len(B) == 40 and set(A).isdisjoint(B)):
        raise SystemExit(f"bad halves: |A|={len(A)} |B|={len(B)} disjoint={set(A).isdisjoint(B)}")

    planned, problems = [], []
    for seed, grp, idx in scope():
        sub = RES / grp / str(idx)
        exps = sorted(sub.glob("*/vla-post-training_*"))
        if not exps:
            problems.append((f"{grp}/{idx}", "no exp dir (training not started / capstor down?)")); continue
        exp = exps[0]
        cfg_name, exp_name = exp.parent.name, exp.name
        try:
            pol = sorted(int(p.name) for p in exp.iterdir() if p.name.isdigit())
            rls = sorted(int(p.name) for p in (exp / "rl_state").iterdir() if p.name.isdigit())
            orig_manifest = json.loads((exp / "runtime_state" / "resume_state.json").read_text())
        except FileNotFoundError as e:
            problems.append((f"{grp}/{idx}", f"checkpoint tree unreadable: {e}")); continue
        cands = sorted(q for q in sub.glob("job_*.sbatch.sh") if "eval" not in q.name)
        if not cands:
            problems.append((f"{grp}/{idx}", "no training job_*.sbatch.sh to template from")); continue
        srun_line = next(l for l in cands[0].read_text().splitlines() if l.startswith("srun "))

        # steps present in BOTH policy and rl_state -- an eval needs the pair
        both = sorted(set(pol) & set(rls))
        for tag, boundary in TARGETS.items():
            if not both:
                problems.append((f"{grp}/{idx}", f"{tag}: no paired policy+critic steps")); continue
            S = min(both, key=lambda x: abs(x - boundary))
            if abs(S - boundary) > TOL:
                problems.append((f"{grp}/{idx}", f"{tag}: nearest paired step {S} is >{TOL} from {boundary} "
                                                 f"(training not there yet; have {both[-3:]})")); continue
            stage_base = sub / f"speval_{tag}"
            ckpt_dir = stage_base / cfg_name / exp_name

            if args.apply:
                # --- staging tree (once per (seed, tag); shared by both halves) ---
                (ckpt_dir / "rl_state").mkdir(parents=True, exist_ok=True)
                (ckpt_dir / "runtime_state" / "replay_shards").mkdir(parents=True, exist_ok=True)
                for link, target in ((ckpt_dir / str(S), exp / str(S)),
                                     (ckpt_dir / "rl_state" / str(S), exp / "rl_state" / str(S))):
                    if link.is_symlink() or link.exists():
                        link.unlink()
                    link.symlink_to(target)
                manifest = dict(orig_manifest)
                manifest["step"] = S
                manifest["replay_shard_dir"] = str(ckpt_dir / "runtime_state" / "replay_shards")
                (ckpt_dir / "runtime_state" / "resume_state.json").write_text(json.dumps(manifest, indent=2))
                shutil.copy2(exp / "wandb_id.txt", ckpt_dir / "wandb_id.txt")

            for half, tasks in HALVES.items():
                selector = f"{grp}/{tag}/{half}"
                if args.only and args.only not in selector:
                    continue
                prefix = f"m64eo_{tag}_{half}"       # distinct sidecar per half+tag
                planned.append(dict(seed=seed, grp=grp, tag=tag, half=half, S=S,
                                    prefix=prefix, ntasks=len(tasks)))
                if not args.apply:
                    continue

                # --- job script (one per half) --------------------------------
                line = srun_line
                line = line.replace("scripts/exp.py", "scripts/eval.py")
                if "env XLA_PYTHON_CLIENT_MEM_FRACTION" in line:
                    line = line.replace("env XLA_PYTHON_CLIENT_MEM_FRACTION",
                                        f"env EVAL_METRIC_PREFIX={prefix} XLA_PYTHON_CLIENT_MEM_FRACTION")
                else:
                    line = line.replace("uv run ", f"env EVAL_METRIC_PREFIX={prefix} uv run ", 1)
                line = re.sub(r"--checkpoint_base_dir \S+", f"--checkpoint_base_dir {stage_base}", line)
                # eval_tasks in the training script is the full 80; replace with this half.
                line = re.sub(r"--collect\.eval_tasks(?: molmo_\d+)+",
                              "--collect.eval_tasks " + " ".join(tasks), line)
                line = re.sub(r"--collect\.eval_env_num \d+", f"--collect.eval_env_num {args.eval_env_num}", line)
                line = re.sub(r"--collect\.num_eval_rollouts \d+", "", line)
                line = re.sub(r"--collect\.eval_interval \d+", f"--collect.eval_interval {S}", line)
                line = line.replace(" --free_buffer_before_eval", "")
                line = line.replace(" --requeue_before_eval", "")
                line = line.replace(" || child_status=$?", "")
                line += f" --collect.num_eval_rollouts {args.num_eval_rollouts}"

                # The srun line also carries --collect.tasks (all 64 TRAINING tasks),
                # which overlaps both halves -- so the task checks must look only at
                # the --collect.eval_tasks segment, never the whole line.
                eval_seg = line.split("--collect.eval_tasks", 1)[1].split("--", 1)[0] if "--collect.eval_tasks" in line else ""
                eval_seg_tasks = set(re.findall(r"molmo_\d+", eval_seg))
                checks = {
                    f"EVAL_METRIC_PREFIX={prefix}": f"EVAL_METRIC_PREFIX={prefix}" in line,
                    "eval.py": "scripts/eval.py" in line and "scripts/exp.py" not in line,
                    "staging base": f"--checkpoint_base_dir {stage_base}" in line,
                    "this half's tasks": eval_seg_tasks == set(tasks),
                    "no other half's tasks": eval_seg_tasks.isdisjoint(HALVES["B" if half == "A" else "A"]),
                    "no requeue_before_eval": "--requeue_before_eval" not in line,
                    "no free_buffer": "--free_buffer_before_eval" not in line,
                    "resume": "--resume" in line,
                    "single num_eval_rollouts": line.count("--collect.num_eval_rollouts") == 1,
                    f"eval_interval == {S}": f"--collect.eval_interval {S}" in line,
                    "40 eval tasks": len(eval_seg_tasks) == 40,
                }
                missing = [k for k, ok in checks.items() if not ok]
                if missing:
                    raise SystemExit(f"REFUSING to write {selector} [S={S}]: failed checks {missing}")
                job = sub / f"job_speval_{tag}_{half}.sbatch.sh"
                job.write_text(
                    "#!/bin/bash\nset -euo pipefail\n\n"
                    "cd /users/ralfroemer/vla-post-training\n\n"
                    "ulimit -c 0\n\n"
                    f"# Molmo-64 EO split eval, {tag} checkpoint (step {S}), half {half} "
                    f"({len(tasks)} of 80 tasks).\n"
                    f"# Sidecar: eval_metrics_{prefix}_step{S}.json. Merge with "
                    f"merge_molmo64_eo_split_eval.py.\n"
                    f"{line}\n")
                job.chmod(0o755)

    for p in planned:
        print(f"seed{p['seed']} {p['tag']:5s} half {p['half']} S={p['S']:<7d} "
              f"tasks={p['ntasks']} prefix={p['prefix']}")
    print(f"\nplanned jobs: {len(planned)}   problems: {len(problems)}")
    for g, why in problems:
        print(f"  PROBLEM {g.replace('vla-post-training_multitask_','')}: {why}")
    print("\nSubmit each generated job with:")
    print("  sbatch --account=ab037 --time=11:59:00 --partition=normal --requeue \\")
    print("    --open-mode=append --output=<subrun>/slurm-%j.out <subrun>/job_speval_<tag>_<half>.sbatch.sh")
    if not args.apply:
        print("\n(dry run -- re-run with --apply to stage trees and write job scripts)")


if __name__ == "__main__":
    main()
