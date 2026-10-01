#!/usr/bin/env python3
"""Stage + generate the "new task split" re-eval jobs.

For each (subrun, target step S) it builds a staging checkpoint tree that points
at the run's real policy and critic without touching the run itself:

    <subrun>/reeval_<tag>/<config_name>/<exp_name>/
        <S>                        -> symlink to the nearest policy checkpoint
        rl_state/<S>               -> symlink to rl_state/<S>
        runtime_state/resume_state.json   {step: S, ...RNG copied from the run}
        runtime_state/replay_shards/      EMPTY (skips a 12-25 GB pointless load)
        wandb_id.txt               copy, so results land in the ORIGINAL wandb run

`eval.py --resume` takes its step solely from resume_state.json, so this is how
a specific checkpoint gets selected. The policy may sit 1-2 steps from the
critic: policy checkpoints exist only at round boundaries (100000/500000) while
the terminal critic is at the eval step (99999/99998/499998/499999). For BoN
that pairing is exact (the policy is never trained); for expert-only it is at
most one actor update out of 5000.
"""
import argparse, json, os, re, shutil, sys
from pathlib import Path

RES = Path("/capstor/store/cscs/swissai/a0220/ralfroemer/results")
NEW_TASKS = ["molmo_666", "molmo_602", "molmo_543", "molmo_273",
             "molmo_916", "molmo_505", "molmo_769", "molmo_675"]
PREFIX = "eval_newtasks"


def scope():
    """(family, split, group, index, {tag: target_step}) for every re-eval."""
    out = []
    for s in (0, 1, 2):
        out.append(("BoN", "1", f"vla-post-training_multitask_tasks1_bofn_molmo_v0_seed{s}", 0, {"final": 99999}))
        for i in (0, 1, 2):
            out.append(("BoN", "1", f"vla-post-training_multitask_tasks1_bofn_molmo_v0_seed{s}_rest", i, {"final": 99999}))
        out.append(("BoN", "4", f"vla-post-training_multitask_tasks4_bofn_molmo_v0_seed{s}", 0, {"final": 499998}))
        out.append(("BoN", "16", f"vla-post-training_multitask_tasks16_bofn_molmo_v0_seed{s}", 0, {"final": 499998}))
    # BoN molmo 8 == "8 inst. x cat. 1" == index 0; seed 0 lives in the _50k dir
    # (see scripts/results_registry.yaml molmo: 8).
    for grp in ("vla-post-training_multitask_tasks8_bofn_molmo_v0_seed0_50k",
                "vla-post-training_multitask_tasks8_bofn_molmo_v0_seed1",
                "vla-post-training_multitask_tasks8_bofn_molmo_v0_seed2"):
        out.append(("BoN", "8", grp, 0, {"final": 499999}))
    for s in (0, 1, 2):
        eo1 = {"mid": 50000, "final": 99998}
        out.append(("EO", "1", f"vla-post-training_multitask_tasks1_bofn_plus_fsft_eo_molmo_v0_seed{s}", 0, eo1))
        for i in (0, 1, 2):
            out.append(("EO", "1", f"vla-post-training_multitask_tasks1_bofn_plus_fsft_eo_molmo_v0_seed{s}_rest", i, dict(eo1)))
        eom = {"mid": 250000, "final": 499998}
        for n in ("4", "8", "16"):
            out.append(("EO", n, f"vla-post-training_multitask_tasks{n}_bofn_plus_fsft_eo_molmo_v0_seed{s}", 0, dict(eom)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_env_num", type=int, default=8)
    ap.add_argument("--num_eval_rollouts", type=int, default=32)
    ap.add_argument("--only", default=None, help="substring filter on <group>/<index>")
    ap.add_argument("--apply", action="store_true", help="actually create dirs + scripts")
    args = ap.parse_args()

    planned, problems = [], []
    for fam, split, grp, idx, targets in scope():
        if args.only and args.only not in f"{grp}/{idx}":
            continue
        sub = RES / grp / str(idx)
        exps = sorted(sub.glob("*/vla-post-training_*"))
        if not exps:
            problems.append((f"{grp}/{idx}", "no exp dir")); continue
        exp = exps[0]
        cfg_name, exp_name = exp.parent.name, exp.name
        pol = sorted(int(p.name) for p in exp.iterdir() if p.name.isdigit())
        rls = sorted(int(p.name) for p in (exp / "rl_state").iterdir() if p.name.isdigit())
        orig_manifest = json.loads((exp / "runtime_state" / "resume_state.json").read_text())
        # The launcher names it job_<positional index>.sbatch.sh, so it is not
        # always job_0000 (the `_rest` subruns use job_0001 / job_0002).
        cands = sorted(q for q in sub.glob("job_*.sbatch.sh") if "reeval" not in q.name)
        if not cands:
            problems.append((f"{grp}/{idx}", "no job_*.sbatch.sh to template from")); continue
        srun_line = next(l for l in cands[0].read_text().splitlines() if l.startswith("srun "))

        for tag, S in sorted(targets.items()):
            if S not in rls:
                problems.append((f"{grp}/{idx}", f"{tag}: no critic at {S} (have {rls[-3:]})")); continue
            if not pol:
                problems.append((f"{grp}/{idx}", f"{tag}: no policy checkpoints at all")); continue
            src_pol = min(pol, key=lambda p: abs(p - S))
            gap = abs(src_pol - S)
            if gap > 2:
                problems.append((f"{grp}/{idx}", f"{tag}: nearest policy {src_pol} is {gap} steps from critic {S}")); continue

            stage_base = sub / f"reeval_{tag}"
            ckpt_dir = stage_base / cfg_name / exp_name
            planned.append(dict(fam=fam, split=split, grp=grp, idx=idx, tag=tag, S=S,
                                src_pol=src_pol, gap=gap, exp=str(exp),
                                stage_base=str(stage_base), ckpt_dir=str(ckpt_dir)))
            if not args.apply:
                continue

            # --- staging tree -------------------------------------------------
            (ckpt_dir / "rl_state").mkdir(parents=True, exist_ok=True)
            (ckpt_dir / "runtime_state" / "replay_shards").mkdir(parents=True, exist_ok=True)
            for link, target in ((ckpt_dir / str(S), exp / str(src_pol)),
                                 (ckpt_dir / "rl_state" / str(S), exp / "rl_state" / str(S))):
                if link.is_symlink() or link.exists():
                    link.unlink()
                link.symlink_to(target)
            manifest = dict(orig_manifest)
            manifest["step"] = S
            manifest["replay_shard_dir"] = str(ckpt_dir / "runtime_state" / "replay_shards")
            (ckpt_dir / "runtime_state" / "resume_state.json").write_text(json.dumps(manifest, indent=2))
            shutil.copy2(exp / "wandb_id.txt", ckpt_dir / "wandb_id.txt")
            (ckpt_dir / "REEVAL_README.txt").write_text(
                f"Staging tree for the new-task re-eval (metric prefix {PREFIX}/).\n"
                f"Source run : {exp}\n"
                f"Target step: {S}   critic rl_state/{S}   policy <- {src_pol} (gap {gap})\n"
                f"Tasks      : {' '.join(NEW_TASKS)}\n"
                f"replay_shards is intentionally EMPTY; eval never samples the buffer.\n"
                f"Nothing here is written back into the source run.\n")

            # --- job script ---------------------------------------------------
            line = srun_line
            line = line.replace("scripts/exp.py", "scripts/eval.py")
            # Subruns submitted before the launcher gained --mem_fraction have no
            # `env ...` segment at all, so keying the injection off it silently
            # did nothing and the eval would have logged under `eval/`, clobbering
            # the run's original final eval. Handle both shapes, then assert.
            if "env XLA_PYTHON_CLIENT_MEM_FRACTION" in line:
                line = line.replace("env XLA_PYTHON_CLIENT_MEM_FRACTION",
                                    f"env EVAL_METRIC_PREFIX={PREFIX} XLA_PYTHON_CLIENT_MEM_FRACTION")
            else:
                line = line.replace("uv run ", f"env EVAL_METRIC_PREFIX={PREFIX} uv run ", 1)
            line = re.sub(r"--checkpoint_base_dir \S+", f"--checkpoint_base_dir {stage_base}", line)
            line = re.sub(r"--collect\.eval_tasks(?: molmo_\d+)+",
                          "--collect.eval_tasks " + " ".join(NEW_TASKS), line)
            line = re.sub(r"--collect\.eval_env_num \d+", f"--collect.eval_env_num {args.eval_env_num}", line)
            line = re.sub(r"--collect\.num_eval_rollouts \d+", "", line)
            line = line.replace(" --requeue_before_eval", "")
            line = re.sub(r"--collect\.eval_interval \d+", f"--collect.eval_interval {S}", line)
            line = line.replace(" || child_status=$?", "")
            line += f" --collect.num_eval_rollouts {args.num_eval_rollouts}"

            # Never emit a job that could write into the `eval/` namespace or
            # load the wrong tree: a silent miss here corrupts recorded results.
            checks = {
                f"EVAL_METRIC_PREFIX={PREFIX}": f"EVAL_METRIC_PREFIX={PREFIX}" in line,
                "eval.py": "scripts/eval.py" in line and "scripts/exp.py" not in line,
                "staging base": f"--checkpoint_base_dir {stage_base}" in line,
                "new eval tasks": all(t in line for t in NEW_TASKS),
                "no requeue_before_eval": "--requeue_before_eval" not in line,
                "resume": "--resume" in line,
                "single num_eval_rollouts": line.count("--collect.num_eval_rollouts") == 1,
                f"eval_interval == {S}": f"--collect.eval_interval {S}" in line,
            }
            missing = [k for k, ok in checks.items() if not ok]
            if missing:
                raise SystemExit(f"REFUSING to write {grp}/{idx} [{tag}]: failed checks {missing}")
            job = sub / f"job_reeval_{tag}.sbatch.sh"
            job.write_text(
                "#!/bin/bash\nset -euo pipefail\n\n"
                "cd /users/ralfroemer/vla-post-training\n\n"
                "ulimit -c 0\n\n"
                f"# New-task re-eval, {tag} checkpoint (step {S}); policy from {src_pol} (gap {gap}).\n"
                f"# Writes {PREFIX}/* into the ORIGINAL wandb run; source run dir is untouched.\n"
                f"{line}\n")
            job.chmod(0o755)

    for p in planned:
        print(f"{p['fam']:3s} {p['split']:>3s} {p['tag']:5s} S={p['S']:<7d} pol<-{p['src_pol']:<7d} gap={p['gap']} "
              f"{p['grp'].replace('vla-post-training_multitask_','')}/{p['idx']}")
    print(f"\nplanned: {len(planned)}   problems: {len(problems)}")
    for g, why in problems:
        print(f"  PROBLEM {g.replace('vla-post-training_multitask_','')}: {why}")
    if args.apply:
        print("\n--apply: staging trees and job scripts written.")


if __name__ == "__main__":
    main()
