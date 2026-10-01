"""
Delete wandb runs that are provably empty crash artifacts.

Context: repeated startup crashes leave short FAILED runs in
lsy-tum/vla-post-training. They clutter the run list but carry no data. This
deletes only the ones that are safe on BOTH a local and a server-side check.

WHY THE SERVER CHECK IS MANDATORY: the candidate list is built from local
mirrors under <repo>/wandb/, and those can lie. We have already seen
`wandb sync` truncate a run's wandb-summary.json to 0 bytes, which makes a
data-bearing run look empty locally. So every candidate is re-verified against
the wandb API and skipped unless the SERVER also says it is empty.

A wid is deleted only if ALL of these hold:
  local  - not referenced by any <subrun>/.../wandb_id.txt   (a live resume target)
  local  - no eval/success_rate* keys in its summary
  local  - no collection `success_rate` keys      (these feed the ablation curves)
  local  - _step == 0 and no non-underscore summary keys
  server - run exists, state is not running
  server - summary has no eval/success_rate* and no success_rate keys
  server - lastHistoryStep <= 0

Deleting a wid is IRREVERSIBLE and breaks any future `--resume` into it
(wandb refuses to resume a deleted id: "was previously created and deleted").

Usage (needs a node with uv + wandb login, e.g. clariden-ln004):
    cd /users/ralfroemer/vla-post-training
    uv run scripts/delete_dead_wandb_runs.py            # dry run, prints plan
    uv run scripts/delete_dead_wandb_runs.py --apply    # actually delete
"""
import argparse
import glob
import json
import os
import re
import sys

import wandb

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = "/capstor/store/cscs/swissai/a0220/ralfroemer/results"
ENTITY, PROJECT = "lsy-tum", "vla-post-training"

EVAL_PREFIX = "eval/success_rate"


def live_wids():
    """wids a subrun would resume into -- never deletable."""
    out = set()
    for f in glob.glob(f"{RESULTS}/*/[0-9]*/**/wandb_id.txt", recursive=True):
        try:
            out.add(open(f).read().strip())
        except OSError:
            pass
    return out


def local_candidates(protected):
    mirrors = {}
    for d in glob.glob(f"{REPO}/wandb/run-*"):
        m = re.search(r"run-\d+_\d+-([0-9a-z]+)$", d)
        if m:
            mirrors.setdefault(m.group(1), []).append(d)

    cands = []
    for wid, dirs in mirrors.items():
        if wid in protected:
            continue
        bad = False
        maxstep = 0
        nkeys = 0
        for d in sorted(dirs):
            p = os.path.join(d, "files/wandb-summary.json")
            if not (os.path.exists(p) and os.path.getsize(p) > 0):
                continue
            try:
                s = json.load(open(p))
            except Exception:
                continue
            if any(k.startswith(EVAL_PREFIX) for k in s):
                bad = True
            if any(k == "success_rate" or k.startswith("success_rate/") for k in s):
                bad = True
            nkeys = max(nkeys, len([k for k in s if not k.startswith("_")]))
            try:
                maxstep = max(maxstep, int(s.get("_step", 0) or 0))
            except Exception:
                pass
        if bad or maxstep != 0 or nkeys != 0:
            continue
        cands.append(wid)
    return sorted(cands)


def server_is_empty(api, wid):
    """Re-verify against wandb. Returns (ok, reason)."""
    try:
        run = api.run(f"{ENTITY}/{PROJECT}/{wid}")
    except Exception as e:
        return False, f"cannot fetch ({type(e).__name__})"
    if run.state == "running":
        return False, "state=running"
    summ = {k: v for k, v in dict(run.summary).items()}
    if any(k.startswith(EVAL_PREFIX) for k in summ):
        return False, "server summary HAS eval data"
    if any(k == "success_rate" or k.startswith("success_rate/") for k in summ):
        return False, "server summary HAS collection success_rate"
    step = run.lastHistoryStep
    if step is not None and step > 0:
        return False, f"server lastHistoryStep={step}"
    return True, f"empty (state={run.state})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry run)")
    ap.add_argument("--limit", type=int, default=0, help="cap how many to process")
    args = ap.parse_args()

    protected = live_wids()
    cands = local_candidates(protected)
    if args.limit:
        cands = cands[: args.limit]
    print(f"protected (live resume targets): {len(protected)}")
    print(f"local candidates:                {len(cands)}")
    print(f"mode: {'APPLY -- will delete' if args.apply else 'DRY RUN'}\n")

    api = wandb.Api(timeout=120)
    deleted = kept = missing = 0
    for i, wid in enumerate(cands, 1):
        ok, reason = server_is_empty(api, wid)
        if not ok:
            if reason.startswith("cannot fetch"):
                missing += 1
            else:
                kept += 1
                print(f"[{i}/{len(cands)}] KEEP   {wid}: {reason}")
            continue
        if args.apply:
            try:
                api.run(f"{ENTITY}/{PROJECT}/{wid}").delete()
                deleted += 1
            except Exception as e:
                print(f"[{i}/{len(cands)}] ERROR  {wid}: {e}")
                continue
        else:
            deleted += 1
        if i % 50 == 0 or not args.apply and i <= 5:
            print(f"[{i}/{len(cands)}] {'DELETED' if args.apply else 'would delete'} {wid}: {reason}")

    print(f"\n{'deleted' if args.apply else 'would delete'}: {deleted}")
    print(f"kept (server disagreed with local): {kept}")
    print(f"not on server / already gone:       {missing}")
    if not args.apply:
        print("\nRe-run with --apply to perform the deletion.")


if __name__ == "__main__":
    sys.exit(main())
