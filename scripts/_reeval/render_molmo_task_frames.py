"""Render one initial-observation frame per Molmo task in the 144-task split.

Produces the pictures behind the task-split figure: for each of the 162 eval
tasks of the 144-task runs (144 train + 12 val_in + 6 val_out), the
exterior-camera RGB the policy sees at reset. These include all 80 eval tasks
of the 64+16 split. Uses the same
MolmoSpacesBenchmarkGymEnv the collect loop uses, so a frame here is exactly
what a rollout starts from (bar the +/-1cm init randomisation, which is left ON
so the frame is a representative draw rather than a canonical pose).

Run on a compute node (needs a GPU for the renderer):
    sbatch scripts/_reeval/render_molmo_task_frames.sbatch.sh
"""
import argparse, json, os, re, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CFG = REPO / "scripts/configs/multitask/bofn/molmo/bofn_molmo_tasks144_seed0_v0.yaml"


def eval_tasks():
    """-> [(task_id, group)] for the 162 eval tasks; group is train | val_in | val_out.
    Same parsing as build_molmo144_split_eval.eval_groups()."""
    eval_block = CFG.read_text().split("# training split (144)")[2]
    train_txt, rest = eval_block.split("# val_in")
    val_in_txt, val_out_txt = rest.split("# val_out")
    groups = {g: [int(x) for x in re.findall(r"molmo_(\d+)", t)]
              for g, t in (("train", train_txt), ("val_in", val_in_txt), ("val_out", val_out_txt))}
    assert [len(v) for v in groups.values()] == [144, 12, 6], {g: len(v) for g, v in groups.items()}
    return [(t, g) for g, ts in groups.items() for t in ts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.environ.get("SCRATCH", "."), "molmo_task_frames_tasks144"))
    # Native frames are 624x352 (1.77:1). An earlier version resized to a fixed
    # SQUARE, which compressed every frame horizontally by ~1.8x. Default is now
    # native resolution; --max-width only ever scales proportionally.
    ap.add_argument("--max-width", type=int, default=0,
                    help="downscale to this width, preserving aspect (0 = native)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    from PIL import Image
    from src.envs.molmo import MolmoSpacesBenchmarkGymEnv, MolmoSpacesGymConfig

    tasks = eval_tasks()
    todo = tasks[: args.limit] if args.limit else tasks
    print(f"rendering {len(todo)} tasks -> {out}", flush=True)

    env = MolmoSpacesBenchmarkGymEnv(episode_id=todo[0][0], render_device=0,
                                     config=MolmoSpacesGymConfig())
    meta = {}
    t0 = time.time()
    for n, (tid, group) in enumerate(todo):
        try:
            obs, info = env.reset(options={"task_id": f"molmo_{tid}"})
        except Exception as e:                      # one bad episode must not kill the batch
            print(f"  molmo_{tid}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        for cam in ("exo_camera_1", "wrist_camera"):
            img = obs.get(cam)
            if img is None:
                continue
            a = np.asarray(img)
            if a.dtype != np.uint8:
                a = (np.clip(a, 0, 1) * 255).astype(np.uint8) if a.max() <= 1.0 else a.astype(np.uint8)
            im = Image.fromarray(a[..., :3])
            if n == 0 and cam == "exo_camera_1":
                print(f"  native frame size: {im.size} (w,h)", flush=True)
            if args.max_width and im.width > args.max_width:
                h = round(im.height * args.max_width / im.width)
                im = im.resize((args.max_width, h), Image.LANCZOS)
            tag = "exo" if cam.startswith("exo") else "wrist"
            im.save(out / f"molmo_{tid}_{tag}.jpg", quality=82, optimize=True)
        meta[f"molmo_{tid}"] = {"task_description": info.get("task_description"),
                                "group": group, "in_train": group == "train"}
        # rewritten every task so a run killed at the time limit keeps its metadata
        (out / "frames_meta.json").write_text(json.dumps(meta, indent=1))
        print(f"  [{n+1}/{len(todo)}] molmo_{tid} ({group})  {info.get('task_description')}  "
              f"({time.time()-t0:.0f}s)", flush=True)
    (out / "frames_meta.json").write_text(json.dumps(meta, indent=1))
    env.close()
    print(f"done: {len(meta)} tasks in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
