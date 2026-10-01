"""Render one initial-observation frame per Molmo task in the 64+16 split.

Produces the pictures behind the task-split figure: for each of the 80 eval
tasks, the exterior-camera RGB the policy sees at reset. Uses the same
MolmoSpacesBenchmarkGymEnv the collect loop uses, so a frame here is exactly
what a rollout starts from (bar the +/-1cm init randomisation, which is left ON
so the frame is a representative draw rather than a canonical pose).

Run on a compute node (needs a GPU for the renderer):
    sbatch scripts/_reeval/render_molmo_task_frames.sbatch.sh
"""
import argparse, json, os, re, sys, time
from pathlib import Path

import numpy as np

REPO = Path("/users/ralfroemer/vla-post-training")
CFG = REPO / "scripts/configs/multitask/bofn_plus_fsft_eo/molmo/bofn_plus_fsft_molmo_tasks64_seed0_v0.yaml"


def eval_task_ids():
    txt = CFG.read_text()
    region = txt[txt.index("collect.eval_tasks]"):txt.index("collect.store_prefix_rep")]
    lists = re.findall(r"\[([^\[\]]*molmo_[^\[\]]*)\]", region, re.S)
    train = [int(x) for x in re.findall(r"molmo_(\d+)", lists[0])]
    ev = [int(x) for x in re.findall(r"molmo_(\d+)", lists[1])]
    assert len(train) == 64 and len(ev) == 80, (len(train), len(ev))
    return train, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/capstor/scratch/cscs/ralfroemer/molmo_task_frames")
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

    train, ev = eval_task_ids()
    todo = ev[: args.limit] if args.limit else ev
    print(f"rendering {len(todo)} tasks -> {out}", flush=True)

    env = MolmoSpacesBenchmarkGymEnv(episode_id=todo[0], render_device=0,
                                     config=MolmoSpacesGymConfig())
    meta = {}
    t0 = time.time()
    for n, tid in enumerate(todo):
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
                                "in_train": tid in set(train)}
        print(f"  [{n+1}/{len(todo)}] molmo_{tid}  {info.get('task_description')}  "
              f"({time.time()-t0:.0f}s)", flush=True)
    (out / "frames_meta.json").write_text(json.dumps(meta, indent=1))
    env.close()
    print(f"done: {len(meta)} tasks in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
