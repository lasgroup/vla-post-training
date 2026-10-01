"""Render one initial-observation frame per LIBERO-90 task in the 64 + 25 split.

Uses the same make_env_libero path as collection, so each frame is what a rollout
starts from: reset to one of the task's recorded init states, then the
WarmUpOnResetWrapper's settle steps. The agent-view image is flipped [::-1, ::-1]
exactly as src/envs/wrappers.py does before it reaches the policy, i.e. upright.
Frames are saved at native resolution (collect.env_resolution, square).

    sbatch scripts/_reeval/render_libero_task_frames.sbatch.sh
"""
import json, re, time
from pathlib import Path
import numpy as np

REPO = Path("/users/ralfroemer/vla-post-training")
CFG = REPO / "scripts/configs/multitask/bofn/libero/bofn_libero_tasks64_seed0_v0.yaml"
OUT = Path("/capstor/scratch/cscs/ralfroemer/libero_task_frames")


def split():
    t = CFG.read_text()
    reg = t[t.index("collect.eval_tasks]"):]
    ls = re.findall(r"\[([^\[\]]*libero_90_[^\[\]]*)\]", reg, re.S)
    tr = [int(x) for x in re.findall(r"libero_90_(\d+)", ls[0])]
    ev = [int(x) for x in re.findall(r"libero_90_(\d+)", ls[1])]
    return tr, [x for x in ev if x not in set(tr)]


def main():
    from PIL import Image
    from libero.libero import benchmark
    from src.training import config as _config
    from src.envs.libero import make_env_libero

    config_name = re.search(r"^config_name:\s*(\S+)", CFG.read_text(), re.M).group(1)
    config = _config.get_config(config_name)
    suite = benchmark.get_benchmark_dict()["libero_90"]()
    tr, un = split()
    todo = tr + un
    OUT.mkdir(parents=True, exist_ok=True)
    env = make_env_libero(config, [f"libero_90_{todo[0]}"], num_devices=1)(0)
    meta, t0 = {}, time.time()
    for n, tid in enumerate(todo):
        try:
            obs, info = env.reset(options={"task_id": f"libero_90_{tid}"})
        except Exception as e:
            print(f"  libero_90_{tid}: FAILED {type(e).__name__}: {e}", flush=True); continue
        img = np.ascontiguousarray(np.asarray(obs["agentview_image"])[::-1, ::-1])
        if n == 0:
            print(f"  native frame size: {img.shape}", flush=True)
        Image.fromarray(img[..., :3].astype(np.uint8)).save(OUT / f"libero_90_{tid}.jpg", quality=90)
        task = suite.get_task(tid)
        meta[tid] = {"name": task.name, "language": task.language,
                     "split": "train" if tid in set(tr) else "unseen"}
        print(f"  [{n+1}/{len(todo)}] libero_90_{tid}  {task.name}  ({time.time()-t0:.0f}s)", flush=True)
    (OUT / "frames_meta.json").write_text(json.dumps(meta, indent=1))
    env.close()
    print(f"done: {len(meta)} tasks in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
