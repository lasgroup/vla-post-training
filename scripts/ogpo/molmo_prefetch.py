"""Install MolmoSpaces base assets (on import) + the scenes / objects / grasps of the given benchmark episodes.

Run in .venv-molmo with MLSPACES_ASSETS_DIR / MLSPACES_CACHE_DIR / NLTK_DATA set (scripts/ogpo/download_molmo_assets.sbatch).
"""
import argparse
import os
import time
from pathlib import Path

BENCH_REL = ("benchmarks/molmospaces-bench-v1/procthor-10k/FrankaPickDroidMiniBench/"
             "FrankaPickDroidMiniBench_json_benchmark_20251231")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True, help="comma list of benchmark episode ids, e.g. 164,86")
    args = p.parse_args()
    ids = [int(t) for t in args.tasks.split(",")]

    t = time.time()
    # Importing the env path builds the resource manager (eager install of robots / benchmarks / objects / grasps)
    # and runs nltk.download (texture.py:10, synset_utils.py:6-13).
    from molmo_spaces.evaluation.benchmark_schema import load_all_episodes
    from molmo_spaces.molmo_spaces_constants import ASSETS_DIR, get_scenes
    from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler  # noqa: F401
    from molmo_spaces.utils.lazy_loading_utils import install_scene_with_objects_and_grasps_from_path, install_uid
    print(f"[prefetch] base install + imports: {time.time() - t:.0f}s | ASSETS_DIR={ASSETS_DIR}", flush=True)

    bench = Path(ASSETS_DIR) / BENCH_REL
    if not bench.is_dir():
        raise FileNotFoundError(f"benchmark dir missing after base install: {bench}")
    episodes = load_all_episodes(bench)
    print(f"[prefetch] {len(episodes)} episodes in {bench}", flush=True)

    for i in ids:
        ep = episodes[i]
        t = time.time()
        variants = get_scenes(ep.scene_dataset, ep.data_split)[ep.data_split][ep.house_index]
        paths = [variants] if isinstance(variants, (str, Path)) else [v for v in variants.values() if v]
        for path in paths:
            install_scene_with_objects_and_grasps_from_path(path)
        for rel in ep.scene_modifications.added_objects.values():
            if not (Path(ASSETS_DIR) / rel).is_file():
                install_uid(Path(rel).stem)
        print(f"[prefetch] molmo_{i}: house {ep.house_index} ({ep.scene_dataset}/{ep.data_split}), "
              f"{len(paths)} scene variant(s), {len(ep.scene_modifications.added_objects)} added object(s), "
              f"{time.time() - t:.0f}s | {ep.language.task_description!r}", flush=True)
    print("[prefetch] DONE", flush=True)


if __name__ == "__main__":
    main()
