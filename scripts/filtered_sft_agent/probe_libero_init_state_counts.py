#!/usr/bin/env python3
"""Measure LIBERO initial-state pool sizes for fixed evaluation manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from libero.libero import benchmark

from src.envs.libero import get_task_init_states


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--descriptions-output", type=Path)
    args = parser.parse_args()

    if not args.tasks or any(task_id < 0 or task_id >= 90 for task_id in args.tasks):
        raise ValueError("--tasks must contain LIBERO-90 indices in [0, 89]")
    if len(set(args.tasks)) != len(args.tasks):
        raise ValueError("--tasks must not contain duplicates")

    suite = benchmark.get_benchmark_dict()["libero_90"]()
    counts = {}
    descriptions = {}
    details = {}
    for task_id in args.tasks:
        task = f"libero_90_{task_id}"
        states = get_task_init_states(suite, task_id)
        task_description = str(suite.get_task(task_id).language)
        counts[task] = int(states.shape[0])
        descriptions[task] = task_description
        details[task] = {
            "state_count": int(states.shape[0]),
            "state_shape": [int(value) for value in states.shape[1:]],
            "task_description": task_description,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".partial")
    temp.write_text(
        json.dumps(counts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temp.replace(args.output)
    if args.descriptions_output is not None:
        args.descriptions_output.parent.mkdir(parents=True, exist_ok=True)
        descriptions_temp = args.descriptions_output.with_suffix(
            args.descriptions_output.suffix + ".partial"
        )
        descriptions_temp.write_text(
            json.dumps(descriptions, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        descriptions_temp.replace(args.descriptions_output)
    print(
        json.dumps(
            {"counts": counts, "descriptions": descriptions, "details": details},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
