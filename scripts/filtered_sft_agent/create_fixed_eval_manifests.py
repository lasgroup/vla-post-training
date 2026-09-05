#!/usr/bin/env python3
"""Create fixed LIBERO evaluation manifests from measured state-pool sizes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

_TASK_RE = re.compile(r"libero_90_(0|[1-9][0-9]?)")


def _seed(base_seed: int, repeat_index: int, state_index: int) -> int:
    material = (
        f"libero90-fixed-eval-v1\0base_seed={base_seed}\0"
        f"repeat={repeat_index}\0state={state_index}"
    ).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], byteorder="big")


def _payload(rows: list[dict]) -> bytes:
    return "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    ).encode()


def _atomic_write_idempotent(path: Path, payload: bytes) -> str:
    sha256 = hashlib.sha256(payload).hexdigest()
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace non-matching artifact: {path}")
        return sha256
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        temp.write_bytes(payload)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return sha256


def _task_index(task: str) -> int:
    match = _TASK_RE.fullmatch(task)
    if match is None:
        raise ValueError(f"invalid LIBERO-90 task name: {task!r}")
    task_index = int(match.group(1))
    if not 0 <= task_index < 90:
        raise ValueError(f"LIBERO-90 task index out of range: {task_index}")
    return task_index


def _manifest_rows(
    *, task: str, state_count: int, rollouts: int, base_seed: int
) -> list[dict]:
    if (
        not isinstance(state_count, int)
        or isinstance(state_count, bool)
        or state_count <= 0
    ):
        raise ValueError(f"invalid state count for {task}: {state_count!r}")
    if rollouts <= 0 or rollouts % state_count != 0:
        raise ValueError(
            f"rollouts={rollouts} must be positive and divisible by "
            f"{task} state_count={state_count}"
        )
    rows = []
    for repeat_index in range(rollouts // state_count):
        for state_index in range(state_count):
            rows.append(
                {
                    "episode_id": f"{task}_state{state_index:03d}_repeat{repeat_index:02d}",
                    "manifest_index": len(rows),
                    "task": task,
                    "initial_state_index": state_index,
                    "repeat_index": repeat_index,
                    "policy_seed": _seed(base_seed, repeat_index, state_index),
                }
            )
    return rows


def build_manifests(
    *, state_counts: Path, output_dir: Path, base_seed: int, rollouts: int
) -> dict:
    counts = json.loads(state_counts.read_text(encoding="utf-8"))
    if not isinstance(counts, dict) or not counts:
        raise TypeError(
            "state-counts must be a non-empty JSON object keyed by task name"
        )

    campaign_rows = []
    for task in sorted(counts, key=_task_index):
        state_count = counts[task]
        rows = _manifest_rows(
            task=task,
            state_count=state_count,
            rollouts=rollouts,
            base_seed=base_seed,
        )
        task_index = _task_index(task)
        path = output_dir / f"task{task_index}_fixed_{rollouts}.jsonl"
        manifest_sha256 = _atomic_write_idempotent(path, _payload(rows))
        campaign_rows.append(
            {
                "task": task,
                "state_count": state_count,
                "repeat_count": rollouts // state_count,
                "rollout_count": len(rows),
                "manifest": str(path.resolve()),
                "manifest_sha256": manifest_sha256,
            }
        )

    campaign = {
        "schema_version": 1,
        "evaluation_version": "libero90-fixed-eval-v1",
        "base_seed": base_seed,
        "rollouts_per_task": rollouts,
        "state_counts_source": str(state_counts.resolve()),
        "tasks": campaign_rows,
    }
    campaign_payload = (json.dumps(campaign, indent=2, sort_keys=True) + "\n").encode()
    _atomic_write_idempotent(output_dir / "evaluation_manifests.json", campaign_payload)
    return campaign


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-counts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--rollouts", type=int, default=100)
    args = parser.parse_args()

    campaign = build_manifests(
        state_counts=args.state_counts,
        output_dir=args.output_dir,
        base_seed=args.base_seed,
        rollouts=args.rollouts,
    )
    print(json.dumps(campaign, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
