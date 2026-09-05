#!/usr/bin/env python3
"""Evaluate a restored filtered-SFT checkpoint on a fixed rollout manifest."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
from pathlib import Path

import jax

import src.training.config as _config
from scripts.filtered_sft_agent.preloaded_sft_exp import _make_eval_env
from scripts.filtered_sft_agent.source_release import verify_source_release
from src.rl.filtered_sft_agent.filtered_sft_learner import FilteredSFTLearner
from src.training.collect import evaluate_policy
from src.training.utils import init_logging

LOGGER = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_inventory_sha256(checkpoint_dir: Path, train_state_step: int) -> str:
    if checkpoint_dir.is_symlink():
        raise ValueError("checkpoint directory must not be a symlink")
    checkpoint_root = checkpoint_dir.resolve(strict=True)
    step_path = checkpoint_root / str(train_state_step)
    if step_path.is_symlink():
        raise ValueError("checkpoint step directory must not be a symlink")
    step_root = step_path.resolve(strict=True)
    try:
        step_root.relative_to(checkpoint_root)
    except ValueError as error:
        raise ValueError("checkpoint step path escapes checkpoint directory") from error
    if not step_root.is_dir():
        raise ValueError(f"checkpoint step path is not a directory: {step_root}")

    inventory = []
    for path in sorted(step_root.rglob("*")):
        relative_path = path.relative_to(step_root).as_posix()
        if path.is_symlink():
            raise ValueError(f"checkpoint inventory forbids symlinks: {relative_path}")
        if path.is_file():
            inventory.append(
                {
                    "path": relative_path,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    if not inventory:
        raise ValueError(f"checkpoint step contains no artifact files: {step_root}")
    canonical = json.dumps(
        inventory, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _checkpoint_id(
    checkpoint_dir: Path, train_state_step: int, base_id: str
) -> tuple[str, str]:
    base_id = base_id.strip()
    if not base_id or "#" in base_id or "\n" in base_id or "\r" in base_id:
        raise ValueError(
            "VLA_EVAL_CHECKPOINT_ID must be a non-empty portable identity without '#' or newlines"
        )
    inventory_sha256 = _checkpoint_inventory_sha256(checkpoint_dir, train_state_step)
    checkpoint_id = (
        f"{base_id}#train_state_step={train_state_step}"
        f"#artifact_inventory_sha256={inventory_sha256}"
    )
    return checkpoint_id, inventory_sha256


def _json_value(value):
    value = jax.device_get(value)
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temp.replace(path)


def main(config: _config.OnlineTrainConfig) -> None:
    init_logging()
    source_receipt = verify_source_release()
    LOGGER.info("Verified immutable source release %s", source_receipt["source_sha"])
    if not config.resume:
        raise ValueError("fixed checkpoint evaluation requires --resume")
    if config.collect.eval_env_num != 1:
        raise ValueError(
            "fixed checkpoint evaluation requires --collect.eval_env_num=1"
        )

    manifest_path = Path(os.environ["VLA_FIXED_EVAL_MANIFEST"]).resolve(strict=True)
    results_path = Path(os.environ["VLA_FIXED_EVAL_RESULTS"]).resolve()
    summary_path = Path(os.environ["VLA_FIXED_EVAL_SUMMARY"]).resolve()
    expected_train_steps = int(os.environ.get("VLA_EXPECTED_TRAIN_STEPS", "4001"))

    agent = FilteredSFTLearner(config)
    if not agent._resuming:
        raise RuntimeError(
            f"checkpoint manager did not resume from {config.checkpoint_dir}"
        )
    restored_train_steps = int(agent.training_steps)
    if restored_train_steps != expected_train_steps:
        raise RuntimeError(
            f"restored train-state step {restored_train_steps}; expected {expected_train_steps}"
        )

    checkpoint_id, inventory_sha256 = _checkpoint_id(
        Path(config.checkpoint_dir),
        restored_train_steps,
        os.environ.get("VLA_EVAL_CHECKPOINT_ID", ""),
    )
    # The fixed-manifest evaluator reads this value and binds every resumed or new
    # result row to the inventory verified above.
    os.environ["VLA_EVAL_CHECKPOINT_ID"] = checkpoint_id

    env = _make_eval_env(config)
    try:
        metrics = evaluate_policy(
            agent=agent,
            env=env,
            config=config,
            step=restored_train_steps,
        )
    finally:
        env.close()

    result_rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    successes = sum(int(row["success"]) for row in result_rows)
    if len(result_rows) != int(config.collect.num_eval_rollouts) * len(
        config.collect.tasks
    ):
        raise RuntimeError(f"unexpected result count: {len(result_rows)}")
    if int(metrics["eval/successes"]) != successes:
        raise RuntimeError(
            "success count disagrees between metrics and episode results"
        )

    summary = {
        "schema_version": 1,
        "host": platform.node(),
        "checkpoint_dir": str(Path(config.checkpoint_dir).resolve()),
        "checkpoint_id": checkpoint_id,
        "checkpoint_artifact_inventory_sha256": inventory_sha256,
        "restored_train_steps": restored_train_steps,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "results_path": str(results_path),
        "results_sha256": _sha256(results_path),
        "episode_count": len(result_rows),
        "successes": successes,
        "success_rate": successes / len(result_rows),
        "metrics": {key: _json_value(value) for key, value in metrics.items()},
    }
    _write_json_atomic(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(_config.cli())
