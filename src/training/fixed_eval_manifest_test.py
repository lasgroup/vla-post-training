import json
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from scripts.filtered_sft_agent.create_fixed_eval_manifests import build_manifests
from src.training.collect import evaluate_policy


class _FakePolicy:
    _rng = None


class _FakeAgent:
    def __init__(self):
        self._rng = None
        self._policy = _FakePolicy()
        self.total_collected_episodes = 0

    def sample_actions(self, obs, task_description):
        del obs, task_description
        return np.zeros((1, 2, 7), dtype=np.float32)


class _FakeEnv:
    env_num = 1

    def __init__(self):
        self.current_state = None
        self.reset_count = 0

    def reset(self, options):
        self.reset_count += 1
        self.current_state = int(options["init_state_index"][0])
        return {}, {
            "task_description": np.array(["test task"]),
            "init_state_index": np.array([self.current_state]),
        }

    def step(self, action):
        del action
        terminate = np.zeros((1, 2), dtype=bool)
        truncate = np.zeros((1, 2), dtype=bool)
        if self.current_state == 0:
            terminate[0, -1] = True
        else:
            truncate[0, -1] = True
        return {}, np.zeros((1, 2)), terminate, truncate, {}


def _write_manifest(path):
    rows = [
        {
            "episode_id": "state0",
            "task": "libero_90_0",
            "initial_state_index": 0,
            "policy_seed": 11,
        },
        {
            "episode_id": "state1",
            "task": "libero_90_0",
            "initial_state_index": 1,
            "policy_seed": 12,
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _config():
    return SimpleNamespace(
        collect=SimpleNamespace(
            tasks=["libero_90_0"],
            num_eval_rollouts=2,
            replan_steps=2,
        )
    )


def _configure_eval(monkeypatch, manifest, results):
    monkeypatch.setenv("VLA_FIXED_EVAL_MANIFEST", str(manifest))
    monkeypatch.setenv("VLA_FIXED_EVAL_RESULTS", str(results))
    monkeypatch.setenv("VLA_EVAL_CHECKPOINT_ID", "test-checkpoint")


def test_fixed_manifest_evaluation_and_idempotent_resume(tmp_path, monkeypatch):
    manifest = tmp_path / "eval.jsonl"
    _write_manifest(manifest)
    results = tmp_path / "results.jsonl"
    _configure_eval(monkeypatch, manifest, results)

    env = _FakeEnv()
    metrics = evaluate_policy(
        cast(Any, _FakeAgent()), cast(Any, env), _config(), step=4001
    )
    assert metrics["eval/episodes"] == 2
    assert metrics["eval/successes"] == 1
    assert metrics["eval/success_rate"] == 0.5
    assert env.reset_count == 2

    persisted = [json.loads(line) for line in results.read_text().splitlines()]
    assert [row["initial_state_index"] for row in persisted] == [0, 1]
    assert [row["policy_seed"] for row in persisted] == [11, 12]

    retry_env = _FakeEnv()
    retry_metrics = evaluate_policy(
        cast(Any, _FakeAgent()), cast(Any, retry_env), _config(), step=4001
    )
    assert retry_metrics["eval/success_rate"] == 0.5
    assert retry_env.reset_count == 0


def test_manifest_builder_is_deterministic_and_idempotent(tmp_path):
    state_counts = tmp_path / "state_counts.json"
    state_counts.write_text('{"libero_90_0": 2}\n', encoding="utf-8")
    output_dir = tmp_path / "manifests"

    first = build_manifests(
        state_counts=state_counts,
        output_dir=output_dir,
        base_seed=17,
        rollouts=4,
    )
    second = build_manifests(
        state_counts=state_counts,
        output_dir=output_dir,
        base_seed=17,
        rollouts=4,
    )
    assert second == first
    manifest_path = output_dir / "task0_fixed_4.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    assert [row["initial_state_index"] for row in rows] == [0, 1, 0, 1]
    assert len({row["policy_seed"] for row in rows}) == 4
    assert first["tasks"][0]["manifest_sha256"]

    manifest_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="non-matching artifact"):
        build_manifests(
            state_counts=state_counts,
            output_dir=output_dir,
            base_seed=17,
            rollouts=4,
        )


def test_fixed_manifest_resume_rejects_malformed_completed_rows(tmp_path, monkeypatch):
    manifest = tmp_path / "eval.jsonl"
    _write_manifest(manifest)
    results = tmp_path / "results.jsonl"
    _configure_eval(monkeypatch, manifest, results)

    evaluate_policy(
        cast(Any, _FakeAgent()), cast(Any, _FakeEnv()), _config(), step=4001
    )
    rows = [json.loads(line) for line in results.read_text().splitlines()]
    rows[0]["success"] = 1
    results.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    with pytest.raises(TypeError, match="success must be a boolean"):
        evaluate_policy(
            cast(Any, _FakeAgent()), cast(Any, _FakeEnv()), _config(), step=4001
        )
