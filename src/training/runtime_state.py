import dataclasses
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any


@dataclasses.dataclass(frozen=True)
class ResumeState:
    step: int
    total_collected_episodes: int
    replay_shard_dir: Path
    agent_rng_state_json: str
    replay_rng_state_json: str


def _checkpoint_dir(config: Any) -> Path:
    return Path(os.fspath(config.checkpoint_dir))


def _runtime_state_dir(config: Any) -> Path:
    return _checkpoint_dir(config) / "runtime_state"


def replay_shard_dir(config: Any) -> Path:
    return _runtime_state_dir(config) / "replay_shards"


def replay_shard_path(config: Any, step: int) -> Path:
    return replay_shard_dir(config) / f"step_{int(step):08d}.h5"


def resume_state_path(config: Any) -> Path:
    return _runtime_state_dir(config) / "resume_state.json"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp_file:
        tmp_file.write(text)
        tmp_path = Path(tmp_file.name)
    os.replace(tmp_path, path)


def load_resume_state(config: Any) -> ResumeState:
    """Load resumable training metadata and validate its replay snapshot.

    Returns `None` when no manifest has been written yet. Relative replay
    snapshot paths are resolved from the manifest directory, and a missing
    snapshot raises `FileNotFoundError`.
    """
    manifest_path = resume_state_path(config)
    if not manifest_path.exists():
        raise FileNotFoundError(f"No resume state manifest found at {manifest_path}")

    payload = json.loads(manifest_path.read_text())
    return ResumeState(**payload)


def save_epoch_state(
    agent: Any,
    config: Any,
    prepare_for_resume: bool = False,
) -> ResumeState | None:
    step = agent.training_steps
    agent.save_checkpoint(step=step)
    agent._checkpoint_manager.wait_until_finished()

    if not prepare_for_resume:
        logging.info("Saved checkpoint at step %d to %s", step, _checkpoint_dir(config))
        return

    shard_path = replay_shard_path(config, step)
    agent._online_data_buffer.save_shard(shard_path)
    manifest_path = resume_state_path(config)
    payload = {
        "step": step,
        "agent_rng_state_json": agent.rng_state_json(),
        "total_collected_episodes": agent.total_collected_episodes,
        "replay_shard_dir": str(replay_shard_dir(config)),
        "replay_rng_state_json": agent._online_data_buffer.rng_state_json(),
    }
    _atomic_write_text(manifest_path, json.dumps(payload, indent=2, sort_keys=True))
    logging.info(
        "Saved resumable epoch state at step %d (manifest=%s)",
        payload["step"],
        manifest_path,
    )
