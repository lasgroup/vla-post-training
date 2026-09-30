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
    replay_shard_dir: str
    agent_rng_state_json: str
    replay_rng_state_json: str


def _runtime_state_dir(config: Any) -> Path:
    return Path(os.fspath(config.checkpoint_dir)) / "runtime_state"


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
    """Load the resumable training metadata; raises FileNotFoundError if none was saved."""
    manifest_path = _runtime_state_dir(config) / "resume_state.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No resume state manifest found at {manifest_path}")
    return ResumeState(**json.loads(manifest_path.read_text()))


def save_epoch_state(agent: Any, config: Any) -> None:
    """Save the model checkpoint, the new replay shard and the resume manifest."""
    step = agent.training_steps
    agent.save_checkpoint(step=step)

    shard_dir = _runtime_state_dir(config) / "replay_shards"
    agent._online_data_buffer.save_shard(shard_dir / f"step_{int(step):08d}.h5")
    manifest_path = _runtime_state_dir(config) / "resume_state.json"
    payload = {
        "step": step,
        "agent_rng_state_json": agent.rng_state_json(),
        "total_collected_episodes": agent.total_collected_episodes,
        "replay_shard_dir": str(shard_dir),
        "replay_rng_state_json": agent._online_data_buffer.rng_state_json(),
    }
    _atomic_write_text(manifest_path, json.dumps(payload, indent=2, sort_keys=True))
    logging.info("Saved resumable epoch state at step %d (manifest=%s)", step, manifest_path)
