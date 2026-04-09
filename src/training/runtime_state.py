import dataclasses
import datetime as dt
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Callable


@dataclasses.dataclass(frozen=True)
class ResumeState:
    step: int
    replay_size: int
    replay_total_inserted: int
    replay_shard_dir: Path
    manifest_path: Path
    latest_replay_shard_path: Path | None = None
    replay_rng_state_json: str | None = None
    timestamp: str | None = None


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


def write_resume_state(
    config: Any,
    *,
    step: int,
    replay_size: int,
    replay_total_inserted: int,
    replay_shards: Path,
    latest_replay_shard: Path | None,
    replay_rng_state_json: str,
) -> ResumeState:
    manifest_path = resume_state_path(config)
    payload = {
        "step": int(step),
        "replay_size": int(replay_size),
        "replay_total_inserted": int(replay_total_inserted),
        "replay_shard_dir": str(replay_shards),
        "latest_replay_shard_path": (
            None if latest_replay_shard is None else str(latest_replay_shard)
        ),
        "replay_rng_state_json": replay_rng_state_json,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    _atomic_write_text(manifest_path, json.dumps(payload, indent=2, sort_keys=True))
    return ResumeState(
        step=int(payload["step"]),
        replay_size=int(payload["replay_size"]),
        replay_total_inserted=int(payload["replay_total_inserted"]),
        replay_shard_dir=Path(payload["replay_shard_dir"]),
        manifest_path=manifest_path,
        latest_replay_shard_path=(
            None
            if payload["latest_replay_shard_path"] is None
            else Path(payload["latest_replay_shard_path"])
        ),
        replay_rng_state_json=payload["replay_rng_state_json"],
        timestamp=payload["timestamp"],
    )


def load_resume_state(config: Any) -> ResumeState | None:
    """Load resumable training metadata and validate its replay snapshot.

    Returns `None` when no manifest has been written yet. Relative replay
    snapshot paths are resolved from the manifest directory, and a missing
    snapshot raises `FileNotFoundError`.
    """
    manifest_path = resume_state_path(config)
    if not manifest_path.exists():
        return None

    payload = json.loads(manifest_path.read_text())
    shard_dir = Path(payload["replay_shard_dir"])
    if not shard_dir.is_absolute():
        shard_dir = manifest_path.parent / shard_dir
    latest_shard = payload.get("latest_replay_shard_path")
    if latest_shard is not None:
        latest_shard = Path(latest_shard)
        if not latest_shard.is_absolute():
            latest_shard = manifest_path.parent / latest_shard
        if not latest_shard.exists():
            raise FileNotFoundError(
                f"Resume state at {manifest_path} points to a missing replay shard: {latest_shard}"
            )
    elif int(payload["replay_total_inserted"]) > 0:
        raise FileNotFoundError(
            f"Resume state at {manifest_path} is missing its latest replay shard path."
        )

    return ResumeState(
        step=int(payload["step"]),
        replay_size=int(payload["replay_size"]),
        replay_total_inserted=int(payload["replay_total_inserted"]),
        replay_shard_dir=shard_dir,
        manifest_path=manifest_path,
        latest_replay_shard_path=latest_shard,
        replay_rng_state_json=payload.get("replay_rng_state_json"),
        timestamp=payload.get("timestamp"),
    )


def restore_train_state(
    restore_fn: Callable[..., Any],
    checkpoint_manager: Any,
    train_state: Any,
    data_loader: Any,
    *,
    resume_state: ResumeState | None,
) -> Any:
    restore_step = None if resume_state is None else int(resume_state.step)
    return restore_fn(
        checkpoint_manager,
        train_state,
        data_loader,
        step=restore_step,
    )


def current_training_step(agent: Any) -> int:
    if hasattr(agent, "training_steps"):
        return int(agent.training_steps)

    train_state = getattr(agent, "_train_state", None)
    if train_state is not None and hasattr(train_state, "step"):
        step = train_state.step
        try:
            import jax
        except ModuleNotFoundError:
            return int(step)
        return int(jax.device_get(step))

    raise AttributeError("Agent does not expose training_steps or train-state step.")


def save_epoch_state(
    agent: Any,
    config: Any,
) -> ResumeState | None:
    step = current_training_step(agent)

    agent.save_checkpoint(step=step)
    checkpoint_manager = getattr(agent, "_checkpoint_manager", None)
    if checkpoint_manager is not None:
        checkpoint_manager.wait_until_finished()

    if not bool(getattr(config, "requeue", False)):
        logging.info("Saved checkpoint at step %d to %s", step, _checkpoint_dir(config))
        return None

    replay_buffer = getattr(agent, "_online_data_buffer", None)
    if replay_buffer is None:
        raise AttributeError("Agent is missing _online_data_buffer for epoch-state saves.")

    shard_path = replay_shard_path(config, step)
    shard_info = replay_buffer.save_shard(shard_path, step=step)
    resume_state = write_resume_state(
        config,
        step=step,
        replay_size=int(shard_info["size"]),
        replay_total_inserted=int(shard_info["total_inserted"]),
        replay_shards=replay_shard_dir(config),
        latest_replay_shard=(
            None
            if shard_info["path"] is None
            else Path(str(shard_info["path"]))
        ),
        replay_rng_state_json=replay_buffer.rng_state_json(),
    )
    logging.info(
        "Saved resumable epoch state at step %d (replay transitions=%d, latest shard=%s, manifest=%s)",
        resume_state.step,
        resume_state.replay_size,
        resume_state.latest_replay_shard_path,
        resume_state.manifest_path,
    )
    return resume_state
