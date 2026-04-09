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
    replay_snapshot_path: Path
    manifest_path: Path
    timestamp: str | None = None


def _checkpoint_dir(config: Any) -> Path:
    return Path(os.fspath(config.checkpoint_dir))


def _runtime_state_dir(config: Any) -> Path:
    return _checkpoint_dir(config) / "runtime_state"


def replay_snapshot_path(config: Any) -> Path:
    return _runtime_state_dir(config) / "replay_buffer_latest.h5"


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
    replay_snapshot: Path,
) -> ResumeState:
    manifest_path = resume_state_path(config)
    payload = {
        "step": int(step),
        "replay_size": int(replay_size),
        "replay_snapshot_path": str(replay_snapshot),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    _atomic_write_text(manifest_path, json.dumps(payload, indent=2, sort_keys=True))
    return ResumeState(
        step=int(payload["step"]),
        replay_size=int(payload["replay_size"]),
        replay_snapshot_path=Path(payload["replay_snapshot_path"]),
        manifest_path=manifest_path,
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
    replay_path = Path(payload["replay_snapshot_path"])
    if not replay_path.is_absolute():
        replay_path = manifest_path.parent / replay_path
    if not replay_path.exists():
        raise FileNotFoundError(
            f"Resume state at {manifest_path} points to a missing replay snapshot: {replay_path}"
        )

    return ResumeState(
        step=int(payload["step"]),
        replay_size=int(payload["replay_size"]),
        replay_snapshot_path=replay_path,
        manifest_path=manifest_path,
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

    snapshot_path = replay_snapshot_path(config)
    snapshot_info = replay_buffer.save_snapshot(snapshot_path, step=step)
    resume_state = write_resume_state(
        config,
        step=step,
        replay_size=int(snapshot_info["size"]),
        replay_snapshot=snapshot_path,
    )
    logging.info(
        "Saved resumable epoch state at step %d (replay transitions=%d, manifest=%s)",
        resume_state.step,
        resume_state.replay_size,
        resume_state.manifest_path,
    )
    return resume_state
