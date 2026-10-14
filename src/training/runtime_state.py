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
    # Set while a collection round is paused mid-way (see collect.max_runtime).
    collection_progress: dict | None = None


def _checkpoint_dir(config: Any) -> Path:
    return Path(os.fspath(config.checkpoint_dir))


def _runtime_state_dir(config: Any) -> Path:
    return _checkpoint_dir(config) / "runtime_state"


def replay_shard_dir(config: Any) -> Path:
    return _runtime_state_dir(config) / "replay_shards"


def replay_shard_path(config: Any, step: int, collection_part: int | None = None) -> Path:
    # A paused collection round saves `step_<S>_c<part>.h5`: the round-boundary shard
    # `step_<S>.h5` already holds the previous round's data, and restore_shards loads
    # shards in sorted order, where "step_<S>.h5" < "step_<S>_c01.h5" < "step_<S+1>.h5".
    if collection_part is None:
        return replay_shard_dir(config) / f"step_{int(step):08d}.h5"
    return replay_shard_dir(config) / f"step_{int(step):08d}_c{int(collection_part):02d}.h5"


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
    collection_progress: dict | None = None,
) -> ResumeState | None:
    step = agent.training_steps
    # A paused collection round has not changed the policy or critic since the
    # round-boundary save at this step, so only round 0 (step 0, never saved) needs one.
    if collection_progress is None or step not in agent._checkpoint_manager.all_steps():
        agent.save_checkpoint(step=step)
    agent._checkpoint_manager.wait_until_finished()
    # The RL-state checkpointer (critic/value/normalizer) is a SEPARATE async
    # checkpointer and is NOT covered by the manager's wait above. Callers exit
    # with sys.exit(REQUEUE_EXIT_CODE) immediately after this returns, which
    # kills an in-flight write and leaves only `<step>.orbax-checkpoint-tmp-*`
    # while the manifest below already advertises `step` -- the run then cannot
    # resume at all. This was masked while saves only happened at round
    # boundaries (the large policy save gave the RL write time to land) and
    # started killing runs once mid-round requeue saves were enabled.
    rl_ckptr = getattr(agent, "_rl_state_checkpointer", None)
    if rl_ckptr is not None and hasattr(rl_ckptr, "wait_until_finished"):
        rl_ckptr.wait_until_finished()

    if not prepare_for_resume:
        logging.info("Saved checkpoint at step %d to %s", step, _checkpoint_dir(config))
        return

    collection_part = None if collection_progress is None else collection_progress["part"]
    shard_path = replay_shard_path(config, step, collection_part)
    agent._online_data_buffer.save_shard(shard_path)
    write_resume_manifest(agent, config, collection_progress)


def write_resume_manifest(agent: Any, config: Any, collection_progress: dict | None = None) -> None:
    """Write resume_state.json for the agent's current step (no checkpoint/shard save)."""
    step = agent.training_steps
    manifest_path = resume_state_path(config)
    payload = {
        "step": step,
        "agent_rng_state_json": agent.rng_state_json(),
        "total_collected_episodes": agent.total_collected_episodes,
        "replay_shard_dir": str(replay_shard_dir(config)),
        "replay_rng_state_json": agent._online_data_buffer.rng_state_json(),
    }
    if collection_progress is not None:
        payload["collection_progress"] = collection_progress
    _atomic_write_text(manifest_path, json.dumps(payload, indent=2, sort_keys=True))
    logging.info(
        "Saved resumable epoch state at step %d (manifest=%s)",
        payload["step"],
        manifest_path,
    )
