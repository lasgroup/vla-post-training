import dataclasses
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Callable


@dataclasses.dataclass(frozen=True)
class ResumeState:
    step: int
    total_collected_episodes: int
    # `str`, not `Path`: the manifest is JSON, which has no Path type, and
    # `ResumeState(**payload)` hands the raw string straight through.
    replay_shard_dir: str
    agent_rng_state_json: str
    replay_rng_state_json: str
    # Learner-owned resume payload (`FilteredSFTLearner.save_extra_resume_state`;
    # OGPO's success buffer, task ranges, advantage scale). One nested field
    # so pre-change five-key manifests still construct here and later additions
    # need no schema edit — an unknown TOP-LEVEL key still raises TypeError.
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)


def _checkpoint_dir(config: Any) -> Path:
    return Path(os.fspath(config.checkpoint_dir))


def _runtime_state_dir(config: Any) -> Path:
    return _checkpoint_dir(config) / "runtime_state"


def replay_shard_dir(config: Any) -> Path:
    return _runtime_state_dir(config) / "replay_shards"


def replay_shard_path(config: Any, step: int) -> Path:
    return replay_shard_dir(config) / f"step_{int(step):08d}.h5"


def success_shard_dir(config: Any) -> Path:
    # Own directory so `restore_shards`'s `step_*.h5` glob can never mix the
    # success buffer's shards with the online buffer's.
    return _runtime_state_dir(config) / "success_shards"


def success_shard_path(config: Any, step: int) -> Path:
    return success_shard_dir(config) / f"step_{int(step):08d}.h5"


def resume_state_path(config: Any) -> Path:
    return _runtime_state_dir(config) / "resume_state.json"


def step_manifest_path(config: Any, step: int) -> Path:
    return _runtime_state_dir(config) / f"resume_state_{int(step):08d}.json"


def step_manifest_steps(config: Any) -> set[int]:
    """Steps that have a per-step resume manifest on disk."""
    state_dir = _runtime_state_dir(config)
    if not state_dir.exists():
        return set()
    steps = set()
    for path in state_dir.glob("resume_state_*.json"):
        stem = path.stem[len("resume_state_"):]
        if not stem.isdigit():
            raise ValueError(
                f"Resume manifest {path} does not encode a step as "
                "resume_state_<digits>.json. Atomic writes go through a leading-dot "
                "temp name, so this is a foreign file; move it out of the runtime "
                "state directory."
            )
        steps.add(int(stem))
    return steps


def resolve_resume_step(
    orbax_steps: set[int],
    manifest_steps: set[int],
    required_ok: Callable[[int], bool],
    pointer_step: int | None,
) -> int:
    """Newest step that is complete in every component, i.e. resumable.

    `save_epoch_state` writes the replay shards and the per-step manifest before
    committing orbax, and openpi pins `max_to_keep=1`
    (openpi/src/openpi/training/checkpoints.py:48) so that commit deletes the
    previous step. A crash therefore leaves either the previous step complete or
    the new one complete, never a half state — provided the resume picks the
    newest step that has an orbax checkpoint, a per-step manifest, and whatever
    sidecars the learner declares (`required_ok`).

    Pre-change checkpoint directories have no per-step manifests at all; they
    fall back to the `resume_state.json` pointer.
    """
    if not manifest_steps:
        if pointer_step is not None and pointer_step in orbax_steps:
            return pointer_step
        raise FileNotFoundError(
            "Resume requested, but this checkpoint directory has no per-step "
            f"manifest, and resume_state.json names step {pointer_step} while the "
            f"committed orbax steps are {sorted(orbax_steps)} (step None = no "
            "resume_state.json at all, e.g. a checkpoint dir not written by "
            "scripts/exp.py). Point 'step' in resume_state.json at one of the "
            "committed steps, or start over with --overwrite (FRESH=1 in the "
            "shell recipes)."
        )
    for step in sorted(orbax_steps & manifest_steps, reverse=True):
        if required_ok(step):
            return step
    # Mixed tree: a pre-change directory whose FIRST post-change save wrote
    # `resume_state_<N>.json` and then died before the orbax commit has manifests
    # {N} against orbax {M} — an empty intersection, while M is complete and
    # named by the pointer. Every in-flight run passes through that window once.
    if (
        pointer_step is not None
        and pointer_step in orbax_steps
        and required_ok(pointer_step)
    ):
        return pointer_step
    raise FileNotFoundError(
        "Resume requested, but no step has all of {orbax checkpoint, per-step "
        "manifest, learner rl_state}: committed orbax steps "
        f"{sorted(orbax_steps)}, per-step manifests {sorted(manifest_steps)}, "
        f"pointer step {pointer_step}. Restore the missing component for one of "
        "those steps, or start over with --overwrite (FRESH=1 in the shell "
        "recipes)."
    )


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


def load_resume_state(config: Any, step: int | None = None) -> ResumeState:
    """Load resumable training metadata.

    `step` reads that step's own manifest; `None` reads the `resume_state.json`
    pointer, i.e. the last step `save_epoch_state` completed. Raises
    `FileNotFoundError` when the manifest does not exist.
    """
    manifest_path = (
        resume_state_path(config) if step is None else step_manifest_path(config, step)
    )
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
    if not prepare_for_resume:
        agent.save_checkpoint(step=step)
        agent._checkpoint_manager.wait_until_finished()
        logging.info("Saved checkpoint at step %d to %s", step, _checkpoint_dir(config))
        return

    # ORDER IS LOAD-BEARING (crash atomicity, not taste). openpi pins
    # max_to_keep=1 (openpi/src/openpi/training/checkpoints.py:48), so committing
    # step N's orbax checkpoint deletes step M's. Everything a resume of N needs
    # is therefore durable BEFORE that commit: a crash before it leaves M whole,
    # a crash after it leaves N whole, and never a manifest pointing at a
    # checkpoint that no longer exists. `resolve_resume_step` picks whichever of
    # the two survived; the shards written for N are excluded from an M resume by
    # `restore_shards(max_step=M)`.
    agent._online_data_buffer.save_shard(replay_shard_path(config, step))
    extra = agent.save_extra_resume_state(step)
    payload = {
        "step": step,
        "agent_rng_state_json": agent.rng_state_json(),
        "total_collected_episodes": agent.total_collected_episodes,
        "replay_shard_dir": str(replay_shard_dir(config)),
        "replay_rng_state_json": agent._online_data_buffer.rng_state_json(),
        "extra": extra,
    }
    manifest_text = json.dumps(payload, indent=2, sort_keys=True)
    _atomic_write_text(step_manifest_path(config, step), manifest_text)
    agent.save_checkpoint(step=step)
    agent._checkpoint_manager.wait_until_finished()
    # Pointer last: it names the newest completed step for humans and for
    # pre-per-step-manifest directories; the resolver never depends on it once
    # per-step manifests exist.
    _atomic_write_text(resume_state_path(config), manifest_text)
    logging.info(
        "Saved resumable epoch state at step %d (manifest=%s)",
        payload["step"],
        step_manifest_path(config, step),
    )
