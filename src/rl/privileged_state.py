"""Key names and extraction for the *privileged* critic observation.

The default OGPO critic sees a mean-pooled PaliGemma prefix embedding plus the
policy's proprioceptive ``state`` vector. The privileged critic instead sees the
simulator's own state — robot proprioception and object poses — so the critic's
representation is no longer bottlenecked by what survives mean-pooling a VLM
prefix. It is a diagnostic upper bound, not a deployable critic: the privileged
vector does not exist on a real robot.

Two observation keys carry it through the collection pipeline and the replay
buffer (mirroring ``prefix_embedding``):

  ``privileged_state``    zero-padded float32 vector, fixed width across tasks
  ``privileged_task_id``  index into the run's train-task list, ``-1`` if the
                          episode's task is not one of them (held-out eval)
"""
from __future__ import annotations

import numpy as np


PRIVILEGED_STATE_NAME = "privileged_state"
PRIVILEGED_TASK_ID_NAME = "privileged_task_id"

# Sentinel for an episode whose task is not in the run's train-task list (the
# held-out eval block). One-hot of -1 is the zero vector, so every per-task
# critic head contributes nothing rather than an arbitrary task's Q.
UNKNOWN_TASK_ID = -1.0

# robosuite concatenates every enabled+active observable into one vector per
# modality. `robot0_proprio-state` is the Panda's joint/eef/gripper block
# (LIBERO additionally activates `robot0_joint_pos`); `object-state` is
# pos/quat/to_eef_pos/to_eef_quat for each object named by the task's BDDL.
# Together they are the full task-relevant simulator state.
LIBERO_PRIVILEGED_OBS_KEYS = ("robot0_proprio-state", "object-state")


def extract_libero_privileged_state(obs: dict, dim: int) -> np.ndarray:
    """Concatenate the LIBERO privileged observables and zero-pad to ``dim``.

    The width of ``object-state`` is 14 per object, so it varies from task to
    task; padding to a fixed ``dim`` keeps one replay-buffer schema for the
    whole run. Each per-task critic sees a constant padding region, so the
    padding carries no signal it could confuse for state.
    """
    parts = []
    for key in LIBERO_PRIVILEGED_OBS_KEYS:
        if key not in obs:
            raise KeyError(
                f"LIBERO observation is missing {key!r}, required for the privileged "
                f"critic. Available keys: {sorted(obs)}"
            )
        parts.append(np.asarray(obs[key], dtype=np.float32).reshape(-1))
    flat = np.concatenate(parts, axis=0)
    if flat.shape[0] > dim:
        raise ValueError(
            f"Privileged state is {flat.shape[0]} wide but collect.privileged_state_dim "
            f"is {dim}; raise the config value."
        )
    out = np.zeros((dim,), dtype=np.float32)
    out[: flat.shape[0]] = flat
    return out


def privileged_task_ids(tasks) -> list[str]:
    """Ordered, de-duplicated train-task list — the per-task critic's index space.

    ``collect.tasks`` may repeat a task (the ``x4`` multiplier) purely to weight
    collection; the critic gets one head per DISTINCT task.
    """
    seen: dict[str, None] = {}
    for task in tasks:
        seen.setdefault(str(task), None)
    return list(seen)
