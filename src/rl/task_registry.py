"""Host-side task-description -> critic-slot registry for per-task critics.

Per-task critics (``rl.critic.num_tasks``, ``src/rl/networks/per_task_critic.py``)
route every critic call by an integer ``task_index`` stored in the replay buffer.
The key is the task ID (``libero_90_79``), threaded from ``collect.py``'s
slot-aligned ``current_task_ids`` through ``save_episode(task_id=...)`` and
``sample_actions(task_id=...)`` — NOT the language string in
``info["task_description"]``: libero_90 prompts are not unique across ids
(79 and 82 share one; 12 collide overall), and a prompt-keyed registry silently
merged them (verifier finding F1). Note ``_success_task_ranges`` and the
``normalize_advantage_per_task`` prompt hash still key on the prompt.

The mapping MUST be persisted with the RL checkpoint (``to_json`` / ``from_json``):
an unpersisted registry would re-assign slots in whatever order tasks happen to be
seen after a resume, silently handing every task another task's critic. See
``docs/changes/2026-08-21-per-task-critics/`` (decision D3).

Fail-fast by design: there is no fallback slot. A task beyond ``num_tasks``
raises with the fix in the message (decision D4).
"""
import json
from pathlib import Path


class TaskRegistry:
    def __init__(self, num_tasks: int):
        if num_tasks < 1:
            raise ValueError(f"TaskRegistry needs num_tasks >= 1, got {num_tasks}.")
        self.num_tasks = int(num_tasks)
        self._slots: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._slots)

    @property
    def tasks(self) -> dict[str, int]:
        """Read-only view: task id -> slot."""
        return dict(self._slots)

    def index_for(self, task_id: str) -> int:
        """Slot for ``task_id`` (e.g. ``libero_90_79``); assigned first-seen, never reassigned."""
        key = str(task_id)
        if key in self._slots:
            return self._slots[key]
        if len(self._slots) >= self.num_tasks:
            registered = sorted(self._slots, key=self._slots.__getitem__)
            raise ValueError(
                f"Per-task critics: task {key!r} has no critic slot — all "
                f"{self.num_tasks} slots are taken by {registered}. Per-task "
                "critics cover exactly the collect.tasks set (rl.critic.num_tasks "
                "== number of distinct train tasks); a held-out eval task can only "
                "be scored with rl.n_samples=1 (no best-of-N), and a new train "
                "task needs a fresh run with a larger rl.critic.num_tasks."
            )
        slot = len(self._slots)
        self._slots[key] = slot
        return slot

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"num_tasks": self.num_tasks, "tasks": self._slots}, indent=2)
        )

    @classmethod
    def from_json(cls, path: str | Path, num_tasks: int) -> "TaskRegistry":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Per-task critic registry {path} not found. A per-task-critic run "
                "can only resume from a checkpoint written by a per-task-critic run "
                "(decision D8, docs/changes/2026-08-21-per-task-critics/); start a "
                "fresh run or resume with rl.critic.num_tasks=None."
            )
        payload = json.loads(path.read_text())
        if int(payload["num_tasks"]) != int(num_tasks):
            raise ValueError(
                f"Per-task critic registry {path} was written with num_tasks="
                f"{payload['num_tasks']} but the config says rl.critic.num_tasks="
                f"{num_tasks}. The critic param tree has one slot per task, so "
                "these must match; resume with the checkpoint's value."
            )
        slots = {str(k): int(v) for k, v in payload["tasks"].items()}
        if len(slots) > int(num_tasks):
            raise ValueError(
                f"Per-task critic registry {path} holds {len(slots)} tasks but the "
                f"critic has only num_tasks={num_tasks} slots; the file is corrupt."
            )
        if sorted(slots.values()) != list(range(len(slots))):
            raise ValueError(
                f"Per-task critic registry {path} has non-contiguous slots "
                f"{slots}; the file is corrupt."
            )
        reg = cls(num_tasks)
        reg._slots = slots
        return reg
