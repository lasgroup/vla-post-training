from __future__ import annotations

from typing import Any


PREFIX_EMBEDDING_NAME = "prefix_embedding"


def unpack_action_and_prefix(
    action_chunk: Any,
) -> tuple[Any, Any | None]:
    """Return `(actions, prefix_embedding)` from supported action payload formats."""
    if (
        isinstance(action_chunk, (tuple, list))
        and len(action_chunk) == 2
    ):
        return action_chunk[0], action_chunk[1]

    if isinstance(action_chunk, dict):
        actions = action_chunk.get("action", action_chunk.get("actions", action_chunk))
        return actions, action_chunk.get(PREFIX_EMBEDDING_NAME)

    return action_chunk, None
