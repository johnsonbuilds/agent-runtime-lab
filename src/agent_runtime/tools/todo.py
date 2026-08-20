"""The todo_write meta-tool: a structured task list as workspace state.

Semantics follow Claude Code's TodoWrite: each call **replaces the whole
list**, which keeps the tool idempotent and merge-free.  State lives in
``.todo.json`` inside the workspace (not in runtime memory), so it
survives turns, works through any ``Workspace`` implementation, and
stays inspectable by the developer.  The rendered checklist comes back
as the observation so the model immediately sees its own plan.
"""

from __future__ import annotations

import json
from typing import Any

from agent_runtime.execution.base import Workspace
from agent_runtime.execution.local import LocalWorkspace


TODO_PATH = ".todo.json"
STATUSES = ("pending", "in_progress", "completed")
MAX_TODOS = 100


def _render(todos: list[dict[str, str]]) -> str:
    return "\n".join(f"{index}. [{item['status']}] {item['content']}"
                     for index, item in enumerate(todos, 1))


def _validate(todos: Any) -> list[dict[str, str]]:
    if not isinstance(todos, list):
        raise ValueError("todos must be a list of {content, status} objects")
    if len(todos) > MAX_TODOS:
        raise ValueError(f"at most {MAX_TODOS} todos per call")
    cleaned: list[dict[str, str]] = []
    for position, item in enumerate(todos):
        if not isinstance(item, dict):
            raise ValueError(f"todo #{position + 1} must be an object")
        content = item.get("content")
        status = item.get("status")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"todo #{position + 1} needs non-empty content")
        if status not in STATUSES:
            raise ValueError(f"todo #{position + 1} status must be one of "
                             f"{list(STATUSES)}, got {status!r}")
        cleaned.append({"content": content.strip(), "status": status})
    in_progress = sum(1 for item in cleaned if item["status"] == "in_progress")
    if in_progress > 1:
        raise ValueError("at most one todo may be in_progress at a time")
    return cleaned


async def todo_write(todos: list[dict[str, Any]], *,
                     workspace: Workspace | None = None) -> dict[str, Any]:
    """Replace the task list stored at ``.todo.json`` in the workspace."""
    cleaned = _validate(todos)
    ws = workspace or LocalWorkspace()
    written = await ws.write_file(TODO_PATH,
                                  json.dumps(cleaned, ensure_ascii=False,
                                             indent=2) + "\n")
    if "error" in written:
        return written
    return {"path": TODO_PATH, "todos": cleaned, "todo_count": len(cleaned),
            "completed": sum(1 for item in cleaned
                             if item["status"] == "completed"),
            "rendered": _render(cleaned) or "(empty task list)"}


__all__ = ["MAX_TODOS", "STATUSES", "TODO_PATH", "todo_write"]
