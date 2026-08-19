"""File meta-tools: thin LLM-facing shells over a Workspace.

All real work (path containment, encoding, paging, error shaping)
lives in the execution layer; these handlers only bind the protocol
into tool-callable functions, mirroring how ``run_command`` relates
to ``ShellExecutor``.
"""

from __future__ import annotations

from typing import Any

from agent_runtime.execution.base import Workspace
from agent_runtime.execution.local import LocalWorkspace


DEFAULT_READ_LIMIT = 2000


async def write_file(path: str, content: str, *,
                     workspace: Workspace | None = None) -> dict[str, Any]:
    """Create or overwrite a file with the given full content."""
    return await (workspace or LocalWorkspace()).write_file(path, content)


async def read_file(path: str, offset: int = 1, limit: int | None = DEFAULT_READ_LIMIT, *,
                    workspace: Workspace | None = None) -> dict[str, Any]:
    """Read a file as text, one page of lines at a time."""
    return await (workspace or LocalWorkspace()).read_file(path, offset, limit)


async def list_dir(path: str = ".", *,
                   workspace: Workspace | None = None) -> dict[str, Any]:
    """List a directory's entries (directories first, then files)."""
    return await (workspace or LocalWorkspace()).list_dir(path)


async def edit_file(path: str, old_str: str, new_str: str, *,
                    workspace: Workspace | None = None) -> dict[str, Any]:
    """Replace exactly one occurrence of ``old_str`` with ``new_str``.

    ``old_str`` must match exactly once: zero matches or several
    matches raise, which pushes the model to read the file and
    disambiguate with more surrounding lines.
    """
    if not old_str:
        raise ValueError("old_str must not be empty")
    ws = workspace or LocalWorkspace()
    read = await ws.read_file(path)
    if "error" in read:
        return read
    content = read["content"]
    occurrences = content.count(old_str)
    if occurrences == 0:
        raise ValueError(f"old_str not found in {path}")
    if occurrences > 1:
        raise ValueError(
            f"old_str appears {occurrences} times in {path}; "
            "include more surrounding lines to make it unique")
    updated = content.replace(old_str, new_str, 1)
    written = await ws.write_file(path, updated)
    if "error" in written:
        return written
    return {"path": path, "occurrences_replaced": 1,
            "bytes_written": written["bytes_written"]}


__all__ = [
    "DEFAULT_READ_LIMIT", "edit_file", "list_dir", "read_file", "write_file",
]
