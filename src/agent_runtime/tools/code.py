"""The execute_code composite tool: write a script, run it, observe once.

This is the "code interpreter" execution pattern: instead of N round
trips (write file, run command, read output), the model ships one
complete script per tool call.  Intermediate data stays in workspace
files or in-memory variables; only what the script prints comes back.

Observation policy: stdout is returned in full when small; when it
exceeds the limit the result carries a head/tail window and the full
text is spilled to ``.outputs/`` where it can be paged with
``read_file``.  stderr and exit_code are never truncated.
"""

from __future__ import annotations

import os
import re
import shlex
from pathlib import PurePosixPath
from typing import Any

from agent_runtime.execution.base import ShellExecutor, Workspace
from agent_runtime.execution.local import LocalShellExecutor, LocalWorkspace


LANGUAGES = {"python": "py", "bash": "sh"}
SCRIPTS_DIR = ".scripts"
OUTPUTS_DIR = ".outputs"
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_OUTPUT_CHARS = 16_000
OUTPUT_WINDOW_CHARS = 4_000

_SEQUENCE = re.compile(r"^(\d{4})\.")


def _python_binary() -> str:
    return os.getenv("AGENT_RUNTIME_PYTHON", "python3")


def _max_output_chars() -> int:
    raw = os.getenv("AGENT_RUNTIME_MAX_OUTPUT_CHARS")
    try:
        value = int(raw) if raw else DEFAULT_MAX_OUTPUT_CHARS
    except ValueError:
        return DEFAULT_MAX_OUTPUT_CHARS
    return value if value > 0 else DEFAULT_MAX_OUTPUT_CHARS


async def _next_script_name(workspace: Workspace, extension: str) -> str:
    """Continue the .scripts/NNNN.ext sequence across all languages."""
    listing = await workspace.list_dir(SCRIPTS_DIR)
    highest = 0
    for entry in listing.get("entries", []):
        match = _SEQUENCE.match(entry.get("name", ""))
        if match:
            highest = max(highest, int(match.group(1)))
    return f"{SCRIPTS_DIR}/{highest + 1:04d}.{extension}"


def _run_command_for(language: str, script: str) -> str:
    if language == "python":
        return f"{_python_binary()} {shlex.quote(script)}"
    return f"bash {shlex.quote(script)}"


async def execute_code(code: str, language: str = "python",
                       path: str | None = None, timeout: float | None = DEFAULT_TIMEOUT,
                       *, workspace: Workspace | None = None,
                       executor: ShellExecutor | None = None) -> dict[str, Any]:
    """Write ``code`` to the workspace, execute it, return one result.

    Composes the Workspace and ShellExecutor protocols: the script is
    saved (default ``.scripts/NNNN.ext``, re-runnable later via
    ``run_command``), executed from the workspace root, and the command
    result is returned with ``script_path`` and ``language`` attached.
    """
    ws = workspace or LocalWorkspace()
    ex = executor or LocalShellExecutor()
    if language not in LANGUAGES:
        raise ValueError(f"unsupported language: {language!r} "
                         f"(choose from {sorted(LANGUAGES)})")

    script = path or await _next_script_name(ws, LANGUAGES[language])
    written = await ws.write_file(script, code)
    if "error" in written:
        return {**written, "script_path": script, "language": language}

    cwd = str(ws.root) if ws.root is not None else None
    result = await ex.execute(_run_command_for(language, script), cwd=cwd,
                              timeout=timeout)

    stdout: str = result.get("stdout") or ""
    if len(stdout) > _max_output_chars():
        head, tail = stdout[:OUTPUT_WINDOW_CHARS], stdout[-OUTPUT_WINDOW_CHARS:]
        omitted = len(stdout) - len(head) - len(tail)
        spill = f"{OUTPUTS_DIR}/{PurePosixPath(script).stem}.stdout.txt"
        saved = await ws.write_file(spill, stdout)
        if "error" not in saved:
            marker = (f"\n... [{omitted} characters omitted; full output saved "
                      f"to {spill}, page through it with read_file] ...\n")
            result["stdout_full_path"] = spill
        else:
            marker = f"\n... [{omitted} characters omitted] ...\n"
        result["stdout"] = head + marker + tail
        result["stdout_chars"] = len(stdout)

    return {**result, "script_path": script, "language": language}


__all__ = [
    "DEFAULT_MAX_OUTPUT_CHARS", "DEFAULT_TIMEOUT", "LANGUAGES", "OUTPUTS_DIR",
    "SCRIPTS_DIR", "execute_code",
]
