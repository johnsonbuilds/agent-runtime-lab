"""Shell execution tool with structured, non-throwing command results."""

from __future__ import annotations

from typing import Any

from agent_runtime.execution.base import ShellExecutor
from agent_runtime.execution.local import LocalShellExecutor


async def run_command(command: str, cwd: str | None = None,
                timeout: float | None = 30.0,
                executor: ShellExecutor | None = None,
                default_cwd: str | None = None) -> dict[str, Any]:
    """Run a shell command without raising for command-level failures.

    A non-zero exit code is not an exception. Process startup failures and
    timeouts are represented by the optional structured ``error`` field.
    Duration is measured in seconds.  ``default_cwd`` is the working
    directory used when the call does not pass ``cwd`` — the registry
    binds it to the workspace root so shell commands and file tools
    share one notion of "the workspace".
    """
    shell_executor = executor if executor is not None else LocalShellExecutor()
    return await shell_executor.execute(
        command, cwd if cwd is not None else default_cwd, timeout)


__all__ = ["run_command"]
