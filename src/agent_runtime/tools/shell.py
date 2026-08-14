"""Shell execution tool with structured, non-throwing command results."""

from __future__ import annotations

from typing import Any

from agent_runtime.execution.base import ShellExecutor
from agent_runtime.execution.local import LocalShellExecutor


async def run_command(command: str, cwd: str | None = None,
                timeout: float | None = 30.0,
                executor: ShellExecutor | None = None) -> dict[str, Any]:
    """Run a shell command without raising for command-level failures.

    A non-zero exit code is not an exception. Process startup failures and
    timeouts are represented by the optional structured ``error`` field.
    Duration is measured in seconds.
    """
    shell_executor = executor if executor is not None else LocalShellExecutor()
    return await shell_executor.execute(command, cwd, timeout)


__all__ = ["run_command"]
