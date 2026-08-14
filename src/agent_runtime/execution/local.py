"""Local subprocess-backed shell execution."""

from __future__ import annotations

import asyncio
import subprocess
import time
from typing import Any


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _error_result(started: float, exc: BaseException,
                  stdout: Any = "", stderr: Any = "") -> dict[str, Any]:
    return {
        "stdout": _text(stdout),
        "stderr": _text(stderr),
        "exit_code": None,
        "duration": time.monotonic() - started,
        "error": {"type": type(exc).__name__, "message": str(exc)},
    }


class LocalShellExecutor:
    """Execute shell commands in the current Python process environment."""

    async def execute(
        self,
        command: str,
        cwd: str | None = None,
        timeout: float | None = 30.0,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self._execute, command, cwd, timeout)

    def _execute(
        self,
        command: str,
        cwd: str | None = None,
        timeout: float | None = 30.0,
    ) -> dict[str, Any]:
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                timeout=timeout,
                capture_output=True,
                text=True,
                shell=True,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return _error_result(started, exc, exc.stdout, exc.stderr)
        except (OSError, ValueError, TypeError) as exc:
            return _error_result(started, exc)

        return {
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "exit_code": completed.returncode,
            "duration": time.monotonic() - started,
        }


__all__ = ["LocalShellExecutor"]
