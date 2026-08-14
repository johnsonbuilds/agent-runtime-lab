"""Harbor environment-backed shell execution."""

from __future__ import annotations

import asyncio
import inspect
import math
import threading
import time
from collections.abc import Awaitable
from queue import Queue
from typing import Any, Protocol


class HarborEnvironment(Protocol):
    def exec(
        self,
        command: str,
        cwd: str | None = None,
        timeout_sec: int | None = None,
    ) -> Awaitable[Any]:
        """Execute a command in the Harbor environment."""


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _run_awaitable(awaitable: Awaitable[Any]) -> Any:
    """Run Harbor's async API from the synchronous tool interface."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    result: Queue[tuple[Any, BaseException | None]] = Queue(maxsize=1)

    def run() -> None:
        try:
            result.put((asyncio.run(awaitable), None))
        except Exception as exc:
            result.put((None, exc))

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    value, error = result.get()
    if error is not None:
        raise error
    return value


class HarborShellExecutor:
    """Adapt a Harbor environment's async ``exec`` method to ShellExecutor."""

    def __init__(self, environment: HarborEnvironment) -> None:
        self.environment = environment

    def execute(
        self,
        command: str,
        cwd: str | None = None,
        timeout: float | None = 30.0,
    ) -> dict[str, Any]:
        started = time.monotonic()
        try:
            timeout_sec = None if timeout is None else max(1, math.ceil(timeout))
            result = self.environment.exec(
                command,
                cwd=cwd,
                timeout_sec=timeout_sec,
            )
            if inspect.isawaitable(result):
                result = _run_awaitable(result)
        except BaseException as exc:
            return {
                "stdout": "",
                "stderr": "",
                "exit_code": None,
                "duration": time.monotonic() - started,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }

        return {
            "stdout": _text(result.stdout),
            "stderr": _text(result.stderr),
            "exit_code": result.return_code,
            "duration": time.monotonic() - started,
        }


__all__ = ["HarborEnvironment", "HarborShellExecutor"]
