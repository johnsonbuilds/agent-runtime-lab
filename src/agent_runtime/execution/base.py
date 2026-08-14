"""Interfaces for executing commands in an environment."""

from __future__ import annotations

from typing import Any, Protocol


class ShellExecutor(Protocol):
    def execute(
        self,
        command: str,
        cwd: str | None = None,
        timeout: float | None = 30.0,
    ) -> dict[str, Any]:
        """Execute a shell command and return its structured result."""
