"""Local subprocess-backed shell execution and file workspace."""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from .base import paginate_lines


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


__all__ = ["LocalShellExecutor", "LocalWorkspace"]


def _default_workspace_root() -> Path:
    raw = os.getenv("AGENT_RUNTIME_WORKSPACE")
    return Path(raw).expanduser().resolve() if raw else Path.cwd()


def _structured_error(exc: BaseException) -> dict[str, Any]:
    return {"error": {"type": type(exc).__name__, "message": str(exc)}}


class LocalWorkspace:
    """Workspace backed by the local filesystem, rooted and contained.

    The root defaults to ``AGENT_RUNTIME_WORKSPACE`` or the current
    working directory.  Every path is resolved before use and must stay
    inside the root, so ``../`` escapes and absolute paths pointing
    elsewhere are rejected with ``ValueError``.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        base = Path(root).expanduser() if root is not None else _default_workspace_root()
        self.root: Path = base.resolve()

    def _resolve(self, path: str) -> Path:
        candidate = Path(path)
        resolved = (candidate.resolve() if candidate.is_absolute()
                    else (self.root / candidate).resolve())
        if not resolved.is_relative_to(self.root):
            raise ValueError(f"path escapes the workspace: {path}")
        return resolved

    async def read_file(self, path: str, offset: int = 1,
                        limit: int | None = None) -> dict[str, Any]:
        target = self._resolve(path)
        return await asyncio.to_thread(self._read_file, path, target, offset, limit)

    def _read_file(self, display_path: str, target: Path, offset: int,
                   limit: int | None) -> dict[str, Any]:
        try:
            text = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return {"path": display_path, **_structured_error(exc)}
        return {"path": display_path, **paginate_lines(text, offset, limit)}

    async def write_file(self, path: str, content: str) -> dict[str, Any]:
        target = self._resolve(path)
        return await asyncio.to_thread(self._write_file, path, target, content)

    def _write_file(self, display_path: str, target: Path,
                    content: str) -> dict[str, Any]:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as exc:
            return {"path": display_path, **_structured_error(exc)}
        return {"path": display_path, "bytes_written": len(content.encode("utf-8"))}

    async def list_dir(self, path: str = ".") -> dict[str, Any]:
        target = self._resolve(path)
        return await asyncio.to_thread(self._list_dir, path, target)

    def _list_dir(self, display_path: str, target: Path) -> dict[str, Any]:
        try:
            items = list(target.iterdir())
        except OSError as exc:
            return {"path": display_path, **_structured_error(exc)}
        entries: list[dict[str, Any]] = []
        for item in items:
            try:
                is_dir = item.is_dir()
                size = item.stat().st_size
            except OSError:
                continue  # broken symlinks and other unreadable entries
            entries.append({"name": item.name,
                            "type": "dir" if is_dir else "file",
                            "size": size})
        entries.sort(key=lambda entry: (entry["type"] != "dir", entry["name"].lower()))
        return {"path": display_path, "entries": entries}
