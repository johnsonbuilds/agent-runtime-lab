"""Harbor environment-backed shell execution and file workspace."""

from __future__ import annotations

import base64
import binascii
import math
import posixpath
import shlex
import time
from collections.abc import Awaitable
from typing import Any, Protocol

from .base import paginate_lines


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


class HarborShellExecutor:
    """Adapt a Harbor environment's async ``exec`` method to ShellExecutor."""

    def __init__(self, environment: HarborEnvironment) -> None:
        self.environment = environment

    async def execute(
        self,
        command: str,
        cwd: str | None = None,
        timeout: float | None = 30.0,
    ) -> dict[str, Any]:
        started = time.monotonic()
        try:
            timeout_sec = None if timeout is None else max(1, math.ceil(timeout))
            result = await self.environment.exec(
                command,
                cwd=cwd,
                timeout_sec=timeout_sec,
            )
        except Exception as exc:
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


__all__ = ["HarborEnvironment", "HarborShellExecutor", "HarborWorkspace"]


class HarborWorkspace:
    """Workspace backed by a Harbor environment's ``exec``.

    File content crosses the boundary base64-encoded, so no quoting or
    escaping can corrupt it.  The container itself is the sandbox: with
    no root configured, paths are used as-is (relative paths resolve
    against the environment's default working directory).  With a root
    configured, paths are normalized and must stay inside it; symlinks
    cannot be resolved remotely, so containment is purely lexical.

    Listing relies on GNU find's ``-printf`` (present in the Ubuntu
    images used by terminal-bench).
    """

    def __init__(self, environment: HarborEnvironment,
                 root: str | None = None) -> None:
        self.environment = environment
        self.root = root

    def _resolve(self, path: str) -> str:
        if self.root is None:
            return posixpath.normpath(path)
        candidate = path if path.startswith("/") else posixpath.join(self.root, path)
        resolved = posixpath.normpath(candidate)
        root = posixpath.normpath(self.root)
        if resolved != root and not resolved.startswith(root + "/"):
            raise ValueError(f"path escapes the workspace: {path}")
        return resolved

    async def _exec(self, command: str) -> dict[str, Any]:
        """Run a helper command; a non-zero exit becomes a structured error."""
        try:
            result = await self.environment.exec(command, cwd=None, timeout_sec=60)
        except Exception as exc:
            return {"error": {"type": type(exc).__name__, "message": str(exc)}}
        code = result.return_code
        if code != 0:
            message = _text(result.stderr).strip() or f"exit code {code}"
            return {"error": {"type": "CommandError", "message": message}}
        return {"stdout": _text(result.stdout), "stderr": _text(result.stderr)}

    async def read_file(self, path: str, offset: int = 1,
                        limit: int | None = None) -> dict[str, Any]:
        target = self._resolve(path)
        outcome = await self._exec(f"base64 < {shlex.quote(target)}")
        if "error" in outcome:
            return {"path": path, "error": outcome["error"]}
        try:
            text = base64.b64decode(outcome["stdout"]).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError) as exc:
            return {"path": path,
                    "error": {"type": type(exc).__name__, "message": str(exc)}}
        return {"path": path, **paginate_lines(text, offset, limit)}

    async def write_file(self, path: str, content: str) -> dict[str, Any]:
        target = self._resolve(path)
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        parent = posixpath.dirname(target)
        command = (f"mkdir -p {shlex.quote(parent)} && "
                   f"printf %s {shlex.quote(encoded)} | base64 -d > {shlex.quote(target)}")
        outcome = await self._exec(command)
        if "error" in outcome:
            return {"path": path, "error": outcome["error"]}
        return {"path": path, "bytes_written": len(content.encode("utf-8"))}

    async def list_dir(self, path: str = ".") -> dict[str, Any]:
        target = self._resolve(path)
        message = shlex.quote(f"not a directory: {target}")
        command = (f"if test -d {shlex.quote(target)}; then "
                   f"find {shlex.quote(target)} -mindepth 1 -maxdepth 1 "
                   f"-printf '%y\\t%s\\t%f\\n'; "
                   f"else echo {message} >&2; exit 1; fi")
        outcome = await self._exec(command)
        if "error" in outcome:
            return {"path": path, "error": outcome["error"]}
        entries: list[dict[str, Any]] = []
        for line in outcome["stdout"].splitlines():
            parts = line.split("\t", 2)
            if len(parts) != 3 or not parts[1].isdigit():
                continue
            kind, raw_size, name = parts
            entries.append({"name": name,
                            "type": {"f": "file", "d": "dir"}.get(kind, "other"),
                            "size": int(raw_size)})
        entries.sort(key=lambda entry: (entry["type"] != "dir", entry["name"].lower()))
        return {"path": path, "entries": entries}
