"""Tool registry and the small built-in examples used by the demo."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

from agent_runtime.execution.base import ShellExecutor
from agent_runtime.execution.local import LocalShellExecutor

from .shell import run_command


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]

    @property
    def schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": self.parameters,
        }}


class ToolRegistry:
    def __init__(self, specs: list[ToolSpec] | None = None) -> None:
        self._tools = {spec.name: spec for spec in specs or []}

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [spec.schema for spec in self._tools.values()]

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def execute(self, name: str, arguments: Mapping[str, Any]) -> Any:
        try:
            spec = self._tools[name]
        except KeyError as exc:
            raise ValueError(f"Unknown tool: {name}") from exc
        return spec.handler(**dict(arguments))

def create_default_registry(executor: ShellExecutor | None = None) -> ToolRegistry:
    shell_executor = executor if executor is not None else LocalShellExecutor()
    return ToolRegistry([
        ToolSpec("run_command", "Run a shell command and return its output and exit code.",
                 {"type": "object", "properties": {
                     "command": {"type": "string", "description": "Shell command to run"},
                     "cwd": {"type": "string",
                             "description": "Working directory for the command"},
                     "timeout": {"type": "number", "description": "Timeout in seconds",
                                 "default": 30}},
                    "required": ["command"]},
                 partial(run_command, executor=shell_executor)),
     ])
