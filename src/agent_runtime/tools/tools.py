"""Tool registry and the small built-in examples used by the demo."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from functools import partial
import inspect
from typing import Any

from agent_runtime.execution.base import ShellExecutor
from agent_runtime.execution.local import LocalShellExecutor

from .shell import run_command


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Awaitable[Any]]

    @property
    def schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": self.parameters,
        }}


class ToolRegistry:
    def __init__(self, specs: list[ToolSpec] | None = None) -> None:
        self._tools: dict[str, ToolSpec] = {}
        for spec in specs or []:
            self.register(spec)

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [spec.schema for spec in self._tools.values()]

    def register(self, spec: ToolSpec) -> None:
        if not inspect.iscoroutinefunction(spec.handler):
            raise TypeError(f"Tool handler must be async: {spec.name}")
        self._tools[spec.name] = spec

    async def execute(self, name: str, arguments: Mapping[str, Any]) -> Any:
        try:
            spec = self._tools[name]
        except KeyError as exc:
            raise ValueError(f"Unknown tool: {name}") from exc
        return await spec.handler(**dict(arguments))


def _run_command_spec(executor: ShellExecutor) -> ToolSpec:
    return ToolSpec("run_command", "Run a shell command and return its output and exit code.",
                    {"type": "object", "properties": {
                        "command": {"type": "string", "description": "Shell command to run"},
                        "cwd": {"type": "string",
                                "description": "Working directory for the command"},
                        "timeout": {"type": "number", "description": "Timeout in seconds",
                                    "default": 30}},
                     "required": ["command"]},
                    partial(run_command, executor=executor))


def builtin_tool_specs(executor: ShellExecutor) -> list[ToolSpec]:
    """Every tool the runtime knows how to build."""
    return [_run_command_spec(executor)]


def create_default_registry(executor: ShellExecutor | None = None,
                            enabled: list[str] | None = None) -> ToolRegistry:
    """Build the tool registry, optionally filtered by the harness gene.

    ``enabled=None`` keeps every built-in tool; otherwise the registry
    exposes exactly the named tools, in the given order.
    """
    shell_executor = executor if executor is not None else LocalShellExecutor()
    specs = builtin_tool_specs(shell_executor)
    if enabled is None:
        return ToolRegistry(specs)
    by_name = {spec.name: spec for spec in specs}
    unknown = [name for name in enabled if name not in by_name]
    if unknown:
        raise ValueError(f"Unknown tools in harness: {unknown}")
    return ToolRegistry([by_name[name] for name in enabled])
