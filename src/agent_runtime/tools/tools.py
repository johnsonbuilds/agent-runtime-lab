"""Tool registry and the built-in tools used by the runtime."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from functools import partial
import inspect
from typing import Any

from agent_runtime.execution.base import ShellExecutor, Workspace
from agent_runtime.execution.local import LocalShellExecutor, LocalWorkspace

from .code import execute_code
from .files import DEFAULT_READ_LIMIT, edit_file, list_dir, read_file, write_file
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


def _write_file_spec(workspace: Workspace) -> ToolSpec:
    return ToolSpec("write_file",
                    "Create or overwrite a file with the given full content. "
                    "Parent directories are created automatically.",
                    {"type": "object", "properties": {
                        "path": {"type": "string",
                                 "description": "File path inside the workspace"},
                        "content": {"type": "string",
                                    "description": "Complete file content"}},
                     "required": ["path", "content"]},
                    partial(write_file, workspace=workspace))


def _read_file_spec(workspace: Workspace) -> ToolSpec:
    return ToolSpec("read_file",
                    "Read a file as text. Long files are returned one page of "
                    "lines at a time; continue with a larger offset when the "
                    "result reports truncated: true.",
                    {"type": "object", "properties": {
                        "path": {"type": "string",
                                 "description": "File path inside the workspace"},
                        "offset": {"type": "integer",
                                   "description": "First line to return (1-based)",
                                   "default": 1},
                        "limit": {"type": "integer",
                                  "description": "Maximum number of lines to return",
                                  "default": DEFAULT_READ_LIMIT}},
                     "required": ["path"]},
                    partial(read_file, workspace=workspace))


def _list_dir_spec(workspace: Workspace) -> ToolSpec:
    return ToolSpec("list_dir",
                    "List the entries of a directory (directories first, "
                    "then files).",
                    {"type": "object", "properties": {
                        "path": {"type": "string",
                                 "description": "Directory path inside the workspace",
                                 "default": "."}},
                     "required": []},
                    partial(list_dir, workspace=workspace))


def _edit_file_spec(workspace: Workspace) -> ToolSpec:
    return ToolSpec("edit_file",
                    "Replace exactly one occurrence of old_str with new_str in "
                    "a file. old_str must match exactly once; include more "
                    "surrounding lines when it is ambiguous.",
                    {"type": "object", "properties": {
                        "path": {"type": "string",
                                 "description": "File path inside the workspace"},
                        "old_str": {"type": "string",
                                    "description": "Exact text to replace"},
                        "new_str": {"type": "string",
                                    "description": "Replacement text (empty to delete)"}},
                     "required": ["path", "old_str", "new_str"]},
                    partial(edit_file, workspace=workspace))


def _execute_code_spec(executor: ShellExecutor, workspace: Workspace) -> ToolSpec:
    return ToolSpec("execute_code",
                    "Write a complete script to the workspace and execute it "
                    "in one step, returning exit code, stdout, and stderr. "
                    "Prefer this over run_command whenever the step needs "
                    "loops, branches, multi-step data processing, or error "
                    "handling: keep intermediate data in files or variables "
                    "and print only the final result. Each script is saved "
                    "under .scripts/ and can be re-run with run_command after "
                    "fixing the environment.",
                    {"type": "object", "properties": {
                        "code": {"type": "string",
                                 "description": "Complete script content"},
                        "language": {"type": "string", "enum": ["python", "bash"],
                                     "description": "Script language",
                                     "default": "python"},
                        "path": {"type": "string",
                                 "description": "Script path inside the workspace; "
                                                "defaults to .scripts/NNNN.ext"},
                        "timeout": {"type": "number",
                                    "description": "Timeout in seconds",
                                    "default": 120}},
                     "required": ["code"]},
                    partial(execute_code, workspace=workspace, executor=executor))


def builtin_tool_specs(executor: ShellExecutor | None = None,
                       workspace: Workspace | None = None) -> list[ToolSpec]:
    """Every tool the runtime knows how to build."""
    shell_executor = executor if executor is not None else LocalShellExecutor()
    file_workspace = workspace if workspace is not None else LocalWorkspace()
    return [
        _run_command_spec(shell_executor),
        _write_file_spec(file_workspace),
        _read_file_spec(file_workspace),
        _list_dir_spec(file_workspace),
        _edit_file_spec(file_workspace),
        _execute_code_spec(shell_executor, file_workspace),
    ]


def create_default_registry(executor: ShellExecutor | None = None,
                            enabled: list[str] | None = None,
                            workspace: Workspace | None = None) -> ToolRegistry:
    """Build the tool registry, optionally filtered by the harness gene.

    ``enabled=None`` keeps every built-in tool; otherwise the registry
    exposes exactly the named tools, in the given order.
    """
    specs = builtin_tool_specs(executor, workspace)
    if enabled is None:
        return ToolRegistry(specs)
    by_name = {spec.name: spec for spec in specs}
    unknown = [name for name in enabled if name not in by_name]
    if unknown:
        raise ValueError(f"Unknown tools in harness: {unknown}")
    return ToolRegistry([by_name[name] for name in enabled])
