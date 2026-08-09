"""Tool registry and the small built-in examples used by the demo."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


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


def get_current_weather(location: str) -> str:
    return f"{location}: 24 degrees Celsius, sunny"


def get_current_time() -> str:
    return "2026-08-06 15:45:00 (mock)"


def create_default_registry() -> ToolRegistry:
    return ToolRegistry([
        ToolSpec("get_current_weather", "Get the current weather for a city or region.",
                 {"type": "object", "properties": {
                     "location": {"type": "string", "description": "City or region name"}},
                  "required": ["location"]}, get_current_weather),
        ToolSpec("get_current_time", "Get the current time.",
                 {"type": "object", "properties": {}}, get_current_time),
    ])
