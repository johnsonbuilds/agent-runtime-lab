"""The minimal tool-calling loop, independent of providers and tools."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Protocol


class ChatModel(Protocol):
    def chat(self, messages: list[dict[str, Any]],
             tools: list[dict[str, Any]] | None = None) -> dict[str, Any]: ...


class ToolExecutor(Protocol):
    @property
    def schemas(self) -> list[dict[str, Any]]: ...

    def execute(self, name: str, arguments: Mapping[str, Any]) -> Any: ...


def run_agent_loop(user_message: str, llm: ChatModel, tools: ToolExecutor,
                   max_iterations: int = 10) -> str:
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
    for _ in range(max_iterations):
        response = llm.chat(messages, tools.schemas)
        tool_calls = response.get("tool_calls") or []
        messages.append({"role": "assistant", "content": response.get("content") or "",
                         "tool_calls": tool_calls})
        if not tool_calls:
            return response.get("content", "")
        for tool_call in tool_calls:
            function = tool_call["function"]
            arguments = json.loads(function.get("arguments") or "{}")
            result = tools.execute(function["name"], arguments)
            messages.append({"role": "tool", "tool_call_id": tool_call["id"],
                             "content": str(result)})

    messages.append({"role": "user", "content":
                     "Iteration limit reached. Summarize the progress and give the "
                     "best possible final answer. Do not call tools."})
    return llm.chat(messages).get("content", "")
