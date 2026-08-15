"""OpenAI-compatible model adapter and runtime-neutral chat responses."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Sequence
from typing import Any


class OpenAICompatibleLLM:
    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None, **client_kwargs: Any) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("Install the openai package to use this provider") from exc
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"),
                                  base_url=base_url or os.getenv("OPENAI_BASE_URL"),
                                  **client_kwargs)

    async def chat(self, messages: Sequence[dict[str, Any]],
                   tools: list[dict[str, Any]] | None = None,
                   **request_kwargs: Any) -> dict[str, Any]:
        kwargs = {"model": self.model, "messages": list(messages), **request_kwargs}
        if tools is not None:
            kwargs["tools"] = tools
        response = await self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        calls = [call.model_dump(exclude_none=True) for call in (message.tool_calls or [])]
        return {"content": message.content or "", "tool_calls": calls}

    async def stream(self, messages: Sequence[dict[str, Any]],
                     tools: list[dict[str, Any]] | None = None,
                     **request_kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        """Yield provider-independent content and tool-call deltas."""
        kwargs = {"model": self.model, "messages": list(messages),
                  "stream": True, **request_kwargs}
        if tools is not None:
            kwargs["tools"] = tools
        response = await self.client.chat.completions.create(**kwargs)
        async for chunk in response:
            choice = chunk.choices[0]
            delta = choice.delta
            data: dict[str, Any] = {"content": delta.content or ""}
            calls = []
            for call in (delta.tool_calls or []):
                function = call.function
                calls.append({
                    "index": call.index,
                    "id": call.id,
                    "function": {
                        "name": function.name,
                        "arguments": function.arguments or "",
                    },
                })
            if calls:
                data["tool_calls"] = calls
            if data["content"] or calls:
                yield data

OpenAILLM = OpenAICompatibleLLM
