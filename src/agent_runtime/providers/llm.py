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
        self.model = model or os.getenv("MODEL_ID", "glm-5.3")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", 0.0))
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("LLM_API_KEY"),
                                  base_url=base_url or os.getenv("LLM_BASE_URL"),
                                  **client_kwargs)

    async def chat(self, messages: Sequence[dict[str, Any]],
                   tools: list[dict[str, Any]] | None = None,
                   **request_kwargs: Any) -> dict[str, Any]:
        kwargs = {"model": self.model, "messages": list(messages),
                  "temperature": self.temperature, **request_kwargs}
        if tools is not None:
            kwargs["tools"] = tools
        response = await self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        calls = [call.model_dump(exclude_none=True) for call in (message.tool_calls or [])]
        result = {"content": message.content or "", "tool_calls": calls}
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning:
            result["reasoning_content"] = reasoning
        return result

    async def stream(self, messages: Sequence[dict[str, Any]],
                     tools: list[dict[str, Any]] | None = None,
                     **request_kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        """Yield provider-independent content and tool-call deltas."""
        kwargs = {"model": self.model, "messages": list(messages),
                  "stream": True, "temperature": self.temperature, **request_kwargs}
        if tools is not None:
            kwargs["tools"] = tools
        response = await self.client.chat.completions.create(**kwargs)
        async for chunk in response:
            if not chunk.choices:
                # Some providers append a final chunk with empty choices
                # (e.g. a usage summary); it carries no delta payload.
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            reasoning = getattr(delta, "reasoning_content", None) or ""
            data: dict[str, Any] = {"content": delta.content or "",
                                    "reasoning_content": reasoning}
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
            if data["content"] or calls or reasoning:
                yield data

OpenAILLM = OpenAICompatibleLLM
