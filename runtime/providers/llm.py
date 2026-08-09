"""OpenAI-compatible model adapter."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
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

    async def achat(self, messages: Sequence[dict[str, Any]],
                    tools: list[dict[str, Any]] | None = None,
                    **request_kwargs: Any) -> dict[str, Any]:
        kwargs = {"model": self.model, "messages": list(messages), **request_kwargs}
        if tools is not None:
            kwargs["tools"] = tools
        response = await self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        calls = [call.model_dump(exclude_none=True) for call in (message.tool_calls or [])]
        return {"content": message.content or "", "tool_calls": calls}

    def chat(self, messages: Sequence[dict[str, Any]],
             tools: list[dict[str, Any]] | None = None,
             **request_kwargs: Any) -> dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.achat(messages, tools, **request_kwargs))
        raise RuntimeError("chat() cannot run inside an event loop; use achat()")


OpenAILLM = OpenAICompatibleLLM
