"""OpenAI-compatible chat-completion client used by the agent loop.

The adapter keeps the small dictionary-based interface used by ``agent_loop``
while using the official asynchronous OpenAI SDK underneath.  It works with
OpenAI and compatible providers by setting ``OPENAI_BASE_URL``.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from typing import Any


class OpenAICompatibleLLM:
    """Call an OpenAI-compatible provider through ``AsyncOpenAI``.

    ``api_key``, ``base_url`` and ``model`` default to the corresponding
    environment variables: ``OPENAI_API_KEY``, ``OPENAI_BASE_URL`` and
    ``OPENAI_MODEL``.  ``OPENAI_MODEL`` defaults to ``gpt-4o-mini``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        **client_kwargs: Any,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required. Install it with: pip install openai"
            ) from exc

        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            **client_kwargs,
        )

    async def achat(
        self,
        messages: Sequence[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **request_kwargs: Any,
    ) -> dict[str, Any]:
        """Send one chat-completion request and normalize its response."""
        kwargs = {"model": self.model, "messages": list(messages), **request_kwargs}
        if tools is not None:
            kwargs["tools"] = tools

        response = await self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        tool_calls = []
        for tool_call in message.tool_calls or []:
            tool_calls.append(
                {
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                }
            )

        return {"content": message.content or "", "tool_calls": tool_calls}

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **request_kwargs: Any,
    ) -> dict[str, Any]:
        """Synchronous bridge for the existing synchronous agent loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.achat(messages, tools, **request_kwargs))

        raise RuntimeError(
            "chat() cannot be called from a running event loop; use await achat() instead"
        )


OpenAILLM = OpenAICompatibleLLM
