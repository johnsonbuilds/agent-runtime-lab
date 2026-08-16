"""The minimal tool-calling loop, independent of providers and tools."""

from __future__ import annotations

import json
import asyncio
import logging
import os
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from agent_runtime.trace import RunTrace


logger = logging.getLogger(__name__)


def _stream_idle_timeout() -> float | None:
    raw = os.getenv("AGENT_RUNTIME_STREAM_IDLE_TIMEOUT", "120")
    try:
        timeout = float(raw)
    except ValueError:
        return 120.0
    return timeout if timeout > 0 else None


class ChatModel(Protocol):
    async def chat(self, messages: list[dict[str, Any]],
                    tools: list[dict[str, Any]] | None = None) -> dict[str, Any]: ...

    def stream(self, messages: list[dict[str, Any]],
               tools: list[dict[str, Any]] | None = None) -> AsyncIterator[dict[str, Any]]: ...


class ToolExecutor(Protocol):
    @property
    def schemas(self) -> list[dict[str, Any]]: ...

    async def execute(self, name: str, arguments: Mapping[str, Any]) -> Any: ...


class Conversation:
    """Conversation state shared by multiple agent turns."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def append(self, message: dict[str, Any]) -> None:
        self.messages.append(message)


@dataclass(frozen=True)
class ValidatedToolCall:
    """A tool call that is safe to pass to the executor."""

    id: str
    name: str
    arguments: dict[str, Any]


def _schema_for_tool(schemas: list[dict[str, Any]], name: str) -> Mapping[str, Any] | None:
    for schema in schemas:
        function = schema.get("function")
        if isinstance(function, Mapping) and function.get("name") == name:
            return function
    return None


def _matches_type(value: Any, expected: str) -> bool:
    # bool is a subclass of int, so it needs to be checked separately.
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }.get(expected, True)


def validate_tool_call(tool_call: Any, schemas: list[dict[str, Any]]) -> ValidatedToolCall:
    """Validate and normalize an untrusted LLM tool call."""
    if not isinstance(tool_call, Mapping):
        raise ValueError("tool call must be an object")

    call_id = tool_call.get("id")
    if not isinstance(call_id, str) or not call_id:
        raise ValueError("tool_call_id is required")

    function = tool_call.get("function")
    if not isinstance(function, Mapping):
        raise ValueError("function is required")

    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("tool name is required")
    schema = _schema_for_tool(schemas, name)
    if schema is None:
        raise ValueError(f"unknown tool: {name}")

    raw_arguments = function.get("arguments") or "{}"
    if not isinstance(raw_arguments, str):
        raise ValueError("tool arguments must be a JSON string")
    try:
        arguments = json.loads(raw_arguments)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("tool arguments are malformed JSON") from exc
    if not isinstance(arguments, dict):
        raise ValueError("tool arguments must be a JSON object")

    parameters = schema.get("parameters")
    if not isinstance(parameters, Mapping):
        parameters = {}
    properties = parameters.get("properties") or {}
    if not isinstance(properties, Mapping):
        properties = {}
    required = parameters.get("required") or []
    for field in required:
        if field not in arguments:
            raise ValueError(f"missing required argument: {field}")
    for field, value in arguments.items():
        definition = properties.get(field)
        if not isinstance(definition, Mapping):
            continue
        expected = definition.get("type")
        if isinstance(expected, str) and not _matches_type(value, expected):
            raise ValueError(f"argument '{field}' must be {expected}")

    return ValidatedToolCall(call_id, name, arguments)


def _append_tool_observation(messages: list[dict[str, Any]], tool_call: Any,
                             content: str) -> None:
    call_id = tool_call.get("id", "") if isinstance(tool_call, Mapping) else ""
    if isinstance(call_id, str) and call_id:
        messages.append({"role": "tool", "tool_call_id": call_id, "content": content})
    else:
        # The model must retry because a tool message cannot be correlated without an ID.
        messages.append({
            "role": "user",
            "content": f"{content} Please retry this tool call with a valid tool_call_id.",
        })


def _tool_trace_metadata(tool_call: Any) -> dict[str, Any]:
    """Return identifiers without copying the model's full tool payload."""
    if not isinstance(tool_call, Mapping):
        return {}
    function = tool_call.get("function")
    return {
        "tool_call_id": tool_call.get("id"),
        "tool": function.get("name") if isinstance(function, Mapping) else None,
    }


class _TurnFailure(Exception):
    """An expected turn failure that should be returned to the caller."""


def _merge_stream_tool_call(calls: list[dict[str, Any]], delta: Mapping[str, Any]) -> None:
    index = delta.get("index", len(calls))
    if not isinstance(index, int):
        index = len(calls)
    while len(calls) <= index:
        calls.append({"id": "", "type": "function",
                      "function": {"name": "", "arguments": ""}})
    target = calls[index]
    if isinstance(delta.get("id"), str):
        target["id"] = delta["id"]
    function = delta.get("function")
    if isinstance(function, Mapping):
        target_function = target["function"]
        if isinstance(function.get("name"), str):
            target_function["name"] += function["name"]
        if isinstance(function.get("arguments"), str):
            target_function["arguments"] += function["arguments"]


async def _chat_response(llm: ChatModel, messages: list[dict[str, Any]],
                        tools: list[dict[str, Any]] | None,
                        trace: RunTrace, iteration: int,
                        use_stream: bool) -> dict[str, Any]:
    """Consume either runtime streaming or a single runtime response."""
    if not use_stream:
        chat = llm.chat
        return await chat(messages, tools)

    content: list[str] = []
    reasoning: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    logger.debug("llm.stream.start iteration=%d", iteration)
    stream_iterator = llm.stream(messages, tools).__aiter__()
    while True:
        try:
            timeout = _stream_idle_timeout()
            if timeout is None:
                chunk = await anext(stream_iterator)
            else:
                async with asyncio.timeout(timeout):
                    chunk = await anext(stream_iterator)
        except StopAsyncIteration:
            break
        except TimeoutError as exc:
            logger.error("llm.stream.timeout iteration=%d timeout_seconds=%s",
                         iteration, timeout)
            raise RuntimeError(
                f"LLM stream produced no chunk for {timeout:g} seconds"
            ) from exc
        if not isinstance(chunk, Mapping):
            raise TypeError("LLM stream chunks must be objects")
        text = chunk.get("content") or ""
        if text:
            content.append(str(text))
        reasoning_text = chunk.get("reasoning_content") or ""
        if reasoning_text:
            reasoning.append(str(reasoning_text))
        for delta in chunk.get("tool_calls") or []:
            if not isinstance(delta, Mapping):
                raise TypeError("LLM tool call chunks must be objects")
            _merge_stream_tool_call(tool_calls, delta)
        logger.debug("llm.chunk iteration=%d content=%r reasoning=%r tool_calls=%r",
                     iteration, text, reasoning_text, chunk.get("tool_calls") or [])
        trace.emit("llm.chunk", iteration,
                   content=text, tool_call_count=len(chunk.get("tool_calls") or []),
                   reasoning_chars=len(reasoning_text))
    logger.debug("llm.stream.end iteration=%d", iteration)
    response = {"content": "".join(content), "tool_calls": tool_calls}
    if reasoning:
        response["reasoning_content"] = "".join(reasoning)
    return response


async def _run_turn(user_message: str, llm: ChatModel, tools: ToolExecutor,
              max_iterations: int, conversation: Conversation | None,
              trace: RunTrace, agent_meta: dict[str, Any],
              use_stream: bool) -> str:
    conversation = conversation or Conversation()
    conversation.append({"role": "user", "content": user_message})
    messages = conversation.messages

    for iteration in range(1, max_iterations + 1):
        try:
            logger.debug(
                "llm.request.start iteration=%d messages=%d tools=%d stream=%s",
                iteration, len(messages), len(tools.schemas), use_stream)
            with trace.span("llm", iteration,
                            message_count=len(messages),
                            tool_count=len(tools.schemas)) as span_meta:
                response = await _chat_response(llm, messages, tools.schemas, trace,
                                                iteration, use_stream)
                tool_calls = response.get("tool_calls") or []
                content = response.get("content") or ""

                span_meta.update(
                    tool_count=len(tool_calls),
                    tools=[_tool_trace_metadata(call).get("tool") for call in tool_calls],
                    final=not tool_calls,
                )
                logger.debug(
                    "llm.request.end iteration=%d content_chars=%d tool_calls=%s",
                    iteration, len(content),
                    [_tool_trace_metadata(call).get("tool") for call in tool_calls])
        except Exception as exc:
            logger.error("llm.request.error iteration=%d error=%s", iteration, exc)
            error = f"LLM error: {exc}"
            agent_meta.update(stage="llm", error=error)
            raise _TurnFailure(error) from exc

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": response.get("content") or "",
        }
        reasoning_content = response.get("reasoning_content")
        if reasoning_content:
            assistant_message["reasoning_content"] = reasoning_content
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        messages.append(assistant_message)

        if not tool_calls:
            answer = response.get("content", "")
            logger.debug("agent.final iteration=%d answer_chars=%d", iteration, len(answer))
            return answer
        for tool_call in tool_calls:
            tool_meta = _tool_trace_metadata(tool_call)
            logger.debug("tool.dispatch iteration=%d tool=%r tool_call_id=%r",
                         iteration, tool_meta.get("tool"), tool_meta.get("tool_call_id"))
            try:
                with trace.span("tool", iteration, **tool_meta) as span_meta:
                    validated = validate_tool_call(tool_call, tools.schemas)
                    result = await tools.execute(validated.name, validated.arguments)
                    span_meta.update(tool=validated.name, tool_call_id=validated.id)
                logger.debug(
                    "tool.result iteration=%d tool=%r result_chars=%d",
                    iteration, validated.name, len(str(result)))
            except Exception as exc:
                logger.error("tool.error iteration=%d tool=%r error=%s",
                             iteration, tool_meta.get("tool"), exc)
                _append_tool_observation(messages, tool_call, f"Tool error: {exc}")
                continue
            _append_tool_observation(messages, tool_call, str(result))

    messages.append({"role": "user", "content":
                     "Iteration limit reached. Summarize the progress and give the "
                     "best possible final answer. Do not call tools."})
    iteration = max_iterations + 1
    try:
        logger.debug("llm.request.start iteration=%d messages=%d tools=0 stream=%s",
                     iteration, len(messages), use_stream)
        with trace.span("llm", iteration,
                        message_count=len(messages), tool_count=0) as span_meta:
            response = await _chat_response(llm, messages, None, trace, iteration, use_stream)
            span_meta.update(tool_count=0, tools=[], final=True)
            logger.debug("llm.request.end iteration=%d content_chars=%d tool_calls=[]",
                         iteration, len(response.get("content") or ""))
    except Exception as exc:
        logger.error("llm.request.error iteration=%d error=%s", iteration, exc)
        error = f"LLM error: {exc}"
        agent_meta.update(stage="llm", error=error)
        raise _TurnFailure(error) from exc
    answer = response.get("content", "")
    logger.debug("agent.final iteration=%d answer_chars=%d", iteration, len(answer))
    return answer


async def run_turn(user_message: str, llm: ChatModel, tools: ToolExecutor,
              max_iterations: int = 10,
              *, conversation: Conversation | None = None,
              stream: bool = False,
              trace: RunTrace | None = None) -> str:
    trace = trace or RunTrace()
    try:
        with trace.span("agent") as agent_meta:
            return await _run_turn(user_message, llm, tools, max_iterations,
                                   conversation, trace, agent_meta, stream)
    except _TurnFailure as exc:
        return str(exc)
    finally:
        await trace.flush()
