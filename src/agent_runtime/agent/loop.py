"""The minimal tool-calling loop, independent of providers and tools.

Tool calls are validated before they enter the conversation history: only
canonical, validated calls are replayed to the provider, so one malformed
tool call cannot poison later requests.
"""

from __future__ import annotations

import json
import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from agent_runtime.agent.memory import apply_memory_strategy
from agent_runtime.events import EventEmitter
from agent_runtime.harness import HarnessSpec, default_harness
from agent_runtime.trace import RunTrace


logger = logging.getLogger(__name__)


def _message_list_chars(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        total += len(message.get("content") or "")
        for call in message.get("tool_calls") or []:
            total += len((call.get("function") or {}).get("arguments") or "")
    return total


def _stream_idle_timeout() -> float | None:
    raw = os.getenv("AGENT_RUNTIME_STREAM_IDLE_TIMEOUT", "120")
    try:
        timeout = float(raw)
    except ValueError:
        return 120.0
    return timeout if timeout > 0 else None


def _reasoning_budget_chars() -> int | None:
    """Max accumulated reasoning chars before a thinking-only turn is cut.

    Guards against reasoning models that spiral into unbounded chain of
    thought without ever emitting content or tool calls (observed on
    ARS-class tasks).  0 or a negative value disables the guard.
    """
    raw = os.getenv("AGENT_RUNTIME_MAX_REASONING_CHARS", "50000")
    try:
        budget = int(raw)
    except ValueError:
        return 50_000
    return budget if budget > 0 else None


class _ReasoningBudgetExceeded(RuntimeError):
    """A thinking-only turn blew the reasoning budget; nudge and retry."""

    def __init__(self, reasoning_chars: int, budget: int) -> None:
        super().__init__(
            f"reasoning exceeded {budget} chars ({reasoning_chars} produced) "
            f"with no content or tool calls yet")
        self.reasoning_chars = reasoning_chars
        self.budget = budget


def use_streaming() -> bool:
    """Whether the runtime should consume LLM streaming (AGENT_RUNTIME_STREAM)."""
    value = os.getenv("AGENT_RUNTIME_STREAM", "1").lower()
    return value not in {"0", "false", "no", "off"}


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


@dataclass(frozen=True)
class ToolCallOutcome:
    """Validation verdict for one untrusted LLM tool call.

    ``validated`` and ``rejection`` are mutually exclusive: a call is
    either safe to execute or carries the reason it was rejected.
    """

    tool_call: Any
    validated: ValidatedToolCall | None
    rejection: Exception | None


def classify_tool_calls(tool_calls: list[Any],
                        schemas: list[dict[str, Any]]) -> list[ToolCallOutcome]:
    """Validate every tool call up front; nothing raises."""
    outcomes: list[ToolCallOutcome] = []
    for tool_call in tool_calls:
        try:
            validated = validate_tool_call(tool_call, schemas)
        except Exception as exc:
            outcomes.append(ToolCallOutcome(tool_call, None, exc))
        else:
            outcomes.append(ToolCallOutcome(tool_call, validated, None))
    return outcomes


def _canonical_tool_call(validated: ValidatedToolCall) -> dict[str, Any]:
    """Serialize a validated call so history only ever holds valid JSON."""
    return {"id": validated.id, "type": "function",
            "function": {"name": validated.name,
                         "arguments": json.dumps(validated.arguments)}}


def canonical_tool_calls(outcomes: list[ToolCallOutcome]) -> list[dict[str, Any]]:
    """The tool calls allowed into the conversation history, canonicalized."""
    return [_canonical_tool_call(outcome.validated)
            for outcome in outcomes if outcome.validated is not None]


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


def _rejected_observation_content(tool_call: Any, observation: str) -> str:
    """Name the rejected tool so the model can retry the corrected call."""
    tool = _tool_call_payload(tool_call).get("tool")
    name = tool if isinstance(tool, str) and tool else "an unknown tool"
    return f"Tool call to {name} was rejected and not executed. {observation}"


def _append_rejected_observation(messages: list[dict[str, Any]], tool_call: Any,
                                 content: str) -> None:
    """Observe a tool call rejected before execution.

    The rejected call never enters the conversation history, so the
    observation must be a user message: a ``role: "tool"`` message would
    reference a tool_call_id that no assistant message carries.
    """
    call_id = tool_call.get("id") if isinstance(tool_call, Mapping) else None
    if isinstance(call_id, str) and call_id:
        retry = "Please retry this tool call with valid JSON arguments."
    else:
        retry = "Please retry this tool call with a valid tool_call_id."
    messages.append({"role": "user", "content": f"{content} {retry}"})


def _trace_tool_rejection(trace: RunTrace, iteration: int,
                          tool_meta: dict[str, Any], error: Exception) -> None:
    """Record a rejected tool call with the normal tool span lifecycle."""
    trace.emit("tool.start", iteration, **tool_meta)
    error_meta = dict(tool_meta, error=str(error), duration_ms=0.0)
    trace.emit("tool.error", iteration, **error_meta)
    end_meta = dict(tool_meta, duration_ms=0.0, status="error")
    trace.emit("tool.end", iteration, **end_meta)


def _ensure_system_prompt(conversation: Conversation, system: str) -> None:
    """Insert the harness system prompt once, before the first user message."""
    if not system:
        return
    if any(message.get("role") == "system" for message in conversation.messages):
        return
    conversation.messages.insert(0, {"role": "system", "content": system})


def _tool_trace_metadata(tool_call: Any) -> dict[str, Any]:
    """Return identifiers without copying the model's full tool payload."""
    if not isinstance(tool_call, Mapping):
        return {}
    function = tool_call.get("function")
    return {
        "tool_call_id": tool_call.get("id"),
        "tool": function.get("name") if isinstance(function, Mapping) else None,
    }


def _tail_text(text: Any, limit: int = 2000) -> str:
    """Return the trailing part of a tool result for user-facing events."""
    value = text if isinstance(text, str) else ("" if text is None else str(text))
    value = value.rstrip()
    if len(value) <= limit:
        return value
    return "..." + value[-limit:]


def _tool_call_payload(tool_call: Any) -> dict[str, Any]:
    """Best-effort user-facing view of an untrusted tool call."""
    if not isinstance(tool_call, Mapping):
        return {"tool": None, "call_id": None, "arguments": str(tool_call)}
    function = tool_call.get("function")
    name = raw_arguments = None
    if isinstance(function, Mapping):
        name = function.get("name")
        raw_arguments = function.get("arguments")
    arguments: Any = raw_arguments
    if isinstance(raw_arguments, str) and raw_arguments:
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            arguments = raw_arguments
    return {"tool": name, "call_id": tool_call.get("id"), "arguments": arguments}


def _tool_result_summary(result: Any) -> dict[str, Any]:
    """Summarize a tool result for user-facing events."""
    if isinstance(result, Mapping) and "exit_code" in result and "stdout" in result:
        summary: dict[str, Any] = {
            "exit_code": result.get("exit_code"),
            "stdout_tail": _tail_text(result.get("stdout")),
            "stderr_tail": _tail_text(result.get("stderr")),
        }
        error = result.get("error")
        if isinstance(error, Mapping):
            summary["error"] = f"{error.get('type', 'Error')}: {error.get('message', '')}"
        return summary
    return {"result": _tail_text(result)}


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


class AgentTurn:
    """One agent turn: collaborators and policy bound as attributes.

    ``run`` drives the tool-calling loop; helpers read ``self`` instead
    of threading a dozen parameters through every call.
    """

    def __init__(self, user_message: str, llm: ChatModel, tools: ToolExecutor,
                 max_iterations: int | None = None, *,
                 harness: HarnessSpec | None = None,
                 conversation: Conversation | None = None,
                 stream: bool = False,
                 trace: RunTrace | None = None,
                 events: EventEmitter | None = None) -> None:
        self.harness = harness or default_harness()
        self.max_iterations = (
            max_iterations if max_iterations is not None
            else self.harness.control.max_iterations)
        self.trace = trace or RunTrace(harness=self.harness)
        self.events = events or EventEmitter(run_id=self.trace.run_id)
        self.user_message = user_message
        self.llm = llm
        self.tools = tools
        self.conversation = conversation or Conversation()
        self.stream = stream

    async def run(self) -> str:
        """Execute the turn and always flush the trace."""
        agent_meta: dict[str, Any] = {}
        try:
            with self.trace.span("agent") as agent_meta:
                return await self._run(agent_meta)
        except _TurnFailure as exc:
            self.events.emit("runtime.error",
                             stage=agent_meta.get("stage", "agent"),
                             error=str(exc))
            return str(exc)
        except Exception as exc:
            self.events.emit("runtime.error",
                             stage=agent_meta.get("stage", "agent"),
                             error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            await self.trace.flush()

    async def _run(self, agent_meta: dict[str, Any]) -> str:
        events, trace = self.events, self.trace
        tools, harness = self.tools, self.harness
        messages = self.conversation.messages

        events.emit("agent.started", message=self.user_message)
        _ensure_system_prompt(self.conversation, harness.prompt.system)
        self.conversation.append({"role": "user", "content": self.user_message})

        for iteration in range(1, self.max_iterations + 1):
            try:
                request_messages = apply_memory_strategy(
                    messages, harness.memory.strategy)
                if request_messages is not messages:
                    trace.emit("memory.compact", iteration,
                               original_chars=_message_list_chars(messages),
                               request_chars=_message_list_chars(
                                   request_messages),
                               original_count=len(messages),
                               request_count=len(request_messages))
                logger.debug(
                    "llm.request.start iteration=%d messages=%d tools=%d stream=%s",
                    iteration, len(messages), len(tools.schemas), self.stream)
                with trace.span("llm", iteration,
                                message_count=len(request_messages),
                                tool_count=len(tools.schemas),
                                messages=list(request_messages)) as span_meta:
                    response = await self._chat_response(
                        request_messages, tools.schemas, iteration)
                    tool_calls = response.get("tool_calls") or []
                    content = response.get("content") or ""

                    span_meta.update(
                        tool_count=len(tool_calls),
                        tools=[_tool_trace_metadata(call).get("tool")
                               for call in tool_calls],
                        final=not tool_calls,
                    )
                    logger.debug(
                        "llm.request.end iteration=%d content_chars=%d tool_calls=%s",
                        iteration, len(content),
                        [_tool_trace_metadata(call).get("tool") for call in tool_calls])
            except _ReasoningBudgetExceeded as exc:
                # The model spiralled into thinking without acting.  Append
                # an assistant placeholder plus a nudge and retry on the
                # next iteration (consuming one iteration of the budget) —
                # resending identical input would just reproduce the spiral.
                logger.warning(
                    "llm.reasoning_budget_exceeded iteration=%d %s", iteration, exc)
                messages.append({"role": "assistant",
                                 "content": "(extended reasoning omitted)"})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your last turn produced {exc.reasoning_chars} characters "
                        f"of reasoning without any content or tool calls. Stop "
                        f"reasoning and act now: call a tool, or give your "
                        f"final answer."),
                })
                continue
            except Exception as exc:
                logger.error("llm.request.error iteration=%d error=%s", iteration, exc)
                error = f"LLM error: {exc}"
                agent_meta.update(stage="llm", error=error)
                raise _TurnFailure(error) from exc

            outcomes = classify_tool_calls(tool_calls, tools.schemas)

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": content,
            }
            reasoning_content = response.get("reasoning_content")
            if reasoning_content:
                assistant_message["reasoning_content"] = reasoning_content
            canonical_calls = canonical_tool_calls(outcomes)
            if canonical_calls:
                assistant_message["tool_calls"] = canonical_calls
            if tool_calls or content or reasoning_content:
                messages.append(assistant_message)

            if not tool_calls:
                answer = response.get("content", "")
                logger.debug("agent.final iteration=%d answer_chars=%d",
                             iteration, len(answer))
                events.emit("agent.completed", iteration, iterations=iteration, answer=answer)
                return answer
            for outcome in outcomes:
                tool_call = outcome.tool_call
                tool_meta = _tool_trace_metadata(tool_call)
                logger.debug("tool.dispatch iteration=%d tool=%r tool_call_id=%r",
                             iteration, tool_meta.get("tool"),
                             tool_meta.get("tool_call_id"))
                payload = _tool_call_payload(tool_call)
                events.emit("tool.started", iteration, call_id=payload["call_id"],
                            tool=payload["tool"], arguments=payload["arguments"])
                tool_started = time.monotonic()
                if outcome.rejection is not None:
                    self._reject_tool_call(outcome, iteration, messages)
                    continue
                validated = outcome.validated
                try:
                    with trace.span("tool", iteration, **tool_meta) as span_meta:
                        result = await tools.execute(validated.name,
                                                     validated.arguments)
                        span_meta.update(tool=validated.name,
                                         tool_call_id=validated.id)
                    logger.debug(
                        "tool.result iteration=%d tool=%r result_chars=%d",
                        iteration, validated.name, len(str(result)))
                    events.emit("tool.completed", iteration, call_id=payload["call_id"],
                                duration=round(time.monotonic() - tool_started, 3),
                                **_tool_result_summary(result))
                except Exception as exc:
                    logger.error("tool.error iteration=%d tool=%r error=%s",
                                 iteration, validated.name, exc)
                    events.emit("tool.failed", iteration, call_id=payload["call_id"],
                                tool=payload["tool"], error=str(exc))
                    observation = harness.tool_error_observation(
                        exc, tool=payload["tool"])
                    _append_tool_observation(messages, tool_call, observation)
                    continue
                _append_tool_observation(messages, tool_call, str(result))

        messages.append({"role": "user", "content":
                         harness.prompt.iteration_limit_notice})
        iteration = self.max_iterations + 1
        try:
            request_messages = apply_memory_strategy(
                messages, harness.memory.strategy)
            if request_messages is not messages:
                trace.emit("memory.compact", iteration,
                           original_chars=_message_list_chars(messages),
                           request_chars=_message_list_chars(
                               request_messages),
                           original_count=len(messages),
                           request_count=len(request_messages))
            logger.debug("llm.request.start iteration=%d messages=%d tools=0 stream=%s",
                         iteration, len(request_messages), self.stream)
            with trace.span("llm", iteration,
                            message_count=len(request_messages), tool_count=0,
                            messages=list(request_messages)) as span_meta:
                response = await self._chat_response(
                    request_messages, None, iteration)
                span_meta.update(tool_count=0, tools=[], final=True)
                logger.debug("llm.request.end iteration=%d content_chars=%d tool_calls=[]",
                             iteration, len(response.get("content") or ""))
        except _ReasoningBudgetExceeded as exc:
            # The forced summary turn has no tools, so nudging makes no
            # sense; surface the partial reasoning as the final answer
            # instead of failing the whole task at the finish line.
            logger.warning("llm.reasoning_budget_exceeded iteration=%d %s",
                           iteration, exc)
            answer = ("(reasoning budget exceeded while summarising; "
                      "the work above stands as delivered)")
            logger.debug("agent.final iteration=%d answer_chars=%d", iteration,
                         len(answer))
            events.emit("agent.completed", iteration, iterations=iteration,
                        answer=answer)
            return answer
        except Exception as exc:
            logger.error("llm.request.error iteration=%d error=%s", iteration, exc)
            error = f"LLM error: {exc}"
            agent_meta.update(stage="llm", error=error)
            raise _TurnFailure(error) from exc
        answer = response.get("content", "")
        logger.debug("agent.final iteration=%d answer_chars=%d", iteration, len(answer))
        events.emit("agent.completed", iteration, iterations=iteration, answer=answer)
        return answer

    def _reject_tool_call(self, outcome: ToolCallOutcome, iteration: int,
                          messages: list[dict[str, Any]]) -> None:
        """Observe a tool call rejected by validation; it never executed."""
        tool_call = outcome.tool_call
        rejection = outcome.rejection
        assert rejection is not None
        payload = _tool_call_payload(tool_call)
        logger.error("tool.error iteration=%d tool=%r error=%s",
                     iteration, payload["tool"], rejection)
        _trace_tool_rejection(self.trace, iteration,
                              _tool_trace_metadata(tool_call), rejection)
        self.events.emit("tool.failed", iteration, call_id=payload["call_id"],
                         tool=payload["tool"], error=str(rejection))
        observation = _rejected_observation_content(
            tool_call,
            self.harness.tool_error_observation(rejection, tool=payload["tool"]))
        _append_rejected_observation(messages, tool_call, observation)

    async def _chat_response(self, messages: list[dict[str, Any]],
                             tools: list[dict[str, Any]] | None,
                             iteration: int) -> dict[str, Any]:
        """Consume either runtime streaming or a single runtime response."""
        self.events.emit("assistant.started", iteration)
        if not self.stream:
            chat = self.llm.chat
            response = await chat(messages, tools)
            content = response.get("content") or ""
            reasoning = response.get("reasoning_content") or ""
            if content or reasoning:
                self.events.emit("assistant.delta", iteration,
                                 content=content, reasoning=reasoning)
        else:
            response = await self._consume_stream(messages, tools, iteration)
        self.events.emit("assistant.completed", iteration,
                         tool_calls=[_tool_trace_metadata(call).get("tool")
                                     for call in response.get("tool_calls") or []])
        return response

    async def _consume_stream(self, messages: list[dict[str, Any]],
                              tools: list[dict[str, Any]] | None,
                              iteration: int) -> dict[str, Any]:
        """Assemble one response from provider streaming chunks."""
        content: list[str] = []
        reasoning: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        finish_reason: str | None = None
        reasoning_budget = _reasoning_budget_chars()
        logger.debug("llm.stream.start iteration=%d", iteration)
        stream_iterator = self.llm.stream(messages, tools).__aiter__()
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
            reason = chunk.get("finish_reason")
            if isinstance(reason, str) and reason:
                finish_reason = reason
                continue
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
            if reasoning_budget is not None and reasoning and not content \
                    and not tool_calls:
                total_reasoning = sum(len(part) for part in reasoning)
                if total_reasoning > reasoning_budget:
                    logger.error(
                        "llm.reasoning_budget iteration=%d budget=%d "
                        "reasoning_chars=%d", iteration, reasoning_budget,
                        total_reasoning)
                    self.trace.emit("llm.reasoning_budget", iteration,
                                    budget=reasoning_budget,
                                    reasoning_chars=total_reasoning)
                    raise _ReasoningBudgetExceeded(total_reasoning,
                                                   reasoning_budget)
            if text or reasoning_text:
                self.events.emit("assistant.delta", iteration,
                                 content=text, reasoning=reasoning_text)
            logger.debug("llm.chunk iteration=%d content=%r reasoning=%r tool_calls=%r",
                         iteration, text, reasoning_text, chunk.get("tool_calls") or [])
            self.trace.emit("llm.chunk", iteration,
                            content=text, tool_call_count=len(chunk.get("tool_calls") or []),
                            reasoning_chars=len(reasoning_text))
        logger.debug("llm.stream.end iteration=%d finish_reason=%r", iteration,
                     finish_reason)
        self.trace.emit("llm.stream.finish", iteration, finish_reason=finish_reason)
        response = {"content": "".join(content), "tool_calls": tool_calls}
        if reasoning:
            response["reasoning_content"] = "".join(reasoning)
        if finish_reason is None:
            # The server cut the stream without a finish marker (observed on
            # free-tier reasoning models after very long thinking).  Treat the
            # truncated stream as a failed turn so it can be retried instead
            # of silently becoming an empty final answer.
            raise RuntimeError(
                "LLM stream ended without a finish_reason "
                "(stream may have been truncated by the server)")
        if not content and not reasoning and not tool_calls:
            raise RuntimeError(
                "LLM stream ended without any content, reasoning, or tool calls")
        return response


async def run_turn(user_message: str, llm: ChatModel, tools: ToolExecutor,
               max_iterations: int | None = None, *,
               harness: HarnessSpec | None = None,
               conversation: Conversation | None = None,
               stream: bool = False,
               trace: RunTrace | None = None,
               events: EventEmitter | None = None) -> str:
    """Run one agent turn under the given harness.

    Convenience wrapper around :class:`AgentTurn`.  ``max_iterations``
    explicitly overrides ``harness.control``; the harness is otherwise
    the source of truth for how the agent behaves.
    """
    return await AgentTurn(user_message, llm, tools, max_iterations,
                           harness=harness, conversation=conversation,
                           stream=stream, trace=trace, events=events).run()
