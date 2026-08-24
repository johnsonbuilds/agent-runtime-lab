"""Context management strategies: the request view of conversation history.

The conversation itself is the source of truth and is never mutated by
these strategies.  A strategy only produces the *view* of the history
that gets sent to the LLM on each request, which keeps the on-disk
trace, future phases (summarisation), and post-hoc analysis intact.

Phase 1 ships ``compact_observations``: a deterministic, LLM-free
window that replaces stale tool observations with one-line summaries
once the request would exceed a character budget.  Message structure is
preserved — a compacted tool message stays a tool message with the same
``tool_call_id`` — so the result is always a protocol-legal sequence.
"""

from __future__ import annotations

import os
from typing import Any

DEFAULT_CONTEXT_BUDGET = 60_000
DEFAULT_WINDOW_ROUNDS = 12
DEFAULT_KEEP_RECENT_ROUNDS = 2
DEFAULT_HEAD_CHARS = 200

STRATEGY_FULL_HISTORY = "full_history"
STRATEGY_COMPACT_OBSERVATIONS = "compact_observations"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw else default
    except ValueError:
        return default
    return value if value > 0 else default


def _context_budget() -> int:
    return _env_int("AGENT_RUNTIME_CONTEXT_BUDGET", DEFAULT_CONTEXT_BUDGET)


def _window_rounds() -> int:
    return _env_int("AGENT_RUNTIME_CONTEXT_WINDOW", DEFAULT_WINDOW_ROUNDS)


def _message_chars(message: dict[str, Any]) -> int:
    total = len(message.get("content") or "")
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        total += len(function.get("arguments") or "")
    return total


def _total_chars(messages: list[dict[str, Any]]) -> int:
    return sum(_message_chars(message) for message in messages)


def _round_starts(messages: list[dict[str, Any]]) -> list[int]:
    """Indices where each agent round begins.

    A round is anchored by its first assistant message; the initial
    user task belongs to round 0 (everything before the first assistant
    message), so it can never fall inside a compaction cutoff.
    """
    starts: list[int] = []
    previous_role = None
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "system":
            continue
        if role == "assistant" and previous_role in (None, "user", "tool"):
            starts.append(index)
        previous_role = role
    return starts


def _compact_tool_message(message: dict[str, Any], tool: str,
                          head_chars: int) -> dict[str, Any]:
    content = message.get("content") or ""
    if len(content) <= head_chars:
        return message
    summary = (f"[observation compacted: {tool} returned {len(content)} chars "
               f"(exit code and first {head_chars} chars below); "
               f"re-run the tool or read_file to see it again]\n")
    compacted = dict(message)
    compacted["content"] = summary + content[:head_chars]
    return compacted


def _tool_name_for(messages: list[dict[str, Any]],
                   index: int) -> str:
    """Best-effort tool name for the tool message at ``index``."""
    tool_call_id = messages[index].get("tool_call_id")
    if tool_call_id:
        for message in messages[:index]:
            for call in message.get("tool_calls") or []:
                if call.get("id") == tool_call_id:
                    return (call.get("function") or {}).get("name") or "tool"
    return "tool"


def compact_observations(messages: list[dict[str, Any]], *,
                         budget: int = DEFAULT_CONTEXT_BUDGET,
                         window_rounds: int = DEFAULT_WINDOW_ROUNDS,
                         keep_recent_rounds: int = DEFAULT_KEEP_RECENT_ROUNDS,
                         head_chars: int = DEFAULT_HEAD_CHARS,
                         ) -> list[dict[str, Any]]:
    """Return a compacted copy of ``messages`` when it exceeds ``budget``.

    Two passes, oldest first:

    1. Compact tool observations in rounds older than the last
       ``window_rounds`` rounds (the comfort zone that is normally kept
       intact).
    2. If the request would still exceed ``budget``, keep compacting
       tool observations round by round from the oldest round still
       intact — but never inside the most recent ``keep_recent_rounds``
       rounds.  Those are the active working context; if they alone
       exceed the budget the budget yields, not the context.

    Below budget the original list is returned unchanged.
    """
    if _total_chars(messages) <= budget:
        return messages
    starts = _round_starts(messages)
    if not starts:
        return messages
    total_rounds = len(starts)
    never_touch_from_round = max(0, total_rounds - keep_recent_rounds)

    def _round_of(index: int) -> int:
        position = 0
        for start in starts:
            if start > index:
                break
            position += 1
        return position

    def _compact_at(cutoff: int) -> tuple[list[dict[str, Any]], bool]:
        """Compact tool messages in rounds 1..cutoff (inclusive)."""
        result: list[dict[str, Any]] = []
        compacted_any = False
        for index, message in enumerate(messages):
            role = message.get("role")
            if role == "tool" and 0 < _round_of(index) <= cutoff:
                tool = _tool_name_for(messages, index)
                compacted = _compact_tool_message(message, tool, head_chars)
                if compacted is not message:
                    compacted_any = True
                result.append(compacted)
            else:
                result.append(message)
        return result, compacted_any

    # Pass 1: everything outside the comfort window.
    window_cutoff = max(0, total_rounds - window_rounds)
    result, _ = _compact_at(window_cutoff)

    # Pass 2: under budget pressure, shrink the window round by round
    # until the budget is met or only the keep-recent core remains.
    cutoff = window_cutoff
    while _total_chars(result) > budget and cutoff < never_touch_from_round:
        cutoff += 1
        result, compacted_any = _compact_at(cutoff)
        if not compacted_any:
            break
    return result


def apply_memory_strategy(messages: list[dict[str, Any]], strategy: str,
                          **options: Any) -> list[dict[str, Any]]:
    """Build the request view of ``messages`` for the given strategy."""
    if strategy == STRATEGY_FULL_HISTORY:
        return messages
    if strategy == STRATEGY_COMPACT_OBSERVATIONS:
        defaults = {"budget": _context_budget(),
                    "window_rounds": _window_rounds()}
        defaults.update(options)
        return compact_observations(messages, **defaults)
    raise ValueError(f"unknown memory strategy: {strategy!r}")


__all__ = [
    "DEFAULT_CONTEXT_BUDGET", "DEFAULT_HEAD_CHARS",
    "DEFAULT_KEEP_RECENT_ROUNDS", "DEFAULT_WINDOW_ROUNDS",
    "STRATEGY_COMPACT_OBSERVATIONS", "STRATEGY_FULL_HISTORY",
    "apply_memory_strategy", "compact_observations",
]
