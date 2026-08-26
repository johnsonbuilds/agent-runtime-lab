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

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_BUDGET = 60_000
DEFAULT_WINDOW_ROUNDS = 12
DEFAULT_KEEP_RECENT_ROUNDS = 2
DEFAULT_HEAD_CHARS = 200
DEFAULT_SUMMARY_TAIL_ROUNDS = 2

STRATEGY_FULL_HISTORY = "full_history"
STRATEGY_COMPACT_OBSERVATIONS = "compact_observations"
STRATEGY_LLM_SUMMARY = "llm_summary"

SUMMARY_PREFIX = "[SYSTEM CONTEXT COMPACTION SUMMARY]\n"

SUMMARIZER_SYSTEM = (
    "You are a professional context-compaction summarizer for an agent."
)

SUMMARIZER_INSTRUCTION = """\
Analyze the conversation history above. Produce a dense, structured summary of the CURRENT \
execution state that a fresh reasoning step can rely on.

Rules:
- Output ONLY the markdown below, with no preamble and no code fences.
- Focus strictly on tracking progress, active attempts, error lessons, and working context.
- For every failed or blocked attempt, record the exact error and root-cause lesson to prevent repeated failures (avoid Sisyphean loops).
- Be extremely specific with key entities, file paths, parameters, tool invocations, exit codes, and intermediate values.
- Keep it concise, precise, and information-dense.

Use exactly these sections:

# Agent Context Snapshot

## 1. Work State
### Completed
- ...
### Active (In-Progress)
- ...
### Blocked / Failure Lessons
- ...

## 2. Next Move
- ...

## 3. Working Context & Anchors
- **Relevant Files / Artifacts**: ...
- **Environment State**: ..."""


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


async def apply_memory_strategy(messages: list[dict[str, Any]], strategy: str,
                                 **options: Any) -> list[dict[str, Any]]:
    """Build the request view of ``messages`` for the given strategy.

    The ``llm_summary`` strategy uses its own summarizer LLM (see
    :func:`_get_summary_llm`); the other strategies are deterministic.
    """
    if strategy == STRATEGY_FULL_HISTORY:
        return messages
    if strategy == STRATEGY_COMPACT_OBSERVATIONS:
        defaults = {"budget": _context_budget(),
                    "window_rounds": _window_rounds()}
        defaults.update(options)
        return compact_observations(messages, **defaults)
    if strategy == STRATEGY_LLM_SUMMARY:
        return await summarize_history(messages, **options)
    raise ValueError(f"unknown memory strategy: {strategy!r}")


_summary_llm: Any = None


def set_summary_llm(llm: Any) -> None:
    """Override the summarizer LLM (used by tests and custom wiring)."""
    global _summary_llm
    _summary_llm = llm


def _get_summary_llm() -> Any:
    """Lazily build the summarizer LLM used by ``llm_summary``.

    A dedicated model can be selected via ``AGENT_RUNTIME_SUMMARY_MODEL`` so
    summarization can run on a different (e.g. cheaper/faster) model than the
    agent's main LLM; it otherwise falls back to the default model.
    """
    global _summary_llm
    if _summary_llm is None:
        from agent_runtime.providers.llm import OpenAICompatibleLLM
        _summary_llm = OpenAICompatibleLLM(
            model=os.getenv("AGENT_RUNTIME_SUMMARY_MODEL") or None)
    return _summary_llm


def _summary_tail_rounds() -> int:
    return _env_int("AGENT_RUNTIME_SUMMARY_TAIL", DEFAULT_SUMMARY_TAIL_ROUNDS)


def _session_memory_path() -> Path | None:
    """Default on-disk location for the latest session summary.

    ``AGENT_RUNTIME_WORKSPACE`` selects the root; the summary is written as
    ``<workspace>/memory/session_memory.md`` (one file per task, overwritten).
    """
    workspace = os.getenv("AGENT_RUNTIME_WORKSPACE") or "."
    return Path(workspace) / "memory" / "session_memory.md"


def _write_session_memory(content: str, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _strip_leading_system(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop any leading ``system`` messages (the agent's persona/system prompt).

    The summarizer must see the interaction *history*, not be re-primed with the
    agent's own system prompt, which would make it re-enter the agent/tool-calling
    role and emit a tool call instead of a summary.
    """
    index = 0
    while index < len(messages) and messages[index].get("role") == "system":
        index += 1
    return messages[index:]


async def _generate_summary(messages: list[dict[str, Any]], llm: Any,
                            prompt: str | None) -> str:
    """Ask the LLM for a structured summary of the full conversation.

    The payload is assembled as: a fixed summarizer-identity system message, the
    raw interaction history (agent system prompt stripped), and a final user
    instruction carrying the extraction schema. Ordering the instruction last
    exploits recency to dominate the model's output, and removing the agent's
    system prompt stops the model from imitating the agent's tool calls.
    """
    history = _strip_leading_system(messages)
    instruction = prompt or SUMMARIZER_INSTRUCTION
    summarizer_messages: list[dict[str, Any]] = [
        {"role": "system", "content": SUMMARIZER_SYSTEM},
        *history,
        {"role": "user", "content": instruction},
    ]
    response = await llm.chat(summarizer_messages, tools=None)
    return response.get("content") or ""


async def summarize_history(messages: list[dict[str, Any]], *,
                           budget: int | None = None,
                           tail_rounds: int | None = None,
                           prompt: str | None = None,
                           session_memory_path: Path | None = None
                           ) -> list[dict[str, Any]]:
    """Return a summarized copy of ``messages`` once over ``budget``.

    The summarizer reads the FULL history and produces a structured summary
    (see ``SUMMARIZER_INSTRUCTION``). The request view keeps the system
    prompt and original task, injects the summary as a ``user`` message, and
    appends the most recent ``tail_rounds`` rounds verbatim (preserving
    tool-call pairs and short-term coherence). The summary is also written to
    ``session_memory.md`` (overwriting any prior summary for this task).

    Below budget, or when there are no older rounds to collapse, the original
    list is returned unchanged. If the summarizer LLM call fails, the strategy
    falls back to the deterministic ``compact_observations`` so the turn still
    proceeds.
    """
    if budget is None:
        budget = _context_budget()
    if tail_rounds is None:
        tail_rounds = _summary_tail_rounds()
    llm = _get_summary_llm()
    if _total_chars(messages) <= budget:
        return messages
    starts = _round_starts(messages)
    total_rounds = len(starts)
    if total_rounds <= tail_rounds:
        return messages
    logger.debug("memory.summary.start rounds=%d tail=%d", total_rounds, tail_rounds)
    try:
        summary = await _generate_summary(messages, llm, prompt)
    except Exception:
        logger.warning("memory.summary.failed falling back to compact_observations")
        return compact_observations(messages, budget=budget)
    leading = messages[:starts[0]]
    tail_start = starts[-tail_rounds]
    tail = messages[tail_start:]
    summary_msg = {"role": "user", "content": SUMMARY_PREFIX + summary}
    view = leading + [summary_msg] + tail
    path = session_memory_path or _session_memory_path()
    if path is not None:
        await asyncio.to_thread(_write_session_memory, summary, path)
    return view


__all__ = [
    "DEFAULT_CONTEXT_BUDGET", "DEFAULT_HEAD_CHARS", "DEFAULT_SUMMARY_TAIL_ROUNDS",
    "DEFAULT_KEEP_RECENT_ROUNDS", "DEFAULT_WINDOW_ROUNDS",
    "STRATEGY_COMPACT_OBSERVATIONS", "STRATEGY_FULL_HISTORY", "STRATEGY_LLM_SUMMARY",
    "apply_memory_strategy", "compact_observations", "set_summary_llm",
    "summarize_history",
]
