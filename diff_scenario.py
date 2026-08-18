"""Deterministic run_turn scenario used to diff behavior across refactors.

Run with the old code and the new code, then diff the two JSONL traces
(ignoring run_id / event_id / timestamps / durations) to prove the
refactor changed nothing observable.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_runtime.agent import Conversation, run_turn
from agent_runtime.trace import RunTrace
from agent_runtime.tools.tools import ToolRegistry, ToolSpec


WEATHER_SCHEMA = {
    "type": "object",
    "properties": {"location": {"type": "string"}},
    "required": ["location"],
}


class ScriptedLLM:
    """Replays fixed responses; records every request payload."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    async def chat(self, messages, tools=None):
        self.requests.append({"messages": messages.copy(), "tools": tools})
        return self.responses.pop(0)


async def failing_tool(location: str) -> str:
    raise RuntimeError("service unavailable")


def make_registry() -> ToolRegistry:
    return ToolRegistry([
        ToolSpec("weather", "", WEATHER_SCHEMA, failing_tool),
    ])


async def main(output: str) -> None:
    # Scenario: tool call -> tool raises -> observation -> recovery answer,
    # then a second turn that hits the iteration limit forcing the final
    # no-tools summarization request.
    final_answers = [
        {"content": "", "tool_calls": [{"id": "1", "function": {
            "name": "weather", "arguments": '{"location":"Paris"}'}}]},
        {"content": "fallback", "tool_calls": []},
    ]
    iteration_answers = [{"content": "", "tool_calls": [{"id": "2", "function": {
        "name": "weather", "arguments": '{"location":"Tokyo"}'}}]}}] * 3
    iteration_answers.append({"content": "summarized", "tool_calls": []})
    llm = ScriptedLLM(final_answers + iteration_answers)
    conversation = Conversation()
    trace = RunTrace(run_id="diff", output_path=output)

    await run_turn("weather", llm, make_registry(), conversation=conversation,
                   trace=trace)
    await run_turn("loop three times", llm, make_registry(), max_iterations=3,
                   conversation=conversation, trace=trace)
    await trace.flush()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
