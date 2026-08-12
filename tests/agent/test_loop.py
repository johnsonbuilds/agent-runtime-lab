import sys
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from agent_runtime.agent.loop import Conversation, run_turn
from agent_runtime.trace import RunTrace
from agent_runtime.tools.tools import ToolRegistry, ToolSpec


WEATHER_SCHEMA = {
    "type": "object",
    "properties": {"location": {"type": "string"}},
    "required": ["location"],
}


class FakeLLM:
    def __init__(self, responses: list[dict[str, Any]] | None = None,
                 error: Exception | None = None) -> None:
        self.responses = responses or []
        self.error = error
        self.messages: list[list[dict[str, Any]]] = []

    def chat(self, messages: list[dict[str, Any]],
             tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        self.messages.append(messages.copy())
        if self.error:
            raise self.error
        return self.responses.pop(0)


def make_registry(handler: Any = lambda location: location) -> ToolRegistry:
    return ToolRegistry([ToolSpec("weather", "", WEATHER_SCHEMA, handler)])


class AgentLoopErrorHandlingTests(unittest.TestCase):
    def test_run_trace_records_agent_loop_and_tool_events(self) -> None:
        llm = FakeLLM([
            {"content": "", "tool_calls": [{"id": "1", "function": {
                "name": "weather", "arguments": '{"location":"Singapore"}'}}]},
            {"content": "It is sunny", "tool_calls": []},
        ])
        trace = RunTrace(run_id="r1")

        self.assertEqual(run_turn("weather", llm, make_registry(), trace=trace),
                         "It is sunny")
        self.assertEqual(
            [event.event_type for event in trace.events],
            ["agent.start", "llm.request", "llm.response", "tool.request",
             "tool.response", "llm.request", "llm.response", "agent.end"],
        )
        self.assertEqual({event.run_id for event in trace.events}, {"r1"})
        self.assertEqual(len({event.event_id for event in trace.events}), len(trace.events))
        llm_response = next(event for event in trace.events
                            if event.event_type == "llm.response")
        self.assertEqual(llm_response.data, {
            "tool_count": 1, "tools": ["weather"], "final": False,
        })
        self.assertNotIn("tool_call", trace.events[4].data)
        tool_response = next(event for event in trace.events
                             if event.event_type == "tool.response")
        self.assertNotIn("result", tool_response.data)

    def test_run_trace_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs" / "run-001.jsonl"
            trace = RunTrace(run_id="r1", output_path=path)
            trace.emit("agent.start")

            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(records[0]["event_type"], "agent.start")
            self.assertEqual(records[0]["run_id"], "r1")

    def test_malformed_json_becomes_observation_and_loop_continues(self) -> None:
        llm = FakeLLM([
            {"content": "", "tool_calls": [{"id": "1", "function": {
                "name": "weather", "arguments": "{"}}]},
            {"content": "recovered", "tool_calls": []},
        ])

        self.assertEqual(run_turn("weather", llm, make_registry()), "recovered")
        self.assertIn("malformed JSON", llm.messages[1][-1]["content"])

    def test_unknown_tool_and_invalid_arguments_are_observations(self) -> None:
        llm = FakeLLM([
            {"content": "", "tool_calls": [
                {"id": "unknown", "function": {"name": "missing", "arguments": "{}"}},
                {"id": "bad", "function": {"name": "weather", "arguments": "{\"location\": 1}"}},
            ]},
            {"content": "recovered", "tool_calls": []},
        ])

        self.assertEqual(run_turn("weather", llm, make_registry()), "recovered")
        observations = [message["content"] for message in llm.messages[1][-2:]]
        self.assertIn("unknown tool", observations[0])
        self.assertIn("must be string", observations[1])

    def test_tool_exception_becomes_observation(self) -> None:
        def failing_tool(location: str) -> str:
            raise RuntimeError("service unavailable")

        llm = FakeLLM([
            {"content": "", "tool_calls": [{"id": "1", "function": {
                "name": "weather", "arguments": '{"location":"Paris"}'}}]},
            {"content": "fallback", "tool_calls": []},
        ])

        self.assertEqual(run_turn("weather", llm, make_registry(failing_tool)), "fallback")
        self.assertIn("service unavailable", llm.messages[1][-1]["content"])

    def test_missing_tool_call_id_does_not_crash(self) -> None:
        llm = FakeLLM([
            {"content": "", "tool_calls": [{"function": {
                "name": "weather", "arguments": '{"location":"Paris"}'}}]},
            {"content": "recovered", "tool_calls": []},
        ])

        self.assertEqual(run_turn("weather", llm, make_registry()), "recovered")
        self.assertEqual(llm.messages[1][-1]["role"], "user")
        self.assertIn("tool_call_id is required", llm.messages[1][-1]["content"])
        self.assertIn("retry this tool call with a valid tool_call_id", llm.messages[1][-1]["content"])

    def test_llm_exception_is_returned(self) -> None:
        self.assertEqual(
            run_turn("hello", FakeLLM(error=RuntimeError("network down")), make_registry()),
            "LLM error: network down",
        )

    def test_conversation_preserves_context_between_turns(self) -> None:
        llm = FakeLLM([
            {"content": "first answer", "tool_calls": []},
            {"content": "second answer", "tool_calls": []},
        ])
        conversation = Conversation()

        self.assertEqual(run_turn("user input 1", llm, make_registry(),
                                  conversation=conversation),
                         "first answer")
        self.assertEqual(run_turn("user input 2", llm, make_registry(),
                                  conversation=conversation),
                         "second answer")
        self.assertEqual([message["content"] for message in conversation.messages
                          if message["role"] == "user"], ["user input 1", "user input 2"])
        self.assertEqual([message["content"] for message in llm.messages[1]
                          if message["role"] == "user"], ["user input 1", "user input 2"])
        self.assertNotIn("tool_calls", conversation.messages[1])


if __name__ == "__main__":
    unittest.main()
