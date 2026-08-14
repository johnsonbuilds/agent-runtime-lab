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
            ["agent.start", "llm.start", "llm.end", "tool.start", "tool.end",
             "llm.start", "llm.end", "agent.end"],
        )
        self.assertEqual({event.run_id for event in trace.events}, {"r1"})
        self.assertEqual(len({event.event_id for event in trace.events}), len(trace.events))
        llm_end = next(event for event in trace.events
                       if event.event_type == "llm.end")
        self.assertEqual(llm_end.data["tool_count"], 1)
        self.assertEqual(llm_end.data["tools"], ["weather"])
        self.assertFalse(llm_end.data["final"])
        self.assertIn("duration_ms", llm_end.data)
        tool_end = next(event for event in trace.events
                        if event.event_type == "tool.end")
        self.assertEqual(tool_end.data["tool"], "weather")
        self.assertNotIn("result", tool_end.data)

    def test_trace_span_records_lifecycle(self) -> None:
        trace = RunTrace(run_id="r1")

        with trace.span("operation", iteration=2, request_id="req-1") as meta:
            meta["result_count"] = 3

        with self.assertRaisesRegex(RuntimeError, "failed"):
            with trace.span("failing_operation"):
                raise RuntimeError("failed")

        self.assertEqual(
            [event.event_type for event in trace.events],
            ["operation.start", "operation.end", "failing_operation.start",
             "failing_operation.error", "failing_operation.end"],
        )
        self.assertEqual(trace.events[1].data["result_count"], 3)
        self.assertIn("duration_ms", trace.events[1].data)
        self.assertEqual(trace.events[3].data["error"], "failed")

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

    def test_tool_trace_records_error_and_loop_continues(self) -> None:
        def failing_tool(location: str) -> str:
            raise RuntimeError("service unavailable")

        llm = FakeLLM([
            {"content": "", "tool_calls": [{"id": "1", "function": {
                "name": "weather", "arguments": '{"location":"Paris"}'}}]},
            {"content": "fallback", "tool_calls": []},
        ])
        trace = RunTrace(run_id="r1")

        self.assertEqual(run_turn("weather", llm, make_registry(failing_tool),
                                  trace=trace), "fallback")
        self.assertEqual(
            [event.event_type for event in trace.events],
            ["agent.start", "llm.start", "llm.end", "tool.start",
             "tool.error", "tool.end", "llm.start", "llm.end", "agent.end"],
        )
        tool_error = next(event for event in trace.events
                          if event.event_type == "tool.error")
        self.assertEqual(tool_error.data["error"], "service unavailable")
        self.assertEqual(tool_error.data["tool"], "weather")
        self.assertEqual(len(llm.messages), 2)
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
        trace = RunTrace(run_id="r1")
        self.assertEqual(
            run_turn("hello", FakeLLM(error=RuntimeError("network down")),
                     make_registry(), trace=trace),
            "LLM error: network down",
        )
        self.assertEqual(
            [event.event_type for event in trace.events],
            ["agent.start", "llm.start", "llm.error", "llm.end",
             "agent.error", "agent.end"],
        )
        llm_error = next(event for event in trace.events
                         if event.event_type == "llm.error")
        self.assertEqual(llm_error.data["error"], "network down")
        agent_error = next(event for event in trace.events
                           if event.event_type == "agent.error")
        self.assertEqual(agent_error.data["error"], "LLM error: network down")
        self.assertEqual(agent_error.data["stage"], "llm")

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
