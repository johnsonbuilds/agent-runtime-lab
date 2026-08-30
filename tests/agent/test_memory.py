import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from agent_runtime.agent.memory import (
    DEFAULT_CONTEXT_BUDGET,
    SUMMARY_PREFIX,
    apply_memory_strategy,
    compact_observations,
    set_summary_llm,
    summarize_history,
)


def tool_message(call_id: str, content: str,
                 metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "tool", "tool_call_id": call_id,
                               "content": content}
    if metadata:
        message["metadata"] = metadata
    return message


def assistant_message(call_id: str, name: str = "run_command",
                      arguments: str = "{}") -> dict[str, Any]:
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": name,
                                         "arguments": arguments}}]}


def build_rounds(rounds: int, obs_chars: int) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "do the task"},
    ]
    for index in range(rounds):
        call_id = f"call-{index}"
        messages.append(assistant_message(call_id))
        messages.append(tool_message(call_id, "x" * obs_chars))
    return messages


def _chars(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        total += len(message.get("content") or "")
        for call in message.get("tool_calls") or []:
            total += len((call.get("function") or {}).get("arguments") or "")
    return total


class CompactObservationsTests(unittest.TestCase):
    def test_below_budget_is_identity(self) -> None:
        messages = build_rounds(3, 100)

        result = compact_observations(messages, budget=DEFAULT_CONTEXT_BUDGET)

        self.assertIs(result, messages)

    def test_recent_window_is_never_compacted(self) -> None:
        messages = build_rounds(20, 5_000)

        result = compact_observations(messages, budget=10_000, window_rounds=6)

        # Rounds older than the window are compacted down to head + summary.
        old_obs = result[3]  # round 0's tool message
        self.assertIn("observation compacted", old_obs["content"])
        self.assertLess(len(old_obs["content"]), 500)
        # Rounds inside the window are untouched.
        recent_obs = result[-1]
        self.assertEqual(recent_obs["content"], "x" * 5_000)
        # Structure is preserved: same roles, same ids, same count.
        self.assertEqual(len(result), len(messages))
        self.assertEqual([m["role"] for m in result],
                         [m["role"] for m in messages])
        self.assertEqual([m.get("tool_call_id") for m in result],
                         [m.get("tool_call_id") for m in messages])

    def test_keeps_recent_rounds_even_when_budget_still_exceeded(self) -> None:
        messages = build_rounds(20, 5_000)

        result = compact_observations(messages, budget=1, window_rounds=2,
                                      keep_recent_rounds=2)

        # The final two rounds stay intact even though the budget is tiny.
        self.assertEqual(result[-1]["content"], "x" * 5_000)
        self.assertEqual(result[-3]["content"], "x" * 5_000)
        self.assertIn("observation compacted", result[3]["content"])

    def test_budget_pressure_shrinks_window_round_by_round(self) -> None:
        # 10 rounds of 5k = ~50k. Comfort window 8 would keep ~40k+ of
        # observations; a 30k budget forces the window to shrink below 8.
        messages = build_rounds(10, 5_000)

        result = compact_observations(messages, budget=30_000,
                                      window_rounds=8, keep_recent_rounds=2)

        # The keep-recent core is intact...
        self.assertEqual(result[-1]["content"], "x" * 5_000)
        self.assertEqual(result[-3]["content"], "x" * 5_000)
        # ...everything else (short of it) is compacted, and the result
        # now fits the budget.
        self.assertLessEqual(_chars(result), 30_000)
        self.assertIn("observation compacted", result[3]["content"])

    def test_keeps_recent_core_even_when_it_alone_exceeds_budget(self) -> None:
        # 3 rounds of 5k; keep_recent=3 protects all of them from a
        # budget that they alone exceed — context wins over budget.
        messages = build_rounds(3, 5_000)

        result = compact_observations(messages, budget=100,
                                      window_rounds=3, keep_recent_rounds=3)

        for index in (3, 5, 7):
            self.assertEqual(result[index]["content"], "x" * 5_000)

    def test_first_round_tool_is_compactable(self) -> None:
        # Regression: round numbering must start at the first assistant
        # message, so the very first tool observation is inside the
        # compaction range (it used to be off by one and untouchable).
        messages = build_rounds(2, 5_000)

        result = compact_observations(messages, budget=100,
                                      window_rounds=1, keep_recent_rounds=1)

        self.assertIn("observation compacted", result[3]["content"])
        self.assertEqual(result[5]["content"], "x" * 5_000)

    def test_short_observations_are_left_alone(self) -> None:
        messages = build_rounds(20, 50)

        result = compact_observations(messages, budget=1, window_rounds=6)

        old_obs = result[3]
        self.assertEqual(old_obs["content"], "x" * 50)

    def test_compacted_summary_names_the_tool(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            assistant_message("c1", name="read_file"),
            tool_message("c1", "y" * 3_000),
        ]

        result = compact_observations(messages, budget=10, window_rounds=0)

        self.assertIn("read_file", result[3]["content"])
        self.assertIn("3000 chars", result[3]["content"])

    def test_spilled_observation_points_to_metadata_spill_path(self) -> None:
        # The spill reference travels as structured message metadata, not
        # as a string scanned out of the content.
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            assistant_message("c1", name="run_command"),
            tool_message("c1", "y" * 3_000,
                         metadata={"spill_path": ".outputs/obs-0001.txt"}),
        ]

        result = compact_observations(messages, budget=10, window_rounds=0)

        content = result[3]["content"]
        self.assertIn(".outputs/obs-0001.txt", content)
        self.assertIn("read_output", content)
        self.assertIn("do NOT re-run", content)

    def test_unspilled_observation_admits_re_run_is_the_only_recovery(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            assistant_message("c1", name="run_command"),
            tool_message("c1", "y" * 3_000),
        ]

        result = compact_observations(messages, budget=10, window_rounds=0)

        content = result[3]["content"]
        self.assertNotIn("Full output file", content)
        self.assertIn("only if you truly need it", content)

    def test_system_and_user_messages_are_never_touched(self) -> None:
        messages = build_rounds(20, 5_000)
        messages[1] = {"role": "user", "content": "precious instructions"}

        result = compact_observations(messages, budget=1, window_rounds=2)

        self.assertEqual(result[0]["content"], "sys")
        self.assertEqual(result[1]["content"], "precious instructions")

    def test_assistant_tool_call_arguments_are_never_compacted(self) -> None:
        arguments = '{"command": "' + "z" * 3_000 + '"}'
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            assistant_message("c1", arguments=arguments),
            tool_message("c1", "x" * 3_000),
        ]

        result = compact_observations(messages, budget=10, window_rounds=0)

        self.assertEqual(result[2]["tool_calls"][0]["function"]["arguments"],
                         arguments)


class ApplyMemoryStrategyTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_history_is_identity(self) -> None:
        messages = build_rounds(3, 100)

        self.assertIs(await apply_memory_strategy(messages, "full_history"),
                      messages)

    async def test_compact_strategy_is_routed(self) -> None:
        messages = build_rounds(20, 5_000)

        result = await apply_memory_strategy(messages, "compact_observations",
                                             budget=10_000, window_rounds=6)

        self.assertIn("observation compacted", result[3]["content"])

    async def test_unknown_strategy_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown memory strategy"):
            await apply_memory_strategy([], "summarize")


class FakeSummaryLLM:
    def __init__(self, summary: str | None = None,
                 *, error: Exception | None = None) -> None:
        self.error = error
        self.call_count = 0
        if summary is not None:
            self.summary = summary
        else:
            self.summary = (
                "# Agent Context Snapshot\n\n"
                "## 1. Work State\n"
                "### Completed\n"
                "- nothing yet\n\n"
                "## 2. Next Move\n"
                "- keep going\n\n"
                "## 3. Working Context & Anchors\n"
                "- **Relevant Files**: none"
            )

    async def chat(self, messages: list[dict[str, Any]],
                   tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        self.call_count += 1
        if self.error is not None:
            raise self.error
        return {"content": self.summary, "tool_calls": []}


class SummarizeHistoryTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self) -> None:
        set_summary_llm(None)

    async def test_below_budget_is_identity(self) -> None:
        messages = build_rounds(3, 100)
        llm = FakeSummaryLLM()
        set_summary_llm(llm)
        result = await summarize_history(messages, budget=DEFAULT_CONTEXT_BUDGET)
        self.assertIs(result, messages)
        self.assertEqual(llm.call_count, 0)

    async def test_collapses_old_rounds_keeps_tail_and_task(self) -> None:
        messages = build_rounds(10, 5_000)
        llm = FakeSummaryLLM(summary=(
            "# Agent Context Snapshot\n\n"
            "## 1. Work State\n"
            "### Completed\n"
            "- WORK STATE: mid-flight\n\n"
            "## 2. Next Move\n"
            "- continue\n\n"
            "## 3. Working Context & Anchors\n"
            "- **Files**: ars.R"))
        set_summary_llm(llm)
        result = await summarize_history(messages, budget=1_000, tail_rounds=2)

        self.assertIsNot(result, messages)
        # system + original task preserved at the front
        self.assertEqual(result[0]["content"], "sys")
        self.assertEqual(result[1]["content"], "do the task")
        # summary injected as a user message with the marker prefix
        summary = result[2]
        self.assertEqual(summary["role"], "user")
        self.assertTrue(summary["content"].startswith(SUMMARY_PREFIX))
        self.assertIn("WORK STATE: mid-flight", summary["content"])
        # only the last two rounds survive as tool messages
        self.assertEqual(sum(1 for m in result if m["role"] == "tool"), 2)
        # tail (last 2 rounds) is verbatim
        self.assertEqual(result[-4:], messages[-4:])

    async def test_failure_falls_back_to_compact(self) -> None:
        messages = build_rounds(20, 5_000)
        llm = FakeSummaryLLM(error=RuntimeError("summarizer down"))
        set_summary_llm(llm)
        result = await summarize_history(messages, budget=10, tail_rounds=2)
        # Deterministic compaction keeps the turn viable.
        self.assertIn("observation compacted", result[3]["content"])

    async def test_writes_session_memory_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session_memory.md"
            messages = build_rounds(10, 5_000)
            set_summary_llm(FakeSummaryLLM(summary=(
                "# Agent Context Snapshot\n\n"
                "## 1. Work State\n"
                "### Completed\n"
                "- LATEST SNAPSHOT\n\n"
                "## 2. Next Move\n"
                "- continue\n\n"
                "## 3. Working Context & Anchors\n"
                "- **Files**: ars.R")))
            await summarize_history(messages, budget=1_000,
                                    tail_rounds=2,
                                    session_memory_path=path)
            self.assertTrue(path.is_file())
            content = path.read_text(encoding="utf-8")
            self.assertIn("LATEST SNAPSHOT", content)
            self.assertIn("## 1. Work State", content)

    async def test_invalid_summary_falls_back_to_compact(self) -> None:
        messages = build_rounds(10, 5_000)
        # Missing "## 3. Working Context & Anchors" → invalid → fallback
        llm = FakeSummaryLLM(summary=(
            "# Agent Context Snapshot\n\n"
            "## 1. Work State\n"
            "### Completed\n"
            "- done\n\n"
            "## 2. Next Move\n"
            "- continue"))
        set_summary_llm(llm)
        result = await summarize_history(messages, budget=1_000, tail_rounds=2)
        self.assertIn("observation compacted", result[3]["content"])


class ApplyMemoryStrategyLLMRoutingTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self) -> None:
        set_summary_llm(None)

    async def test_llm_summary_routes_to_summarizer(self) -> None:
        messages = build_rounds(10, 5_000)
        set_summary_llm(FakeSummaryLLM())
        result = await apply_memory_strategy(
            messages, "llm_summary", budget=1_000, tail_rounds=2)
        self.assertEqual(sum(1 for m in result if m["role"] == "tool"), 2)

    async def test_unknown_strategy_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            await apply_memory_strategy([], "bogus")


if __name__ == "__main__":
    unittest.main()
