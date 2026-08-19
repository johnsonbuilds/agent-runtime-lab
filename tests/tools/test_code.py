import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from agent_runtime.agent.loop import run_turn
from agent_runtime.execution.local import LocalWorkspace
from agent_runtime.trace import RunTrace
from agent_runtime.tools.tools import create_default_registry


CODE_TOOLS = ["execute_code", "read_file"]


def make_registry(directory: str):
    return create_default_registry(workspace=LocalWorkspace(directory),
                                   enabled=CODE_TOOLS)


class FakeLLM:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.messages: list[list[dict[str, Any]]] = []

    async def chat(self, messages: list[dict[str, Any]],
                   tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        self.messages.append(messages.copy())
        return self.responses.pop(0)


class ExecuteCodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_runs_python_script_and_saves_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = make_registry(directory)

            result = await registry.execute("execute_code", {
                "code": "print('hello')\n", "language": "python"})
            saved = await registry.execute("read_file", {"path": ".scripts/0001.py"})

        self.assertEqual(result["stdout"], "hello\n")
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["script_path"], ".scripts/0001.py")
        self.assertEqual(result["language"], "python")
        self.assertEqual(saved["content"], "print('hello')\n")

    async def test_sequence_numbers_advance_across_languages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = make_registry(directory)

            first = await registry.execute("execute_code", {
                "code": "print('py')\n"})
            second = await registry.execute("execute_code", {
                "code": "echo bash\n", "language": "bash"})

        self.assertEqual(first["script_path"], ".scripts/0001.py")
        self.assertEqual(second["script_path"], ".scripts/0002.sh")
        self.assertEqual(second["stdout"], "bash\n")

    async def test_explicit_path_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = make_registry(directory)

            result = await registry.execute("execute_code", {
                "code": "print('x')\n", "path": "analysis/run.py"})
            saved = await registry.execute("read_file", {"path": "analysis/run.py"})

        self.assertEqual(result["script_path"], "analysis/run.py")
        self.assertEqual(saved["content"], "print('x')\n")

    async def test_failing_script_returns_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = make_registry(directory)

            result = await registry.execute("execute_code", {
                "code": "raise ValueError('boom')\n"})

        self.assertNotEqual(result["exit_code"], 0)
        self.assertIn("ValueError: boom", result["stderr"])
        self.assertEqual(result["script_path"], ".scripts/0001.py")

    async def test_unsupported_language_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = make_registry(directory)

            with self.assertRaisesRegex(ValueError, "unsupported language"):
                await registry.execute("execute_code", {
                    "code": "print('x')\n", "language": "cobol"})

    async def test_small_stdout_is_returned_in_full(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = make_registry(directory)

            result = await registry.execute("execute_code", {
                "code": "print('small')\n"})

        self.assertEqual(result["stdout"], "small\n")
        self.assertNotIn("stdout_chars", result)
        self.assertNotIn("stdout_full_path", result)

    async def test_oversized_stdout_is_windowed_and_spilled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = make_registry(directory)

            result = await registry.execute("execute_code", {
                "code": "print('x' * 20000)\n"})
            spilled = await registry.execute("read_file", {
                "path": ".outputs/0001.stdout.txt"})

        window = result["stdout"]
        self.assertEqual(result["stdout_chars"], 20001)
        self.assertEqual(result["stdout_full_path"], ".outputs/0001.stdout.txt")
        self.assertIn("characters omitted", window)
        self.assertTrue(window.startswith("x" * 4000))
        self.assertTrue(window.endswith("x" * 3999 + "\n"))
        self.assertLess(len(window), result["stdout_chars"])
        self.assertEqual(spilled["content"], "x" * 20000 + "\n")

    async def test_run_turn_feeds_one_observation_back_to_the_llm(self) -> None:
        llm = FakeLLM([
            {"content": "", "tool_calls": [{"id": "1", "function": {
                "name": "execute_code",
                "arguments": '{"code": "print(21 * 2)"}'}}]},
            {"content": "42", "tool_calls": []},
        ])

        with tempfile.TemporaryDirectory() as directory:
            registry = make_registry(directory)
            answer = await run_turn("double 21", llm, registry,
                                    trace=RunTrace(run_id="code"))

            saved = await registry.execute("read_file", {"path": ".scripts/0001.py"})

        self.assertEqual(answer, "42")
        observation = llm.messages[1][-1]
        self.assertEqual(observation["role"], "tool")
        self.assertEqual(observation["tool_call_id"], "1")
        self.assertIn("42", observation["content"])
        self.assertIn(".scripts/0001.py", observation["content"])
        self.assertEqual(saved["content"], "print(21 * 2)")


if __name__ == "__main__":
    unittest.main()
