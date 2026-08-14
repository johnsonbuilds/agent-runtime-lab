import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from agent_runtime.tools.shell import run_command
from agent_runtime.tools.tools import create_default_registry
from agent_runtime.tools.tools import ToolRegistry, ToolSpec


class RecordingExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, float | None]] = []

    async def execute(self, command: str, cwd: str | None = None,
                      timeout: float | None = 30.0) -> dict[str, Any]:
        self.calls.append((command, cwd, timeout))
        return {"stdout": "sandbox", "stderr": "", "exit_code": 0, "duration": 0.0}


class RunCommandTests(unittest.IsolatedAsyncioTestCase):
    def test_sync_handler_is_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be async"):
            ToolRegistry([ToolSpec("sync", "", {"type": "object"}, lambda: None)])

    async def test_captures_output_exit_code_and_duration(self) -> None:
        result = await run_command("printf 'hello'; printf 'bad' >&2; exit 3")

        self.assertEqual(result["stdout"], "hello")
        self.assertEqual(result["stderr"], "bad")
        self.assertEqual(result["exit_code"], 3)
        self.assertIsInstance(result["duration"], float)
        self.assertNotIn("error", result)

    async def test_uses_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = await run_command("pwd", cwd=directory)

        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["stdout"].strip(), directory)

    async def test_timeout_is_structured_error(self) -> None:
        result = await run_command("sleep 1", timeout=0.01)

        self.assertIsNone(result["exit_code"])
        self.assertEqual(result["error"]["type"], "TimeoutExpired")
        self.assertIn("duration", result)

    async def test_invalid_cwd_is_structured_error(self) -> None:
        result = await run_command("pwd", cwd="/path/that/does/not/exist")

        self.assertIsNone(result["exit_code"])
        self.assertEqual(result["error"]["type"], "FileNotFoundError")

    def test_default_registry_includes_shell_tool(self) -> None:
        names = [schema["function"]["name"] for schema in create_default_registry().schemas]

        self.assertIn("run_command", names)

    async def test_registry_uses_injected_executor(self) -> None:
        executor = RecordingExecutor()
        registry = create_default_registry(executor)

        result = await registry.execute("run_command", {
            "command": "printf from-sandbox",
            "cwd": "/workspace",
            "timeout": 5,
        })

        self.assertEqual(result["stdout"], "sandbox")
        self.assertEqual(executor.calls, [("printf from-sandbox", "/workspace", 5)])


if __name__ == "__main__":
    unittest.main()
