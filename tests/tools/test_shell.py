import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from agent_runtime.tools.shell import run_command
from agent_runtime.tools.tools import create_default_registry


class RunCommandTests(unittest.TestCase):
    def test_captures_output_exit_code_and_duration(self) -> None:
        result = run_command("printf 'hello'; printf 'bad' >&2; exit 3")

        self.assertEqual(result["stdout"], "hello")
        self.assertEqual(result["stderr"], "bad")
        self.assertEqual(result["exit_code"], 3)
        self.assertIsInstance(result["duration"], float)
        self.assertNotIn("error", result)

    def test_uses_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_command("pwd", cwd=directory)

        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["stdout"].strip(), directory)

    def test_timeout_is_structured_error(self) -> None:
        result = run_command("sleep 1", timeout=0.01)

        self.assertIsNone(result["exit_code"])
        self.assertEqual(result["error"]["type"], "TimeoutExpired")
        self.assertIn("duration", result)

    def test_invalid_cwd_is_structured_error(self) -> None:
        result = run_command("pwd", cwd="/path/that/does/not/exist")

        self.assertIsNone(result["exit_code"])
        self.assertEqual(result["error"]["type"], "FileNotFoundError")

    def test_default_registry_includes_shell_tool(self) -> None:
        names = [schema["function"]["name"] for schema in create_default_registry().schemas]

        self.assertIn("run_command", names)


if __name__ == "__main__":
    unittest.main()
