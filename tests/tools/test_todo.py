import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from agent_runtime.execution.local import LocalWorkspace
from agent_runtime.tools.todo import TODO_PATH
from agent_runtime.tools.tools import create_default_registry


def make_registry(directory: str):
    return create_default_registry(workspace=LocalWorkspace(directory),
                                   enabled=["write_file", "read_file",
                                            "todo_write"])


class TodoWriteTests(unittest.IsolatedAsyncioTestCase):
    async def test_writes_checklist_and_persists_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = make_registry(directory)
            result = await registry.execute("todo_write", {"todos": [
                {"content": "read the code", "status": "in_progress"},
                {"content": "run tests", "status": "pending"},
            ]})
            stored = await registry.execute("read_file", {"path": TODO_PATH})

        self.assertEqual(result["todo_count"], 2)
        self.assertEqual(result["completed"], 0)
        self.assertIn("1. [in_progress] read the code", result["rendered"])
        self.assertIn("2. [pending] run tests", result["rendered"])
        self.assertEqual(json.loads(stored["content"]), [
            {"content": "read the code", "status": "in_progress"},
            {"content": "run tests", "status": "pending"},
        ])

    async def test_each_call_replaces_the_whole_list(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = make_registry(directory)
            await registry.execute("todo_write", {"todos": [
                {"content": "first", "status": "completed"},
                {"content": "second", "status": "pending"},
            ]})
            result = await registry.execute("todo_write", {"todos": [
                {"content": "only task", "status": "pending"},
            ]})

        self.assertEqual(result["todo_count"], 1)
        self.assertNotIn("first", result["rendered"])

    async def test_invalid_status_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = make_registry(directory)
            with self.assertRaisesRegex(ValueError, "status"):
                await registry.execute("todo_write", {"todos": [
                    {"content": "task", "status": "done-ish"},
                ]})

    async def test_two_in_progress_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = make_registry(directory)
            with self.assertRaisesRegex(ValueError, "in_progress"):
                await registry.execute("todo_write", {"todos": [
                    {"content": "a", "status": "in_progress"},
                    {"content": "b", "status": "in_progress"},
                ]})

    async def test_empty_content_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = make_registry(directory)
            with self.assertRaisesRegex(ValueError, "content"):
                await registry.execute("todo_write", {"todos": [
                    {"content": "  ", "status": "pending"},
                ]})

    async def test_empty_list_clears_the_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = make_registry(directory)
            await registry.execute("todo_write", {"todos": [
                {"content": "only task", "status": "completed"},
            ]})
            result = await registry.execute("todo_write", {"todos": []})

        self.assertEqual(result["todo_count"], 0)
        self.assertEqual(result["rendered"], "(empty task list)")


if __name__ == "__main__":
    unittest.main()
