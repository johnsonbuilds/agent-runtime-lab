import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from agent_runtime.execution.local import LocalWorkspace
from agent_runtime.tools.tools import create_default_registry


def make_registry(directory: str):
    return create_default_registry(workspace=LocalWorkspace(directory),
                                   enabled=["write_file", "read_file",
                                            "apply_patch"])


ORIGINAL = "def add(a, b):\n    return a + b\n"


def block(path: str, old: str, new: str) -> str:
    return (f"{path}\n<<<<<<< SEARCH\n{old}\n=======\n{new}\n>>>>>>> REPLACE\n"
            if old else
            f"{path}\n<<<<<<< SEARCH\n=======\n{new}\n>>>>>>> REPLACE\n")


class ApplyPatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_edit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = make_registry(directory)
            await registry.execute("write_file", {"path": "calc.py",
                                                  "content": ORIGINAL})
            result = await registry.execute("apply_patch", {
                "patch": block("calc.py", "return a + b",
                               "return a - b")})
            read = await registry.execute("read_file", {"path": "calc.py"})

        self.assertEqual(result["blocks_applied"], 1)
        self.assertEqual(result["files_updated"], ["calc.py"])
        self.assertIn("return a - b", read["content"])

    async def test_multiple_blocks_on_one_file_apply_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = make_registry(directory)
            await registry.execute("write_file", {
                "path": "app.py",
                "content": "TITLE = 'old'\nNAME = 'old'\n"})
            patch = (block("app.py", "TITLE = 'old'", "TITLE = 'new'")
                     + "\n"
                     + block("app.py", "NAME = 'old'", "NAME = 'new'"))
            result = await registry.execute("apply_patch", {"patch": patch})
            read = await registry.execute("read_file", {"path": "app.py"})

        self.assertEqual(result["blocks_applied"], 2)
        self.assertEqual(read["content"], "TITLE = 'new'\nNAME = 'new'\n")

    async def test_edit_across_multiple_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = make_registry(directory)
            await registry.execute("write_file", {"path": "a.py",
                                                  "content": "x = 1\n"})
            await registry.execute("write_file", {"path": "b.py",
                                                  "content": "y = 2\n"})
            patch = (block("a.py", "x = 1", "x = 10")
                     + "\n" + block("b.py", "y = 2", "y = 20"))
            result = await registry.execute("apply_patch", {"patch": patch})
            a = await registry.execute("read_file", {"path": "a.py"})
            b = await registry.execute("read_file", {"path": "b.py"})

        self.assertEqual(sorted(result["files_updated"]), ["a.py", "b.py"])
        self.assertEqual(a["content"], "x = 10\n")
        self.assertEqual(b["content"], "y = 20\n")

    async def test_empty_search_section_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = make_registry(directory)
            result = await registry.execute("apply_patch", {
                "patch": block("new.py", "", "print('hi')\n")})
            read = await registry.execute("read_file", {"path": "new.py"})

        self.assertEqual(result["files_created"], ["new.py"])
        self.assertEqual(read["content"], "print('hi')\n")

    async def test_failed_block_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = make_registry(directory)
            await registry.execute("write_file", {"path": "a.py",
                                                  "content": "keep me\n"})
            patch = (block("a.py", "keep me", "changed")
                     + "\n" + block("a.py", "not present", "boom"))
            with self.assertRaisesRegex(ValueError, "not found"):
                await registry.execute("apply_patch", {"patch": patch})
            read = await registry.execute("read_file", {"path": "a.py"})

        self.assertEqual(read["content"], "keep me\n")

    async def test_ambiguous_search_text_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = make_registry(directory)
            await registry.execute("write_file", {
                "path": "dup.py", "content": "same\nsame\n"})
            with self.assertRaisesRegex(ValueError, "2 times"):
                await registry.execute("apply_patch", {
                    "patch": block("dup.py", "same", "other")})

    async def test_create_existing_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = make_registry(directory)
            await registry.execute("write_file", {"path": "a.py",
                                                  "content": "x\n"})
            with self.assertRaisesRegex(ValueError, "already exists"):
                await registry.execute("apply_patch", {
                    "patch": block("a.py", "", "y\n")})

    async def test_edit_missing_file_reports_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = make_registry(directory)
            result = await registry.execute("apply_patch", {
                "patch": block("ghost.py", "old text", "new text")})

        self.assertIn("error", result)

    async def test_malformed_patch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = make_registry(directory)
            with self.assertRaisesRegex(ValueError, "no edit block"):
                await registry.execute("apply_patch", {"patch": "hello"})
            with self.assertRaisesRegex(ValueError, "unterminated"):
                await registry.execute("apply_patch", {
                    "patch": "a.py\n<<<<<<< SEARCH\nx\n=======\ny\n"})
            with self.assertRaisesRegex(ValueError, "file path"):
                await registry.execute("apply_patch", {
                    "patch": "<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE\n"})

    async def test_empty_patch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = make_registry(directory)
            with self.assertRaisesRegex(ValueError, "non-empty"):
                await registry.execute("apply_patch", {"patch": "  "})


if __name__ == "__main__":
    unittest.main()
