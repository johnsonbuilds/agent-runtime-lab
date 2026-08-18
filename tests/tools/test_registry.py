import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from agent_runtime.tools.tools import create_default_registry


class DefaultRegistryTests(unittest.TestCase):
    def test_default_registry_exposes_all_builtins(self) -> None:
        registry = create_default_registry()
        self.assertEqual([schema["function"]["name"] for schema in registry.schemas],
                         ["run_command"])

    def test_enabled_filter_selects_named_tools_in_order(self) -> None:
        registry = create_default_registry(enabled=["run_command"])
        self.assertEqual([schema["function"]["name"] for schema in registry.schemas],
                         ["run_command"])

    def test_enabled_filter_can_disable_everything(self) -> None:
        registry = create_default_registry(enabled=[])
        self.assertEqual(registry.schemas, [])

    def test_unknown_enabled_tool_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown tools in harness"):
            create_default_registry(enabled=["read_file"])


if __name__ == "__main__":
    unittest.main()
