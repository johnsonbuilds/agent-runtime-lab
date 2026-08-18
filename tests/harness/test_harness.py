import asyncio
import sys
import json
import tempfile
import unittest
from pathlib import Path
from textwrap import dedent


sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from agent_runtime.harness import (
    DEFAULT_HARNESS,
    ControlGenome,
    HarnessError,
    HarnessSpec,
    from_dict,
    load_harness,
    resolve_harness,
)
from agent_runtime.trace import RunTrace


REPO_ROOT = Path(__file__).parents[2]
BASELINE_YAML = REPO_ROOT / "harnesses" / "baseline-v0.yaml"


class HarnessDefaultTests(unittest.TestCase):
    def test_repo_manifest_stays_in_sync_with_builtin_default(self) -> None:
        loaded = load_harness(BASELINE_YAML)
        self.assertEqual(loaded.id, DEFAULT_HARNESS.id)
        self.assertEqual(loaded.genes_hash, DEFAULT_HARNESS.genes_hash)

    def test_genes_hash_ignores_lineage_metadata(self) -> None:
        base = HarnessSpec()
        annotated = HarnessSpec(id="baseline-v1", parent="baseline-v0",
                                mutation="enable verification",
                                reason="test the verification gene",
                                verification=base.verification)
        self.assertEqual(base.genes_hash, annotated.genes_hash)

    def test_genes_hash_changes_when_a_gene_changes(self) -> None:
        mutated = HarnessSpec(control=ControlGenome(max_iterations=20))
        self.assertNotEqual(DEFAULT_HARNESS.genes_hash, mutated.genes_hash)

    def test_to_dict_roundtrips_through_from_dict(self) -> None:
        spec = HarnessSpec(id="h1", parent="baseline-v0",
                           mutation="bump iterations",
                           control=ControlGenome(max_iterations=5))
        replica = from_dict(spec.to_dict())
        self.assertEqual(spec.genes_hash, replica.genes_hash)
        self.assertEqual(spec.prompt, replica.prompt)
        self.assertEqual(spec.control, replica.control)


class HarnessValidationTests(unittest.TestCase):
    def assert_harness_error(self, yaml_text: str, fragment: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "h.yaml"
            path.write_text(dedent(yaml_text))
            with self.assertRaises(HarnessError) as caught:
                load_harness(path)
            self.assertIn(fragment, str(caught.exception))

    def test_unknown_manifest_key_is_rejected(self) -> None:
        self.assert_harness_error(
            """
            id: h1
            model:
              stream: env
            """,
            "unknown manifest keys: ['model']",
        )

    def test_unknown_section_key_is_rejected(self) -> None:
        self.assert_harness_error(
            """
            control:
              max_iterations: 3
              timeout: 30
            """,
            "unknown control keys: ['timeout']",
        )

    def test_bad_max_iterations_is_rejected(self) -> None:
        self.assert_harness_error(
            """
            control:
              max_iterations: 0
            """,
            "max_iterations must be an integer >= 1",
        )

    def test_unknown_recovery_strategy_is_rejected(self) -> None:
        self.assert_harness_error(
            """
            recovery:
              tool_error: panic
            """,
            "recovery.tool_error",
        )

    def test_unknown_memory_strategy_is_rejected(self) -> None:
        self.assert_harness_error(
            """
            memory:
              strategy: summarized
            """,
            "memory.strategy",
        )

    def test_blank_id_is_rejected(self) -> None:
        self.assert_harness_error(
            """
            id: ""
            """,
            "manifest.id must be non-empty text",
        )

    def test_empty_notice_is_rejected(self) -> None:
        self.assert_harness_error(
            """
            prompt:
              system: ""
              iteration_limit_notice: ""
            """,
            "iteration_limit_notice must be non-empty text",
        )

    def test_missing_file_is_rejected(self) -> None:
        with self.assertRaises(HarnessError):
            load_harness("/nonexistent/harness.yaml")

    def test_unresolvable_reference_lists_candidates(self) -> None:
        with self.assertRaisesRegex(HarnessError, "no-such-harness"):
            resolve_harness("no-such-harness")

    def test_resolve_none_returns_builtin_default(self) -> None:
        self.assertIs(resolve_harness(None), DEFAULT_HARNESS)
        self.assertIs(resolve_harness(""), DEFAULT_HARNESS)


class HarnessResolveTests(unittest.TestCase):
    def test_resolve_by_id_finds_repo_manifest(self) -> None:
        spec = resolve_harness("baseline-v0")
        self.assertEqual(spec.id, "baseline-v0")
        self.assertEqual(Path(spec.source).resolve(), BASELINE_YAML.resolve())

    def test_resolve_by_path_loads_file(self) -> None:
        self.assertEqual(resolve_harness(str(BASELINE_YAML)).id, "baseline-v0")


class TraceHarnessHeaderTests(unittest.TestCase):
    def test_jsonl_leads_with_harness_header_then_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            spec = HarnessSpec(id="header-check")
            trace = RunTrace(run_id="r1", output_path=path, harness=spec)
            trace.emit("agent.start")
            trace.emit("agent.end")

            self.assertEqual(len(trace.events), 2)
            asyncio.run(trace.flush())

            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(records[0]["record_type"], "harness")
            self.assertEqual(records[0]["harness_id"], "header-check")
            self.assertEqual(records[0]["genes_hash"], spec.genes_hash)
            self.assertEqual(records[0]["harness"]["id"], "header-check")
            self.assertEqual([r["event_type"] for r in records[1:]],
                             ["agent.start", "agent.end"])

    def test_trace_without_harness_writes_no_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            trace = RunTrace(run_id="r1", output_path=path)
            trace.emit("agent.start")

            asyncio.run(trace.flush())

            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(records[0]["event_type"], "agent.start")


if __name__ == "__main__":
    unittest.main()
