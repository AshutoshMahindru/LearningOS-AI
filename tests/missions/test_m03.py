from __future__ import annotations

import json
import re
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
MISSION = ROOT / "missions" / "M03"
NOTEBOOK = ROOT / "labs" / "M03_python_modification.ipynb"


class M03MissionPackageTests(unittest.TestCase):
    def test_required_artifacts_exist(self) -> None:
        required = [
            "manifest.yaml",
            "README.md",
            "content.yaml",
            "experiments.yaml",
            "code_reading.md",
            "no_ai_gate.md",
            "controlled_failure.md",
            "assessment.yaml",
            "evidence_contract.yaml",
            "flagship_integration.md",
            "status.yaml",
        ]

        for name in required:
            self.assertTrue((MISSION / name).is_file(), name)

        self.assertTrue(NOTEBOOK.is_file())
        self.assertTrue((ROOT / "requirements" / "m03.txt").is_file())

    def test_manifest_declares_m03_contract(self) -> None:
        text = (MISSION / "manifest.yaml").read_text(encoding="utf-8")

        for expected in [
            "id: M03",
            "phase: P1",
            "flagship: V01",
            "pedagogy: whole-first",
            "cpu_only: true",
            "requires_secrets: false",
            "requires_paid_api: false",
            "network_required: false",
            "- python-tutorial",
        ]:
            self.assertIn(expected, text)

    def test_source_registry_contains_authoritative_python_reference(self) -> None:
        registry = json.loads(
            (ROOT / "data" / "source_registry.json").read_text(encoding="utf-8")
        )
        source_ids = {source["id"] for source in registry["sources"]}
        self.assertIn("python-tutorial", source_ids)

    def test_notebook_is_substantial_and_has_stable_cell_ids(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(len(notebook["cells"]), 25)

        ids = [cell.get("id") for cell in notebook["cells"]]
        self.assertTrue(all(ids))
        self.assertEqual(len(ids), len(set(ids)))

    def test_notebook_covers_required_python_modification_and_debugging(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "markdown"
        )

        for token in [
            "FREE_SHIPPING_THRESHOLD",
            "MEMBER_DISCOUNT",
            "def final_total",
            "for order in",
            "observe_exception",
            "NameError",
            "TypeError",
            "KeyError",
            "shipping_fee_broken",
            "last_id_broken",
            "buggy_final_total",
            "trace_buggy_total",
            "assert",
        ]:
            self.assertIn(token, code)

        for phrase in [
            "Predict before running",
            "Controlled failure",
            "Code reading",
            "No-AI Gate",
            "Explain",
        ]:
            self.assertIn(phrase, markdown)

    def test_notebook_code_is_syntactically_valid_and_network_free(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

        code_cells = [
            cell
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        ]

        for index, cell in enumerate(code_cells):
            source = "".join(cell.get("source", []))
            compile(source, f"M03-cell-{index}", "exec")

        all_code = "\n".join(
            "".join(cell.get("source", []))
            for cell in code_cells
        )

        for forbidden in [
            "import requests",
            "import httpx",
            "from openai",
            "import openai",
            "api_key",
            "os.environ",
            "socket.",
        ]:
            self.assertNotIn(forbidden, all_code)

    def test_source_notebook_contains_no_prefilled_execution_outputs(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs", []), [])

    def test_debugging_and_transfer_contract_is_explicit(self) -> None:
        controlled = (
            MISSION / "controlled_failure.md"
        ).read_text(encoding="utf-8").lower()

        no_ai = (
            MISSION / "no_ai_gate.md"
        ).read_text(encoding="utf-8").lower()

        assessment = (
            MISSION / "assessment.yaml"
        ).read_text(encoding="utf-8").lower()

        for term in [
            "one seeded root cause",
            "hypothesis",
            "trace",
            "smallest",
            "repair",
            "verification",
        ]:
            self.assertIn(term, controlled)

        self.assertIn("without ai-generated code", no_ai)
        self.assertIn("fresh program", no_ai)
        self.assertIn("transfer_required: true", assessment)
        self.assertIn("predict_behavior_before_execution", assessment)
        self.assertIn("diagnose_failure_from_evidence", assessment)

    def test_evidence_contract_has_no_prefilled_learner_evidence(self) -> None:
        text = (
            MISSION / "evidence_contract.yaml"
        ).read_text(encoding="utf-8")

        self.assertIn("required_evidence:", text)
        self.assertIsNone(
            re.search(r"(?m)^\s*learner_evidence\s*:", text)
        )
        self.assertIsNone(
            re.search(r"(?m)^\s*learner_response\s*:", text)
        )

    def test_v01_integration_is_explicit(self) -> None:
        text = (
            MISSION / "flagship_integration.md"
        ).read_text(encoding="utf-8")

        self.assertIn("V01", text)
        self.assertIn("Structured Data Workbench", text)
        self.assertIn("trace", text.lower())
        self.assertIn("unfamiliar Python", text)


if __name__ == "__main__":
    unittest.main()
