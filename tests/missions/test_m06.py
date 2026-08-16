from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import statistics
import unittest


ROOT = Path(__file__).resolve().parents[2]
MISSION = ROOT / "missions" / "M06"
NOTEBOOK = ROOT / "labs" / "M06_see_the_data.ipynb"
GUIDED_DATA = ROOT / "datasets" / "M06" / "support_tickets.csv"
FRESH_DATA = ROOT / "datasets" / "M06" / "community_programs_fresh.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class M06MissionPackageTests(unittest.TestCase):
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
            "review_brief.md",
        ]
        missing = [name for name in required if not (MISSION / name).is_file()]
        self.assertEqual(missing, [])
        self.assertTrue(NOTEBOOK.is_file())
        self.assertTrue(GUIDED_DATA.is_file())
        self.assertTrue(FRESH_DATA.is_file())
        self.assertTrue((ROOT / "requirements" / "m06.txt").is_file())

    def test_manifest_declares_m06_runtime_and_reasoning_contract(self) -> None:
        text = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        for expected in [
            "id: M06",
            "phase: P1",
            "flagship: V01",
            "pedagogy: question-first",
            "- M04",
            "cpu_only: true",
            "stable_cell_ids: true",
            "source_outputs_prefilled: false",
            "requires_secrets: false",
            "requires_paid_api: false",
            "network_required: false",
            "- visible_observation",
            "- inference",
            "- what_cannot_yet_be_concluded",
        ]:
            self.assertIn(expected, text)

    def test_guided_dataset_contains_contractual_quality_conditions(self) -> None:
        rows = read_csv(GUIDED_DATA)
        self.assertEqual(len(rows), 48)
        self.assertEqual(len({row["ticket_id"] for row in rows}), len(rows))

        targets = [int(row["escalated"]) for row in rows]
        positive_rate = sum(targets) / len(targets)
        self.assertGreater(positive_rate, 0.15)
        self.assertLess(positive_rate, 0.35)

        self.assertGreater(sum(not row["customer_tenure_months"] for row in rows), 0)
        self.assertGreater(sum(not row["satisfaction_score"] for row in rows), 0)

        response_times = [float(row["first_response_minutes"]) for row in rows]
        q1, _, q3 = statistics.quantiles(
            response_times, n=4, method="inclusive"
        )
        upper_fence = q3 + 1.5 * (q3 - q1)
        self.assertGreater(max(response_times), upper_fence)

        for row in rows:
            expected_priority = "urgent" if row["escalated"] == "1" else "standard"
            self.assertEqual(row["post_case_priority"], expected_priority)

    def test_fresh_no_ai_fixture_is_distinct_and_supports_denominators(self) -> None:
        guided = read_csv(GUIDED_DATA)
        fresh = read_csv(FRESH_DATA)
        self.assertGreaterEqual(len(fresh), 20)
        self.assertNotEqual(set(guided[0]), set(fresh[0]))
        self.assertEqual({row["day_type"] for row in fresh}, {"weekday", "weekend"})
        for row in fresh:
            registered = int(row["registered"])
            attended = int(row["attended"])
            self.assertGreater(registered, 0)
            self.assertGreaterEqual(attended, 0)
            self.assertLessEqual(attended, registered)

    def test_notebook_is_substantial_and_has_stable_unique_cell_ids(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(len(notebook["cells"]), 34)
        ids = [cell.get("id") for cell in notebook["cells"]]
        self.assertTrue(all(ids))
        self.assertEqual(len(ids), len(set(ids)))

    def test_notebook_source_is_clean_syntactically_valid_and_network_free(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        code_cells = [
            cell for cell in notebook["cells"] if cell.get("cell_type") == "code"
        ]
        for index, cell in enumerate(code_cells):
            self.assertIsNone(cell.get("execution_count"))
            self.assertEqual(cell.get("outputs", []), [])
            compile("".join(cell.get("source", [])), f"M06-cell-{index}", "exec")

        all_code = "\n".join("".join(cell.get("source", [])) for cell in code_cells)
        for forbidden in [
            "import requests",
            "import httpx",
            "from openai",
            "import openai",
            "api_key",
            "os.environ",
            "socket.",
            "read_html(",
            "sklearn",
            ".fit(",
        ]:
            self.assertNotIn(forbidden, all_code)

    def test_notebook_covers_each_visual_interrogation_and_reasoning_step(self) -> None:
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
            ".hist(",
            ".isna()",
            "iqr_outliers",
            ".boxplot(",
            ".scatter(",
            ".groupby(",
            "pd.crosstab",
            "class_counts",
            "set_ylim(0, 50)",
        ]:
            self.assertIn(token, code)

        for phrase in [
            "Start with questions, not chart types",
            "Predict before running",
            "Distribution",
            "Missingness",
            "Outliers",
            "Relationships",
            "Groups",
            "Class imbalance",
            "Possible leakage",
            "Visible observation",
            "Inference",
            "What cannot yet be concluded",
            "Controlled failure",
            "No-AI Gate",
        ]:
            self.assertIn(phrase, markdown)

    def test_controlled_failure_and_no_ai_gate_are_evidence_bearing(self) -> None:
        controlled = (MISSION / "controlled_failure.md").read_text(
            encoding="utf-8"
        ).lower()
        gate = (MISSION / "no_ai_gate.md").read_text(encoding="utf-8")

        for term in [
            "truncates",
            "underlying records",
            "identical",
            "wrongly infer",
            "repair",
            "causation",
        ]:
            self.assertIn(term, controlled)

        for phrase in [
            "fresh fixture",
            "without an AI assistant",
            "Choose a chart",
            "Visible observation",
            "Inference",
            "What cannot yet be concluded",
            "limitations",
        ]:
            self.assertIn(phrase, gate)

    def test_evidence_contract_does_not_prefill_learner_evidence(self) -> None:
        text = (MISSION / "evidence_contract.yaml").read_text(encoding="utf-8")
        self.assertIn("required_evidence:", text)
        self.assertIn("prefilled_learner_evidence: prohibited", text)
        self.assertIsNone(re.search(r"(?m)^\s*learner_evidence\s*:", text))
        self.assertIsNone(re.search(r"(?m)^\s*learner_response\s*:", text))

    def test_requirements_cover_visual_notebook_and_test_runtime(self) -> None:
        requirements = (ROOT / "requirements" / "m06.txt").read_text(
            encoding="utf-8"
        )
        for package in ["pandas", "numpy", "matplotlib", "nbformat", "nbconvert", "jupyter", "pytest"]:
            self.assertRegex(requirements, rf"(?m)^{package}[<=>]")


if __name__ == "__main__":
    unittest.main()
