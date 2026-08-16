from __future__ import annotations

import ast
import csv
import json
import math
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
MISSION = ROOT / "missions" / "M11"
NOTEBOOK = ROOT / "labs" / "M11_decision_tree.ipynb"
DATASET = ROOT / "datasets" / "M11" / "learner_readiness.csv"


def notebook_cells() -> list[dict]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def gini(labels: list[int]) -> float:
    if not labels:
        return 0.0
    proportions = [labels.count(value) / len(labels) for value in set(labels)]
    return 1.0 - sum(proportion**2 for proportion in proportions)


class M11MissionPackageTests(unittest.TestCase):
    def test_required_standard_package_is_complete(self) -> None:
        mission_files = {
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
            "adr_prompt.md",
        }
        self.assertEqual(
            mission_files,
            {path.name for path in MISSION.iterdir() if path.is_file()},
        )
        self.assertTrue(NOTEBOOK.is_file())
        self.assertTrue(DATASET.is_file())
        self.assertTrue((DATASET.parent / "README.md").is_file())
        self.assertTrue((ROOT / "requirements" / "m11.txt").is_file())

    def test_manifest_matches_canonical_mission_and_runtime_contract(self) -> None:
        text = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        for expected in [
            "id: M11",
            "phase: P2",
            "flagship: V03",
            "decision trees",
            "model interpretation",
            "- M09",
            "cpu_only: true",
            "requires_secrets: false",
            "requires_paid_api: false",
            "network_required: false",
            "learner_evidence_prepopulated: false",
        ]:
            self.assertIn(expected, text)

        canonical = json.loads((ROOT / "data" / "missions.json").read_text())
        m11 = next(item for item in canonical["missions"] if item["id"] == "M11")
        self.assertEqual(m11["title"], "Interrogate a Decision Tree")
        self.assertIn(m11["objective"], text)

    def test_source_references_resolve_without_registry_edits(self) -> None:
        source_registry = json.loads(
            (ROOT / "data" / "source_registry.json").read_text(encoding="utf-8")
        )
        source_ids = {item["id"] for item in source_registry["sources"]}
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        self.assertIn("sklearn-guide", source_ids)
        self.assertIn("stanford-cs229", source_ids)
        self.assertIn("- sklearn-guide", manifest)
        self.assertIn("- stanford-cs229", manifest)

    def test_dataset_has_valid_schema_balance_and_ranges(self) -> None:
        with DATASET.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        expected_fields = [
            "learner_id",
            "study_hours_week",
            "practice_accuracy",
            "attendance_pct",
            "sleep_hours",
            "ready_for_assessment",
        ]
        self.assertEqual(list(rows[0]), expected_fields)
        self.assertEqual(len(rows), 96)
        self.assertEqual(len({row["learner_id"] for row in rows}), len(rows))

        labels = [int(row["ready_for_assessment"]) for row in rows]
        self.assertEqual(set(labels), {0, 1})
        self.assertGreaterEqual(min(labels.count(0), labels.count(1)), 40)
        for row in rows:
            self.assertTrue(2 <= int(row["study_hours_week"]) <= 12)
            self.assertTrue(0.0 <= float(row["practice_accuracy"]) <= 1.0)
            self.assertTrue(0 <= int(row["attendance_pct"]) <= 100)
            self.assertTrue(0.0 < float(row["sleep_hours"]) < 24.0)

    def test_impurity_intuition_has_a_real_numerical_invariant(self) -> None:
        parent = [0, 0, 1, 1]
        mixed_left, mixed_right = [0, 1], [0, 1]
        pure_left, pure_right = [0, 0], [1, 1]

        def weighted(left: list[int], right: list[int]) -> float:
            total = len(left) + len(right)
            return len(left) / total * gini(left) + len(right) / total * gini(right)

        self.assertTrue(math.isclose(gini(parent), 0.5))
        self.assertTrue(math.isclose(weighted(mixed_left, mixed_right), 0.5))
        self.assertTrue(math.isclose(weighted(pure_left, pure_right), 0.0))
        self.assertGreater(
            gini(parent) - weighted(pure_left, pure_right),
            gini(parent) - weighted(mixed_left, mixed_right),
        )

    def test_notebook_is_substantial_stable_and_output_free(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(len(notebook["cells"]), 30)
        ids = [cell.get("id") for cell in notebook["cells"]]
        self.assertTrue(all(ids))
        self.assertEqual(len(ids), len(set(ids)))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs", []), [])

    def test_notebook_code_compiles_and_has_no_network_or_secret_access(self) -> None:
        code_cells = [cell for cell in notebook_cells() if cell["cell_type"] == "code"]
        for cell in code_cells:
            source = "".join(cell.get("source", []))
            ast.parse(source, filename=f"M11-{cell['id']}")

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
            "read_csv(\"http",
            "read_csv('http",
        ]:
            self.assertNotIn(forbidden, all_code)

    def test_notebook_covers_required_tree_mechanics_and_experiments(self) -> None:
        cells = notebook_cells()
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in cells
            if cell["cell_type"] == "code"
        )
        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in cells
            if cell["cell_type"] == "markdown"
        )
        for symbol in [
            "DecisionTreeClassifier",
            "export_text",
            "decision_path",
            "tree_",
            "feature_importances_",
            "max_depth",
            "min_samples_leaf",
            "accuracy_score",
            "train_test_split",
            "describe_nodes",
            "explain_path",
            "generalization_gap",
        ]:
            self.assertIn(symbol, code)
        for phrase in [
            "threshold",
            "left branch",
            "right branch",
            "leaf",
            "impurity",
            "Predict before running",
            "Controlled failure",
            "causal",
            "No-AI gate",
            "ADR",
        ]:
            self.assertIn(phrase, markdown)

    def test_prediction_prompts_precede_each_consequential_action(self) -> None:
        ids = [cell["id"] for cell in notebook_cells()]
        pairs = [
            ("path-prediction", "path-action"),
            ("depth-prediction", "depth-action"),
            ("compare-prediction", "compare-action"),
            ("perturb-prediction", "perturb-action"),
            ("leaf-prediction", "leaf-action"),
            ("failure-prediction", "failure-action"),
        ]
        for prediction, action in pairs:
            self.assertLess(ids.index(prediction), ids.index(action))

    def test_experiment_and_evidence_contracts_cover_interpretation_limits(self) -> None:
        experiments = (MISSION / "experiments.yaml").read_text(encoding="utf-8")
        self.assertEqual(experiments.count("prediction_required: true"), 7)
        for term in [
            "Individual decision paths",
            "max_depth sweep",
            "Shallow versus deep",
            "One-feature perturbation",
            "Minimum leaf size",
            "Controlled overfitting failure",
            "Causal-claim audit",
        ]:
            self.assertIn(term, experiments)

        evidence = (MISSION / "evidence_contract.yaml").read_text(encoding="utf-8")
        self.assertIn("required_evidence:", evidence)
        self.assertIn("prefilled_learner_evidence: prohibited", evidence)
        self.assertIsNone(re.search(r"(?m)^\s*learner_evidence\s*:", evidence))
        self.assertIsNone(re.search(r"(?m)^\s*learner_response\s*:", evidence))

    def test_controlled_failure_and_adr_require_evidence_not_causal_claims(self) -> None:
        controlled = (MISSION / "controlled_failure.md").read_text(encoding="utf-8").lower()
        for term in [
            "unconstrained decision tree",
            "training accuracy",
            "test accuracy",
            "generalization gap",
            "smallest leaf",
            "causal overreach",
            "does not identify interventions",
        ]:
            self.assertIn(term, controlled)

        adr = (MISSION / "adr_prompt.md").read_text(encoding="utf-8")
        for heading in [
            "## Decision",
            "## Context",
            "## Alternatives considered",
            "## Evidence",
            "## Trade-offs",
            "## Revisit conditions",
            "## Status",
        ]:
            self.assertIn(heading, adr)
        self.assertIn("do not deploy", adr.lower())
        self.assertIn("real learners", adr)


if __name__ == "__main__":
    unittest.main()
