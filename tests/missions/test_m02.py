from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
MISSION = ROOT / "missions" / "M02"
NOTEBOOK = ROOT / "labs" / "M02_first_ml_system.ipynb"
DATASET = ROOT / "datasets" / "M02" / "wine.csv"

EXPECTED_COLUMNS = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280_od315_of_diluted_wines",
    "proline",
    "target",
]


class M02MissionPackageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.code = "\n".join(
            "".join(cell.get("source", []))
            for cell in self.notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        self.markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in self.notebook["cells"]
            if cell.get("cell_type") == "markdown"
        )

    def test_required_artifacts_exist_and_manifest_paths_resolve(self) -> None:
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
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
        for name in required:
            self.assertTrue((MISSION / name).is_file(), name)

        artifact_paths = re.findall(
            r"^  - ((?:missions|labs|datasets|requirements|tests)/\S+)$",
            manifest,
            re.MULTILINE,
        )
        self.assertGreaterEqual(len(artifact_paths), 15)
        for relative_path in artifact_paths:
            self.assertTrue((ROOT / relative_path).is_file(), relative_path)

    def test_manifest_declares_complete_m02_contract(self) -> None:
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        for token in [
            "id: M02",
            "phase: P0",
            "flagship: V00",
            "pedagogy: whole-first",
            "formal_engineering_review: true",
            "cpu_only: true",
            "requires_secrets: false",
            "requires_paid_api: false",
            "network_required: false",
            "deterministic_seed: 7",
            "learner_evidence_prepopulated: false",
        ]:
            self.assertIn(token, manifest)
        for stage in [
            "raw data",
            "features and target",
            "train/test split",
            "model fit",
            "predictions",
            "evaluation",
            "interrogation",
        ]:
            self.assertIn(f"  - {stage}", manifest)

    def test_local_wine_fixture_has_expected_numeric_schema_and_classes(self) -> None:
        with DATASET.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(178, len(rows))
        self.assertEqual(EXPECTED_COLUMNS, list(rows[0]))

        class_counts = {0: 0, 1: 0, 2: 0}
        for row in rows:
            target = int(row["target"])
            self.assertIn(target, class_counts)
            class_counts[target] += 1
            for column in EXPECTED_COLUMNS[:-1]:
                self.assertTrue(math.isfinite(float(row[column])), column)
        self.assertEqual({0: 59, 1: 71, 2: 48}, class_counts)

    def test_notebook_is_substantial_with_stable_unique_cell_ids(self) -> None:
        self.assertEqual(4, self.notebook["nbformat"])
        self.assertGreaterEqual(len(self.notebook["cells"]), 30)
        ids = [cell.get("id") for cell in self.notebook["cells"]]
        self.assertTrue(all(ids))
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(
            all(re.fullmatch(r"m02-[a-z0-9-]+", cell_id) for cell_id in ids)
        )

    def test_source_notebook_has_no_prefilled_execution_evidence(self) -> None:
        for cell in self.notebook["cells"]:
            if cell.get("cell_type") == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual([], cell.get("outputs", []))
        for forbidden in ["learner_response:", "observed answer:", "my prediction is"]:
            self.assertNotIn(forbidden, self.markdown.lower())

    def test_notebook_code_compiles_and_has_no_network_or_secret_access(self) -> None:
        for index, cell in enumerate(self.notebook["cells"]):
            if cell.get("cell_type") == "code":
                compile("".join(cell.get("source", [])), f"M02-cell-{index}", "exec")
        for forbidden in [
            "import requests",
            "import httpx",
            "urllib",
            "from openai",
            "import openai",
            "api_key",
            "os.environ",
            "socket.",
            "http://",
            "https://",
        ]:
            self.assertNotIn(forbidden, self.code.lower())

    def test_whole_first_pipeline_boundaries_are_explicit_and_ordered(self) -> None:
        tokens = [
            "pd.read_csv(DATA_PATH)",
            "X = raw_data[FEATURE_COLUMNS].copy()",
            "train_test_split(",
            "baseline_model.fit(X_train, y_train)",
            "baseline_model.predict(X_test)",
            "accuracy_score(y_test, baseline_predictions)",
            "error_rows = raw_data.loc[test_indices].copy()",
        ]
        positions = [self.code.index(token) for token in tokens]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("fit_boundary", self.code)
        self.assertIn("prediction_boundary", self.code)
        self.assertIn("evaluation_boundary", self.code)

    def test_required_experiment_dimensions_and_prediction_prompts_are_present(self) -> None:
        experiments = (MISSION / "experiments.yaml").read_text(encoding="utf-8")
        for experiment_id in [
            "split",
            "features",
            "model",
            "hyperparameter",
            "labels",
            "evaluation-setup",
        ]:
            self.assertRegex(
                experiments,
                rf"(?m)^  - id: {re.escape(experiment_id)}$",
            )
        self.assertGreaterEqual(self.markdown.count("Predict before running"), 6)
        for symbol in [
            "split_results",
            "feature_results",
            "model_results",
            "hyperparameter_results",
            "corrupted_y_train",
            "cross_val_score",
        ]:
            self.assertIn(symbol, self.code)

    def test_controlled_failures_include_diagnosis_and_executable_guards(self) -> None:
        controlled = (
            MISSION / "controlled_failure.md"
        ).read_text(encoding="utf-8").lower()
        for term in [
            "corrupted training labels",
            "invalid self-comparison",
            "root cause",
            "repair",
            "verification",
        ]:
            self.assertIn(term, controlled)
        for code_token in [
            "label_disagreement_rate",
            "corrupted_label_accuracy",
            "invalid_accuracy",
            "assert invalid_accuracy == 1.0",
            "assert corrupted_label_accuracy < baseline_accuracy",
        ]:
            self.assertIn(code_token, self.code)

    def test_no_ai_gate_requires_fresh_run_interrogation_and_explanation(self) -> None:
        no_ai = (MISSION / "no_ai_gate.md").read_text(encoding="utf-8").lower()
        for phrase in [
            "without ai-generated code",
            "blank notebook or script",
            "fresh supervised classifier",
            "prediction written before execution",
            "interrogation of at least three errors",
            "fit, prediction, and evaluation boundaries",
            "limitations",
        ]:
            self.assertIn(phrase, no_ai)

    def test_evidence_contract_is_unpopulated_and_formal_review_is_complete(self) -> None:
        evidence = (
            MISSION / "evidence_contract.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("required_evidence:", evidence)
        self.assertNotRegex(
            evidence,
            r"(?m)^\s*learner_(?:evidence|response)\s*:",
        )

        review = (MISSION / "review_brief.md").read_text(encoding="utf-8")
        for heading in [
            "## System map",
            "## Evidence to inspect",
            "## Required experiments",
            "## Failure diagnosis",
            "## Validation criteria",
            "## Limitations and residual uncertainty",
            "## V00 integration decision",
        ]:
            self.assertIn(heading, review)

    def test_status_does_not_overclaim_learner_completion(self) -> None:
        status = (MISSION / "status.yaml").read_text(encoding="utf-8")
        self.assertIn("mission_id: M02", status)
        self.assertIn("implementation_status: implemented", status)
        self.assertIn("learner_evidence_status: intentionally_unpopulated", status)
        self.assertIn("formal_engineering_review: learner_review_required", status)
        self.assertIn("status: passed", status)
        self.assertNotIn("learner_evidence_status: complete", status)

    def test_requirements_are_bounded_and_cover_notebook_runtime(self) -> None:
        requirements = (
            ROOT / "requirements" / "m02.txt"
        ).read_text(encoding="utf-8").splitlines()
        expected = {
            "jupyter",
            "matplotlib",
            "nbconvert",
            "numpy",
            "pandas",
            "pytest",
            "scikit-learn",
        }
        packages = {
            re.split(r"[<>=!~]", line, maxsplit=1)[0]
            for line in requirements
            if line
        }
        self.assertEqual(expected, packages)
        self.assertTrue(all(">=" in line and ",<" in line for line in requirements))


if __name__ == "__main__":
    unittest.main()
