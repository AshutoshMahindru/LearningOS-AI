from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import unittest


ROOT = Path(__file__).resolve().parents[2]
MISSION = ROOT / "missions" / "M08"
NOTEBOOK = ROOT / "labs" / "M08_regression.ipynb"
DATASET = ROOT / "datasets" / "M08" / "housing_regression.csv"

SAFE_FEATURES = [
    "floor_area_m2",
    "bedrooms",
    "building_age_years",
    "distance_to_transit_km",
    "neighborhood_score",
    "renovation_quality",
    "energy_efficiency",
    "local_job_access_score",
]


class M08MissionPackageTests(unittest.TestCase):
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
            "adr_prompt.md",
        ]
        for name in required:
            self.assertTrue((MISSION / name).is_file(), name)

        self.assertTrue(NOTEBOOK.is_file())
        self.assertTrue(DATASET.is_file())
        self.assertTrue((ROOT / "datasets" / "M08" / "generate_dataset.py").is_file())
        self.assertTrue((ROOT / "requirements" / "m08.txt").is_file())

    def test_manifest_declares_complete_runtime_and_review_contract(self) -> None:
        text = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        for expected in [
            "id: M08",
            "phase: P2",
            "flagship: V02",
            "pedagogy: whole-first",
            "formal_engineering_review: true",
            "adr_required: true",
            "cpu_only: true",
            "requires_secrets: false",
            "requires_paid_api: false",
            "network_required: false",
            "restart_run_all_required: true",
            "- baseline",
            "- train/test split",
            "- residuals",
            "- diagnose",
        ]:
            self.assertIn(expected, text)

    def test_registered_sources_cover_regression_and_evaluation(self) -> None:
        registry = json.loads((ROOT / "data" / "source_registry.json").read_text(encoding="utf-8"))
        sources = {source["id"]: source for source in registry["sources"]}
        self.assertIn("regression", sources["sklearn-guide"]["topics"])
        self.assertIn("metrics", sources["sklearn-guide"]["topics"])
        self.assertIn("regression", sources["stanford-cs229"]["topics"])

    def test_dataset_schema_values_and_controlled_leakage_fixture(self) -> None:
        with DATASET.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        expected_columns = [
            "transaction_id",
            *SAFE_FEATURES,
            "post_sale_assessment_k",
            "sale_price_k",
        ]
        self.assertEqual(list(rows[0]), expected_columns)
        self.assertEqual(len(rows), 320)
        self.assertEqual(len({row["transaction_id"] for row in rows}), 320)

        numeric_columns = [*SAFE_FEATURES, "post_sale_assessment_k", "sale_price_k"]
        for row in rows:
            for column in numeric_columns:
                self.assertTrue(math.isfinite(float(row[column])), (row["transaction_id"], column))

        target = [float(row["sale_price_k"]) for row in rows]
        proxy = [float(row["post_sale_assessment_k"]) for row in rows]
        self.assertGreater(statistics.pstdev(target), 80.0)
        self.assertGreater(self._correlation(target, proxy), 0.999)

    def test_dataset_is_reproducible_from_the_committed_generator(self) -> None:
        generator = ROOT / "datasets" / "M08" / "generate_dataset.py"
        namespace: dict[str, object] = {
            "__name__": "m08_dataset_generator_test",
            "__file__": str(generator),
        }
        exec(compile(generator.read_text(encoding="utf-8"), str(generator), "exec"), namespace)

        rows = namespace["build_rows"]()
        self.assertEqual(len(rows), 320)
        self.assertEqual(rows[0]["transaction_id"], "TX-0001")
        self.assertEqual(rows[-1]["transaction_id"], "TX-0320")

        with DATASET.open(encoding="utf-8", newline="") as handle:
            committed_rows = list(csv.DictReader(handle))
        normalized_generated_rows = [
            {key: str(value) for key, value in row.items()}
            for row in rows
        ]
        self.assertEqual(normalized_generated_rows, committed_rows)

        digest = hashlib.sha256(DATASET.read_bytes()).hexdigest()
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual(
            digest,
            "aea908839b9353f305c55514d1a017d6f4ff59d7b6b547a90212926f3e2c4a6c",
        )

    def test_notebook_is_substantial_stable_and_output_free(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(len(notebook["cells"]), 30)

        ids = [cell.get("id") for cell in notebook["cells"]]
        self.assertTrue(all(ids))
        self.assertEqual(len(ids), len(set(ids)))

        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs", []), [])

    def test_notebook_follows_the_whole_first_sequence(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "markdown"
        )
        sequence = [
            "## 2. Baseline",
            "## 3. Train/test split",
            "## 4. Fit",
            "## 5. Predict",
            "## 6. Metrics",
            "## 7. Residuals",
            "## 8. Diagnose",
        ]
        positions = [markdown.index(label) for label in sequence]
        self.assertEqual(positions, sorted(positions))

    def test_notebook_covers_metrics_diagnostics_cv_influence_and_leakage(self) -> None:
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
            "DummyRegressor",
            "train_test_split",
            ".fit(",
            ".predict(",
            "mean_absolute_error",
            "root_mean_squared_error",
            "r2_score",
            "residual",
            "KFold",
            "cross_validate",
            "permutation_importance",
            "DecisionTreeRegressor",
            "post_sale_assessment_k",
            "FEATURE_AVAILABILITY",
            "invalid_evaluation_metrics",
        ]:
            self.assertIn(token, code)

        for phrase in [
            "Prediction before action",
            "Controlled failure",
            "underfitting",
            "overfitting",
            "not causal",
            "No-AI Gate",
            "ADR",
        ]:
            self.assertIn(phrase, markdown)

    def test_predictions_precede_material_experiment_code(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cells = notebook["cells"]
        experiment_ids = {
            "fit-models",
            "residual-plots",
            "cross-validation",
            "capacity-diagnosis",
            "permutation-influence",
            "leakage-failure",
            "training-evaluation-failure",
        }
        for index, cell in enumerate(cells):
            if cell.get("id") in experiment_ids:
                self.assertGreater(index, 0)
                previous = "".join(cells[index - 1].get("source", []))
                self.assertIn("Prediction before action", previous, cell.get("id"))

    def test_notebook_code_is_syntactically_valid_deterministic_and_network_free(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
        for index, cell in enumerate(code_cells):
            source = "".join(cell.get("source", []))
            compile(source, f"M08-cell-{index}", "exec")

        all_code = "\n".join("".join(cell.get("source", [])) for cell in code_cells)
        self.assertIn("random_state=RANDOM_STATE", all_code)
        self.assertIn("n_jobs=1", all_code)
        for forbidden in [
            "import requests",
            "import httpx",
            "from openai",
            "import openai",
            "api_key",
            "os.environ",
            "socket.",
            "read_html",
            "read_json(\"http",
            "read_csv(\"http",
        ]:
            self.assertNotIn(forbidden, all_code)

    def test_failure_review_and_adr_contracts_are_explicit(self) -> None:
        controlled = (MISSION / "controlled_failure.md").read_text(encoding="utf-8").lower()
        review = (MISSION / "review_brief.md").read_text(encoding="utf-8").lower()
        adr = (MISSION / "adr_prompt.md").read_text(encoding="utf-8").lower()

        for term in ["target leakage", "availability", "cross-validation", "training rows", "prevention"]:
            self.assertIn(term, controlled)
        for term in ["architecture", "failure analysis", "known uncertainty", "accepted", "rejected", "deferred"]:
            self.assertIn(term, review)
        for heading in ["## decision", "## context", "## alternatives considered", "## evidence", "## trade-offs", "## revisit conditions", "## status"]:
            self.assertIn(heading, adr)

    def test_evidence_is_required_but_not_prefilled(self) -> None:
        text = (MISSION / "evidence_contract.yaml").read_text(encoding="utf-8")
        self.assertIn("required_evidence:", text)
        self.assertIn("prefilled_learner_evidence: prohibited", text)
        self.assertIsNone(re.search(r"(?m)^\s*learner_evidence\s*:", text))
        self.assertIsNone(re.search(r"(?m)^\s*learner_response\s*:", text))

    def test_model_contract_beats_baseline_and_exposes_leakage_when_dependencies_exist(self) -> None:
        try:
            import pandas as pd
            from sklearn.dummy import DummyRegressor
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_absolute_error, r2_score
            from sklearn.model_selection import train_test_split
        except ImportError as exc:
            self.skipTest(f"M08 optional runtime dependencies not installed: {exc}")

        frame = pd.read_csv(DATASET)
        train, test = train_test_split(frame, test_size=0.2, random_state=42)
        baseline = DummyRegressor(strategy="mean").fit(train[SAFE_FEATURES], train["sale_price_k"])
        model = RandomForestRegressor(
            n_estimators=160,
            min_samples_leaf=3,
            max_features=0.8,
            random_state=42,
            n_jobs=1,
        ).fit(train[SAFE_FEATURES], train["sale_price_k"])
        leaky = LinearRegression().fit(
            train[[*SAFE_FEATURES, "post_sale_assessment_k"]], train["sale_price_k"]
        )

        baseline_prediction = baseline.predict(test[SAFE_FEATURES])
        model_prediction = model.predict(test[SAFE_FEATURES])
        leaky_prediction = leaky.predict(test[[*SAFE_FEATURES, "post_sale_assessment_k"]])
        baseline_mae = mean_absolute_error(test["sale_price_k"], baseline_prediction)
        model_mae = mean_absolute_error(test["sale_price_k"], model_prediction)
        model_r2 = r2_score(test["sale_price_k"], model_prediction)
        leaky_r2 = r2_score(test["sale_price_k"], leaky_prediction)

        self.assertLess(model_mae, baseline_mae * 0.55)
        self.assertGreater(model_r2, 0.75)
        self.assertGreater(leaky_r2, 0.995)
        self.assertGreater(leaky_r2, model_r2)

    @staticmethod
    def _correlation(left: list[float], right: list[float]) -> float:
        left_mean = statistics.mean(left)
        right_mean = statistics.mean(right)
        numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
        denominator = math.sqrt(
            sum((x - left_mean) ** 2 for x in left)
            * sum((y - right_mean) ** 2 for y in right)
        )
        return numerator / denominator


if __name__ == "__main__":
    unittest.main()
