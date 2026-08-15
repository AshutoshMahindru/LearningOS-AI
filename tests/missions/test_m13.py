from __future__ import annotations

import csv
import importlib.util
import json
import math
import re
from pathlib import Path
import statistics
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import yaml


ROOT = Path(__file__).resolve().parents[2]
MISSION = ROOT / "missions" / "M13"
NOTEBOOK = ROOT / "labs" / "M13_knn.ipynb"
DATASET = ROOT / "datasets" / "M13" / "knn_scale_cases.csv"
GENERATOR = ROOT / "datasets" / "M13" / "generate_dataset.py"
REQUIREMENTS = ROOT / "requirements" / "m13.txt"


def load_notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET)


def fixed_split(data: pd.DataFrame):
    return train_test_split(
        data.index,
        test_size=0.25,
        random_state=13,
        stratify=data["learning_route"],
    )


class M13MissionPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = yaml.safe_load(
            (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        )

    def test_manifest_contract_and_all_declared_artifacts_exist(self) -> None:
        self.assertEqual(self.manifest["id"], "M13")
        self.assertEqual(self.manifest["title"], "Learn from Neighbors")
        self.assertEqual(self.manifest["phase"], "P2")
        self.assertEqual(self.manifest["flagship"], "V03")
        self.assertEqual(self.manifest["competencies"], ["KNN", "distance reasoning"])
        self.assertEqual(self.manifest["prerequisites"]["blocking"], ["M09"])
        self.assertEqual(self.manifest["prerequisites"]["helpful"], ["M15"])
        self.assertTrue(self.manifest["adr_required"])
        self.assertFalse(self.manifest["formal_engineering_review"])

        runtime = self.manifest["execution"]
        self.assertTrue(runtime["cpu_only"])
        self.assertTrue(runtime["restart_run_all_required"])
        self.assertFalse(runtime["requires_secrets"])
        self.assertFalse(runtime["requires_paid_api"])
        self.assertFalse(runtime["network_required"])

        artifacts = self.manifest["mission_artifacts"]
        self.assertGreaterEqual(len(artifacts), 17)
        for relative_path in artifacts:
            self.assertTrue((ROOT / relative_path).is_file(), relative_path)

    def test_canonical_metadata_supports_m13_contract(self) -> None:
        missions = json.loads(
            (ROOT / "data" / "missions.json").read_text(encoding="utf-8")
        )["missions"]
        m13 = next(mission for mission in missions if mission["id"] == "M13")
        self.assertEqual(m13["phase"], "P2")
        self.assertEqual(m13["flagship"], "V03")
        self.assertEqual(m13["competencies"], ["KNN", "distance reasoning"])

        dependencies = json.loads(
            (ROOT / "data" / "mission_dependencies.json").read_text(encoding="utf-8")
        )["dependencies"]["M13"]
        self.assertEqual(dependencies, {"blocking": ["M09"], "helpful": ["M15"]})

        source_ids = {
            source["id"]
            for source in json.loads(
                (ROOT / "data" / "source_registry.json").read_text(encoding="utf-8")
            )["sources"]
        }
        self.assertIn("sklearn-guide", source_ids)

    def test_dataset_schema_balance_ranges_and_weak_feature(self) -> None:
        with DATASET.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(
            list(rows[0]),
            [
                "case_id",
                "practice_hours",
                "assessment_score",
                "interface_event_count",
                "learning_route",
            ],
        )
        self.assertEqual(len(rows), 96)
        self.assertEqual(
            [row["case_id"] for row in rows],
            [f"M13-{index:03d}" for index in range(1, 97)],
        )

        labels = [row["learning_route"] for row in rows]
        self.assertEqual(labels.count("guided"), 48)
        self.assertEqual(labels.count("independent"), 48)

        practice = [float(row["practice_hours"]) for row in rows]
        scores = [float(row["assessment_score"]) for row in rows]
        events = [float(row["interface_event_count"]) for row in rows]
        encoded = [1.0 if label == "independent" else 0.0 for label in labels]

        self.assertLess(max(practice) - min(practice), 12)
        self.assertLess(max(scores) - min(scores), 50)
        self.assertGreater(max(events) - min(events), 8000)
        self.assertGreater(
            statistics.pstdev(events) / statistics.pstdev(practice),
            500,
        )

        weak_correlation = float(np.corrcoef(events, encoded)[0, 1])
        self.assertTrue(math.isfinite(weak_correlation))
        self.assertLess(abs(weak_correlation), 0.10)

    def test_checked_in_dataset_is_exact_generator_output(self) -> None:
        spec = importlib.util.spec_from_file_location("m13_dataset_generator", GENERATOR)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        generated = module.build_dataset()
        checked_in = load_dataset()
        assert_frame_equal(generated, checked_in, check_dtype=False)

    def test_notebook_is_substantial_stable_and_output_free(self) -> None:
        notebook = load_notebook()
        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(len(notebook["cells"]), 40)
        self.assertEqual(notebook["metadata"]["mission"]["id"], "M13")
        self.assertFalse(notebook["metadata"]["mission"]["network_required"])

        cell_ids = [cell.get("id") for cell in notebook["cells"]]
        self.assertTrue(all(cell_ids))
        self.assertEqual(len(cell_ids), len(set(cell_ids)))

        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell.get("execution_count"), cell["id"])
                self.assertEqual(cell.get("outputs", []), [], cell["id"])

    def test_prediction_checkpoints_precede_every_required_action(self) -> None:
        notebook = load_notebook()
        cell_ids = [cell["id"] for cell in notebook["cells"]]
        required_pairs = [
            ("predict-visual", "run-visual"),
            ("predict-query", "run-query"),
            ("predict-k", "run-k"),
            ("predict-metric", "run-metric"),
            ("predict-scale", "run-scale"),
            ("predict-failure", "run-failure"),
            ("predict-repair", "run-repair"),
            ("predict-boundary", "run-boundary"),
        ]
        for prediction_id, action_id in required_pairs:
            self.assertLess(
                cell_ids.index(prediction_id),
                cell_ids.index(action_id),
                f"{prediction_id} must precede {action_id}",
            )

        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        self.assertGreaterEqual(markdown.count("Prediction checkpoint"), 8)
        self.assertIn("prediction → run → observation", markdown)

    def test_notebook_covers_knn_distance_scaling_and_failure_mechanisms(self) -> None:
        notebook = load_notebook()
        all_code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )

        for token in [
            "KNeighborsClassifier",
            ".kneighbors(",
            "n_neighbors=5",
            'metric="euclidean"',
            '"manhattan"',
            "StandardScaler",
            "make_pipeline",
            "inverse_transform",
            "interface_event_count",
            "squared_contributions",
            "weak_distance_share",
            "plot_boundary",
            "raw_query_prediction != scaled_query_prediction",
        ]:
            self.assertIn(token, all_code)

        for phrase in [
            "nearest neighbors",
            "Controlled failure",
            "Code reading trace",
            "No-AI Gate",
            "ADR",
            "Boundary observation",
        ]:
            self.assertIn(phrase, markdown)

    def test_notebook_code_compiles_and_has_no_runtime_network_path(self) -> None:
        notebook = load_notebook()
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        for cell in code_cells:
            source = "".join(cell.get("source", []))
            compile(source, f"M13-{cell['id']}", "exec")

        all_code = "\n".join("".join(cell.get("source", [])) for cell in code_cells)
        for forbidden in [
            "import requests",
            "import httpx",
            "urlopen(",
            "fetch_openml",
            "api_key",
            "os.environ",
            "socket.",
            "http://",
            "https://",
        ]:
            self.assertNotIn(forbidden, all_code)

    def test_controlled_failure_is_behaviorally_reproducible_and_repaired(self) -> None:
        data = load_dataset()
        train_index, test_index = fixed_split(data)
        target = "learning_route"
        y_train = data.loc[train_index, target]
        y_test = data.loc[test_index, target]
        features = ["practice_hours", "assessment_score", "interface_event_count"]
        X_train = data.loc[train_index, features]
        X_test = data.loc[test_index, features]

        raw = KNeighborsClassifier(n_neighbors=5, metric="euclidean").fit(X_train, y_train)
        scaled = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
        ).fit(X_train, y_train)

        raw_accuracy = accuracy_score(y_test, raw.predict(X_test))
        scaled_accuracy = accuracy_score(y_test, scaled.predict(X_test))
        self.assertLessEqual(raw_accuracy, 0.65)
        self.assertGreaterEqual(scaled_accuracy, 0.85)
        self.assertGreaterEqual(scaled_accuracy - raw_accuracy, 0.25)

        query = pd.DataFrame([[6.0, 67.0, 9200]], columns=features)
        distances, positions = raw.kneighbors(query, n_neighbors=1)
        nearest = X_train.iloc[positions[0, 0]]
        squared = (nearest - query.iloc[0]) ** 2
        weak_share = squared["interface_event_count"] / squared.sum()
        self.assertGreater(weak_share, 0.99)
        self.assertGreater(distances[0, 0], 0)

    def test_scaling_and_k_change_the_fixed_query_for_explainable_reasons(self) -> None:
        data = load_dataset()
        train_index, test_index = fixed_split(data)
        features = ["practice_hours", "assessment_score"]
        X_train = data.loc[train_index, features]
        X_test = data.loc[test_index, features]
        y_train = data.loc[train_index, "learning_route"]
        y_test = data.loc[test_index, "learning_route"]
        query = pd.DataFrame([[6.0, 67.0]], columns=features)

        raw = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
        scaled = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)).fit(
            X_train, y_train
        )
        self.assertEqual(raw.predict(query)[0], "guided")
        self.assertEqual(scaled.predict(query)[0], "independent")
        self.assertGreater(
            accuracy_score(y_test, scaled.predict(X_test)),
            accuracy_score(y_test, raw.predict(X_test)),
        )

        predictions_by_k = {
            k: KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train).predict(query)[0]
            for k in [1, 3, 5, 9, 15, 25]
        }
        self.assertEqual(predictions_by_k[3], "guided")
        self.assertEqual(predictions_by_k[9], "independent")
        self.assertEqual(set(predictions_by_k.values()), {"guided", "independent"})

        metric_query = pd.DataFrame([[2.65, 62.0]], columns=features)
        metric_models = {
            metric: KNeighborsClassifier(n_neighbors=5, metric=metric).fit(X_train, y_train)
            for metric in ["euclidean", "manhattan"]
        }
        metric_predictions = {
            metric: model.predict(metric_query)[0]
            for metric, model in metric_models.items()
        }
        metric_neighbor_sets = {
            metric: set(model.kneighbors(metric_query, n_neighbors=5)[1][0])
            for metric, model in metric_models.items()
        }
        self.assertEqual(
            metric_predictions,
            {"euclidean": "independent", "manhattan": "guided"},
        )
        self.assertEqual(
            len(metric_neighbor_sets["euclidean"] & metric_neighbor_sets["manhattan"]),
            3,
        )

    def test_failure_evidence_and_adr_contract_are_not_prefilled(self) -> None:
        controlled = (MISSION / "controlled_failure.md").read_text(encoding="utf-8").lower()
        for phrase in [
            "incompatible high numeric scale",
            "decompose squared distance",
            "training rows only",
            "feature removal",
            "prevention check",
        ]:
            self.assertIn(phrase, controlled)

        adr = (MISSION / "adr_prompt.md").read_text(encoding="utf-8").lower()
        for phrase in [
            "raw features with euclidean distance",
            "standardized features with manhattan distance",
            "alternatives",
            "trade-offs",
            "revisit",
        ]:
            self.assertIn(phrase, adr)

        evidence = (MISSION / "evidence_contract.yaml").read_text(encoding="utf-8")
        self.assertIn("prefilled_learner_evidence: prohibited", evidence)
        self.assertIsNone(re.search(r"(?m)^\s*learner_response\s*:", evidence))
        self.assertIsNone(re.search(r"(?m)^\s*learner_evidence\s*:", evidence))

    def test_requirements_name_the_offline_cpu_stack(self) -> None:
        requirements = REQUIREMENTS.read_text(encoding="utf-8").lower()
        for package in [
            "numpy",
            "pandas",
            "matplotlib",
            "scikit-learn",
            "jupyter",
            "nbformat",
            "pyyaml",
            "pytest",
        ]:
            self.assertIn(package, requirements)


if __name__ == "__main__":
    unittest.main()
