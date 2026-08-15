from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
MISSION = ROOT / "missions" / "M14"
NOTEBOOK = ROOT / "labs" / "M14_discover_structure_without_labels.ipynb"
DATASET = ROOT / "datasets" / "M14" / "learning_sessions.csv"
REQUIREMENTS = ROOT / "requirements" / "m14.txt"

EXPECTED_FEATURES = [
    "active_minutes",
    "practice_ratio",
    "review_ratio",
    "help_requests",
    "context_switches",
    "completion_fraction",
    "activity_events",
]


class M14MissionContractTests(unittest.TestCase):
    def test_standard_package_and_formal_review_artifacts_are_declared(self) -> None:
        required = {
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
        self.assertEqual(required, {path.name for path in MISSION.iterdir() if path.is_file()})
        self.assertTrue(NOTEBOOK.is_file())
        self.assertTrue(DATASET.is_file())
        self.assertTrue((DATASET.parent / "README.md").is_file())
        self.assertTrue(REQUIREMENTS.is_file())

        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        for contract in [
            "id: M14",
            "phase: P2",
            "flagship: V03",
            "formal_engineering_review: true",
            "adr_required: true",
            "cpu_only: true",
            "requires_secrets: false",
            "network_required: false",
            "target_labels_available_to_lab: false",
        ]:
            self.assertIn(contract, manifest)
        for artifact in required:
            self.assertIn(f"missions/M14/{artifact}", manifest)

    def test_evidence_contract_does_not_prefill_learner_work(self) -> None:
        contract = (MISSION / "evidence_contract.yaml").read_text(encoding="utf-8")
        self.assertIn("required_evidence:", contract)
        self.assertIn("evidence_against_selected_k", contract)
        self.assertIn("clustering_adr", contract)
        for forbidden in ["learner_evidence:", "learner_response:", "learner_score:"]:
            self.assertNotIn(forbidden, contract)

    def test_review_and_adr_cover_consequential_choices_and_limits(self) -> None:
        review = (MISSION / "review_brief.md").read_text(encoding="utf-8").lower()
        adr = (MISSION / "adr_prompt.md").read_text(encoding="utf-8").lower()
        combined = review + adr
        for concept in [
            "feature",
            "standardscaler",
            "euclidean",
            "k-means",
            "silhouette",
            "stability",
            "outlier",
            "visualization",
            "reversal trigger",
            "not true classes",
        ]:
            self.assertIn(concept, combined)
        self.assertIn("approve or reject", review)
        self.assertIn("alternative", adr)
        self.assertIn("evidence against", adr)


class M14DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with DATASET.open(encoding="utf-8", newline="") as handle:
            cls.rows = list(csv.DictReader(handle))
            cls.columns = list(cls.rows[0])

    def test_fixture_is_unlabelled_numeric_and_traceable(self) -> None:
        self.assertEqual(["session_id", *EXPECTED_FEATURES], self.columns)
        self.assertEqual(54, len(self.rows))
        ids = [row["session_id"] for row in self.rows]
        self.assertEqual(len(ids), len(set(ids)))
        forbidden_tokens = ("target", "label", "class", "segment", "cluster", "cohort")
        self.assertFalse(any(token in column.lower() for token in forbidden_tokens for column in self.columns))
        for row in self.rows:
            for feature in EXPECTED_FEATURES:
                value = float(row[feature])
                self.assertTrue(value == value, f"non-finite value in {feature}")
            for ratio in ["practice_ratio", "review_ratio", "completion_fraction"]:
                self.assertGreaterEqual(float(row[ratio]), 0.0)
                self.assertLessEqual(float(row[ratio]), 1.0)

    def test_scale_trap_is_crossed_with_broad_session_profiles(self) -> None:
        event_values = [float(row["activity_events"]) for row in self.rows]
        active_values = [float(row["active_minutes"]) for row in self.rows]
        self.assertGreater(max(event_values) - min(event_values), 100 * (max(active_values) - min(active_values)))

        event_bands = [
            sum(value < 2_000 for value in event_values),
            sum(2_000 <= value < 6_000 for value in event_values),
            sum(value >= 6_000 for value in event_values),
        ]
        self.assertEqual([18, 18, 18], event_bands)
        for block_start in (0, 18, 36):
            block = event_values[block_start : block_start + 18]
            self.assertEqual([6, 6, 6], [
                sum(value < 2_000 for value in block),
                sum(2_000 <= value < 6_000 for value in block),
                sum(value >= 6_000 for value in block),
            ])


class M14NotebookSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
        )
        cls.markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "markdown"
        )

    def test_notebook_has_stable_ids_and_clean_source_state(self) -> None:
        self.assertEqual(4, self.notebook["nbformat"])
        self.assertGreaterEqual(len(self.notebook["cells"]), 20)
        ids = [cell.get("id") for cell in self.notebook["cells"]]
        self.assertTrue(all(ids))
        self.assertEqual(len(ids), len(set(ids)))
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell.get("execution_count"), cell["id"])
                self.assertEqual([], cell.get("outputs"), cell["id"])

    def test_notebook_implements_the_clustering_contract(self) -> None:
        for symbol in [
            "SELECTED_FEATURES",
            "StandardScaler",
            "KMeans",
            "evaluate_candidate",
            "silhouette_score",
            "silhouette_samples",
            "adjusted_rand_score",
            "inverse_transform",
            "distance_to_center",
            "PCA",
        ]:
            self.assertIn(symbol, self.code)
        self.assertIn("range(2, 7)", self.code)
        self.assertIn("MIN_ACCEPTABLE_CLUSTER_SIZE", self.code)
        self.assertIn("MIN_ACCEPTABLE_STABILITY", self.code)
        self.assertGreaterEqual(self.markdown.count("Prediction before action"), 6)
        self.assertIn("clusters are not true classes", self.markdown.lower())

    def test_notebook_has_no_runtime_network_secret_or_target_dependency(self) -> None:
        for forbidden in [
            "import requests",
            "import httpx",
            "from openai",
            "import openai",
            "api_key",
            "os.environ",
            "read_html(",
            "http://",
            "https://",
        ]:
            self.assertNotIn(forbidden, self.code.lower())
        self.assertNotIn("y_true", self.code)
        self.assertNotIn("target_column", self.code)


@unittest.skipUnless(
    importlib.util.find_spec("numpy")
    and importlib.util.find_spec("pandas")
    and importlib.util.find_spec("sklearn"),
    "M14 scientific requirements are not installed",
)
class M14RuntimeInvariantTests(unittest.TestCase):
    def test_scale_failure_candidate_diagnostics_and_outlier_are_real(self) -> None:
        import numpy as np
        import pandas as pd
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score, silhouette_score
        from sklearn.preprocessing import StandardScaler

        data = pd.read_csv(DATASET)
        raw = data[EXPECTED_FEATURES].to_numpy(dtype=float)
        raw_labels = KMeans(n_clusters=3, n_init=20, random_state=14).fit_predict(raw)
        pairwise = raw[:, None, :] - raw[None, :, :]
        shares = np.square(pairwise).sum(axis=(0, 1))
        shares = shares / shares.sum()
        self.assertEqual("activity_events", EXPECTED_FEATURES[int(shares.argmax())])
        self.assertGreater(float(shares.max()), 0.99)
        self.assertGreater(silhouette_score(raw, raw_labels), 0.90)

        scaled = StandardScaler().fit_transform(raw)
        scaled_labels = KMeans(n_clusters=3, n_init=20, random_state=14).fit_predict(scaled)
        self.assertLess(adjusted_rand_score(raw_labels, scaled_labels), 0.10)

        diagnostics = []
        for k in range(2, 7):
            model = KMeans(n_clusters=k, n_init=20, random_state=14).fit(scaled)
            repeat = KMeans(n_clusters=k, n_init=20, random_state=1014 + k).fit(scaled)
            diagnostics.append((
                k,
                model.inertia_,
                silhouette_score(scaled, model.labels_),
                int(np.bincount(model.labels_).min()),
                adjusted_rand_score(model.labels_, repeat.labels_),
            ))
        self.assertTrue(all(left[1] > right[1] for left, right in zip(diagnostics, diagnostics[1:])))
        eligible = [row for row in diagnostics if row[3] >= 5 and row[4] >= 0.90]
        selected_k = max(eligible, key=lambda row: row[2])[0]
        self.assertEqual(2, selected_k)

        baseline = KMeans(n_clusters=selected_k, n_init=20, random_state=14).fit_predict(scaled)
        corrupted = data.copy()
        corrupted.loc[len(corrupted)] = ["LS_CORRUPTED", 250, 0.05, 0.95, 30, 50, 0.10, 60000]
        corrupted_scaled = StandardScaler().fit_transform(corrupted[EXPECTED_FEATURES])
        stressed = KMeans(n_clusters=selected_k, n_init=20, random_state=14).fit_predict(corrupted_scaled)
        self.assertEqual(1, int(np.bincount(stressed).min()))
        self.assertLess(adjusted_rand_score(baseline, stressed[:-1]), 0.10)
        self.assertGreater(silhouette_score(corrupted_scaled, stressed), silhouette_score(scaled, baseline))


if __name__ == "__main__":
    unittest.main()
