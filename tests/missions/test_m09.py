from __future__ import annotations

from contextlib import chdir, redirect_stdout
import csv
import importlib.util
import io
import json
from pathlib import Path
import re
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MISSION = ROOT / "missions" / "M09"
NOTEBOOK = ROOT / "labs" / "M09_binary_classification.ipynb"
DATASET_DIR = ROOT / "datasets" / "M09"
DATASET = DATASET_DIR / "learner_disengagement.csv"
GENERATOR = DATASET_DIR / "generate_dataset.py"


def load_notebook() -> dict[str, object]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def load_generator_module():
    spec = importlib.util.spec_from_file_location("m09_dataset_generator", GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load M09 dataset generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute_notebook_code() -> dict[str, object]:
    notebook = load_notebook()
    namespace: dict[str, object] = {"__name__": "__m09_test__"}
    output = io.StringIO()

    with chdir(ROOT), redirect_stdout(output):
        for cell in notebook["cells"]:
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            cell_id = cell.get("id", "unknown")
            exec(compile(source, f"M09-notebook-{cell_id}", "exec"), namespace)

    namespace["_captured_stdout"] = output.getvalue()
    return namespace


class M09MissionPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.namespace = execute_notebook_code()

    def test_required_standard_package_exists(self) -> None:
        required_mission_files = [
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
        missing = [name for name in required_mission_files if not (MISSION / name).is_file()]
        self.assertEqual(missing, [])
        self.assertTrue(NOTEBOOK.is_file())
        self.assertTrue(DATASET.is_file())
        self.assertTrue(GENERATOR.is_file())
        self.assertTrue((DATASET_DIR / "README.md").is_file())
        self.assertTrue((ROOT / "requirements" / "m09.txt").is_file())

    def test_manifest_declares_m09_execution_contract(self) -> None:
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        expected_fragments = [
            "id: M09",
            "title: Make Binary Decisions",
            "phase: P2",
            "flagship: V02",
            "pedagogy: whole-first",
            "cpu_only: true",
            "requires_secrets: false",
            "requires_paid_api: false",
            "network_required: false",
            "- classification",
            "- probability",
            "datasets/M09/learner_disengagement.csv",
        ]
        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, manifest)

    def test_whole_first_sequence_is_exact_and_ordered(self) -> None:
        content = (MISSION / "content.yaml").read_text(encoding="utf-8")
        sequence = [
            "- id: baseline",
            "- id: split",
            "- id: classifier",
            "- id: predicted_probabilities",
            "- id: default_classification",
            "- id: confusion_matrix",
            "- id: threshold_changes",
            "- id: consequences",
        ]
        positions = [content.index(item) for item in sequence]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(len(positions), len(set(positions)))

    def test_dataset_is_deterministic_binary_and_imbalanced(self) -> None:
        with DATASET.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = reader.fieldnames

        self.assertEqual(
            fieldnames,
            [
                "learner_id",
                "account_age_days",
                "weekly_sessions",
                "overdue_tasks",
                "assessment_score",
                "help_requests",
                "disengaged_next_30_days",
            ],
        )
        self.assertEqual(len(rows), 180)
        self.assertEqual(len({row["learner_id"] for row in rows}), 180)

        targets = [int(row["disengaged_next_30_days"]) for row in rows]
        self.assertEqual(set(targets), {0, 1})
        self.assertEqual(sum(targets), 44)
        self.assertGreater(sum(targets) / len(targets), 0.20)
        self.assertLess(sum(targets) / len(targets), 0.30)

        generator = load_generator_module()
        generated = [
            {key: str(value) for key, value in row.items()}
            for row in generator.generate_rows()
        ]
        self.assertEqual(generated, rows)

    def test_dataset_generator_reproduces_committed_bytes(self) -> None:
        generator = load_generator_module()
        with tempfile.TemporaryDirectory() as directory:
            regenerated = Path(directory) / "regenerated.csv"
            generator.write_dataset(regenerated)
            self.assertEqual(regenerated.read_bytes(), DATASET.read_bytes())

    def test_notebook_has_stable_ids_no_outputs_and_whole_first_route(self) -> None:
        notebook = load_notebook()
        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(len(notebook["cells"]), 30)

        cell_ids = [cell.get("id") for cell in notebook["cells"]]
        self.assertTrue(all(cell_ids))
        self.assertEqual(len(cell_ids), len(set(cell_ids)))

        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                self.assertIsNone(cell.get("execution_count"), cell.get("id"))
                self.assertEqual(cell.get("outputs", []), [], cell.get("id"))

        markdown_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "markdown"
        )
        route = (
            "baseline → split → classifier → predicted probabilities → "
            "default classification → confusion matrix → threshold changes → consequences"
        )
        self.assertIn(route, markdown_text)
        self.assertGreaterEqual(markdown_text.count("Predict before running"), 7)

    def test_notebook_code_is_syntactically_valid_offline_and_secret_free(self) -> None:
        notebook = load_notebook()
        code_texts = []
        for cell in notebook["cells"]:
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            compile(source, f"M09-cell-{cell.get('id')}", "exec")
            code_texts.append(source)

        all_code = "\n".join(code_texts)
        forbidden = [
            "import requests",
            "import httpx",
            "urllib",
            "socket.",
            "from openai",
            "import openai",
            "api_key",
            "os.environ",
            "subprocess",
        ]
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, all_code)

        required_symbols = [
            "stratified_split",
            "fit_logistic_classifier",
            "predict_probabilities",
            "classify",
            "confusion_counts",
            "metric_summary",
            "calibration_bins",
            "brier_score",
        ]
        for symbol in required_symbols:
            with self.subTest(symbol=symbol):
                self.assertIn(f"def {symbol}", all_code)

    def test_executed_notebook_preserves_split_and_probability_invariants(self) -> None:
        namespace = self.namespace
        train_rows = namespace["train_rows"]
        test_rows = namespace["test_rows"]
        probabilities = namespace["test_probabilities"]

        self.assertEqual((len(train_rows), len(test_rows)), (135, 45))
        train_ids = {row["learner_id"] for row in train_rows}
        test_ids = {row["learner_id"] for row in test_rows}
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertEqual(sum(row["disengaged_next_30_days"] for row in test_rows), 11)
        self.assertEqual(len(probabilities), 45)
        self.assertTrue(all(0.0 < probability < 1.0 for probability in probabilities))
        self.assertLess(min(probabilities), 0.01)
        self.assertGreater(max(probabilities), 0.80)

    def test_confusion_matrix_orientation_and_metric_denominators(self) -> None:
        confusion_counts = self.namespace["confusion_counts"]
        metric_summary = self.namespace["metric_summary"]

        counts = confusion_counts([0, 0, 1, 1], [0, 1, 0, 1])
        self.assertEqual(counts, {"tn": 1, "fp": 1, "fn": 1, "tp": 1})
        self.assertEqual(
            metric_summary(counts),
            {"accuracy": 0.5, "precision": 0.5, "recall": 0.5},
        )

        asymmetric = {"tn": 7, "fp": 3, "fn": 1, "tp": 9}
        metrics = metric_summary(asymmetric)
        self.assertAlmostEqual(metrics["accuracy"], 0.8)
        self.assertAlmostEqual(metrics["precision"], 0.75)
        self.assertAlmostEqual(metrics["recall"], 0.9)

    def test_default_threshold_and_imbalance_failure_are_observable(self) -> None:
        self.assertEqual(
            self.namespace["baseline_counts"],
            {"tn": 34, "fp": 0, "fn": 11, "tp": 0},
        )
        self.assertEqual(
            self.namespace["default_counts"],
            {"tn": 33, "fp": 1, "fn": 8, "tp": 3},
        )
        baseline_metrics = self.namespace["baseline_metrics"]
        default_metrics = self.namespace["default_metrics"]
        self.assertGreater(baseline_metrics["accuracy"], 0.75)
        self.assertEqual(baseline_metrics["recall"], 0.0)
        self.assertGreater(default_metrics["accuracy"], baseline_metrics["accuracy"])
        self.assertLess(default_metrics["recall"], 0.30)

    def test_threshold_changes_policy_not_model_and_exposes_tradeoffs(self) -> None:
        results = {
            row["threshold"]: row for row in self.namespace["threshold_results"]
        }
        self.assertEqual(set(results), {0.20, 0.30, 0.50, 0.70})
        self.assertEqual(
            [results[value]["predicted_positive"] for value in [0.20, 0.30, 0.50, 0.70]],
            [16, 10, 4, 2],
        )
        self.assertGreater(results[0.20]["recall"], results[0.50]["recall"])
        self.assertLess(results[0.20]["precision"], results[0.50]["precision"])
        self.assertEqual(results[0.30]["accuracy"], results[0.50]["accuracy"])
        self.assertNotEqual(results[0.30]["fn"], results[0.50]["fn"])
        self.assertEqual(
            tuple(self.namespace["model_weights"]),
            self.namespace["weights_snapshot"],
        )
        self.assertEqual(
            tuple(self.namespace["test_probabilities"]),
            self.namespace["probabilities_snapshot"],
        )

    def test_consequence_policy_and_calibration_checks_are_substantive(self) -> None:
        selected = self.namespace["selected_policy"]
        results = {
            row["threshold"]: row for row in self.namespace["threshold_results"]
        }
        self.assertEqual(selected["threshold"], 0.20)
        self.assertEqual(results[0.20]["consequence_cost"], 23)
        self.assertEqual(results[0.50]["consequence_cost"], 41)
        self.assertLess(results[0.20]["consequence_cost"], results[0.50]["consequence_cost"])

        calibration = self.namespace["calibration_table"]
        self.assertEqual(sum(row["count"] for row in calibration), 45)
        self.assertTrue(all(0.0 <= row["observed_rate"] <= 1.0 for row in calibration))
        self.assertLess(self.namespace["model_brier"], self.namespace["constant_brier"])

    def test_controlled_failure_and_no_ai_gate_require_fresh_reasoning(self) -> None:
        controlled = (MISSION / "controlled_failure.md").read_text(encoding="utf-8").lower()
        no_ai = (MISSION / "no_ai_gate.md").read_text(encoding="utf-8").lower()

        for phrase in [
            "threshold `0.50`",
            "majority baseline",
            "false negatives",
            "accuracy, precision and recall",
            "false-positive cost",
            "false-negative cost",
            "not “always use a lower threshold.”",
        ]:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, controlled)

        for phrase in [
            "without ai-generated analysis or code",
            "predicted defect probability",
            "at most four batches",
            "commit before outcomes",
            "2 × fp + 9 × fn",
            "threshold `0.50`",
            "calibration",
        ]:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, no_ai)

    def test_evidence_contract_does_not_prefill_learner_evidence(self) -> None:
        evidence = (MISSION / "evidence_contract.yaml").read_text(encoding="utf-8")
        self.assertIn("required_evidence:", evidence)
        self.assertIn("prefilled_learner_evidence: prohibited", evidence)
        self.assertIsNone(re.search(r"(?m)^\s*learner_evidence\s*:", evidence))
        self.assertIsNone(re.search(r"(?m)^\s*learner_response\s*:", evidence))

    def test_existing_authoritative_sources_are_reused_without_registry_mutation(self) -> None:
        registry = json.loads(
            (ROOT / "data" / "source_registry.json").read_text(encoding="utf-8")
        )
        source_ids = {source["id"] for source in registry["sources"]}
        self.assertTrue({"sklearn-guide", "stanford-cs229"}.issubset(source_ids))

        content = (MISSION / "content.yaml").read_text(encoding="utf-8")
        self.assertIn("global_content_registry_modified: false", content)
        self.assertIn("- sklearn-guide", content)
        self.assertIn("- stanford-cs229", content)


if __name__ == "__main__":
    unittest.main()
