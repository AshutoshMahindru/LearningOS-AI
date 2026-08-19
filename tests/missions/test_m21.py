from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
MISSION = ROOT / "missions" / "M21"
NOTEBOOK = ROOT / "labs" / "M21_train_neural_network_black_box.ipynb"
REQUIREMENTS = ROOT / "requirements" / "m21.txt"


def load_core():
    spec = importlib.util.spec_from_file_location("m21_training_core", MISSION / "training_core.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load M21 training core")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_core()
RUNTIME_DEPS = all(importlib.util.find_spec(name) is not None for name in ("numpy", "sklearn"))


def cell_source(cell: dict[str, object]) -> str:
    source = cell.get("source", [])
    if isinstance(source, str):
        return source
    return "".join(str(part) for part in source)


def notebook_cells() -> list[dict[str, object]]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def first_code_cell() -> dict[str, object]:
    for cell in notebook_cells():
        if cell.get("cell_type") == "code":
            return cell
    raise AssertionError("notebook has no code cells")


class M21StaticContractTests(unittest.TestCase):
    def test_required_executable_artifacts_exist(self):
        for path in (MISSION / "training_core.py", NOTEBOOK, REQUIREMENTS, ROOT / "tests" / "test_m21.py"):
            with self.subTest(path=path):
                self.assertTrue(path.is_file())

    def test_notebook_is_valid_clean_unique_substantial_and_offline(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        cells = notebook["cells"]
        self.assertGreaterEqual(len(cells), 40)
        ids = [cell.get("id") for cell in cells]
        self.assertTrue(all(ids))
        self.assertEqual(len(ids), len(set(ids)))

        code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
        self.assertGreaterEqual(len(code_cells), 14)
        markdown_chars = sum(
            len(cell_source(cell)) for cell in cells if cell.get("cell_type") == "markdown"
        )
        self.assertGreaterEqual(markdown_chars, 5000)

        for cell in code_cells:
            with self.subTest(cell=cell.get("id")):
                self.assertIsNone(cell.get("execution_count"), cell.get("id"))
                self.assertEqual(cell.get("outputs"), [], cell.get("id"))
                ast.parse(cell_source(cell))

        all_code = "\n".join(cell_source(cell) for cell in code_cells).lower()
        for forbidden in (
            "import requests",
            "import httpx",
            "from openai",
            "import openai",
            "api_key",
            "os.environ",
            "urlopen",
            "socket.",
            "subprocess",
            "import torch",
            "tensorflow",
            "cuda",
            "http://",
            "https://",
            "password",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, all_code)

    def test_first_code_cell_bootstraps_repository_root(self):
        cell = first_code_cell()
        self.assertEqual(cell.get("id"), "setup")
        source = cell_source(cell)
        self.assertIn("Path.cwd()", source)
        self.assertIn("parents", source)
        self.assertIn("missions", source)
        self.assertIn("M21", source)
        self.assertIn("training_core.py", source)
        self.assertIn("sys.path", source)
        self.assertIn("from missions.M21.training_core import", source)

    def test_notebook_code_is_offline_and_secret_free(self):
        source = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        ).lower()
        for forbidden in ("requests.", "urllib", "http://", "https://", "api_key", "token=", "password"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_black_box_boundary_is_not_opened_in_code(self):
        source = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        for forbidden in ("coefs_", "intercepts_", "backprop", "gradient", "activation"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_notebook_enforces_prediction_before_every_action(self):
        cells = notebook_cells()
        positions = {str(cell["id"]): index for index, cell in enumerate(cells)}
        pairs = (
            ("predict-dataset", "inspect-dataset"),
            ("predict-holdout", "inspect-holdout"),
            ("predict-reference", "train-reference"),
            ("predict-error-profile", "inspect-error-profile"),
            ("predict-replay", "run-replay"),
            ("predict-seed-change", "run-seed-change"),
            ("predict-undertraining", "run-undertraining"),
            ("predict-undertraining-repair", "run-undertraining-repair"),
            ("predict-capacity", "run-capacity"),
            ("predict-label-corruption", "run-label-corruption"),
            ("predict-label-repair", "run-label-repair"),
        )
        for prediction, action in pairs:
            with self.subTest(prediction=prediction, action=action):
                self.assertLess(positions[prediction], positions[action])
                self.assertEqual(cells[positions[prediction]].get("cell_type"), "markdown")
                prediction_text = cell_source(cells[positions[prediction]])
                self.assertIn("Predict before running", prediction_text)

        markdown = "\n".join(
            cell_source(cell) for cell in cells if cell.get("cell_type") == "markdown"
        )
        self.assertGreaterEqual(markdown.count("Predict before running"), 10)
        for phrase in (
            "predict → act → observe → explain",
            "timestamp",
            "Controlled failure A",
            "Controlled failure B",
            "UNFILLED BY LEARNER",
            "M20 → M21 → M22",
            "StandardScaler",
            "majority baseline",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, markdown)

    def test_notebook_prints_required_black_box_evidence(self):
        code = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        for token in (
            "inspect_holdout_split",
            "majority_baseline_accuracy",
            "print_run_evidence",
            "loss_curve",
            "validation_scores",
            "confusion_matrix",
            "StandardScaler",
            "most_confused_pair",
            "compact_report",
            "model_seed=2102",
            "max_iter=1",
            "hidden_units=4",
            "shuffle_labels=True",
        ):
            with self.subTest(token=token):
                self.assertIn(token, code)

        setup_index = next(
            index for index, cell in enumerate(notebook_cells()) if cell.get("cell_type") == "code"
        )
        holdout_index = next(
            index
            for index, cell in enumerate(notebook_cells())
            if cell.get("id") == "inspect-holdout"
        )
        train_index = next(
            index
            for index, cell in enumerate(notebook_cells())
            if cell.get("id") == "train-reference"
        )
        self.assertLess(setup_index, holdout_index)
        self.assertLess(holdout_index, train_index)

    def test_requirements_are_bounded_and_cover_runtime(self):
        requirements = REQUIREMENTS.read_text(encoding="utf-8").lower().splitlines()
        for package in ("numpy", "scikit-learn", "matplotlib", "nbclient", "nbformat", "pytest"):
            with self.subTest(package=package):
                matching = [line for line in requirements if line.startswith(package)]
                self.assertEqual(len(matching), 1)
                self.assertIn(">=", matching[0])
                self.assertIn("<", matching[0])

    def test_training_core_defers_optional_imports_until_runtime(self):
        source = (MISSION / "training_core.py").read_text(encoding="utf-8")
        module = ast.parse(source)
        top_level_imports = [
            alias.name
            for node in module.body
            if isinstance(node, ast.Import)
            for alias in node.names
        ] + [
            node.module or ""
            for node in module.body
            if isinstance(node, ast.ImportFrom)
        ]
        self.assertNotIn("numpy", top_level_imports)
        self.assertFalse(any(name.startswith("sklearn") for name in top_level_imports))
        self.assertNotIn("matplotlib", top_level_imports)

    def test_learner_facing_contracts_remain_unfilled(self):
        adr = (MISSION / "adr_prompt.md").read_text(encoding="utf-8")
        no_ai = (MISSION / "no_ai_gate.md").read_text(encoding="utf-8")
        status = (MISSION / "status.yaml").read_text(encoding="utf-8")
        self.assertIn("[UNFILLED BY LEARNER]", adr)
        self.assertIn("Leave all learner responses unfilled", no_ai)
        self.assertIn("intentionally_unpopulated", status)
        notebook_markdown = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "markdown"
        )
        self.assertIn("[UNFILLED BY LEARNER]", notebook_markdown)
        self.assertNotIn("[FILLED", notebook_markdown)


@unittest.skipUnless(
    RUNTIME_DEPS,
    "install requirements/m21.txt to run NumPy/scikit-learn-dependent M21 tests",
)
class M21RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = CORE.train_black_box()
        cls.seed_change = CORE.train_black_box(model_seed=2102)
        cls.under = CORE.train_black_box(max_iter=1)
        cls.tiny = CORE.train_black_box(hidden_units=4)
        cls.shuffle = CORE.train_black_box(shuffle_labels=True, label_seed=2121)

    def test_dataset_summary_matches_bundled_fixture(self):
        summary = CORE.dataset_summary()
        self.assertEqual(summary["samples"], 1797)
        self.assertEqual(summary["features"], 64)
        self.assertEqual(summary["classes"], 10)

    def test_holdout_inspection_matches_reference_split(self):
        holdout = CORE.inspect_holdout_split()
        self.assertEqual(holdout["train_size"], self.reference.train_size)
        self.assertEqual(holdout["test_size"], self.reference.test_size)
        self.assertEqual(holdout["split_seed"], self.reference.split_seed)
        self.assertAlmostEqual(
            holdout["majority_baseline_accuracy"],
            self.reference.majority_baseline_accuracy,
        )
        self.assertIn("StandardScaler", str(holdout["preprocessing"]))

    def test_compact_report_exposes_baseline_and_traces(self):
        report = CORE.compact_report(self.reference)
        self.assertIn("majority_baseline_accuracy", report)
        self.assertEqual(report["n_loss_curve_points"], len(self.reference.loss_curve))
        self.assertEqual(report["n_validation_scores"], len(self.reference.validation_scores))
        self.assertGreater(report["n_loss_curve_points"], 1)

    def test_reference_beats_majority_baseline(self):
        self.assertGreater(self.reference.test_accuracy, 0.90)
        self.assertGreater(self.reference.macro_f1, 0.90)
        self.assertLess(self.reference.majority_baseline_accuracy, 0.11)

    def test_same_seed_replays_exactly(self):
        self.assertEqual(self.reference, CORE.train_black_box())

    def test_different_seed_changes_trace_but_remains_useful(self):
        self.assertNotEqual(self.reference, self.seed_change)
        self.assertGreater(self.seed_change.test_accuracy, 0.90)

    def test_undertraining_failure_is_observable(self):
        self.assertLess(self.under.test_accuracy, 0.60)
        self.assertGreater(self.reference.test_accuracy - self.under.test_accuracy, 0.40)

    def test_shuffled_label_failure_is_near_baseline(self):
        self.assertLess(self.shuffle.test_accuracy, 0.20)
        self.assertLess(abs(self.shuffle.test_accuracy - self.shuffle.majority_baseline_accuracy), 0.06)
        # Held-out labels stay true. train_accuracy is scored against the
        # permuted fit labels, so it is not the near-baseline diagnosis stat.

    def test_tiny_capacity_and_confusion_matrix_are_observable(self):
        self.assertLess(self.tiny.test_accuracy, self.reference.test_accuracy - 0.10)
        self.assertEqual(sum(map(sum, self.reference.confusion_matrix)), self.reference.test_size)
        true_class, predicted_class, count = CORE.most_confused_pair(self.reference)
        self.assertNotEqual(true_class, predicted_class)
        self.assertGreaterEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
