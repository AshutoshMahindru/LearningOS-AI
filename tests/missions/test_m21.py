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


class M21StaticContractTests(unittest.TestCase):
    def load_notebook(self):
        return json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    def test_required_executable_artifacts_exist(self):
        for path in (MISSION / "training_core.py", NOTEBOOK, REQUIREMENTS, ROOT / "tests" / "test_m21.py"):
            with self.subTest(path=path):
                self.assertTrue(path.is_file())

    def test_notebook_has_exact_declared_shape(self):
        nb = self.load_notebook()
        self.assertEqual(len(nb["cells"]), 36)
        self.assertEqual(sum(cell["cell_type"] == "code" for cell in nb["cells"]), 12)

    def test_notebook_cell_ids_are_stable_and_unique(self):
        nb = self.load_notebook()
        ids = [cell["id"] for cell in nb["cells"]]
        self.assertEqual(len(ids), 36)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(ids))

    def test_source_notebook_has_no_prefilled_execution_state(self):
        for cell in self.load_notebook()["cells"]:
            if cell["cell_type"] == "code":
                with self.subTest(cell=cell["id"]):
                    self.assertIsNone(cell["execution_count"])
                    self.assertEqual(cell["outputs"], [])

    def test_every_code_cell_is_syntax_valid(self):
        for cell in self.load_notebook()["cells"]:
            if cell["cell_type"] == "code":
                with self.subTest(cell=cell["id"]):
                    ast.parse("".join(cell["source"]))

    def test_notebook_code_is_offline_and_secret_free(self):
        source = "\n".join(
            "".join(cell["source"]) for cell in self.load_notebook()["cells"] if cell["cell_type"] == "code"
        ).lower()
        for forbidden in ("requests.", "urllib", "http://", "https://", "api_key", "token=", "password"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_black_box_boundary_is_not_opened_in_code(self):
        source = "\n".join(
            "".join(cell["source"]) for cell in self.load_notebook()["cells"] if cell["cell_type"] == "code"
        )
        for forbidden in ("coefs_", "intercepts_", "backprop", "gradient", "activation"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_prediction_checkpoints_precede_each_experiment_action(self):
        ids = [cell["id"] for cell in self.load_notebook()["cells"]]
        pairs = (
            ("predict-reference", "train-reference"),
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
                self.assertLess(ids.index(prediction), ids.index(action))

    def test_requirements_are_bounded_and_cover_runtime(self):
        requirements = REQUIREMENTS.read_text(encoding="utf-8").lower().splitlines()
        for package in ("numpy", "scikit-learn", "nbclient", "nbformat", "pytest"):
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

    def test_tiny_capacity_and_confusion_matrix_are_observable(self):
        self.assertLess(self.tiny.test_accuracy, self.reference.test_accuracy - 0.10)
        self.assertEqual(sum(map(sum, self.reference.confusion_matrix)), self.reference.test_size)
        true_class, predicted_class, count = CORE.most_confused_pair(self.reference)
        self.assertNotEqual(true_class, predicted_class)
        self.assertGreaterEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
