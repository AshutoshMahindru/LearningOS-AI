from __future__ import annotations

import ast
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
MISSION = ROOT / "missions" / "M23"
NOTEBOOK = ROOT / "labs" / "M23_numpy_forward_pass.ipynb"
REQUIREMENTS = ROOT / "requirements" / "m23.txt"


def load_core():
    spec = importlib.util.spec_from_file_location("m23_forward_core", MISSION / "forward_core.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load M23 forward core")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_core()
RUNTIME_DEPS = importlib.util.find_spec("numpy") is not None


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


class M23StaticContractTests(unittest.TestCase):
    def test_required_executable_artifacts_exist(self):
        for path in (MISSION / "forward_core.py", NOTEBOOK, REQUIREMENTS, ROOT / "tests" / "test_m23.py"):
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
        self.assertGreaterEqual(len(code_cells), 12)
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
            "import torch",
            "tensorflow",
            "cuda",
            "http://",
            "https://",
            "sklearn",
            "coefs_",
            "backward",
            "autograd",
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
        self.assertIn("M23", source)
        self.assertIn("forward_core.py", source)
        self.assertIn("sys.path", source)
        self.assertIn("from missions.M23.forward_core import", source)

    def test_future_mission_boundary_stays_closed(self):
        source = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        for forbidden in ("backward", "autograd", "torch", "sklearn", "coefs_"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

        core = (MISSION / "forward_core.py").read_text(encoding="utf-8")
        self.assertNotIn("import torch", core)
        self.assertNotIn("from torch", core)
        self.assertNotIn(".backward(", core)
        self.assertIn("deferred to M24", core)

    def test_notebook_enforces_prediction_before_every_action(self):
        cells = notebook_cells()
        positions = {str(cell["id"]): index for index, cell in enumerate(cells)}
        pairs = (
            ("predict-first-layer", "run-first-layer"),
            ("predict-scalar", "run-scalar"),
            ("predict-vectorized-batch", "run-vectorized-batch"),
            ("predict-compose", "run-compose"),
            ("predict-softmax", "run-softmax"),
            ("predict-shift", "run-shift"),
            ("predict-reference", "run-reference"),
            ("predict-perturbation", "run-perturbation"),
            ("predict-reorder", "run-reorder"),
            ("predict-failure", "run-failure"),
            ("predict-omitted", "run-omitted"),
            ("predict-failure-repair", "run-failure-repair"),
        )
        for prediction, action in pairs:
            with self.subTest(prediction=prediction, action=action):
                self.assertLess(positions[prediction], positions[action])
                self.assertEqual(cells[positions[prediction]].get("cell_type"), "markdown")
                self.assertIn("Predict before running", cell_source(cells[positions[prediction]]))

        markdown = "\n".join(
            cell_source(cell) for cell in cells if cell.get("cell_type") == "markdown"
        )
        self.assertGreaterEqual(markdown.count("Predict before running"), 11)
        for phrase in (
            "predict → act → observe → explain",
            "timestamp",
            "Controlled failure",
            "UNFILLED BY LEARNER",
            "M22 → M23 → M24",
            "named intermediates",
            "class axis",
            "logits",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, markdown)

    def test_notebook_prints_required_forward_evidence(self):
        code = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        for token in (
            "two_layer_forward",
            "two_layer_forward_with_defect",
            "stable_softmax",
            "scalar_two_layer_one_example",
            "m22_reference_forward",
            "intermediate_parity",
            "singleton_batch_parity",
            "perturb_matrix",
            "shift_logits",
            "reorder_rows",
            "GRAPH_NODES",
            "REFERENCE_LOGITS",
            "dense_forward",
            'defect="softmax_axis_batch"',
            'defect="omitted_hidden_activation"',
            "CLASS_AXIS",
        ):
            with self.subTest(token=token):
                self.assertIn(token, code)

    def test_requirements_are_bounded_and_cover_runtime(self):
        requirements = REQUIREMENTS.read_text(encoding="utf-8").lower().splitlines()
        for package in ("numpy", "matplotlib", "nbclient", "nbformat", "pytest"):
            with self.subTest(package=package):
                matching = [line for line in requirements if line.startswith(package)]
                self.assertEqual(len(matching), 1)
                self.assertIn(">=", matching[0])
                self.assertIn("<", matching[0])

    def test_forward_core_defers_optional_imports_until_runtime(self):
        source = (MISSION / "forward_core.py").read_text(encoding="utf-8")
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
        self.assertNotIn("torch", top_level_imports)

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

    def test_mission_package_declares_m22_handoff_and_m24_deferral(self):
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        content = (MISSION / "content.yaml").read_text(encoding="utf-8")
        self.assertIn("M22", manifest)
        self.assertIn("M16", manifest)
        self.assertIn("backpropagation_deferred_to_m24", manifest)
        self.assertIn("named intermediates", content)
        self.assertIn("deferred_to_m24", content)


@unittest.skipUnless(RUNTIME_DEPS, "install requirements/m23.txt to run NumPy-dependent M23 tests")
class M23RuntimeTests(unittest.TestCase):
    def test_reference_first_layer_matches_m22_hand_values(self):
        trace = CORE.two_layer_forward(
            CORE.REFERENCE_X,
            CORE.REFERENCE_W1,
            CORE.REFERENCE_B1,
            CORE.REFERENCE_W2,
            CORE.REFERENCE_B2,
        )
        self.assertTrue(CORE.arrays_close(trace.hidden_preactivation, CORE.REFERENCE_HIDDEN_PREACTIVATION))
        self.assertTrue(CORE.arrays_close(trace.hidden_activation, CORE.REFERENCE_HIDDEN_ACTIVATION))
        self.assertAlmostEqual(float(trace.hidden_activation[0, 0]), 0.0)
        self.assertAlmostEqual(float(trace.hidden_activation[0, 1]), 0.0)
        self.assertAlmostEqual(float(trace.hidden_activation[1, 0]), 1.0)
        self.assertAlmostEqual(float(trace.hidden_activation[1, 1]), 1.5)

        m22 = CORE._load_m22()
        self.assertEqual(CORE.REFERENCE_X, m22.REFERENCE_LAYER_X)
        self.assertEqual(CORE.REFERENCE_W1, m22.REFERENCE_LAYER_W)
        self.assertEqual(CORE.REFERENCE_B1, m22.REFERENCE_LAYER_BIAS)
        layer = m22.dense_forward(
            m22.REFERENCE_LAYER_X,
            m22.REFERENCE_LAYER_W,
            m22.REFERENCE_LAYER_BIAS,
            "relu",
        )
        self.assertTrue(CORE.arrays_close(trace.hidden_activation, layer))

    def test_reference_logits_are_hand_computable(self):
        trace = CORE.reference_forward()
        self.assertTrue(CORE.arrays_close(trace.logits, CORE.REFERENCE_LOGITS))
        self.assertAlmostEqual(float(trace.logits[0, 0]), 0.0)
        self.assertAlmostEqual(float(trace.logits[0, 1]), 0.0)
        self.assertAlmostEqual(float(trace.logits[0, 2]), 0.0)
        self.assertAlmostEqual(float(trace.logits[1, 0]), 1.0)
        self.assertAlmostEqual(float(trace.logits[1, 1]), 1.5)
        self.assertAlmostEqual(float(trace.logits[1, 2]), -0.25)
        self.assertTrue(CORE.arrays_close(trace.probabilities, CORE.REFERENCE_PROBABILITIES))
        self.assertAlmostEqual(float(trace.probabilities[0, 0]), 1.0 / 3.0)
        self.assertEqual(trace.shapes["x"], (2, 3))
        self.assertEqual(trace.shapes["hidden_activation"], (2, 2))
        self.assertEqual(trace.shapes["logits"], (2, 3))
        self.assertEqual(trace.shapes["probabilities"], (2, 3))
        self.assertEqual(CORE.GRAPH_NODES[-1], "probabilities")

    def test_scalar_path_matches_vectorized_rows(self):
        batch = CORE.two_layer_forward(
            CORE.REFERENCE_X,
            CORE.REFERENCE_W1,
            CORE.REFERENCE_B1,
            CORE.REFERENCE_W2,
            CORE.REFERENCE_B2,
        )
        for index, row in enumerate(CORE.REFERENCE_X):
            scalar = CORE.scalar_two_layer_one_example(
                row,
                CORE.REFERENCE_W1,
                CORE.REFERENCE_B1,
                CORE.REFERENCE_W2,
                CORE.REFERENCE_B2,
            )
            self.assertTrue(CORE.arrays_close(scalar.hidden_preactivation, batch.hidden_preactivation[index]))
            self.assertTrue(CORE.arrays_close(scalar.hidden_activation, batch.hidden_activation[index]))
            self.assertTrue(CORE.arrays_close(scalar.logits, batch.logits[index]))
            self.assertTrue(CORE.arrays_close(scalar.probabilities, batch.probabilities[index]))

    def test_batch_rows_match_singleton_forward(self):
        report = CORE.singleton_batch_parity(
            CORE.REFERENCE_X,
            CORE.REFERENCE_W1,
            CORE.REFERENCE_B1,
            CORE.REFERENCE_W2,
            CORE.REFERENCE_B2,
        )
        self.assertTrue(report["all_match"])
        self.assertEqual(report["row_matches"], (True, True))

    def test_stable_softmax_is_on_class_axis_and_shift_invariant(self):
        logits = CORE.REFERENCE_LOGITS
        probabilities = CORE.stable_softmax(logits, axis=CORE.CLASS_AXIS)
        self.assertTrue(CORE.arrays_close(CORE.probability_row_sums(probabilities), (1.0, 1.0)))
        shifted = CORE.shift_logits(logits, example_index=1, constant=5.0)
        shifted_probabilities = CORE.stable_softmax(shifted, axis=CORE.CLASS_AXIS)
        self.assertTrue(CORE.arrays_close(probabilities, shifted_probabilities))
        large = CORE.stable_softmax((1000.0, 1001.0, 999.0))
        self.assertAlmostEqual(float(sum(large)), 1.0)
        self.assertTrue(all(math.isfinite(float(value)) for value in large))
        self.assertGreater(float(large[1]), float(large[0]))

    def test_independent_softmax_formula_matches_row_one(self):
        logits = CORE.REFERENCE_LOGITS[1]
        maximum = max(logits)
        expected = tuple(math.exp(value - maximum) for value in logits)
        total = sum(expected)
        expected = tuple(value / total for value in expected)
        got = CORE.stable_softmax(logits)
        self.assertTrue(CORE.arrays_close(got, expected))

    def test_m22_reference_parity_on_every_named_intermediate(self):
        numpy_stack = CORE.two_layer_forward(
            CORE.REFERENCE_X,
            CORE.REFERENCE_W1,
            CORE.REFERENCE_B1,
            CORE.REFERENCE_W2,
            CORE.REFERENCE_B2,
        )
        trusted = CORE.m22_reference_forward(
            CORE.REFERENCE_X,
            CORE.REFERENCE_W1,
            CORE.REFERENCE_B1,
            CORE.REFERENCE_W2,
            CORE.REFERENCE_B2,
        )
        parity = CORE.intermediate_parity(numpy_stack, trusted)
        self.assertEqual(set(parity), set(CORE.INTERMEDIATE_NAMES))
        self.assertTrue(all(parity.values()))

    def test_single_weight_perturbation_is_localized(self):
        base = CORE.reference_forward()
        weights = CORE.perturb_matrix(CORE.REFERENCE_W2, (0, 0), 0.1)
        changed = CORE.two_layer_forward(
            CORE.REFERENCE_X,
            CORE.REFERENCE_W1,
            CORE.REFERENCE_B1,
            weights,
            CORE.REFERENCE_B2,
        )
        self.assertTrue(CORE.arrays_close(base.hidden_activation, changed.hidden_activation))
        self.assertTrue(CORE.arrays_close(base.logits[0], changed.logits[0]))
        self.assertAlmostEqual(float(changed.logits[1, 0] - base.logits[1, 0]), 0.1)
        self.assertAlmostEqual(float(changed.logits[1, 1]), float(base.logits[1, 1]))
        self.assertFalse(CORE.arrays_close(base.probabilities[1], changed.probabilities[1]))

    def test_batch_reordering_permutes_outputs_without_mixing_rows(self):
        base = CORE.reference_forward()
        shuffled_x = CORE.reorder_rows(CORE.REFERENCE_X, (1, 0))
        shuffled = CORE.two_layer_forward(
            shuffled_x,
            CORE.REFERENCE_W1,
            CORE.REFERENCE_B1,
            CORE.REFERENCE_W2,
            CORE.REFERENCE_B2,
        )
        self.assertTrue(CORE.arrays_close(shuffled.probabilities[0], base.probabilities[1]))
        self.assertTrue(CORE.arrays_close(shuffled.probabilities[1], base.probabilities[0]))
        restored = CORE.reorder_rows(shuffled.probabilities, (1, 0))
        self.assertTrue(CORE.arrays_close(restored, base.probabilities))

    def test_wrong_axis_softmax_breaks_singleton_batch_parity(self):
        correct = CORE.singleton_batch_parity(
            CORE.REFERENCE_X,
            CORE.REFERENCE_W1,
            CORE.REFERENCE_B1,
            CORE.REFERENCE_W2,
            CORE.REFERENCE_B2,
            defect="none",
        )
        broken = CORE.singleton_batch_parity(
            CORE.REFERENCE_X,
            CORE.REFERENCE_W1,
            CORE.REFERENCE_B1,
            CORE.REFERENCE_W2,
            CORE.REFERENCE_B2,
            defect="softmax_axis_batch",
        )
        self.assertTrue(correct["all_match"])
        self.assertFalse(broken["all_match"])

        single = CORE.two_layer_forward_with_defect(
            CORE.REFERENCE_X[0],
            CORE.REFERENCE_W1,
            CORE.REFERENCE_B1,
            CORE.REFERENCE_W2,
            CORE.REFERENCE_B2,
            defect="softmax_axis_batch",
        )
        self.assertTrue(CORE.arrays_close(single.probabilities[0], (1.0, 1.0, 1.0)))
        batch = CORE.two_layer_forward_with_defect(
            CORE.REFERENCE_X,
            CORE.REFERENCE_W1,
            CORE.REFERENCE_B1,
            CORE.REFERENCE_W2,
            CORE.REFERENCE_B2,
            defect="softmax_axis_batch",
        )
        self.assertTrue(CORE.arrays_close(broken["batch_column_sums"], (1.0, 1.0, 1.0)))
        self.assertFalse(CORE.arrays_close(CORE.probability_row_sums(batch.probabilities), (1.0, 1.0)))

    def test_omitted_hidden_activation_is_caught_by_intermediates(self):
        correct = CORE.reference_forward()
        broken = CORE.two_layer_forward_with_defect(
            CORE.REFERENCE_X,
            CORE.REFERENCE_W1,
            CORE.REFERENCE_B1,
            CORE.REFERENCE_W2,
            CORE.REFERENCE_B2,
            hidden_activation="relu",
            defect="omitted_hidden_activation",
        )
        self.assertTrue(CORE.arrays_close(broken.hidden_activation[0], (0.0, -0.5)))
        self.assertFalse(CORE.arrays_close(broken.hidden_activation, correct.hidden_activation))
        self.assertFalse(CORE.arrays_close(broken.logits[0], correct.logits[0]))
        self.assertTrue(CORE.arrays_close(broken.logits[1], correct.logits[1]))
        still_aligned = CORE.singleton_batch_parity(
            CORE.REFERENCE_X,
            CORE.REFERENCE_W1,
            CORE.REFERENCE_B1,
            CORE.REFERENCE_W2,
            CORE.REFERENCE_B2,
            defect="omitted_hidden_activation",
        )
        self.assertTrue(still_aligned["all_match"])

    def test_smallest_repair_restores_class_axis_softmax(self):
        correct = CORE.two_layer_forward_with_defect(
            CORE.REFERENCE_X,
            CORE.REFERENCE_W1,
            CORE.REFERENCE_B1,
            CORE.REFERENCE_W2,
            CORE.REFERENCE_B2,
            defect="none",
        )
        repaired = CORE.two_layer_forward_with_defect(
            CORE.REFERENCE_X,
            CORE.REFERENCE_W1,
            CORE.REFERENCE_B1,
            CORE.REFERENCE_W2,
            CORE.REFERENCE_B2,
            defect="none",
        )
        broken = CORE.two_layer_forward_with_defect(
            CORE.REFERENCE_X,
            CORE.REFERENCE_W1,
            CORE.REFERENCE_B1,
            CORE.REFERENCE_W2,
            CORE.REFERENCE_B2,
            defect="softmax_axis_batch",
        )
        self.assertFalse(CORE.arrays_close(broken.probabilities, correct.probabilities))
        self.assertTrue(CORE.arrays_close(repaired.probabilities, correct.probabilities))

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            CORE.two_layer_forward(
                ((1.0, 2.0),),
                CORE.REFERENCE_W1,
                CORE.REFERENCE_B1,
                CORE.REFERENCE_W2,
                CORE.REFERENCE_B2,
            )
        with self.assertRaises(ValueError):
            CORE.validate_stack_shapes(
                CORE.REFERENCE_X,
                CORE.REFERENCE_W1,
                (0.0,),
                CORE.REFERENCE_W2,
                CORE.REFERENCE_B2,
            )
        listed = CORE.two_layer_forward(
            CORE.REFERENCE_X[0],
            CORE.REFERENCE_W1,
            CORE.REFERENCE_B1,
            CORE.REFERENCE_W2,
            CORE.REFERENCE_B2,
        )
        self.assertEqual(listed.x.shape, (1, 3))
        self.assertEqual(listed.probabilities.shape, (1, 3))


if __name__ == "__main__":
    unittest.main()
