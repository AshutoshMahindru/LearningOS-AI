from __future__ import annotations

import ast
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
MISSION = ROOT / "missions" / "M22"
NOTEBOOK = ROOT / "labs" / "M22_neuron_layer.ipynb"
REQUIREMENTS = ROOT / "requirements" / "m22.txt"


def load_core():
    spec = importlib.util.spec_from_file_location("m22_neuron_layer_core", MISSION / "neuron_layer_core.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load M22 neuron layer core")
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


class M22StaticContractTests(unittest.TestCase):
    def test_required_executable_artifacts_exist(self):
        for path in (MISSION / "neuron_layer_core.py", NOTEBOOK, REQUIREMENTS, ROOT / "tests" / "test_m22.py"):
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
            "softmax",
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
        self.assertIn("M22", source)
        self.assertIn("neuron_layer_core.py", source)
        self.assertIn("sys.path", source)
        self.assertIn("from missions.M22.neuron_layer_core import", source)

    def test_black_box_and_future_mission_boundary_stay_closed(self):
        source = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        for forbidden in ("coefs_", "intercepts_", "backprop", "gradient", "torch", "softmax"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_notebook_enforces_prediction_before_every_action(self):
        cells = notebook_cells()
        positions = {str(cell["id"]): index for index, cell in enumerate(cells)}
        pairs = (
            ("predict-neuron", "run-neuron"),
            ("predict-feature-change", "run-feature-change"),
            ("predict-bias-ablation", "run-bias-ablation"),
            ("predict-activation-sweep", "run-activation-sweep"),
            ("predict-dense-layer", "run-dense-layer"),
            ("predict-batch", "run-batch"),
            ("predict-linearity", "run-linearity"),
            ("predict-failure", "run-failure"),
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
        self.assertGreaterEqual(markdown.count("Predict before running"), 9)
        for phrase in (
            "predict → act → observe → explain",
            "timestamp",
            "Controlled failure",
            "UNFILLED BY LEARNER",
            "M21 → M22 → M23",
            "n_in",
            "n_out",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, markdown)

    def test_notebook_prints_required_layer_evidence(self):
        code = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        for token in (
            "neuron_trace",
            "affine_preactivation",
            "activation_sweep",
            "dense_forward",
            "dense_forward_with_defect",
            "compose_two_layers",
            "collapsed_affine",
            "validate_dense_shapes",
            "REFERENCE_PREACTIVATION",
            'hidden_activation="relu"',
            'defect="transposed_weights"',
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

    def test_training_core_defers_optional_imports_until_runtime(self):
        source = (MISSION / "neuron_layer_core.py").read_text(encoding="utf-8")
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


@unittest.skipUnless(RUNTIME_DEPS, "install requirements/m22.txt to run NumPy-dependent M22 tests")
class M22RuntimeTests(unittest.TestCase):
    def test_reference_neuron_is_hand_computable(self):
        trace = CORE.reference_neuron_relu()
        self.assertEqual(trace.x, (1.0, 2.0))
        self.assertEqual(trace.weights, (0.5, -0.25))
        self.assertEqual(trace.bias, 0.5)
        self.assertAlmostEqual(trace.weighted_sum, 0.0)
        self.assertAlmostEqual(trace.preactivation, 0.5)
        self.assertAlmostEqual(trace.output, 0.5)
        self.assertAlmostEqual(CORE.REFERENCE_PREACTIVATION, 0.5)

    def test_feature_change_moves_preactivation_by_the_weight(self):
        base = CORE.neuron_trace((1.0, 2.0), CORE.REFERENCE_WEIGHTS, CORE.REFERENCE_BIAS, "relu")
        changed = CORE.neuron_trace((3.0, 2.0), CORE.REFERENCE_WEIGHTS, CORE.REFERENCE_BIAS, "relu")
        self.assertAlmostEqual(changed.preactivation - base.preactivation, 1.0)
        self.assertAlmostEqual(changed.output, 1.5)

    def test_bias_ablation_removes_only_the_translation(self):
        with_bias = CORE.neuron_trace(CORE.REFERENCE_X, CORE.REFERENCE_WEIGHTS, 0.5, "identity")
        no_bias = CORE.neuron_trace(CORE.REFERENCE_X, CORE.REFERENCE_WEIGHTS, 0.0, "identity")
        self.assertAlmostEqual(with_bias.preactivation - no_bias.preactivation, 0.5)
        self.assertAlmostEqual(no_bias.preactivation, no_bias.weighted_sum)

    def test_activation_sweep_uses_one_shared_sequence(self):
        zs = (-2.0, -0.5, 0.0, 0.5, 2.0)
        swept = CORE.activation_sweep(zs)
        self.assertEqual(swept["relu"], (0.0, 0.0, 0.0, 0.5, 2.0))
        self.assertAlmostEqual(swept["sigmoid"][2], 0.5)
        self.assertLess(swept["sigmoid"][0], 0.2)
        self.assertGreater(swept["sigmoid"][4], 0.8)
        self.assertAlmostEqual(swept["tanh"][2], 0.0)
        self.assertLess(swept["tanh"][0], 0.0)
        self.assertGreater(swept["tanh"][4], 0.0)
        self.assertTrue(all(-1.0 <= value <= 1.0 for value in swept["tanh"]))

    def test_dense_forward_matches_hand_computed_rows(self):
        output = CORE.dense_forward(
            CORE.REFERENCE_LAYER_X,
            CORE.REFERENCE_LAYER_W,
            CORE.REFERENCE_LAYER_BIAS,
            "relu",
        )
        self.assertEqual(output.shape, (2, 2))
        self.assertAlmostEqual(float(output[0, 0]), 0.0)
        self.assertAlmostEqual(float(output[0, 1]), 0.0)
        self.assertAlmostEqual(float(output[1, 0]), 1.0)
        self.assertAlmostEqual(float(output[1, 1]), 1.5)

    def test_batch_rows_match_singleton_forward(self):
        batch = CORE.dense_forward(
            CORE.REFERENCE_LAYER_X,
            CORE.REFERENCE_LAYER_W,
            CORE.REFERENCE_LAYER_BIAS,
            "relu",
        )
        for index, row in enumerate(CORE.REFERENCE_LAYER_X):
            single = CORE.dense_forward(
                row,
                CORE.REFERENCE_LAYER_W,
                CORE.REFERENCE_LAYER_BIAS,
                "relu",
            )
            self.assertEqual(single.shape, (1, 2))
            self.assertTrue((single[0] == batch[index]).all())

    def test_identity_hidden_composition_collapses_to_one_affine_map(self):
        np = CORE._require_numpy()
        x = CORE.REFERENCE_LAYER_X
        w1 = CORE.REFERENCE_LAYER_W
        b1 = CORE.REFERENCE_LAYER_BIAS
        w2 = ((1.0, 0.0), (0.0, 1.0))
        b2 = (0.25, -0.25)
        composed = CORE.compose_two_layers(x, w1, b1, w2, b2, hidden_activation="identity")
        weights_eq, bias_eq = CORE.collapsed_affine(w1, b1, w2, b2)
        collapsed = CORE.dense_forward(x, weights_eq, bias_eq, "identity")
        self.assertTrue(np.allclose(composed, collapsed))
        nonlinear = CORE.compose_two_layers(x, w1, b1, w2, b2, hidden_activation="relu")
        self.assertFalse(np.allclose(nonlinear, collapsed))

    def test_transposed_square_weights_emit_wrong_but_finite_numbers(self):
        x = ((1.0, 2.0),)
        w = ((1.0, 3.0), (0.0, 2.0))
        b = (0.0, 0.0)
        correct = CORE.dense_forward(x, w, b, "identity")
        broken = CORE.dense_forward_with_defect(x, w, b, "identity", defect="transposed_weights")
        self.assertAlmostEqual(float(correct[0, 0]), 1.0)
        self.assertAlmostEqual(float(correct[0, 1]), 7.0)
        self.assertAlmostEqual(float(broken[0, 0]), 7.0)
        self.assertAlmostEqual(float(broken[0, 1]), 4.0)
        repaired = CORE.dense_forward_with_defect(x, w, b, "identity", defect="none")
        self.assertTrue((repaired == correct).all())

    def test_activation_before_affine_is_a_boundary_defect(self):
        x = ((1.0, -2.0),)
        w = ((1.0, 1.0), (1.0, 1.0))
        b = (0.0, 0.0)
        correct = CORE.dense_forward(x, w, b, "relu")
        broken = CORE.dense_forward_with_defect(
            x, w, b, "relu", defect="activation_before_affine"
        )
        self.assertAlmostEqual(float(correct[0, 0]), 0.0)
        self.assertAlmostEqual(float(correct[0, 1]), 0.0)
        self.assertAlmostEqual(float(broken[0, 0]), 1.0)
        self.assertAlmostEqual(float(broken[0, 1]), 1.0)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            CORE.dense_forward(((1.0, 2.0),), CORE.REFERENCE_LAYER_W, CORE.REFERENCE_LAYER_BIAS)
        with self.assertRaises(ValueError):
            CORE.neuron_trace((1.0,), CORE.REFERENCE_WEIGHTS, 0.0)
        with self.assertRaises(ValueError):
            CORE.apply_activation(0.0, "gelu")

    def test_sigmoid_matches_independent_formula(self):
        value = 0.5
        expected = 1.0 / (1.0 + math.exp(-value))
        self.assertAlmostEqual(float(CORE.sigmoid(value)), expected)


if __name__ == "__main__":
    unittest.main()
