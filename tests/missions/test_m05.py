from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
import unittest


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
NUMPY_SKIP_REASON = (
    "install requirements/m05.txt to run NumPy-dependent M05 tests"
)

if NUMPY_AVAILABLE:
    import numpy as np
else:
    np = None


ROOT = Path(__file__).resolve().parents[2]
MISSION = ROOT / "missions" / "M05"
NOTEBOOK = ROOT / "labs" / "M05_array_vectorization.ipynb"


def cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


@unittest.skipUnless(NUMPY_AVAILABLE, NUMPY_SKIP_REASON)
class M05MissionPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.cells = cls.notebook["cells"]
        cls.cells_by_id = {cell["id"]: cell for cell in cls.cells}

    def test_required_standard_package_exists(self) -> None:
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
        ]

        for name in required:
            self.assertTrue((MISSION / name).is_file(), name)

        self.assertTrue(NOTEBOOK.is_file())
        self.assertTrue((ROOT / "requirements" / "m05.txt").is_file())
        self.assertTrue((ROOT / "datasets" / "M05" / "README.md").is_file())

    def test_manifest_declares_m05_execution_contract(self) -> None:
        text = (MISSION / "manifest.yaml").read_text(encoding="utf-8")

        for expected in [
            "id: M05",
            "phase: P1",
            "flagship: V01",
            "pedagogy: whole-first",
            "formal_engineering_review: false",
            "adr_required: false",
            "cpu_only: true",
            "requires_secrets: false",
            "requires_paid_api: false",
            "network_required: false",
            "controlled_shape_broadcasting_and_axis_failure",
            "no_ai_manual_shape_and_result_prediction",
        ]:
            self.assertIn(expected, text)

    def test_content_is_mission_local_and_uses_official_numpy_sources(self) -> None:
        text = (MISSION / "content.yaml").read_text(encoding="utf-8")

        self.assertIn("registry_scope: mission_local_only", text)
        self.assertIn("global_content_registry_modified: false", text)
        self.assertGreaterEqual(text.count("publisher: NumPy"), 3)
        for path in [
            "absolute_beginners.html",
            "basics.broadcasting.html",
            "generated/numpy.sum.html",
        ]:
            self.assertIn(path, text)

    def test_notebook_is_substantial_and_has_stable_unique_cell_ids(self) -> None:
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertGreaterEqual(len(self.cells), 30)

        ids = [cell.get("id") for cell in self.cells]
        self.assertTrue(all(ids))
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), set(self.cells_by_id))

    def test_whole_first_sequence_is_explicit_in_cell_order(self) -> None:
        ids = [cell["id"] for cell in self.cells]
        milestones = [
            "loop-baseline",
            "safe-scale-fixture",
            "time-python-loop",
            "small-vectorized-whole",
            "scaled-correctness",
            "time-numpy",
            "timing-comparison",
            "shape-dtype-inspection",
            "broadcast-failure",
            "generalize-transfer",
        ]
        positions = [ids.index(milestone) for milestone in milestones]
        self.assertEqual(positions, sorted(positions))

    def test_prediction_prompts_precede_actions_and_no_ai_gate(self) -> None:
        ids = [cell["id"] for cell in self.cells]
        prediction_action_pairs = [
            ("whole-first-problem", "loop-baseline"),
            ("scale-prediction", "time-python-loop"),
            ("vectorize-predict", "small-vectorized-whole"),
            ("correctness-predict", "scaled-correctness"),
            ("broadcast-failure-predict", "broadcast-failure"),
            ("axis-failure-predict", "axis-failure"),
            ("no-ai-manual-prediction", "no-ai-implementation-after-prediction"),
            ("generalize-heading", "generalize-transfer"),
        ]

        for prediction, action in prediction_action_pairs:
            self.assertLess(ids.index(prediction), ids.index(action))
            self.assertIn(
                "predict",
                cell_source(self.cells_by_id[prediction]).lower(),
            )

        gate = (MISSION / "no_ai_gate.md").read_text(encoding="utf-8").lower()
        self.assertIn("without an ai assistant", gate)
        self.assertIn("before implementation", gate)
        self.assertIn("works on a second input", gate)

    def test_notebook_covers_required_array_operations(self) -> None:
        code = "\n".join(
            cell_source(cell)
            for cell in self.cells
            if cell["cell_type"] == "code"
        )
        markdown = "\n".join(
            cell_source(cell)
            for cell in self.cells
            if cell["cell_type"] == "markdown"
        )

        for token in [
            "np.asarray",
            ".shape",
            ".dtype",
            "axis=0",
            "axis=1",
            "np.newaxis",
            "np.where",
            "small_units[0, 1]",
            "small_units[:2, 1:3]",
            "numpy_small_totals[high_value_mask]",
            "np.testing.assert_allclose",
            "perf_counter",
            "default_rng(20260815)",
            "SCALE_ROWS = 200_000",
        ]:
            self.assertIn(token, code)

        for phrase in [
            "shape",
            "dtype",
            "axis",
            "broadcasting",
            "vectorized arithmetic",
            "controlled failure",
            "no-ai gate",
            "generalize",
        ]:
            self.assertIn(phrase, markdown.lower())

    def test_actual_loop_and_vectorized_implementations_agree(self) -> None:
        namespace: dict = {}
        exec(cell_source(self.cells_by_id["m05-core-functions"]), namespace)

        units = [[2, 1, 0, 1], [0, 3, 2, 0], [5, 0, 4, 1]]
        prices = [12.5, 8.0, 5.5, 20.0]
        discounts = [0.10, 0.00, 0.05]

        loop = namespace["python_order_totals"](units, prices, discounts)
        components = namespace["vectorized_order_components"](
            units, prices, discounts
        )

        np.testing.assert_allclose(loop, [47.7, 39.5, 99.275], rtol=0, atol=1e-12)
        np.testing.assert_allclose(components["totals"], loop, rtol=0, atol=1e-12)
        self.assertEqual(components["line_values"].shape, (3, 4))
        self.assertEqual(components["discount_columns"].shape, (3, 1))
        self.assertEqual(components["totals"].shape, (3,))
        self.assertEqual(components["units"].dtype, np.dtype(np.int64))
        self.assertEqual(components["line_values"].dtype, np.dtype(np.float64))

    def test_actual_vectorized_function_validates_shape_contract(self) -> None:
        namespace: dict = {}
        exec(cell_source(self.cells_by_id["m05-core-functions"]), namespace)
        function = namespace["vectorized_order_components"]

        with self.assertRaisesRegex(ValueError, "orders, products"):
            function([1, 2, 3], [1.0, 2.0, 3.0], [0.0])
        with self.assertRaisesRegex(ValueError, "prices"):
            function([[1, 2], [3, 4]], [1.0], [0.0, 0.1])
        with self.assertRaisesRegex(ValueError, "discounts"):
            function([[1, 2], [3, 4]], [1.0, 2.0], [0.0])

    def test_axis_aggregations_have_correct_shapes_and_values(self) -> None:
        namespace: dict = {}
        exec(cell_source(self.cells_by_id["m05-core-functions"]), namespace)
        components = namespace["vectorized_order_components"](
            [[2, 1, 0, 1], [0, 3, 2, 0], [5, 0, 4, 1]],
            [12.5, 8.0, 5.5, 20.0],
            [0.10, 0.00, 0.05],
        )
        line_values = components["line_values"]

        per_order = line_values.sum(axis=1)
        per_product = line_values.sum(axis=0)
        np.testing.assert_allclose(per_order, [53.0, 35.0, 104.5])
        np.testing.assert_allclose(per_product, [87.5, 32.0, 33.0, 40.0])
        self.assertEqual(per_order.shape, (3,))
        self.assertEqual(per_product.shape, (4,))

    def test_controlled_failures_are_executed_captured_and_repaired(self) -> None:
        broadcast = cell_source(self.cells_by_id["broadcast-failure"])
        broadcast_repair = cell_source(self.cells_by_id["broadcast-repair"])
        axis = cell_source(self.cells_by_id["axis-failure"])
        axis_repair = cell_source(self.cells_by_id["axis-repair"])

        self.assertIn("except ValueError", broadcast)
        self.assertIn("raise AssertionError", broadcast)
        self.assertIn("small_discounts[:, np.newaxis]", broadcast_repair)
        self.assertIn("np.testing.assert_allclose", broadcast_repair)
        self.assertIn("wrong_axis_totals = line_values.sum(axis=0)", axis)
        self.assertIn("except AssertionError", axis)
        self.assertIn("line_values.sum(axis=1)", axis_repair)

    def test_benchmark_is_bounded_fair_and_not_hardware_gated(self) -> None:
        fixture = cell_source(self.cells_by_id["safe-scale-fixture"])
        loop_timer = cell_source(self.cells_by_id["time-python-loop"])
        numpy_timer = cell_source(self.cells_by_id["time-numpy"])
        comparison = cell_source(self.cells_by_id["timing-comparison"])

        self.assertIn("SCALE_ROWS = 200_000", fixture)
        self.assertIn("default_rng(20260815)", fixture)
        self.assertIn("repeats=3", loop_timer)
        self.assertIn("repeats=3", numpy_timer)
        self.assertIn("loop_seconds / numpy_seconds", comparison)
        self.assertNotRegex(comparison, r"observed_speedup\s*[><]=?\s*\d")

    def test_notebook_code_is_syntactically_valid_and_network_free(self) -> None:
        code_cells = [
            cell for cell in self.cells if cell.get("cell_type") == "code"
        ]

        for index, cell in enumerate(code_cells):
            compile(cell_source(cell), f"M05-cell-{index}", "exec")

        all_code = "\n".join(cell_source(cell) for cell in code_cells)
        for forbidden in [
            "import requests",
            "import httpx",
            "from openai",
            "import openai",
            "api_key",
            "os.environ",
            "socket.",
            "urlopen",
        ]:
            self.assertNotIn(forbidden, all_code)

    def test_source_notebook_contains_no_prefilled_execution_outputs(self) -> None:
        for cell in self.cells:
            if cell.get("cell_type") == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs", []), [])

    def test_evidence_contract_has_no_prefilled_learner_evidence(self) -> None:
        text = (MISSION / "evidence_contract.yaml").read_text(encoding="utf-8")

        self.assertIn("required_evidence:", text)
        self.assertIn("predictions_recorded_before_implementation", text)
        self.assertIsNone(re.search(r"(?m)^\s*learner_evidence\s*:", text))
        self.assertIsNone(re.search(r"(?m)^\s*learner_prediction\s*:", text))
        self.assertIsNone(re.search(r"(?m)^\s*learner_score\s*:", text))

    def test_requirements_and_v01_integration_are_explicit(self) -> None:
        requirements = (
            ROOT / "requirements" / "m05.txt"
        ).read_text(encoding="utf-8")
        integration = (
            MISSION / "flagship_integration.md"
        ).read_text(encoding="utf-8")

        self.assertIn("numpy>=2.0,<3", requirements)
        self.assertIn("jupyter>=1.0", requirements)
        self.assertIn("V01", integration)
        self.assertIn("Structured Data Workbench", integration)
        self.assertIn("M06", integration)
        self.assertIn("M07", integration)


if __name__ == "__main__":
    unittest.main()
