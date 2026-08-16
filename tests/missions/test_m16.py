from __future__ import annotations

import ast
import csv
import importlib.util
import json
import os
import re
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
MISSION = ROOT / "missions" / "M16"
NOTEBOOK = ROOT / "labs" / "M16_matrix_transformations.ipynb"
DATASET = ROOT / "datasets" / "M16" / "shape_points.csv"
RUNTIME_DEPENDENCIES = ("numpy", "matplotlib")
MISSING_RUNTIME_DEPENDENCIES = tuple(
    name
    for name in RUNTIME_DEPENDENCIES
    if importlib.util.find_spec(name) is None
)


def cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return source if isinstance(source, str) else "".join(source)


class M16MissionPackageTests(unittest.TestCase):
    def test_required_mission_artifacts_exist(self) -> None:
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

        self.assertEqual(
            required,
            {path.name for path in MISSION.iterdir() if path.is_file()},
        )
        self.assertTrue(NOTEBOOK.is_file())
        self.assertTrue(DATASET.is_file())
        self.assertTrue((ROOT / "datasets" / "M16" / "README.md").is_file())
        self.assertTrue((ROOT / "requirements" / "m16.txt").is_file())

    def test_manifest_declares_authoritative_contract_and_runtime(self) -> None:
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")

        for expected in [
            "id: M16",
            "phase: P3",
            "flagship: V04",
            "formal_engineering_review: true",
            "entry: visual-first",
            "experiment_loop: prediction-before-action",
            "- matrices",
            "- linear transformations",
            "- M15",
            "- numpy-quickstart",
            "- 3b1b-linear-algebra",
            "cpu_only: true",
            "requires_secrets: false",
            "requires_paid_api: false",
            "network_required: false",
            "restart_run_all_required: true",
            "batch_transform: X_times_A_transpose",
        ]:
            self.assertIn(expected, manifest)

    def test_requirements_declare_runtime_and_validation_dependencies(self) -> None:
        requirements = {
            line.strip()
            for line in (ROOT / "requirements" / "m16.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }

        self.assertEqual(
            requirements,
            {
                "numpy>=1.26",
                "matplotlib>=3.8",
                "nbformat>=5.10",
                "jupyter>=1.0",
                "pytest>=8",
            },
        )

    def test_registered_sources_cover_numpy_and_visual_linear_algebra(self) -> None:
        registry = json.loads(
            (ROOT / "data" / "source_registry.json").read_text(encoding="utf-8")
        )
        sources = {source["id"]: source for source in registry["sources"]}

        self.assertIn("numpy-quickstart", sources)
        self.assertIn("3b1b-linear-algebra", sources)
        self.assertIn("matrices", sources["numpy-quickstart"]["topics"])
        self.assertIn("matrices", sources["3b1b-linear-algebra"]["topics"])

    def test_dataset_is_ordered_numeric_asymmetric_polygon(self) -> None:
        with DATASET.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 7)
        self.assertEqual([int(row["vertex_order"]) for row in rows], list(range(7)))
        self.assertEqual(len({row["point_id"] for row in rows}), len(rows))
        self.assertEqual(
            {row["point_id"] for row in rows if row["landmark"] == "true"},
            {"tip", "neck_upper"},
        )

        points = [(float(row["x"]), float(row["y"])) for row in rows]
        signed_twice_area = sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
        )
        self.assertGreater(abs(signed_twice_area) / 2.0, 1.0)
        self.assertNotEqual(
            max(x for x, _ in points) - min(x for x, _ in points),
            max(y for _, y in points) - min(y for _, y in points),
        )

    def test_notebook_is_visual_first_substantial_and_has_stable_ids(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(len(notebook["cells"]), 30)
        ids = [cell.get("id") for cell in notebook["cells"]]
        self.assertTrue(all(ids))
        self.assertEqual(len(ids), len(set(ids)))

        self.assertEqual(notebook["cells"][0]["cell_type"], "markdown")
        self.assertIn("Begin visually", cell_source(notebook["cells"][0]))
        first_code = next(
            cell_source(cell)
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertIn("shape_points.csv", first_code)
        self.assertIn("plot_shape", first_code)
        self.assertIn("MYSTERY_POINTS", first_code)

    def test_notebook_covers_transform_composition_batch_and_ml_invariants(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        code = "\n".join(
            cell_source(cell)
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        markdown = "\n".join(
            cell_source(cell)
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )

        for token in [
            "A @ x",
            "def rotation",
            "SCALE",
            "SHEAR_X",
            "ROTATE_90 @ SHEAR_X",
            "WRONG_ORDER_COMPOSITE",
            "wrong_orientation_batch",
            "POINTS @ ASYMMETRIC_A.T",
            "samplewise",
            "LARGE_BATCH",
            "X_BATCH @ WEIGHTS + BIAS",
            "A_COLUMN_VIEW = WEIGHTS.T",
            "def apply_layers",
            "except ValueError",
        ]:
            self.assertIn(token, code)

        for phrase in [
            "Matrix × vector",
            "Scaling",
            "Rotation",
            "Shearing",
            "Composition and order",
            "why transpose is useful",
            "Shape reasoning",
            "Connection to ML batches and layers",
            "No-AI Gate and ADR",
        ]:
            self.assertIn(phrase, markdown)

        prediction_cells = [
            cell
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
            and "Prediction checkpoint" in cell_source(cell)
        ]
        self.assertGreaterEqual(len(prediction_cells), 10)

    def test_notebook_code_is_syntax_valid_and_ast_offline_safe(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        imported_roots = set()

        for index, cell in enumerate(code_cells):
            source = cell_source(cell)
            compile(source, f"M16-cell-{index}", "exec")
            tree = ast.parse(source, filename=f"M16-cell-{index}")
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_roots.update(
                        alias.name.split(".", 1)[0] for alias in node.names
                    )
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported_roots.add(node.module.split(".", 1)[0])

        self.assertTrue({"numpy", "matplotlib"}.issubset(imported_roots))
        self.assertTrue(
            imported_roots.isdisjoint(
                {"requests", "httpx", "urllib", "socket", "openai"}
            )
        )

        all_code = "\n".join(cell_source(cell) for cell in code_cells)
        for forbidden in [
            "import requests",
            "import httpx",
            "urllib.request",
            "from openai",
            "import openai",
            "api_key",
            "os.environ",
            "socket.",
        ]:
            self.assertNotIn(forbidden, all_code)

    def test_source_notebook_has_no_prefilled_execution_state(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs", []), [])

    @unittest.skipUnless(
        not MISSING_RUNTIME_DEPENDENCIES,
        "install requirements/m16.txt to run the M16 notebook runtime test; "
        f"missing: {', '.join(MISSING_RUNTIME_DEPENDENCIES)}",
    )
    def test_every_notebook_code_cell_executes_in_order(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        namespace = {"__name__": "__m16_test__"}
        previous_cwd = Path.cwd()
        previous_backend = os.environ.get("MPLBACKEND")
        os.environ["MPLBACKEND"] = "Agg"
        try:
            os.chdir(ROOT)
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    exec(
                        compile(cell_source(cell), f"M16-cell-{index}", "exec"),
                        namespace,
                    )
        finally:
            os.chdir(previous_cwd)
            if previous_backend is None:
                os.environ.pop("MPLBACKEND", None)
            else:
                os.environ["MPLBACKEND"] = previous_backend
            pyplot = namespace.get("plt")
            if pyplot is not None:
                pyplot.close("all")

        self.assertEqual(namespace["POINTS"].shape, (7, 2))
        self.assertTrue(namespace["np"].allclose(
            namespace["repaired_batch"], namespace["samplewise"]
        ))
        self.assertTrue(namespace["np"].allclose(
            namespace["column_view_outputs"], namespace["pre_activation"]
        ))

    def test_controlled_failures_define_distinct_causes_and_repairs(self) -> None:
        controlled = (MISSION / "controlled_failure.md").read_text(
            encoding="utf-8"
        ).lower()

        for term in [
            "wrong composition order",
            "wrong batch orientation",
            "exactly one seeded root cause",
            "smallest one-vector counterexample",
            "non-symmetric matrix",
            "first intermediate value",
            "sample-wise equivalence",
            "repair",
        ]:
            self.assertIn(term, controlled)

        self.assertIn("x @ a", controlled)
        self.assertIn("x @ a.t", controlled)

    def test_no_ai_gate_and_assessment_preserve_transfer_requirements(self) -> None:
        no_ai = (MISSION / "no_ai_gate.md").read_text(encoding="utf-8").lower()
        assessment = (MISSION / "assessment.yaml").read_text(
            encoding="utf-8"
        ).lower()

        for phrase in [
            "without ai-generated code",
            "prediction before execution",
            "composite matrix",
            "shape table",
            "counterexample",
            "dense ml layer",
        ]:
            self.assertIn(phrase, no_ai)

        for requirement in [
            "transfer_required: true",
            "predict_landmark_motion_before_execution",
            "state_composition_application_order",
            "transform_a_row_major_batch_using_transpose",
            "diagnose_plausible_wrong_order_and_orientation_outputs",
            "complete_representation_adr",
        ]:
            self.assertIn(requirement, assessment)

    def test_review_and_adr_require_engineering_decisions_and_invariants(self) -> None:
        review = (MISSION / "review_brief.md").read_text(encoding="utf-8")
        adr = (MISSION / "adr_prompt.md").read_text(encoding="utf-8")

        for heading in [
            "## Architecture and conventions",
            "## Failure model",
            "## Verification strategy",
            "## Review questions",
            "## Residual risks",
        ]:
            self.assertIn(heading, review)

        for requirement in [
            "column-vector mathematics with row-major batch storage",
            "column-vector mathematics with column-major batch storage",
            "row-vector convention throughout",
            "non-square transform",
            "non-symmetric square transform",
            "non-commuting transforms",
            "dense-layer example",
            "revisiting the decision",
        ]:
            self.assertIn(requirement, adr)

    def test_evidence_contract_does_not_prefill_learner_evidence(self) -> None:
        evidence = (MISSION / "evidence_contract.yaml").read_text(encoding="utf-8")

        self.assertIn("required_evidence:", evidence)
        self.assertIn("require_smallest_counterexample_for_plausible_failures: true", evidence)
        self.assertIsNone(re.search(r"(?m)^\s*learner_evidence\s*:", evidence))
        self.assertIsNone(re.search(r"(?m)^\s*learner_response\s*:", evidence))

    def test_v04_bridge_maps_column_and_row_batch_weight_conventions(self) -> None:
        bridge = (MISSION / "flagship_integration.md").read_text(encoding="utf-8")

        self.assertIn("V04", bridge)
        self.assertIn("Mathematical Instrumentation Layer", bridge)
        self.assertIn("X @ W + b", bridge)
        self.assertIn("A = W.T", bridge)
        self.assertIn("X @ A.T", bridge)
        self.assertIn("sample-wise equivalence", bridge)


if __name__ == "__main__":
    unittest.main()
