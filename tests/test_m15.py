from __future__ import annotations

import importlib.util
import inspect
import json
import math
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
MISSION = ROOT / "missions" / "M15"
NOTEBOOK = ROOT / "labs" / "M15_vectors.ipynb"
FIXTURE = ROOT / "datasets" / "M15" / "vector_fixtures.json"
VECTOR_OPS = MISSION / "vector_ops.py"


def _load_vector_ops():
    spec = importlib.util.spec_from_file_location("m15_vector_ops", VECTOR_OPS)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load M15 vector operations")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OPS = _load_vector_ops()


def _independent_dot(left: list[float], right: list[float]) -> float:
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def _independent_norm(vector: list[float]) -> float:
    return math.sqrt(_independent_dot(vector, vector))


def _independent_cosine(left: list[float], right: list[float]) -> float:
    return _independent_dot(left, right) / (
        _independent_norm(left) * _independent_norm(right)
    )


def _independent_distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(
        math.fsum((a - b) ** 2 for a, b in zip(left, right, strict=True))
    )


class M15VectorOperationTests(unittest.TestCase):
    def test_addition_and_subtraction_preserve_geometric_contract(self) -> None:
        start = (2.0, 1.0)
        displacement = (3.0, -2.0)

        end = OPS.add(start, displacement)

        self.assertEqual(end, (5.0, -1.0))
        self.assertEqual(OPS.subtract(end, start), displacement)
        self.assertEqual(OPS.subtract(OPS.add(start, displacement), start), displacement)

    def test_norm_dot_and_normalization_have_expected_invariants(self) -> None:
        vector = (3.0, 4.0)

        self.assertEqual(OPS.norm(vector), 5.0)
        self.assertEqual(OPS.dot(vector, vector), 25.0)
        self.assertEqual(OPS.dot((1.0, 0.0), (0.0, 1.0)), 0.0)

        unit = OPS.normalize(vector)
        self.assertAlmostEqual(OPS.norm(unit), 1.0)
        self.assertAlmostEqual(unit[0] / unit[1], vector[0] / vector[1])

    def test_zero_vector_direction_operations_fail_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero vector"):
            OPS.normalize((0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "zero vector"):
            OPS.cosine_similarity((1.0, 0.0), (0.0, 0.0))

    def test_cosine_is_scale_invariant_and_bounded(self) -> None:
        self.assertAlmostEqual(OPS.cosine_similarity((1, 0), (5, 0)), 1.0)
        self.assertAlmostEqual(OPS.cosine_similarity((1, 0), (-5, 0)), -1.0)
        self.assertAlmostEqual(OPS.cosine_similarity((1, 0), (0, 9)), 0.0)
        self.assertAlmostEqual(
            OPS.cosine_similarity((2, 1), (7, 3)),
            OPS.cosine_similarity((20, 10), (0.7, 0.3)),
        )
        self.assertLessEqual(abs(OPS.cosine_similarity((2, 1), (7, 3))), 1.0)

    def test_euclidean_distance_is_symmetric_and_has_identity(self) -> None:
        left = (1.0, -2.0, 3.0)
        right = (4.0, 2.0, -1.0)

        self.assertEqual(OPS.euclidean_distance(left, left), 0.0)
        self.assertAlmostEqual(
            OPS.euclidean_distance(left, right),
            OPS.euclidean_distance(right, left),
        )

    def test_input_validation_rejects_invalid_vector_contracts(self) -> None:
        invalid_vectors = [(), (1.0, math.inf), (1.0, math.nan), "1,2"]
        for vector in invalid_vectors:
            with self.subTest(vector=vector):
                with self.assertRaises(ValueError):
                    OPS.as_vector(vector)

        with self.assertRaisesRegex(ValueError, "same dimensions"):
            OPS.add((1.0, 2.0), (1.0,))
        with self.assertRaisesRegex(ValueError, "same dimensions"):
            OPS.dot((1.0,), (1.0, 2.0))

    def test_ranking_requires_explicit_metric_and_has_stable_ties(self) -> None:
        candidates = [
            {"id": "zeta", "vector": [1.0, 1.0]},
            {"id": "alpha", "vector": [1.0, 1.0]},
        ]

        with self.assertRaises(TypeError):
            OPS.rank_vectors((1.0, 0.0), candidates)

        cosine = OPS.rank_vectors((1.0, 0.0), candidates, metric="cosine")
        euclidean = OPS.rank_vectors((1.0, 0.0), candidates, metric="euclidean")
        self.assertEqual([row["id"] for row in cosine], ["alpha", "zeta"])
        self.assertEqual([row["id"] for row in euclidean], ["alpha", "zeta"])

        with self.assertRaisesRegex(ValueError, "metric must be"):
            OPS.rank_vectors((1.0, 0.0), candidates, metric="manhattan")

    def test_ranking_rejects_ambiguous_candidate_identity(self) -> None:
        duplicates = [
            {"id": "same", "vector": [1.0, 0.0]},
            {"id": "same", "vector": [0.0, 1.0]},
        ]

        with self.assertRaisesRegex(ValueError, "duplicate candidate id"):
            OPS.rank_vectors((1.0, 0.0), duplicates, metric="cosine")

    def test_pairwise_report_exposes_v04_measurements(self) -> None:
        report = OPS.compare_vectors((1.0, 0.0), (1.0, 1.0))

        self.assertEqual(
            set(report),
            {"query_norm", "candidate_norm", "dot", "cosine", "euclidean"},
        )
        self.assertEqual(report["query_norm"], 1.0)
        self.assertEqual(report["dot"], 1.0)
        self.assertAlmostEqual(report["candidate_norm"], math.sqrt(2.0))
        self.assertAlmostEqual(report["cosine"], 1.0 / math.sqrt(2.0))
        self.assertEqual(report["euclidean"], 1.0)


class M15FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_fixture_sections_have_finite_fixed_dimensions(self) -> None:
        fixture = self.fixture
        self.assertEqual(fixture["schema_version"], 1)

        numeric = fixture["numeric_features"]
        self.assertEqual(len(numeric["dimensions"]), len(set(numeric["dimensions"])))
        for record in numeric["records"]:
            self.assertEqual(len(record["vector"]), len(numeric["dimensions"]))
            self.assertTrue(all(math.isfinite(value) for value in record["vector"]))

        geometry = fixture["geometry"]
        for key in ["start_position", "displacement", "expected_end_position"]:
            self.assertEqual(len(geometry[key]), len(geometry["dimensions"]))
            self.assertTrue(all(math.isfinite(value) for value in geometry[key]))

        semantic = fixture["semantic_embeddings"]
        semantic_vectors = [semantic["query"]["vector"]] + [
            row["vector"] for row in semantic["documents"]
        ]
        for vector in semantic_vectors:
            self.assertEqual(len(vector), len(semantic["dimensions"]))
            self.assertTrue(all(math.isfinite(value) for value in vector))
            self.assertGreater(_independent_norm(vector), 0.0)

    def test_fixture_geometry_encodes_addition_and_inverse(self) -> None:
        geometry = self.fixture["geometry"]
        calculated_end = [
            start + move
            for start, move in zip(
                geometry["start_position"], geometry["displacement"], strict=True
            )
        ]

        self.assertEqual(calculated_end, geometry["expected_end_position"])
        self.assertEqual(
            [
                end - start
                for end, start in zip(
                    geometry["expected_end_position"],
                    geometry["start_position"],
                    strict=True,
                )
            ],
            geometry["displacement"],
        )

    def test_controlled_fixture_proves_metric_disagreement(self) -> None:
        failure = self.fixture["metric_disagreement"]
        query = failure["query"]
        candidates = failure["candidates"]

        cosine_order = sorted(
            candidates,
            key=lambda row: (-_independent_cosine(query, row["vector"]), row["id"]),
        )
        euclidean_order = sorted(
            candidates,
            key=lambda row: (_independent_distance(query, row["vector"]), row["id"]),
        )

        self.assertEqual(cosine_order[0]["id"], failure["expected_top"]["cosine"])
        self.assertEqual(
            euclidean_order[0]["id"], failure["expected_top"]["euclidean"]
        )
        self.assertNotEqual(cosine_order[0]["id"], euclidean_order[0]["id"])

    def test_normalized_euclidean_order_reconciles_with_cosine(self) -> None:
        failure = self.fixture["metric_disagreement"]
        query = failure["query"]
        candidates = failure["candidates"]

        def unit(vector: list[float]) -> list[float]:
            magnitude = _independent_norm(vector)
            return [value / magnitude for value in vector]

        cosine_ids = [
            row["id"]
            for row in sorted(
                candidates,
                key=lambda row: (
                    -_independent_cosine(query, row["vector"]),
                    row["id"],
                ),
            )
        ]
        normalized_distance_ids = [
            row["id"]
            for row in sorted(
                candidates,
                key=lambda row: (
                    _independent_distance(unit(query), unit(row["vector"])),
                    row["id"],
                ),
            )
        ]

        self.assertEqual(normalized_distance_ids, cosine_ids)

    def test_semantic_fixture_has_expected_cosine_ranking(self) -> None:
        semantic = self.fixture["semantic_embeddings"]
        query = semantic["query"]["vector"]
        calculated = [
            row["id"]
            for row in sorted(
                semantic["documents"],
                key=lambda row: (
                    -_independent_cosine(query, row["vector"]),
                    row["id"],
                ),
            )
        ]

        self.assertEqual(calculated, semantic["expected_cosine_order"])
        self.assertEqual(calculated[0], "vector-operations")
        self.assertEqual(calculated[-1], "bread-recipe")


class M15NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code_cells = [
            cell for cell in cls.notebook["cells"] if cell["cell_type"] == "code"
        ]
        cls.markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        cls.code = "\n".join(
            "".join(cell.get("source", [])) for cell in cls.code_cells
        )

    def test_notebook_is_substantial_with_stable_unique_cell_ids(self) -> None:
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertGreaterEqual(len(self.notebook["cells"]), 30)
        ids = [cell.get("id") for cell in self.notebook["cells"]]
        self.assertTrue(all(isinstance(cell_id, str) and cell_id for cell_id in ids))
        self.assertEqual(len(ids), len(set(ids)))

    def test_source_notebook_has_no_prefilled_execution_state(self) -> None:
        for cell in self.code_cells:
            self.assertIsNone(cell.get("execution_count"), cell["id"])
            self.assertEqual(cell.get("outputs", []), [], cell["id"])

    def test_every_code_cell_is_syntactically_valid(self) -> None:
        for cell in self.code_cells:
            source = "".join(cell.get("source", []))
            compile(source, f"M15-{cell['id']}", "exec")

    def test_notebook_covers_required_operations_and_controlled_failure(self) -> None:
        for token in [
            "add(",
            "subtract(",
            "norm(",
            "dot(",
            "normalize(",
            "cosine_similarity(",
            "euclidean_distance(",
            "rank_vectors(",
            "compare_vectors(",
            "metric='cosine'",
            "metric='euclidean'",
            "metric='dot'",
            "cosine_winner != euclidean_winner",
        ]:
            self.assertIn(token, self.code)

        for phrase in [
            "Controlled failure",
            "normalization mistake",
            "semantic",
            "V04 Mathematical Instrumentation Layer",
            "Code reading",
            "No-AI Gate",
            "Explain and decide",
        ]:
            self.assertIn(phrase, self.markdown)

    def test_prediction_precedes_each_operational_section(self) -> None:
        prediction_count = self.markdown.count("Predict before running")
        self.assertGreaterEqual(prediction_count, 13)

        cells = self.notebook["cells"]
        for index, cell in enumerate(cells):
            if cell["cell_type"] != "code" or cell["id"] == "load-local-contract":
                continue
            earlier_markdown = [
                previous
                for previous in cells[:index]
                if previous["cell_type"] == "markdown"
            ]
            self.assertTrue(earlier_markdown, cell["id"])
            self.assertIn(
                "Predict before running",
                "".join(earlier_markdown[-1].get("source", [])),
                cell["id"],
            )

    def test_notebook_source_is_offline_and_secret_free(self) -> None:
        lowered = self.code.lower()
        for forbidden in [
            "import requests",
            "import httpx",
            "urllib.request",
            "urlopen(",
            "socket.",
            "from openai",
            "import openai",
            "api_key",
            "os.environ",
            "subprocess",
        ]:
            self.assertNotIn(forbidden, lowered)


class M15ContractTests(unittest.TestCase):
    def test_manifest_declares_identity_dependency_runtime_and_artifacts(self) -> None:
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")

        for expected in [
            "id: M15",
            "phase: P3",
            "flagship: V04",
            "- vectors",
            "- similarity",
            "- M05",
            "pedagogy: whole-first",
            "formal_engineering_review: false",
            "notebook: labs/M15_vectors.ipynb",
            "implementation: missions/M15/vector_ops.py",
            "- datasets/M15/vector_fixtures.json",
            "requirements: requirements/m15.txt",
            "mission_test: tests/test_m15.py",
            "cpu_only: true",
            "requires_secrets: false",
            "requires_paid_api: false",
            "network_required: false",
        ]:
            self.assertIn(expected, manifest)

    def test_standard_package_artifacts_have_substantive_content(self) -> None:
        required = [
            "README.md",
            "manifest.yaml",
            "content.yaml",
            "experiments.yaml",
            "code_reading.md",
            "no_ai_gate.md",
            "controlled_failure.md",
            "assessment.yaml",
            "evidence_contract.yaml",
            "flagship_integration.md",
            "status.yaml",
            "adr_prompt.md",
            "vector_ops.py",
        ]

        for name in required:
            with self.subTest(name=name):
                content = (MISSION / name).read_text(encoding="utf-8")
                self.assertGreater(len(content.split()), 15)

        self.assertGreater(NOTEBOOK.stat().st_size, 10_000)
        self.assertGreater(FIXTURE.stat().st_size, 500)

    def test_evidence_contract_is_required_but_never_prefilled(self) -> None:
        evidence = (MISSION / "evidence_contract.yaml").read_text(encoding="utf-8")
        status = (MISSION / "status.yaml").read_text(encoding="utf-8")

        self.assertIn("required_evidence:", evidence)
        self.assertIn("prefilled_learner_evidence: prohibited", evidence)
        self.assertIsNone(re.search(r"(?m)^\s*learner_(evidence|response|answer)\s*:", evidence))
        self.assertIn("implementation_status: implemented", status)
        self.assertIn("learner_evidence_status: intentionally_unpopulated", status)

    def test_controlled_failure_requires_diagnosis_not_metric_dogma(self) -> None:
        controlled = (MISSION / "controlled_failure.md").read_text(
            encoding="utf-8"
        ).lower()

        for term in [
            "cosine",
            "euclidean",
            "different winners",
            "normalization",
            "intended invariant",
            "repair the measurement pipeline",
            "regression check",
        ]:
            self.assertIn(term, controlled)
        self.assertIn("merely says", controlled)

    def test_adr_prompt_matches_repository_decision_template(self) -> None:
        template = (ROOT / "templates" / "ADR.md").read_text(encoding="utf-8")
        prompt = (MISSION / "adr_prompt.md").read_text(encoding="utf-8").lower()

        for heading in [
            "Decision",
            "Context",
            "Alternatives considered",
            "Evidence",
            "Trade-offs",
            "Revisit conditions",
            "Status",
        ]:
            self.assertIn(heading, template)

        for requirement in [
            "representation",
            "preprocessing",
            "metric",
            "ranking direction",
            "zero-vector policy",
            "deterministic tie policy",
            "revisiting the decision",
        ]:
            self.assertIn(requirement, prompt)

    def test_v04_connection_matches_canonical_flagship_and_dependency(self) -> None:
        integration = (MISSION / "flagship_integration.md").read_text(
            encoding="utf-8"
        )
        flagship_registry = (ROOT / "data" / "flagship_versions.yaml").read_text(
            encoding="utf-8"
        )
        dependencies = json.loads(
            (ROOT / "data" / "mission_dependencies.json").read_text(encoding="utf-8")
        )["dependencies"]

        self.assertIn("V04 — Mathematical Instrumentation Layer", integration)
        self.assertIn("id: V04", flagship_registry)
        self.assertIn("name: Mathematical Instrumentation Layer", flagship_registry)
        self.assertEqual(dependencies["M15"], {"blocking": ["M05"], "helpful": []})

    def test_vector_module_has_explicit_metric_and_no_runtime_network(self) -> None:
        signature = inspect.signature(OPS.rank_vectors)
        self.assertEqual(signature.parameters["metric"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIs(signature.parameters["metric"].default, inspect.Parameter.empty)

        source = VECTOR_OPS.read_text(encoding="utf-8").lower()
        for forbidden in ["requests", "httpx", "urllib", "socket", "openai", "api_key"]:
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
