from __future__ import annotations

import ast
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MISSION = ROOT / "missions" / "M36"
NOTEBOOK = ROOT / "labs" / "M36_vector_db_hybrid.ipynb"
REQUIREMENTS = ROOT / "requirements" / "m36.txt"
DATASETS = ROOT / "datasets" / "M36"
M34_QUESTIONS = ROOT / "datasets" / "M34" / "questions.json"


CORE = importlib.import_module("missions.M36.hybrid_retrieval")
RUNTIME_DEPS = importlib.util.find_spec("numpy") is not None
TRANSFER = json.loads((DATASETS / "transfer.json").read_text(encoding="utf-8"))
EXPECTED = json.loads((DATASETS / "expected.json").read_text(encoding="utf-8"))
M34 = json.loads(M34_QUESTIONS.read_text(encoding="utf-8"))


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


def independent_rrf(rankings: list[list[str]], rrf_k: int) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in rankings:
        seen: set[str] = set()
        for rank, chunk_id in enumerate(ranked, start=1):
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (int(rrf_k) + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class M36StaticContractTests(unittest.TestCase):
    def test_required_executable_artifacts_exist(self):
        for path in (
            MISSION / "hybrid_retrieval.py",
            MISSION / "optional_qdrant.py",
            NOTEBOOK,
            REQUIREMENTS,
            ROOT / "tests" / "test_m36.py",
            DATASETS / "expected.json",
            DATASETS / "transfer.json",
            DATASETS / "README.md",
            DATASETS / "freeze_expected.py",
        ):
            with self.subTest(path=path):
                self.assertTrue(path.is_file())

    def test_required_mission_artifacts_are_declared_and_present(self):
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
            "review_brief.md",
            "adr_prompt.md",
            "status.yaml",
            "hybrid_retrieval.py",
            "optional_qdrant.py",
        }
        actual = {path.name for path in MISSION.iterdir() if path.is_file()}
        self.assertTrue(required <= actual, required - actual)
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        for name in required:
            self.assertIn(f"missions/M36/{name}", manifest)

    def test_json_fixtures_parse_and_yaml_is_well_formed(self):
        for name in ("expected.json", "transfer.json"):
            payload = json.loads((DATASETS / name).read_text(encoding="utf-8"))
            self.assertIsInstance(payload, dict)
            self.assertFalse(payload.get("downloaded", False))
            self.assertFalse(payload.get("network_required", False))
        try:
            import yaml
        except ImportError:
            yaml = None
        for path in sorted(MISSION.glob("*.yaml")):
            text = path.read_text(encoding="utf-8")
            self.assertTrue(text.strip())
            self.assertNotIn("\t", text)
            if yaml is not None:
                loaded = yaml.safe_load(text)
                self.assertIsInstance(loaded, dict, path.name)

    def test_notebook_is_valid_clean_unique_substantial_and_offline(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertEqual(notebook.get("nbformat_minor"), 5)
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
            "sentence_transformers",
            "from_pretrained",
            "huggingface",
            "autotokenizer",
            "automodel",
            "tiktoken",
            "transformers",
            "faiss",
            "qdrant",
            "hnsw",
            "nucleus",
            "top-p",
            "temperature",
            "execute_tool",
            "langgraph",
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
        self.assertIn("M36", source)
        self.assertIn("hybrid_retrieval.py", source)
        self.assertIn("sys.path", source)
        self.assertIn("from missions.M33.semantic_search import", source)
        self.assertIn("from missions.M35.retrieval_eval import", source)
        self.assertIn("from missions.M36.hybrid_retrieval import", source)

    def test_future_mission_boundary_stays_closed_in_code(self):
        source = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        for forbidden in (
            "from_pretrained",
            "sentence_transformers",
            "faiss",
            "qdrant",
            "hnsw",
            "FastAPI",
            "flask",
            "ChatCompletion",
            "nucleus",
            "top_p",
            "temperature",
            "execute_tool",
            "langgraph",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_notebook_enforces_prediction_before_every_action(self):
        cells = notebook_cells()
        positions = {str(cell["id"]): index for index, cell in enumerate(cells)}
        pairs = (
            ("predict-baseline", "run-baseline"),
            ("predict-ann", "run-ann"),
            ("predict-effort", "run-effort"),
            ("predict-filter", "run-filter"),
            ("predict-sparse", "run-sparse"),
            ("predict-fusion", "run-fusion"),
            ("predict-slices", "run-slices"),
            ("predict-lifecycle", "run-lifecycle"),
            ("predict-failure", "run-failure"),
            ("predict-failure-repair", "run-failure-repair"),
        )
        for prediction, action in pairs:
            with self.subTest(prediction=prediction, action=action):
                self.assertLess(positions[prediction], positions[action])
                self.assertEqual(cells[positions[prediction]].get("cell_type"), "markdown")
                self.assertIn("Predict before running", cell_source(cells[positions[prediction]]))

        self.assertLess(positions["code-reading"], positions["run-code-reading"])
        self.assertIn("Predict before running", cell_source(cells[positions["code-reading"]]))
        self.assertLess(positions["run-failure"], positions["predict-failure-repair"])

        fusion_src = cell_source(cells[positions["run-fusion"]])
        self.assertIn("fuse_channels", fusion_src)
        self.assertIn("dense_invoice", fusion_src)
        self.assertIn("sparse_invoice", fusion_src)
        self.assertNotIn("hybrid_search", fusion_src)
        self.assertNotIn("candidate_k=5", fusion_src)
        self.assertNotIn("mix_raw_scores", fusion_src)
        self.assertNotIn("insert_chunk", fusion_src)

        inspect_src = cell_source(cells[positions["inspect-store"]])
        self.assertIn("as_evidence keys", inspect_src)
        self.assertIn("memory_proxy", inspect_src)
        self.assertNotIn("doc-tickets::c0", inspect_src)
        self.assertNotIn("doc-tickets::c1", inspect_src)
        self.assertNotIn("scored_candidates", inspect_src)
        self.assertNotIn("response.ids()", inspect_src)

        failure_src = cell_source(cells[positions["run-failure"]])
        repair_src = cell_source(cells[positions["run-failure-repair"]])
        self.assertIn("mix_raw_scores", failure_src)
        self.assertNotIn("repair_fusion", failure_src)
        self.assertIn("repair_fusion", repair_src)
        self.assertNotIn("mix_raw_scores", repair_src)

        effort_src = cell_source(cells[positions["run-effort"]])
        self.assertIn("ef=", effort_src)
        self.assertNotIn("mix_raw_scores", effort_src)
        self.assertNotIn("insert_chunk", effort_src)

        filter_src = cell_source(cells[positions["run-filter"]])
        self.assertIn("late_missed_relevant", filter_src)
        self.assertIn("prefilter_ids", filter_src)

        code_reading = cell_source(cells[positions["run-code-reading"]])
        self.assertIn("inspect.getsource", code_reading)
        self.assertIn("build_adjacency", code_reading)
        self.assertIn("as_evidence keys", code_reading)
        self.assertIn("entry neighbors", code_reading)
        self.assertIn("degree_m", code_reading)
        self.assertNotIn("ceo_low", code_reading)
        self.assertNotIn("late missed", code_reading)

        predict_code = cell_source(cells[positions["code-reading"]])
        self.assertIn("fusion", predict_code)
        self.assertIn("entry_id", predict_code)
        self.assertNotIn("ef=1", predict_code)
        self.assertNotIn("Please reset", predict_code)

        markdown = "\n".join(
            cell_source(cell) for cell in cells if cell.get("cell_type") == "markdown"
        )
        self.assertGreaterEqual(markdown.count("Predict before running"), 11)
        for phrase in (
            "WHOLE",
            "MAP",
            "PREDICT",
            "predict → act → observe → explain",
            "timestamp",
            "Controlled failure",
            "UNFILLED BY LEARNER",
            "M35 → M36",
            "sentence-transformers",
            "qdrant-docs",
            "hnsw-paper",
            "phase-end",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, markdown)

    def test_notebook_prints_required_retrieval_evidence(self):
        code = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        for token in (
            "m35_baseline_report",
            "open_teaching_store",
            "exact_search",
            "approximate_search",
            "compare_to_exact",
            "filter_placement_trace",
            "sparse_search",
            "fuse_channels",
            "hybrid_search",
            "mix_raw_scores",
            "repair_fusion",
            "insert_chunk",
            "update_chunk_text",
            "delete_chunk",
            "rebuild_store",
            "as_evidence",
            "neighbor_recall",
            "late_missed_relevant",
            "memory_proxy",
        ):
            with self.subTest(token=token):
                self.assertIn(token, code)

    def test_requirements_are_bounded_and_cover_runtime(self):
        requirements = REQUIREMENTS.read_text(encoding="utf-8").lower().splitlines()
        for package in ("matplotlib", "nbclient", "nbformat", "pytest", "jupyter", "numpy"):
            with self.subTest(package=package):
                matching = [line for line in requirements if line.startswith(package)]
                self.assertEqual(len(matching), 1)
                self.assertIn(">=", matching[0])
                self.assertIn("<", matching[0])
        joined = "\n".join(requirements)
        for forbidden in ("torch", "sentence-transformers", "transformers", "faiss", "qdrant"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, joined)

    def test_hybrid_retrieval_top_level_imports_are_local_or_stdlib(self):
        source = (MISSION / "hybrid_retrieval.py").read_text(encoding="utf-8")
        module = ast.parse(source)
        top_level_imports = [
            alias.name
            for node in module.body
            if isinstance(node, ast.Import)
            for alias in node.names
        ] + [
            (node.module or "")
            for node in module.body
            if isinstance(node, ast.ImportFrom)
        ]
        allowed_prefixes = (
            "__future__",
            "collections",
            "dataclasses",
            "functools",
            "pathlib",
            "hashlib",
            "heapq",
            "json",
            "math",
            "sys",
            "time",
            "missions.M28",
            "missions.M33",
            "missions.M35",
        )
        for name in top_level_imports:
            with self.subTest(name=name):
                self.assertTrue(
                    name.startswith(allowed_prefixes) or name in allowed_prefixes,
                    name,
                )
        self.assertNotIn("numpy", top_level_imports)
        self.assertNotIn("torch", top_level_imports)
        self.assertNotIn("faiss", " ".join(top_level_imports))
        self.assertNotIn("qdrant_client", " ".join(top_level_imports))
        self.assertNotIn("optional_qdrant", source)
        self.assertIn("as_evidence", source)
        self.assertIn("search(", source)

    def test_optional_qdrant_is_unavailable_and_unimported(self):
        adapter = (MISSION / "optional_qdrant.py").read_text(encoding="utf-8")
        self.assertIn("OptionalQdrantUnavailable", adapter)
        self.assertNotIn("import qdrant_client", adapter)
        spec = importlib.util.spec_from_file_location(
            "m36_optional_qdrant_probe",
            MISSION / "optional_qdrant.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with self.assertRaises(module.OptionalQdrantUnavailable):
            module.open_optional_qdrant()

    def test_package_import_shares_m35_classes(self):
        m35 = importlib.import_module("missions.M35.retrieval_eval")
        self.assertIs(CORE.FrozenQuery, m35.FrozenQuery)
        self.assertIs(CORE.evaluate_set, m35.evaluate_set)

    def test_learner_facing_contracts_remain_unfilled(self):
        adr = (MISSION / "adr_prompt.md").read_text(encoding="utf-8")
        no_ai = (MISSION / "no_ai_gate.md").read_text(encoding="utf-8")
        status = (MISSION / "status.yaml").read_text(encoding="utf-8")
        self.assertIn("[UNFILLED BY LEARNER]", adr)
        self.assertIn("Leave all learner responses unfilled", no_ai)
        self.assertIn("intentionally_unpopulated", status)
        self.assertIn("does not mark M36 repository-executable", status)
        self.assertIn("must not jump executable inventory from this package", status)
        self.assertIn("V09 does not close because this package exists", status)
        notebook_markdown = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "markdown"
        )
        self.assertIn("[UNFILLED BY LEARNER]", notebook_markdown)
        self.assertNotIn("[FILLED", notebook_markdown)

    def test_m34_questions_stay_frozen_and_unedited_here(self):
        self.assertEqual(M34["eval_version"], "m34.eval.v1")
        self.assertEqual(EXPECTED["eval_version"], "m34.eval.v1")
        self.assertEqual(EXPECTED["label_hash"], CORE.label_hash(CORE.load_frozen_queries()))
        self.assertEqual(EXPECTED["questions_sha256"], CORE.questions_sha256())
        self.assertEqual(EXPECTED["source_hash"], EXPECTED["source_hash"])

    def test_rrf_matches_independent_oracle_on_transfer_fixture(self):
        example = TRANSFER["rrf_example"]
        fused = CORE.reciprocal_rank_fusion(
            (example["dense_ids"], example["sparse_ids"]),
            rrf_k=int(TRANSFER["rrf_k"]),
        )
        independent = independent_rrf(
            [list(example["dense_ids"]), list(example["sparse_ids"])],
            int(TRANSFER["rrf_k"]),
        )
        self.assertEqual([chunk_id for chunk_id, _score in fused], TRANSFER["rrf_oracle"]["order"])
        self.assertEqual([chunk_id for chunk_id, _score in fused], [chunk_id for chunk_id, _score in independent])
        for chunk_id, score in fused:
            self.assertAlmostEqual(score, TRANSFER["rrf_oracle"]["scores"][chunk_id], places=12)
        left = CORE.reciprocal_rank_fusion((["a", "b"], ["b", "a"]), rrf_k=60)
        self.assertEqual(left[0][1], left[1][1])
        self.assertEqual([chunk_id for chunk_id, _score in left[:2]], ["a", "b"])

    def test_infra_config_identity_is_deterministic(self):
        left = CORE.InfraConfig(experiment_id="m35-oracle", backend=CORE.BACKEND_EXACT)
        right = CORE.InfraConfig(experiment_id="m35-oracle", backend=CORE.BACKEND_EXACT)
        self.assertEqual(left.identity(), right.identity())
        self.assertEqual(left.identity(), EXPECTED["configs"]["m35-oracle"])
        other = CORE.InfraConfig(
            experiment_id="hybrid-rrf",
            backend=CORE.BACKEND_EXACT,
            fusion=CORE.FUSION_RRF,
            candidate_k=CORE.DEFAULT_CANDIDATE_K,
        )
        self.assertEqual(other.identity(), EXPECTED["configs"]["hybrid-rrf"])
        self.assertNotEqual(left.identity(), other.identity())
        with self.assertRaises(ValueError):
            CORE.InfraConfig(experiment_id="bad", backend=CORE.BACKEND_EXACT, fusion=CORE.FUSION_RAW_SUM)


@unittest.skipUnless(RUNTIME_DEPS, "install requirements/m36.txt to run NumPy-dependent M36 tests")
class M36RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.queries = CORE.load_frozen_queries()
        cls.query_map = CORE.load_query_map()
        cls.store = CORE.open_teaching_store()
        cls.expected = CORE.load_expected_payload()

    def test_m35_oracle_identity_is_unchanged(self):
        report = CORE.m35_baseline_report(queries=self.queries)
        self.assertEqual(report.eval_version, "m34.eval.v1")
        self.assertEqual(report.index_id, "v08-exact-memory")
        self.assertEqual(report.scored_candidates, 14)
        self.assertEqual(CORE.round_metric(report.mean_ndcg_at_k), self.expected["baseline"]["mean_ndcg_at_k"])
        self.assertEqual(
            list(report.row_map()["rag-ticket-4412"].ranked_ids),
            self.expected["baseline"]["ticket_ranked_ids"],
        )
        candidates = CORE.generate_candidates(
            self.query_map["rag-ticket-4412"].text,
            query_id="rag-ticket-4412",
            candidate_k=3,
            index=self.store.exact,
            corpus=self.store.corpus,
        )
        self.assertEqual(candidates.ids()[0], "doc-tickets::c1")
        self.assertEqual(candidates.items[0].evidence["index_id"], "v08-exact-memory")

    def test_approximate_search_can_miss_exact_neighbors(self):
        low = CORE.compare_to_exact(
            self.store,
            self.query_map["rag-ceo"].text,
            query_id="rag-ceo",
            top_k=3,
            ef=CORE.LOW_EF,
        )
        self.assertEqual(list(low.exact_ids), self.expected["ann"]["rag-ceo"]["ef_1"]["exact_ids"])
        self.assertEqual(list(low.approx_ids), self.expected["ann"]["rag-ceo"]["ef_1"]["approx_ids"])
        self.assertLess(low.neighbor_recall, 1.0)
        self.assertLess(low.approx_comparisons, low.exact_comparisons)
        legal = CORE.compare_to_exact(
            self.store,
            self.query_map["rag-legal-forbid"].text,
            query_id="rag-legal-forbid",
            top_k=3,
            ef=CORE.LOW_EF,
        )
        self.assertNotEqual(legal.approx_ids[0], legal.exact_ids[0])
        self.assertFalse(legal.top1_match)

    def test_raising_ef_recovers_exact_neighbors_on_ceo(self):
        low = CORE.compare_to_exact(
            self.store,
            self.query_map["rag-ceo"].text,
            query_id="rag-ceo",
            top_k=3,
            ef=1,
        )
        high = CORE.compare_to_exact(
            self.store,
            self.query_map["rag-ceo"].text,
            query_id="rag-ceo",
            top_k=3,
            ef=4,
        )
        self.assertEqual(list(high.approx_ids), list(high.exact_ids))
        self.assertEqual(high.neighbor_recall, 1.0)
        self.assertGreater(high.neighbor_recall, low.neighbor_recall)
        self.assertGreaterEqual(high.approx_comparisons, low.approx_comparisons)
        self.assertEqual(high.exact_comparisons, 14)

    def test_late_filter_after_small_topk_misses_eligible_relevant(self):
        trace = CORE.filter_placement_trace(
            self.store,
            CORE.FILTER_DEMO_QUERY,
            query_id=CORE.FILTER_DEMO_QUERY_ID,
            filters=CORE.FILTER_DEMO_FILTERS,
            relevant_ids=CORE.FILTER_DEMO_RELEVANT,
            top_k=CORE.FILTER_DEMO_K,
        )
        self.assertEqual(list(trace.unfiltered_ids), ["doc-device-printer::c0"])
        self.assertEqual(list(trace.prefilter_ids), ["doc-account-access::c1"])
        self.assertEqual(list(trace.late_ids), [])
        self.assertEqual(list(trace.late_missed_relevant), ["doc-account-access::c1"])
        self.assertEqual(list(trace.prefilter_missed_relevant), [])
        repaired = CORE.repair_filter_placement(
            self.store,
            CORE.FILTER_DEMO_QUERY,
            query_id=CORE.FILTER_DEMO_QUERY_ID,
            filters=CORE.FILTER_DEMO_FILTERS,
            relevant_ids=CORE.FILTER_DEMO_RELEVANT,
            top_k=CORE.FILTER_DEMO_K,
            broken=trace,
        )
        self.assertEqual(list(repaired.prefilter_ids), ["doc-account-access::c1"])
        self.assertEqual(list(trace.late_ids), [])
        self.assertEqual(list(trace.late_missed_relevant), ["doc-account-access::c1"])

    def test_sparse_promotes_lexical_gold_dense_ranks_second(self):
        ticket = self.query_map["rag-ticket-4412"]
        dense = CORE.exact_search(self.store, ticket.text, query_id=ticket.query_id, top_k=3)
        sparse = CORE.sparse_search(self.store, ticket.text, query_id=ticket.query_id, top_k=3)
        self.assertEqual(list(dense.ids()), self.expected["channels"]["rag-ticket-4412"]["dense_ids"])
        self.assertEqual(list(sparse.ids()), self.expected["channels"]["rag-ticket-4412"]["sparse_ids"])
        self.assertEqual(dense.ids()[0], "doc-tickets::c1")
        self.assertEqual(sparse.ids()[0], "doc-tickets::c0")
        invoice = self.query_map["rag-h-invoice"]
        sparse_invoice = CORE.sparse_search(self.store, invoice.text, query_id=invoice.query_id, top_k=3)
        dense_invoice = CORE.exact_search(self.store, invoice.text, query_id=invoice.query_id, top_k=3)
        self.assertEqual(dense_invoice.ids()[0], "doc-payments::c1")
        self.assertEqual(sparse_invoice.ids()[0], "doc-payments::c2")

    def test_declared_rrf_fuses_ticket_and_invoice_without_raw_scores(self):
        ticket = self.query_map["rag-ticket-4412"]
        hybrid = CORE.hybrid_search(
            self.store,
            ticket.text,
            query_id=ticket.query_id,
            top_k=3,
            candidate_k=5,
        )
        self.assertEqual(hybrid.fusion, CORE.FUSION_RRF)
        self.assertEqual(hybrid.ids()[0], "doc-tickets::c0")
        self.assertEqual(list(hybrid.ids()), self.expected["channels"]["rag-ticket-4412"]["hybrid_ids"])
        invoice = self.query_map["rag-h-invoice"]
        hybrid_invoice = CORE.hybrid_search(
            self.store,
            invoice.text,
            query_id=invoice.query_id,
            top_k=3,
            candidate_k=5,
        )
        self.assertEqual(hybrid_invoice.ids()[0], "doc-payments::c2")
        evidence = hybrid.hits[0].as_evidence()
        self.assertEqual(evidence["chunk_id"], "doc-tickets::c0")
        self.assertEqual(evidence["fusion"], "rrf")
        self.assertEqual(evidence["index_id"], hybrid.index_id)

    def test_raw_score_mix_is_the_broken_path_and_repair_keeps_it(self):
        password = self.query_map["rag-password-procedure"]
        dense = CORE.exact_search(self.store, password.text, query_id=password.query_id, top_k=5)
        sparse = CORE.sparse_search(self.store, password.text, query_id=password.query_id, top_k=5)
        mixed = CORE.mix_raw_scores(dense, sparse, top_k=5, store=self.store)
        self.assertEqual(mixed.fusion, CORE.FUSION_RAW_SUM)
        self.assertEqual(list(mixed.ids()), self.expected["channels"]["rag-password-procedure"]["mix_ids"])
        self.assertEqual(mixed.ids()[1], "doc-refund-policy::c1")
        gold_rank = list(mixed.ids()).index("doc-account-access::c1")
        trap_rank = list(mixed.ids()).index("doc-refund-policy::c1")
        self.assertLess(trap_rank, gold_rank)
        repaired = CORE.repair_fusion(
            broken=mixed,
            dense=dense,
            sparse=sparse,
            store=self.store,
            top_k=5,
        )
        self.assertEqual(repaired.fusion, CORE.FUSION_RRF)
        self.assertEqual(
            list(repaired.ids()),
            self.expected["channels"]["rag-password-procedure"]["repaired_rrf_ids"],
        )
        self.assertIn("doc-account-access::c1", repaired.ids()[:3])
        self.assertEqual(mixed.ids()[1], "doc-refund-policy::c1")
        self.assertNotEqual(list(mixed.ids()), list(repaired.ids()))

    def test_query_slices_separate_lexical_and_semantic_wins(self):
        rows = {
            row["query_id"]: row
            for row in CORE.slice_channel_rows(self.store, self.queries, top_k=3, candidate_k=5)
        }
        ticket = rows["rag-ticket-4412"]
        self.assertEqual(ticket["sparse_top"], "doc-tickets::c0")
        self.assertEqual(ticket["dense_top"], "doc-tickets::c1")
        self.assertEqual(ticket["hybrid_top"], "doc-tickets::c0")
        password = rows["rag-password-procedure"]
        self.assertIsNotNone(password["dense_support_rank"])
        self.assertNotEqual(password["sparse_top"], password["support_id"])

    def test_lifecycle_insert_update_delete_rebuild_fail_closed(self):
        inserted, event = CORE.insert_chunk(
            self.store,
            document_id="doc-tickets",
            chunk_id=CORE.LIFECYCLE_INSERT_ID,
            text="Ticket 4414 is waiting for billing.",
            metadata={"topic": "ticket", "source": "ops-log", "locale": "en"},
        )
        self.assertTrue(inserted.metadata.dirty)
        self.assertEqual(event.op, "insert")
        self.assertEqual(event.chunk_count, 15)
        with self.assertRaises(CORE.StoreStaleError):
            CORE.exact_search(inserted, "ticket 4414", query_id="stale")
        rebuilt, rebuilt_event = CORE.rebuild_store(inserted)
        self.assertFalse(rebuilt.metadata.dirty)
        self.assertEqual(rebuilt_event.op, "rebuild")
        self.assertEqual(len(rebuilt.records()), 15)
        sparse = CORE.sparse_search(rebuilt, "ticket 4414", query_id="insert-query", top_k=3)
        self.assertEqual(sparse.ids()[0], CORE.LIFECYCLE_INSERT_ID)

        updated, _update_event = CORE.update_chunk_text(
            rebuilt,
            "doc-weather::c0",
            "Snow is expected tomorrow in the valley.",
        )
        with self.assertRaises(CORE.StoreStaleError):
            CORE.exact_search(updated, "snow valley", query_id="stale-update")
        rebuilt_update, _ = CORE.rebuild_store(updated)
        self.assertIn("Snow", rebuilt_update.corpus.get_chunk("doc-weather::c0").text)
        snow = CORE.exact_search(
            rebuilt_update,
            "Snow is expected tomorrow in the valley.",
            query_id="snow",
            top_k=3,
        )
        self.assertEqual(snow.ids()[0], "doc-weather::c0")

        deleted, _ = CORE.delete_chunk(rebuilt_update, "doc-legal::c0")
        rebuilt_delete, _ = CORE.rebuild_store(deleted)
        ids = [record.chunk.chunk_id for record in rebuilt_delete.records()]
        self.assertNotIn("doc-legal::c0", ids)
        self.assertEqual(len(ids), 14)
        self.assertFalse(rebuilt_delete.metadata.dirty)

    def test_memory_proxy_counts_graph_edges_beyond_vectors(self):
        proxy = CORE.memory_proxy(self.store)
        self.assertEqual(proxy.n_vectors, 14)
        self.assertGreater(proxy.graph_edges, 0)
        self.assertGreater(proxy.total_bytes, proxy.vector_bytes)
        self.assertEqual(proxy.as_dict()["n_vectors"], self.expected["memory_proxy"]["n_vectors"])
        self.assertEqual(proxy.graph_edges, self.expected["memory_proxy"]["graph_edges"])

    def test_hybrid_search_rejects_raw_sum_fusion_name(self):
        with self.assertRaises(ValueError):
            CORE.hybrid_search(
                self.store,
                "ticket 4412",
                query_id="x",
                fusion=CORE.FUSION_RAW_SUM,
            )


if __name__ == "__main__":
    unittest.main()
