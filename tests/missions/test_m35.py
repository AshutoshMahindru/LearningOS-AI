from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MISSION = ROOT / "missions" / "M35"
NOTEBOOK = ROOT / "labs" / "M35_improve_retrieval.ipynb"
REQUIREMENTS = ROOT / "requirements" / "m35.txt"
DATASETS = ROOT / "datasets" / "M35"
M34_QUESTIONS = ROOT / "datasets" / "M34" / "questions.json"


def load_core():
    spec = importlib.util.spec_from_file_location("m35_retrieval_eval", MISSION / "retrieval_eval.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load M35 retrieval eval")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_core()
RUNTIME_DEPS = importlib.util.find_spec("numpy") is not None
TRANSFER = json.loads((DATASETS / "transfer.json").read_text(encoding="utf-8"))
CHUNK_VERSIONS = json.loads((DATASETS / "chunk_versions.json").read_text(encoding="utf-8"))
HARD_NEGATIVES = json.loads((DATASETS / "hard_negatives.json").read_text(encoding="utf-8"))
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


def independent_dcg(gains: list[float], k: int) -> float:
    return sum(float(gain) / math.log2(index + 1) for index, gain in enumerate(gains[:k], start=1))


def independent_ndcg(ranked_ids: list[str], grades: dict[str, float], k: int) -> float:
    gains = [float(grades.get(chunk_id, 0.0)) for chunk_id in ranked_ids[:k]]
    ideal = sorted((float(gain) for gain in grades.values() if float(gain) > 0.0), reverse=True)
    idcg = independent_dcg(ideal, k)
    return independent_dcg(gains, k) / idcg


class M35StaticContractTests(unittest.TestCase):
    def test_required_executable_artifacts_exist(self):
        for path in (
            MISSION / "retrieval_eval.py",
            MISSION / "optional_model_reranker.py",
            NOTEBOOK,
            REQUIREMENTS,
            ROOT / "tests" / "test_m35.py",
            DATASETS / "chunk_versions.json",
            DATASETS / "hard_negatives.json",
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
            "retrieval_eval.py",
            "optional_model_reranker.py",
        }
        actual = {path.name for path in MISSION.iterdir() if path.is_file()}
        self.assertTrue(required <= actual, required - actual)
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        for name in required:
            self.assertIn(f"missions/M35/{name}", manifest)

    def test_json_fixtures_parse_and_yaml_is_well_formed(self):
        for name in ("chunk_versions.json", "hard_negatives.json", "expected.json", "transfer.json"):
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
        self.assertIn("M35", source)
        self.assertIn("retrieval_eval.py", source)
        self.assertIn("sys.path", source)
        self.assertIn("from missions.M33.semantic_search import", source)
        self.assertIn("from missions.M35.retrieval_eval import", source)

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
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_notebook_enforces_prediction_before_every_action(self):
        cells = notebook_cells()
        positions = {str(cell["id"]): index for index, cell in enumerate(cells)}
        pairs = (
            ("predict-baseline", "run-baseline"),
            ("predict-worst", "run-worst"),
            ("predict-chunking", "run-chunking"),
            ("predict-candidate-k", "run-candidate-k"),
            ("predict-rerank", "run-rerank"),
            ("predict-hard-negatives", "run-hard-negatives"),
            ("predict-slices", "run-slices"),
            ("predict-failure", "run-failure"),
            ("predict-failure-repair", "run-failure-repair"),
            ("predict-relabel", "run-relabel"),
        )
        for prediction, action in pairs:
            with self.subTest(prediction=prediction, action=action):
                self.assertLess(positions[prediction], positions[action])
                self.assertEqual(cells[positions[prediction]].get("cell_type"), "markdown")
                self.assertIn("Predict before running", cell_source(cells[positions[prediction]]))

        self.assertLess(positions["code-reading"], positions["run-code-reading"])
        self.assertIn("Predict before running", cell_source(cells[positions["code-reading"]]))
        self.assertLess(positions["run-failure-repair"], positions["predict-relabel"])

        repair_src = cell_source(cells[positions["run-failure-repair"]])
        relabel_src = cell_source(cells[positions["run-relabel"]])
        failure_src = cell_source(cells[positions["run-failure"]])
        self.assertNotIn("relabel_after_results", repair_src)
        self.assertNotIn("leak_eval_phrasing", relabel_src)
        self.assertNotIn("repair_eval_boundary", failure_src)
        self.assertIn("leak_eval_phrasing", failure_src)
        self.assertIn("repair_eval_boundary", repair_src)
        self.assertIn("relabel_after_results", relabel_src)

        code_reading = cell_source(cells[positions["run-code-reading"]])
        self.assertNotIn("retrieve uses as_evidence", code_reading)
        self.assertIn("candidate ids", code_reading)
        self.assertIn("reranked ids", code_reading)
        self.assertIn("config identity", code_reading)
        self.assertIn("leaked source_hash", code_reading)
        self.assertIn("inspect.getsource", code_reading)

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
            "M34 → M35",
            "sentence-transformers",
            "qdrant-docs",
            "hnsw-paper",
            "candidate recall",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, markdown)

    def test_notebook_prints_required_retrieval_evidence(self):
        code = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        for token in (
            "load_frozen_queries",
            "generate_candidates",
            "rerank_candidates",
            "evaluate_set",
            "project_labels",
            "leak_eval_phrasing",
            "repair_eval_boundary",
            "relabel_after_results",
            "recall_at_k",
            "ndcg_at_k",
            "mean_reciprocal_rank",
            "worst_queries",
            "as_evidence",
            "search",
            "identity()",
            "candidate_ids",
            "ranked_ids",
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

    def test_retrieval_eval_top_level_imports_are_local_or_stdlib(self):
        source = (MISSION / "retrieval_eval.py").read_text(encoding="utf-8")
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
            "json",
            "math",
            "re",
            "sys",
            "time",
            "missions.M33",
        )
        for name in top_level_imports:
            with self.subTest(name=name):
                self.assertTrue(
                    name.startswith(allowed_prefixes) or name in allowed_prefixes,
                    name,
                )
        self.assertNotIn("numpy", top_level_imports)
        self.assertNotIn("torch", top_level_imports)
        self.assertNotIn("sklearn", top_level_imports)
        self.assertNotIn("sentence_transformers", top_level_imports)
        self.assertNotIn("qdrant_client", top_level_imports)
        self.assertNotIn("optional_model_reranker", source)
        self.assertIn("as_evidence", source)
        self.assertIn("search(", source)

    def test_optional_model_reranker_is_unavailable_and_unimported(self):
        adapter = (MISSION / "optional_model_reranker.py").read_text(encoding="utf-8")
        self.assertIn("OptionalModelRerankerUnavailable", adapter)
        self.assertNotIn("import sentence_transformers", adapter)
        spec = importlib.util.spec_from_file_location(
            "m35_optional_model_reranker",
            MISSION / "optional_model_reranker.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with self.assertRaises(module.OptionalModelRerankerUnavailable):
            module.open_optional_model_reranker()

    def test_learner_facing_contracts_remain_unfilled(self):
        adr = (MISSION / "adr_prompt.md").read_text(encoding="utf-8")
        no_ai = (MISSION / "no_ai_gate.md").read_text(encoding="utf-8")
        status = (MISSION / "status.yaml").read_text(encoding="utf-8")
        self.assertIn("[UNFILLED BY LEARNER]", adr)
        self.assertIn("Leave all learner responses unfilled", no_ai)
        self.assertIn("intentionally_unpopulated", status)
        self.assertIn("does not mark M35 repository-executable", status)
        self.assertIn("must not jump executable inventory from this package", status)
        notebook_markdown = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "markdown"
        )
        self.assertIn("[UNFILLED BY LEARNER]", notebook_markdown)
        self.assertNotIn("[FILLED", notebook_markdown)

    def test_m34_questions_stay_frozen_and_unedited_here(self):
        self.assertEqual(M34["eval_version"], "m34.eval.v1")
        self.assertEqual(M34["index_id"], "v08-exact-memory")
        self.assertEqual(M34["corpus_version"], "m33.corpus.v1")
        digest = hashlib.sha256(M34_QUESTIONS.read_bytes()).hexdigest()
        self.assertEqual(digest, EXPECTED["questions_sha256"])
        self.assertEqual(digest, CORE.questions_sha256())
        queries = CORE.load_frozen_queries()
        self.assertEqual(CORE.label_hash(queries), EXPECTED["label_hash"])
        self.assertEqual(queries[0].eval_version, "m34.eval.v1")
        self.assertEqual({query.query_id for query in queries}, set(EXPECTED["query_ids"]))
        self.assertNotIn("rag-reset-login", HARD_NEGATIVES.get("copied_labels", ()))

    def test_metrics_match_independent_oracle_on_transfer_fixture(self):
        ranked = [row["chunk_id"] for row in TRANSFER["candidates"]]
        relevant = list(TRANSFER["gold_relevant"])
        grades = {row["chunk_id"]: float(row["grade"]) for row in TRANSFER["candidates"] if row["grade"]}
        oracle = TRANSFER["metric_oracle"]
        self.assertEqual(CORE.recall_at_k(ranked, relevant, 3), 1.0)
        self.assertEqual(CORE.mean_reciprocal_rank(ranked, relevant), 0.5)
        ndcg = CORE.ndcg_at_k(ranked, grades, 3)
        self.assertAlmostEqual(ndcg, independent_ndcg(ranked, grades, 3), places=12)
        self.assertAlmostEqual(ndcg, oracle["ndcg_at_3"], places=10)
        self.assertAlmostEqual(CORE.dcg_at_k([0.0, 2.0, 1.0], 3), oracle["dcg_at_3"], places=12)
        self.assertIsNone(CORE.recall_at_k(["x"], [], 3))
        self.assertIsNone(CORE.ndcg_at_k(["x"], {}, 3))

    def test_lexical_reranker_does_not_read_labels(self):
        fn_src = inspect_source("lexical_rerank_score")
        self.assertNotIn("relevant_chunk", fn_src)
        self.assertNotIn("gold_answer", fn_src)
        self.assertNotIn("support_chunk", fn_src)
        ticket_gold = CORE.lexical_rerank_score(
            "What is ticket 4412 waiting for?",
            "Ticket 4412 is waiting for inspection.",
        )
        ticket_trap = CORE.lexical_rerank_score(
            "What is ticket 4412 waiting for?",
            "Ticket 4413 is waiting for inspection.",
        )
        self.assertGreater(ticket_gold, ticket_trap)

    def test_window_spans_and_config_identity_are_deterministic(self):
        self.assertEqual(CORE.window_spans("abcdefghij", 4, 0), ((0, 4), (4, 8), (8, 10)))
        self.assertEqual(CORE.window_spans("abcdefghij", 4, 2), ((0, 4), (2, 6), (4, 8), (6, 10)))
        with self.assertRaises(ValueError):
            CORE.window_spans("abc", 4, 4)
        left = CORE.baseline_config()
        right = CORE.baseline_config()
        self.assertEqual(left.identity(), right.identity())
        self.assertEqual(left.identity(), EXPECTED["baseline"]["config_identity"])
        other = CORE.ExperimentConfig(
            experiment_id="rerank-lex",
            corpus_version=CORE.CANONICAL_CORPUS_VERSION,
            chunk_mode="canonical",
            chunk_size=None,
            chunk_overlap=0,
            candidate_k=3,
            reranker_id=CORE.RERANKER_LEX,
        )
        self.assertNotEqual(left.identity(), other.identity())

    def test_relabel_after_results_inflates_without_changing_ranks(self):
        ranked = ("trap::c0", "gold::c0")
        dummy = CORE.QueryReport(
            query_id="toy",
            query_text="toy",
            split="dev",
            answerable=True,
            experiment_id="baseline",
            corpus_version="toy",
            index_id="toy",
            source_hash="abc",
            candidate_k=2,
            reranker_id="identity",
            config_identity="abc",
            candidate_ids=ranked,
            ranked_ids=ranked,
            cosine_ids=ranked,
            relevant_ids=("gold::c0",),
            support_ids=("gold::c0",),
            trap_ids=("trap::c0",),
            mixed_ids=(),
            recall_at_k=1.0,
            mrr=0.5,
            ndcg_at_k=0.63,
            first_relevant_rank=2,
            first_support_rank=2,
            candidate_recall=1.0,
            candidate_support_hit=True,
            trap_at_1=True,
            scored_candidates=2,
            rerank_cost=0,
            proxy_cost=2,
            latency_ms=0.0,
            failure_mode="trap_at_1",
            eval_version=CORE.EVAL_VERSION,
        )
        frozen = CORE.FrozenQuery(
            query_id="toy",
            text="toy",
            experiment="toy",
            split="dev",
            answerable=True,
            gold_answer="gold",
            gold_aliases=(),
            support_chunk_ids=("gold::c0",),
            relevant_chunk_ids=("gold::c0",),
            traps=("trap::c0",),
        )
        gamed = CORE.relabel_after_results((frozen,), (dummy,))
        self.assertEqual(frozen.relevant_chunk_ids, ("gold::c0",))
        self.assertEqual(gamed[0].relevant_chunk_ids, ("gold::c0", "trap::c0"))
        self.assertEqual(CORE.mean_reciprocal_rank(ranked, gamed[0].relevant_chunk_ids), 1.0)
        self.assertEqual(dummy.ranked_ids, ranked)

    def test_chunk_version_fixture_keeps_same_source_docs(self):
        versions = CHUNK_VERSIONS["versions"]
        self.assertEqual(CHUNK_VERSIONS["eval_version"], "m34.eval.v1")
        self.assertEqual(CHUNK_VERSIONS["source_corpus"], "m33.corpus.v1")
        modes = {row["mode"] for row in versions}
        self.assertEqual(modes, {"canonical", "merged", "windows"})
        self.assertTrue(all(item["chunk_id"].startswith("hn::") for item in HARD_NEGATIVES["items"]))


def inspect_source(name: str) -> str:
    import inspect

    return inspect.getsource(getattr(CORE, name))


@unittest.skipUnless(RUNTIME_DEPS, "install requirements/m35.txt to run NumPy-dependent M35 tests")
class M35RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.queries = CORE.load_frozen_queries()
        cls.query_map = CORE.load_query_map()
        cls.source = CORE.load_canonical_corpus()
        cls.index = CORE.load_canonical_index()
        cls.expected = CORE.load_expected_payload()

    def test_canonical_projection_is_identity(self):
        for query in self.queries:
            projected = CORE.project_labels(query, self.source, self.source)
            self.assertEqual(projected.relevant_ids, query.relevant_chunk_ids, query.query_id)
            self.assertEqual(projected.support_ids, query.support_chunk_ids, query.query_id)

    def test_baseline_matches_frozen_expected(self):
        report = CORE.evaluate_set(config=CORE.baseline_config(), queries=self.queries, source_corpus=self.source)
        self.assertEqual(report.label_hash, self.expected["label_hash"])
        self.assertEqual(report.questions_sha256, self.expected["questions_sha256"])
        self.assertEqual(report.index_id, "v08-exact-memory")
        self.assertEqual(report.source_hash, self.expected["source_hash"])
        self.assertEqual(CORE.round_metric(report.mean_mrr), self.expected["baseline"]["mean_mrr"])
        self.assertEqual(CORE.round_metric(report.mean_ndcg_at_k), self.expected["baseline"]["mean_ndcg_at_k"])
        self.assertEqual(report.scored_candidates, 14)
        ticket = report.row_map()["rag-ticket-4412"]
        self.assertEqual(list(ticket.ranked_ids), ["doc-tickets::c1", "doc-tickets::c0", "doc-payments::c2"])
        self.assertEqual(ticket.failure_mode, "trap_at_1")
        self.assertEqual(ticket.eval_version, "m34.eval.v1")

    def test_rerank_keeps_members_and_promotes_ticket_and_invoice(self):
        cosine = CORE.generate_candidates(
            self.query_map["rag-ticket-4412"].text,
            query_id="rag-ticket-4412",
            candidate_k=3,
            index=self.index,
            corpus=self.source,
        )
        reranked = CORE.rerank_candidates(
            cosine,
            reranker_id=CORE.RERANKER_LEX,
            query_text=self.query_map["rag-ticket-4412"].text,
        )
        self.assertEqual(cosine.id_set(), reranked.id_set())
        self.assertEqual(cosine.ids()[0], "doc-tickets::c1")
        self.assertEqual(reranked.ids()[0], "doc-tickets::c0")
        self.assertNotEqual(cosine.ids(), reranked.ids())
        cfg = CORE.ExperimentConfig(
            experiment_id="rerank-lex",
            corpus_version=CORE.CANONICAL_CORPUS_VERSION,
            chunk_mode="canonical",
            chunk_size=None,
            chunk_overlap=0,
            candidate_k=3,
            reranker_id=CORE.RERANKER_LEX,
        )
        report = CORE.evaluate_set(config=cfg, queries=self.queries, source_corpus=self.source)
        self.assertEqual(report.row_map()["rag-ticket-4412"].ranked_ids[0], "doc-tickets::c0")
        self.assertEqual(report.row_map()["rag-h-invoice"].ranked_ids[0], "doc-payments::c2")
        self.assertGreater(report.mean_ndcg_at_k, self.expected["baseline"]["mean_ndcg_at_k"])
        self.assertEqual(CORE.round_metric(report.mean_mrr), 1.0)

    def test_candidate_k_changes_recall_not_scored_candidates(self):
        k1 = CORE.evaluate_set(
            config=CORE.ExperimentConfig(
                experiment_id="candidate-k1",
                corpus_version=CORE.CANONICAL_CORPUS_VERSION,
                chunk_mode="canonical",
                chunk_size=None,
                chunk_overlap=0,
                candidate_k=1,
                reranker_id=CORE.RERANKER_IDENTITY,
            ),
            queries=self.queries,
            source_corpus=self.source,
        )
        k5 = CORE.evaluate_set(
            config=CORE.ExperimentConfig(
                experiment_id="candidate-k5",
                corpus_version=CORE.CANONICAL_CORPUS_VERSION,
                chunk_mode="canonical",
                chunk_size=None,
                chunk_overlap=0,
                candidate_k=5,
                reranker_id=CORE.RERANKER_IDENTITY,
            ),
            queries=self.queries,
            source_corpus=self.source,
        )
        self.assertEqual(k1.scored_candidates, k5.scored_candidates)
        self.assertEqual(k1.scored_candidates, 14)
        self.assertLess(k1.mean_candidate_recall, k5.mean_candidate_recall)
        self.assertEqual(CORE.round_metric(k5.mean_candidate_recall), 1.0)
        self.assertEqual(k1.row_map()["rag-ticket-4412"].candidate_support_hit, False)
        self.assertEqual(k5.row_map()["rag-ticket-4412"].candidate_support_hit, True)

    def test_chunk_versions_change_identity_and_can_mix_spans(self):
        merged = CORE.evaluate_set(
            config=CORE.ExperimentConfig(
                experiment_id="chunk-merged",
                corpus_version="m35.corpus.merged.v1",
                chunk_mode="merged",
                chunk_size=None,
                chunk_overlap=0,
                candidate_k=3,
                reranker_id=CORE.RERANKER_IDENTITY,
            ),
            queries=self.queries,
            source_corpus=self.source,
        )
        self.assertEqual(merged.corpus_version, "m35.corpus.merged.v1")
        self.assertNotEqual(merged.source_hash, self.expected["source_hash"])
        ticket = merged.row_map()["rag-ticket-4412"]
        self.assertIn("doc-tickets::merged", ticket.mixed_ids)
        self.assertEqual(ticket.ranked_ids[0], "doc-tickets::merged")
        self.assertEqual(merged.label_hash, self.expected["label_hash"])

    def test_hard_negatives_degrade_ticket_without_relabeling(self):
        report = CORE.evaluate_set(
            config=CORE.ExperimentConfig(
                experiment_id="hard-negatives",
                corpus_version=CORE.CANONICAL_CORPUS_VERSION,
                chunk_mode="canonical",
                chunk_size=None,
                chunk_overlap=0,
                candidate_k=3,
                reranker_id=CORE.RERANKER_IDENTITY,
                hard_negatives=True,
            ),
            queries=self.queries,
            source_corpus=self.source,
        )
        ticket = report.row_map()["rag-ticket-4412"]
        self.assertEqual(ticket.ranked_ids[0], "hn::ticket-4412-approval")
        self.assertTrue(ticket.trap_at_1)
        self.assertEqual(ticket.relevant_ids, ("doc-tickets::c0",))
        self.assertEqual(report.label_hash, self.expected["label_hash"])
        self.assertEqual(self.query_map["rag-ticket-4412"].relevant_chunk_ids, ("doc-tickets::c0",))

    def test_critical_slice_is_worse_than_aggregate(self):
        report = CORE.evaluate_set(config=CORE.baseline_config(), queries=self.queries, source_corpus=self.source)
        aggregate = report.slices["all"]["mean_ndcg_at_k"]
        critical = report.slices["critical"]["mean_ndcg_at_k"]
        self.assertLess(critical, aggregate)
        worst = CORE.worst_queries(report.rows, n=3)
        self.assertEqual(worst[0].query_id, "rag-h-invoice")
        self.assertEqual(worst[1].query_id, "rag-ticket-4412")

    def test_leakage_improves_metrics_and_repair_restores_boundary(self):
        leaked_corpus = CORE.leak_eval_phrasing(self.source, self.queries)
        self.assertNotEqual(CORE.source_hash(leaked_corpus), CORE.source_hash(self.source))
        self.assertIn(
            "How do I reset my login credentials?",
            leaked_corpus.get_chunk("doc-account-access::c1").text,
        )
        leaked = CORE.evaluate_set(
            config=CORE.ExperimentConfig(
                experiment_id="leak-eval-phrasing",
                corpus_version=CORE.CANONICAL_CORPUS_VERSION,
                chunk_mode="canonical",
                chunk_size=None,
                chunk_overlap=0,
                candidate_k=3,
                reranker_id=CORE.RERANKER_IDENTITY,
                leaked=True,
            ),
            queries=self.queries,
            source_corpus=self.source,
        )
        self.assertGreater(leaked.mean_ndcg_at_k, self.expected["baseline"]["mean_ndcg_at_k"])
        repaired_corpus, repaired_labels = CORE.repair_eval_boundary(
            broken_corpus=leaked_corpus,
            source_corpus=self.source,
            frozen_labels=self.queries,
        )
        self.assertIn(
            "How do I reset my login credentials?",
            leaked_corpus.get_chunk("doc-account-access::c1").text,
        )
        self.assertEqual(CORE.source_hash(repaired_corpus), CORE.source_hash(self.source))
        self.assertEqual(CORE.label_hash(repaired_labels), self.expected["label_hash"])
        repaired = CORE.evaluate_set(
            config=CORE.baseline_config(),
            queries=repaired_labels,
            source_corpus=repaired_corpus,
        )
        self.assertEqual(CORE.round_metric(repaired.mean_ndcg_at_k), self.expected["baseline"]["mean_ndcg_at_k"])

    def test_relabel_gaming_is_separate_from_ranking(self):
        baseline = CORE.evaluate_set(config=CORE.baseline_config(), queries=self.queries, source_corpus=self.source)
        gamed_labels = CORE.relabel_after_results(self.queries, baseline.rows)
        self.assertNotEqual(CORE.label_hash(gamed_labels), CORE.label_hash(self.queries))
        gamed = CORE.rescore_with_labels(
            baseline.rows,
            gamed_labels,
            config=CORE.baseline_config(),
            corpus=self.source,
            index=self.index,
            source_corpus=self.source,
        )
        self.assertGreater(gamed.mean_ndcg_at_k, baseline.mean_ndcg_at_k)
        self.assertEqual(
            gamed.row_map()["rag-ticket-4412"].ranked_ids,
            baseline.row_map()["rag-ticket-4412"].ranked_ids,
        )
        self.assertTrue(gamed.relabeled)

    def test_as_evidence_identity_survives_candidate_generation(self):
        query = self.query_map["rag-reset-login"]
        candidates = CORE.generate_candidates(
            query.text,
            query_id=query.query_id,
            candidate_k=3,
            index=self.index,
            corpus=self.source,
        )
        evidence = candidates.items[0].evidence
        self.assertEqual(evidence["index_id"], candidates.index_id)
        self.assertEqual(evidence["source_hash"], candidates.source_hash)
        self.assertEqual(evidence["chunk_id"], candidates.ids()[0])
        self.assertIn("span", evidence)


if __name__ == "__main__":
    unittest.main()
