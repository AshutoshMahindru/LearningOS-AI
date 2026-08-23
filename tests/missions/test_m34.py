from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MISSION = ROOT / "missions" / "M34"
NOTEBOOK = ROOT / "labs" / "M34_rag.ipynb"
REQUIREMENTS = ROOT / "requirements" / "m34.txt"
DATASETS = ROOT / "datasets" / "M34"
M33 = ROOT / "missions" / "M33"


def load_core():
    spec = importlib.util.spec_from_file_location("m34_rag_pipeline", MISSION / "rag_pipeline.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load M34 RAG pipeline")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_core()
RUNTIME_DEPS = importlib.util.find_spec("numpy") is not None
QUESTIONS = json.loads((DATASETS / "questions.json").read_text(encoding="utf-8"))
EXPECTED = json.loads((DATASETS / "expected.json").read_text(encoding="utf-8"))
TRANSFER = json.loads((DATASETS / "transfer.json").read_text(encoding="utf-8"))


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


def independent_support(claim: str, span: str) -> bool:
    """Second support path used only as a parity oracle."""

    return CORE.span_supports_claim(claim, span)


def independent_pack_ids(rows: list[dict], *, max_chars: int, max_chunks: int) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    dropped: list[str] = []
    used = 0
    sep = len(CORE.SEPARATOR)
    for row in rows:
        text = str(row["text"])
        if len(kept) >= max_chunks:
            dropped.append(str(row["chunk_id"]))
            continue
        extra = len(text) if not kept else sep + len(text)
        if used + extra > max_chars:
            dropped.append(str(row["chunk_id"]))
            continue
        kept.append(str(row["chunk_id"]))
        used += extra
    return kept, dropped


class M34StaticContractTests(unittest.TestCase):
    def test_required_executable_artifacts_exist(self):
        for path in (
            MISSION / "rag_pipeline.py",
            MISSION / "optional_live_llm.py",
            NOTEBOOK,
            REQUIREMENTS,
            ROOT / "tests" / "test_m34.py",
            DATASETS / "questions.json",
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
            "rag_pipeline.py",
            "optional_live_llm.py",
        }
        actual = {path.name for path in MISSION.iterdir() if path.is_file()}
        self.assertTrue(required <= actual, required - actual)
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        for name in required:
            self.assertIn(f"missions/M34/{name}", manifest)

    def test_json_fixtures_parse_and_yaml_is_well_formed(self):
        for name in ("questions.json", "expected.json", "transfer.json"):
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
            "softmax",
            "faiss",
            "qdrant",
            "hnsw",
            "rerank",
            "ndcg",
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
        self.assertIn("M34", source)
        self.assertIn("rag_pipeline.py", source)
        self.assertIn("sys.path", source)
        self.assertIn("from missions.M33.semantic_search import", source)
        self.assertIn("from missions.M34.rag_pipeline import", source)

    def test_future_mission_boundary_stays_closed_in_code(self):
        source = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        for forbidden in (
            "softmax",
            "attention",
            "from_pretrained",
            "sentence_transformers",
            "faiss",
            "qdrant",
            "hnsw",
            "rerank",
            "ndcg",
            "FastAPI",
            "flask",
            "ChatCompletion",
            "nucleus",
            "top_p",
            "temperature",
            "chunk_overlap",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_notebook_enforces_prediction_before_every_action(self):
        cells = notebook_cells()
        positions = {str(cell["id"]): index for index, cell in enumerate(cells)}
        pairs = (
            ("predict-rag", "run-rag"),
            ("predict-with-without", "run-with-without"),
            ("predict-topk", "run-topk"),
            ("predict-budget", "run-budget"),
            ("predict-unanswerable", "run-unanswerable"),
            ("predict-miss", "run-miss"),
            ("predict-citation", "run-citation"),
            ("predict-eval", "run-eval"),
            ("predict-failure", "run-failure"),
            ("predict-failure-repair", "run-failure-repair"),
            ("predict-invented", "run-invented"),
        )
        for prediction, action in pairs:
            with self.subTest(prediction=prediction, action=action):
                self.assertLess(positions[prediction], positions[action])
                self.assertEqual(cells[positions[prediction]].get("cell_type"), "markdown")
                self.assertIn("Predict before running", cell_source(cells[positions[prediction]]))

        self.assertLess(positions["code-reading"], positions["run-code-reading"])
        self.assertIn("Predict before running", cell_source(cells[positions["code-reading"]]))
        self.assertLess(positions["run-failure-repair"], positions["predict-invented"])

        repair_src = cell_source(cells[positions["run-failure-repair"]])
        invented_src = cell_source(cells[positions["run-invented"]])
        self.assertNotIn("invented_support", repair_src)
        self.assertNotIn("invented =", repair_src)
        self.assertIn('defect="invented_support"', invented_src)
        self.assertIn("repair_grounding", invented_src)
        self.assertNotIn('defect="invented_support"', cell_source(cells[positions["predict-failure-repair"]]))

        code_reading = cell_source(cells[positions["run-code-reading"]])
        self.assertNotIn("retrieve uses as_evidence", code_reading)
        self.assertNotIn("answer_query records weights_updated", code_reading)
        self.assertIn("weights_updated on live trace", code_reading)
        self.assertIn("pack ids equal retrieval prefix", code_reading)
        self.assertIn("packed index_id matches as_evidence", code_reading)

        markdown = "\n".join(
            cell_source(cell) for cell in cells if cell.get("cell_type") == "markdown"
        )
        self.assertGreaterEqual(markdown.count("Predict before running"), 11)
        for phrase in (
            "predict → act → observe → explain",
            "timestamp",
            "Controlled failure",
            "UNFILLED BY LEARNER",
            "M33 → M34",
            "provenance",
            "abstain",
            "citation",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, markdown)

    def test_notebook_prints_required_rag_evidence(self):
        code = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        for token in (
            "load_canonical_corpus",
            "load_canonical_index",
            "search",
            "as_evidence",
            "pack_context",
            "answer_query",
            "answer_labeled",
            "verify_support",
            "repair_grounding",
            "classify_failure",
            "evaluate_set",
            "retrieve",
            "synthesize",
            "unsupported_citation",
            "retrieval_enabled=False",
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

    def test_rag_pipeline_top_level_imports_are_local_or_stdlib(self):
        source = (MISSION / "rag_pipeline.py").read_text(encoding="utf-8")
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
        self.assertNotIn("openai", top_level_imports)
        self.assertNotIn("sentence_transformers", top_level_imports)
        self.assertNotIn("qdrant_client", top_level_imports)
        self.assertNotIn("optional_live_llm", source)
        self.assertIn("as_evidence", source)
        self.assertIn("search(", source)

    def test_optional_live_llm_is_unavailable_and_unimported(self):
        adapter = (MISSION / "optional_live_llm.py").read_text(encoding="utf-8")
        self.assertIn("OptionalLiveLLMUnavailable", adapter)
        self.assertNotIn("import openai", adapter)
        spec = importlib.util.spec_from_file_location("m34_optional_live_llm", MISSION / "optional_live_llm.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with self.assertRaises(module.OptionalLiveLLMUnavailable):
            module.open_optional_live_synthesizer()

    def test_learner_facing_contracts_remain_unfilled(self):
        adr = (MISSION / "adr_prompt.md").read_text(encoding="utf-8")
        no_ai = (MISSION / "no_ai_gate.md").read_text(encoding="utf-8")
        status = (MISSION / "status.yaml").read_text(encoding="utf-8")
        self.assertIn("[UNFILLED BY LEARNER]", adr)
        self.assertIn("Leave all learner responses unfilled", no_ai)
        self.assertIn("intentionally_unpopulated", status)
        self.assertIn("does not mark M34 repository-executable", status)
        notebook_markdown = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "markdown"
        )
        self.assertIn("[UNFILLED BY LEARNER]", notebook_markdown)
        self.assertNotIn("[FILLED", notebook_markdown)

    def test_fixture_is_offline_versioned_and_split(self):
        self.assertEqual(QUESTIONS["eval_version"], "m34.eval.v1")
        self.assertEqual(QUESTIONS["index_id"], "v08-exact-memory")
        self.assertEqual(QUESTIONS["corpus_version"], "m33.corpus.v1")
        self.assertFalse(QUESTIONS["downloaded"])
        ids = [row["id"] for row in QUESTIONS["questions"]]
        self.assertEqual(len(ids), len(set(ids)))
        splits = {row["id"]: row["split"] for row in QUESTIONS["questions"]}
        self.assertEqual(splits["rag-reset-login"], "dev")
        self.assertEqual(splits["rag-h-weather"], "holdout")
        self.assertIn("rag-ceo", ids)
        self.assertIn("rag-ticket-4412", ids)
        holdout = [row for row in QUESTIONS["questions"] if row["split"] == "holdout"]
        self.assertGreaterEqual(len(holdout), 4)
        unanswerable = [row for row in QUESTIONS["questions"] if not row["answerable"]]
        self.assertGreaterEqual(len(unanswerable), 3)
        note = QUESTIONS["note"].lower()
        self.assertIn("not", note)
        self.assertIn("holdout", note)
        self.assertEqual(TRANSFER["budget"]["max_chars"], 70)
        self.assertEqual(EXPECTED["source_hash"], "f3ac71c4e290010611c786da3227af664b2f29923b6775e6a8b1b053e3aa75df")
        self.assertEqual(EXPECTED["holdout"]["n_pass"], 6)

    def test_pack_and_support_do_not_need_numpy(self):
        rows = []
        for offset, chunk in enumerate(TRANSFER["chunks"], start=1):
            rows.append(
                {
                    "rank": chunk["rank"],
                    "score": chunk["score"],
                    "document_id": chunk["document_id"],
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "span": chunk["span"],
                    "metadata": {},
                    "m28_id": "",
                    "index_id": "toy",
                    "source_hash": "abc",
                    "corpus_version": "toy",
                    "model": "toy",
                    "version": "1",
                    "metric": "cosine",
                    "normalization": "l2",
                }
            )
        pack = CORE.pack_context(
            rows,
            query_id="t-library-close",
            query_text=TRANSFER["query"]["text"],
            budget=CORE.ContextBudget(max_chars=70, max_chunks=2),
            retrieval_top_k=3,
        )
        kept, dropped = independent_pack_ids(rows, max_chars=70, max_chunks=2)
        self.assertEqual(list(pack.chunk_ids()), kept)
        self.assertEqual(list(pack.dropped_ids()), dropped)
        self.assertEqual(kept, ["lib::c0", "lib::c1"])
        self.assertEqual(dropped, ["mus::c0"])
        self.assertEqual(pack.char_count(), 70)
        self.assertTrue(
            independent_support(
                TRANSFER["supported_claim"]["answer"],
                rows[0]["text"],
            )
        )
        self.assertFalse(
            independent_support(
                TRANSFER["unsupported_fluent"]["answer"],
                rows[0]["text"],
            )
        )
        answer = CORE.synthesize(TRANSFER["query"]["text"], pack, policy=CORE.POLICY_GATED)
        support = CORE.verify_support(answer, pack)
        self.assertTrue(answer.answered)
        self.assertEqual(answer.citation_ids(), ("lib::c0",))
        self.assertTrue(support.ok)

    def test_transfer_failure_traces_are_separable(self):
        labeled = CORE.LabeledRagQuery(
            query_id="t-library-close",
            text=TRANSFER["query"]["text"],
            experiment="transfer",
            split="transfer",
            answerable=True,
            gold_answer=TRANSFER["gold"]["answer"],
            gold_aliases=(),
            support_chunk_ids=tuple(TRANSFER["gold"]["support"]),
            relevant_chunk_ids=tuple(TRANSFER["gold"]["support"]),
            traps=(),
        )

        def pack_from_ids(chunk_ids: list[str], dropped_ids: list[str] | None = None):
            by_id = {chunk["chunk_id"]: chunk for chunk in TRANSFER["chunks"]}
            evidence = []
            for rank, chunk_id in enumerate(chunk_ids, start=1):
                chunk = by_id[chunk_id]
                evidence.append(
                    {
                        "rank": rank,
                        "score": chunk["score"],
                        "document_id": chunk["document_id"],
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk["text"],
                        "span": chunk["span"],
                        "metadata": {},
                        "m28_id": "",
                        "index_id": "toy",
                        "source_hash": "abc",
                        "corpus_version": "toy",
                        "model": "toy",
                        "version": "1",
                        "metric": "cosine",
                        "normalization": "l2",
                    }
                )
            pack = CORE.pack_context(
                evidence,
                query_id=labeled.query_id,
                query_text=labeled.text,
                budget=CORE.ContextBudget(max_chars=400, max_chunks=8),
            )
            if dropped_ids:
                dropped_items = []
                for offset, chunk_id in enumerate(dropped_ids):
                    chunk = by_id[chunk_id]
                    dropped_items.append(
                        CORE.packed_from_evidence(
                            {
                                "rank": 90 + offset,
                                "score": 0.0,
                                "document_id": chunk["document_id"],
                                "chunk_id": chunk["chunk_id"],
                                "text": chunk["text"],
                                "span": chunk["span"],
                                "metadata": {},
                                "m28_id": "",
                                "index_id": "toy",
                                "source_hash": "abc",
                                "corpus_version": "toy",
                                "model": "toy",
                                "version": "1",
                                "metric": "cosine",
                                "normalization": "l2",
                            },
                            packed_offset=len(pack.items) + offset,
                        )
                    )
                pack = CORE.ContextPack(
                    query_id=pack.query_id,
                    query_text=pack.query_text,
                    budget=pack.budget,
                    items=pack.items,
                    dropped=tuple(dropped_items),
                    retrieval_top_k=pack.retrieval_top_k,
                    index_id=pack.index_id,
                    source_hash=pack.source_hash,
                    corpus_version=pack.corpus_version,
                    model=pack.model,
                    version=pack.version,
                    metric=pack.metric,
                    normalization=pack.normalization,
                    scored_candidates=pack.scored_candidates,
                    formatted=pack.formatted,
                    pack_hash=pack.pack_hash,
                )
            return pack

        def trace_for(spec: dict, *, retrieval_ids: list[str], pack, answer: CORE.GroundedAnswer):
            support = CORE.verify_support(answer, pack)
            answer = CORE.GroundedAnswer(
                query_id=answer.query_id,
                query_text=answer.query_text,
                status=answer.status,
                text=answer.text,
                claims=answer.claims,
                citations=answer.citations,
                abstain_reason=answer.abstain_reason,
                support=support,
                policy=answer.policy,
                defect=answer.defect,
            )
            inference = CORE.InferenceConfig()
            return CORE.RagTrace(
                query_id=labeled.query_id,
                query_text=labeled.text,
                inference=inference,
                retrieval_ids=tuple(retrieval_ids),
                retrieval_scores=tuple(0.0 for _ in retrieval_ids),
                scored_candidates=len(retrieval_ids),
                latency_ms=0.0,
                index_id="toy",
                source_hash="abc",
                corpus_version="toy",
                model="toy",
                version="1",
                metric="cosine",
                normalization="l2",
                pack=pack,
                answer=answer,
            )

        expected_primary = {
            "t-retrieval": "retrieval",
            "t-context": "context",
            "t-generation": "generation",
            "t-citation": "citation",
        }
        by_id = {chunk["chunk_id"]: chunk for chunk in TRANSFER["chunks"]}
        for spec in TRANSFER["traces"]:
            pack = pack_from_ids(list(spec["packed"]), list(spec.get("dropped") or []))
            if spec["status"] == "abstained":
                answer = CORE.abstain(
                    labeled.query_id,
                    labeled.text,
                    reason="insufficient_overlap",
                    policy=CORE.POLICY_GATED,
                )
            else:
                cited = by_id[spec["citations"][0]]
                cited_item = pack.get(cited["chunk_id"]) if cited["chunk_id"] in pack.chunk_ids() else CORE.packed_from_evidence(
                    {
                        "rank": 1,
                        "score": cited["score"],
                        "document_id": cited["document_id"],
                        "chunk_id": cited["chunk_id"],
                        "text": cited["text"],
                        "span": cited["span"],
                        "metadata": {},
                        "m28_id": "",
                        "index_id": "toy",
                        "source_hash": "abc",
                        "corpus_version": "toy",
                        "model": "toy",
                        "version": "1",
                        "metric": "cosine",
                        "normalization": "l2",
                    },
                    packed_offset=0,
                )
                source_item = cited_item
                answer = CORE._answered_from_item(
                    labeled.query_id,
                    labeled.text,
                    source_item,
                    policy=CORE.POLICY_GATED,
                    defect=CORE.DEFECT_NONE,
                    citation=cited_item,
                    claim=spec["answer"],
                )
            trace = trace_for(spec, retrieval_ids=list(spec["retrieved"]), pack=pack, answer=answer)
            classified = CORE.classify_failure(trace, labeled)
            with self.subTest(trace=spec["id"]):
                self.assertEqual(classified["primary"], expected_primary[spec["id"]])


@unittest.skipUnless(RUNTIME_DEPS, "install requirements/m34.txt to run NumPy-dependent M34 tests")
class M34RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = CORE.load_canonical_corpus()
        cls.index = CORE.load_canonical_index()
        cls.queries = CORE.load_query_map()
        cls.expected = CORE.load_expected_payload()

    def test_m33_identity_survives_into_pack_and_citations(self):
        self.assertEqual(self.index.metadata.index_id, "v08-exact-memory")
        self.assertEqual(self.index.metadata.embedding.version, "v06.1")
        self.assertEqual(self.index.metadata.source_hash, self.expected["source_hash"])
        response = CORE.retrieve(
            self.queries["rag-reset-login"].text,
            query_id="rag-reset-login",
            top_k=3,
            index=self.index,
            corpus=self.corpus,
        )
        hit = response.hits[0]
        evidence = hit.as_evidence()
        self.assertEqual(evidence["index_id"], response.index_id)
        self.assertEqual(evidence["source_hash"], response.source_hash)
        self.assertEqual(evidence["model"], response.embedding.model)
        self.assertEqual(evidence["version"], response.embedding.version)
        pack = CORE.pack_context(
            response.evidence(),
            query_id="rag-reset-login",
            query_text=self.queries["rag-reset-login"].text,
            retrieval_top_k=3,
            scored_candidates=response.scored_candidates,
        )
        self.assertEqual(pack.items[0].index_id, evidence["index_id"])
        self.assertEqual(pack.items[0].source_hash, evidence["source_hash"])
        self.assertEqual(pack.items[0].chunk_id, hit.chunk_id)
        trace = CORE.answer_labeled("rag-reset-login", index=self.index, corpus=self.corpus)
        self.assertEqual(trace.answer.citations[0].index_id, trace.index_id)
        self.assertEqual(trace.answer.citations[0].source_hash, trace.source_hash)
        self.assertFalse(trace.inference.weights_updated)
        self.assertEqual(trace.inference.decoding, "extractive-copy")

    def test_useful_whole_is_grounded_reset(self):
        trace = CORE.answer_labeled("rag-reset-login", index=self.index, corpus=self.corpus)
        expected = self.expected["default"]["rag-reset-login"]
        self.assertEqual(list(trace.retrieval_ids), expected["retrieval_ids"])
        self.assertEqual(trace.answer.status, "answered")
        self.assertEqual(trace.answer.text, "Please reset the login credentials.")
        self.assertEqual(trace.answer.citation_ids(), ("doc-account-access::c1",))
        self.assertTrue(trace.answer.support.ok)
        self.assertTrue(trace.evaluation["eval_pass"])
        self.assertTrue(trace.evaluation["grounded"])

    def test_with_versus_without_retrieval_keeps_policy_fixed(self):
        on = CORE.answer_labeled("rag-legal-forbid", retrieval_enabled=True, index=self.index, corpus=self.corpus)
        off = CORE.answer_labeled("rag-legal-forbid", retrieval_enabled=False, index=self.index, corpus=self.corpus)
        self.assertEqual(on.inference.policy, off.inference.policy)
        self.assertTrue(on.answer.answered)
        self.assertEqual(on.answer.citation_ids(), ("doc-legal::c0",))
        self.assertTrue(off.answer.abstained)
        self.assertEqual(off.retrieval_ids, ())
        self.assertEqual(off.pack.chunk_ids(), ())
        self.assertEqual(off.evaluation["primary"], "retrieval")
        self.assertEqual(on.inference.synthesizer_id, off.inference.synthesizer_id)

    def test_top_k_and_budget_separate_retrieval_from_context(self):
        k1 = CORE.answer_labeled("rag-password-procedure", top_k=1, index=self.index, corpus=self.corpus)
        k3 = CORE.answer_labeled("rag-password-procedure", top_k=3, index=self.index, corpus=self.corpus)
        budget = CORE.answer_labeled(
            "rag-password-procedure",
            top_k=3,
            budget_chars=80,
            index=self.index,
            corpus=self.corpus,
        )
        self.assertEqual(k1.evaluation["primary"], "retrieval")
        self.assertTrue(k1.answer.abstained)
        self.assertEqual(list(k1.pack.chunk_ids()), ["doc-account-access::c0"])
        self.assertTrue(k3.evaluation["eval_pass"])
        self.assertEqual(k3.answer.citation_ids(), ("doc-account-access::c1",))
        self.assertTrue(budget.evaluation["retrieval_hit"])
        self.assertFalse(budget.evaluation["packed_hit"])
        self.assertEqual(budget.evaluation["primary"], "context")
        self.assertEqual(list(budget.pack.dropped_ids()), ["doc-account-access::c1"])
        self.assertEqual(list(k3.retrieval_ids), list(budget.retrieval_ids))

    def test_unanswerable_abstains_despite_high_score(self):
        ceo = CORE.answer_labeled("rag-ceo", index=self.index, corpus=self.corpus)
        naive = CORE.answer_labeled("rag-ceo", policy=CORE.POLICY_NAIVE, index=self.index, corpus=self.corpus)
        order = CORE.answer_labeled("rag-order-7", index=self.index, corpus=self.corpus)
        self.assertGreater(ceo.retrieval_scores[0], 0.99)
        self.assertEqual(ceo.retrieval_ids[0], "doc-weather::c0")
        self.assertTrue(ceo.answer.abstained)
        self.assertTrue(ceo.evaluation["eval_pass"])
        self.assertTrue(naive.answer.answered)
        self.assertIn("Rain is expected", naive.answer.text)
        self.assertEqual(naive.evaluation["primary"], "generation")
        self.assertTrue(order.answer.abstained)
        self.assertGreater(order.retrieval_scores[0], 0.9)

    def test_retrieval_miss_is_classified_before_generation(self):
        miss = CORE.answer_labeled("rag-ticket-4412", top_k=1, index=self.index, corpus=self.corpus)
        naive = CORE.answer_labeled(
            "rag-ticket-4412",
            top_k=1,
            policy=CORE.POLICY_NAIVE,
            index=self.index,
            corpus=self.corpus,
        )
        recovered = CORE.answer_labeled("rag-ticket-4412", top_k=3, index=self.index, corpus=self.corpus)
        self.assertEqual(miss.retrieval_ids, ("doc-tickets::c1",))
        self.assertTrue(miss.answer.abstained)
        self.assertEqual(miss.evaluation["primary"], "retrieval")
        self.assertEqual(naive.answer.text, "Ticket 4413 is waiting for inspection.")
        self.assertEqual(naive.evaluation["primary"], "retrieval")
        self.assertTrue(naive.answer.support.ok)
        self.assertFalse(naive.evaluation["eval_pass"])
        self.assertEqual(recovered.answer.citation_ids(), ("doc-tickets::c0",))
        self.assertTrue(recovered.evaluation["eval_pass"])

    def test_unsupported_citation_fails_then_repairs_from_broken_objects(self):
        broken = CORE.answer_labeled(
            "rag-reset-login",
            defect=CORE.DEFECT_UNSUPPORTED,
            index=self.index,
            corpus=self.corpus,
        )
        self.assertEqual(broken.answer.text, "Please reset the login credentials.")
        self.assertEqual(broken.answer.citation_ids(), ("doc-account-access::c0",))
        self.assertFalse(broken.answer.support.ok)
        self.assertFalse(broken.evaluation["eval_pass"])
        self.assertEqual(broken.evaluation["primary"], "citation")
        self.assertTrue(broken.evaluation["retrieval_hit"])
        oracle = independent_support(broken.answer.text, broken.pack.get("doc-account-access::c0").text)
        self.assertFalse(oracle)
        repaired = CORE.repair_grounding(broken.answer, broken.pack)
        self.assertTrue(repaired.answered)
        self.assertEqual(repaired.citation_ids(), ("doc-account-access::c1",))
        self.assertTrue(repaired.support.ok)
        still_broken = CORE.verify_support(broken.answer, broken.pack)
        self.assertFalse(still_broken.ok)

    def test_invented_support_on_a_miss_fails_support_and_abstains_on_repair(self):
        invented = CORE.answer_labeled(
            "rag-ticket-4412",
            top_k=1,
            defect=CORE.DEFECT_INVENTED,
            index=self.index,
            corpus=self.corpus,
        )
        classified = CORE.classify_failure(invented, self.queries["rag-ticket-4412"])
        self.assertEqual(classified["primary"], "retrieval")
        self.assertIn("citation", classified["layers"])
        self.assertTrue(classified["answer_correct"])
        self.assertFalse(invented.answer.support.ok)
        repaired = CORE.repair_grounding(invented.answer, invented.pack)
        self.assertTrue(repaired.abstained)
        self.assertEqual(repaired.abstain_reason, "unsupported_claim")

    def test_held_out_eval_is_frozen_and_stable(self):
        report = CORE.evaluate_set(
            split="holdout",
            index=self.index,
            corpus=self.corpus,
        )
        self.assertEqual(report["n"], 6)
        self.assertEqual(report["n_pass"], 6)
        self.assertEqual(report["n_unsupported_citation"], 0)
        self.assertTrue(report["held_out_untuned"])
        holdout_ids = [row["query_id"] for row in report["rows"]]
        self.assertEqual(holdout_ids, self.expected["holdout_ids"])
        self.assertIn("rag-h-email", holdout_ids)
        self.assertNotIn("rag-reset-login", holdout_ids)

    def test_closed_book_guess_is_not_canonical(self):
        guess = CORE.answer_labeled(
            "rag-legal-forbid",
            retrieval_enabled=False,
            defect=CORE.DEFECT_CLOSED,
            index=self.index,
            corpus=self.corpus,
        )
        self.assertTrue(guess.answer.answered)
        self.assertFalse(guess.answer.support.ok)
        self.assertIn("citation_not_in_pack", {issue.kind for issue in guess.answer.support.issues})

    def test_default_dev_set_matches_frozen_expected(self):
        for query_id, expected in self.expected["default"].items():
            trace = CORE.answer_labeled(query_id, index=self.index, corpus=self.corpus)
            self.assertEqual(list(trace.retrieval_ids), expected["retrieval_ids"], query_id)
            self.assertEqual(trace.answer.status, expected["status"], query_id)
            self.assertEqual(list(trace.answer.citation_ids()), expected["citations"], query_id)
            self.assertEqual(trace.answer.support.ok, expected["support_ok"], query_id)
            self.assertEqual(bool(trace.evaluation["eval_pass"]), expected["eval_pass"], query_id)


if __name__ == "__main__":
    unittest.main()
