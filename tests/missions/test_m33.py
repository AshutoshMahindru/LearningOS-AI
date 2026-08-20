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

MISSION = ROOT / "missions" / "M33"
NOTEBOOK = ROOT / "labs" / "M33_semantic_search.ipynb"
REQUIREMENTS = ROOT / "requirements" / "m33.txt"
DATASETS = ROOT / "datasets" / "M33"


def load_core():
    spec = importlib.util.spec_from_file_location("m33_semantic_search", MISSION / "semantic_search.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load M33 semantic search core")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_core()
RUNTIME_DEPS = importlib.util.find_spec("numpy") is not None
CORPUS = json.loads((DATASETS / "corpus.json").read_text(encoding="utf-8"))
VECTORS = json.loads((DATASETS / "vectors.json").read_text(encoding="utf-8"))
QUERIES = json.loads((DATASETS / "queries.json").read_text(encoding="utf-8"))
EXPECTED = json.loads((DATASETS / "expected.json").read_text(encoding="utf-8"))
INCOMPATIBLE = json.loads((DATASETS / "incompatible_vectors.json").read_text(encoding="utf-8"))
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


def independent_brute_force(
    query_vector,
    records,
    *,
    top_k: int,
    filters: dict[str, str] | None = None,
):
    """Second cosine ranking path used only as a parity oracle."""

    import numpy as np

    query = np.asarray(query_vector, dtype=float).reshape(-1)
    query_norm = float(np.linalg.norm(query))
    if query_norm == 0.0:
        raise ValueError("query vector must be nonzero")
    query = query / query_norm
    rows = []
    for record in records:
        meta = dict(record.chunk.metadata)
        if filters and any(meta.get(key) != value for key, value in filters.items()):
            continue
        vector = np.asarray(record.vector, dtype=float).reshape(-1)
        vector_norm = float(np.linalg.norm(vector))
        unit = vector / vector_norm
        score = float(np.clip(float(np.dot(query, unit)), -1.0, 1.0))
        rows.append((record.chunk.chunk_id, score, record))
    rows.sort(key=lambda item: (-item[1], item[0]))
    return rows[: int(top_k)]


class M33StaticContractTests(unittest.TestCase):
    def test_required_executable_artifacts_exist(self):
        for path in (
            MISSION / "semantic_search.py",
            MISSION / "optional_qdrant.py",
            NOTEBOOK,
            REQUIREMENTS,
            ROOT / "tests" / "test_m33.py",
            DATASETS / "corpus.json",
            DATASETS / "vectors.json",
            DATASETS / "queries.json",
            DATASETS / "expected.json",
            DATASETS / "incompatible_vectors.json",
            DATASETS / "transfer.json",
            DATASETS / "README.md",
            DATASETS / "generate_index.py",
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
            "semantic_search.py",
            "optional_qdrant.py",
        }
        actual = {path.name for path in MISSION.iterdir() if path.is_file()}
        self.assertTrue(required <= actual, required - actual)
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        for name in required:
            self.assertIn(f"missions/M33/{name}", manifest)

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
        self.assertIn("M33", source)
        self.assertIn("semantic_search.py", source)
        self.assertIn("sys.path", source)
        self.assertIn("from missions.M33.semantic_search import", source)

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
            "FastAPI",
            "flask",
            "ChatCompletion",
            "generate_answer",
            "cite_sources",
            "abstain",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_notebook_enforces_prediction_before_every_action(self):
        cells = notebook_cells()
        positions = {str(cell["id"]): index for index, cell in enumerate(cells)}
        pairs = (
            ("predict-search", "run-search"),
            ("predict-topk", "run-topk"),
            ("predict-filter", "run-filter"),
            ("predict-baseline", "run-baseline"),
            ("predict-labels", "run-labels"),
            ("predict-hard", "run-hard"),
            ("predict-latency", "run-latency"),
            ("predict-failure", "run-failure"),
            ("predict-failure-repair", "run-failure-repair"),
            ("predict-incompatible", "run-incompatible"),
        )
        for prediction, action in pairs:
            with self.subTest(prediction=prediction, action=action):
                self.assertLess(positions[prediction], positions[action])
                self.assertEqual(cells[positions[prediction]].get("cell_type"), "markdown")
                self.assertIn("Predict before running", cell_source(cells[positions[prediction]]))

        self.assertLess(positions["code-reading"], positions["run-code-reading"])
        self.assertIn("Predict before running", cell_source(cells[positions["code-reading"]]))

        markdown = "\n".join(
            cell_source(cell) for cell in cells if cell.get("cell_type") == "markdown"
        )
        self.assertGreaterEqual(markdown.count("Predict before running"), 10)
        for phrase in (
            "predict → act → observe → explain",
            "timestamp",
            "Controlled failure",
            "UNFILLED BY LEARNER",
            "M28 → M33",
            "cosine",
            "provenance",
            "L2",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, markdown)

    def test_notebook_prints_required_search_evidence(self):
        code = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        for token in (
            "load_canonical_corpus",
            "load_canonical_index",
            "load_incompatible_index",
            "search",
            "search_unchecked",
            "search_labeled",
            "search_report",
            "exact_cosine_rank",
            "evaluate_labeled",
            "replace_chunk_text",
            "rebuild_index",
            "IndexStaleError",
            "IndexIncompatibleError",
            "FilterSchemaError",
            "as_evidence",
            "scored_candidates",
            "enforce_freshness=True",
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

    def test_semantic_search_top_level_imports_are_local_or_stdlib(self):
        source = (MISSION / "semantic_search.py").read_text(encoding="utf-8")
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
            "sys",
            "time",
            "missions.M28",
        )
        for name in top_level_imports:
            with self.subTest(name=name):
                self.assertTrue(
                    name.startswith(allowed_prefixes) or name in allowed_prefixes,
                    name,
                )
        self.assertNotIn("numpy", top_level_imports)
        self.assertNotIn("torch", top_level_imports)
        self.assertNotIn("sentence_transformers", top_level_imports)
        self.assertNotIn("qdrant_client", top_level_imports)
        self.assertNotIn("requests", top_level_imports)
        self.assertIn("_require_numpy", source)
        self.assertNotIn("optional_qdrant", source)

    def test_optional_qdrant_is_unavailable_and_unimported(self):
        adapter = (MISSION / "optional_qdrant.py").read_text(encoding="utf-8")
        self.assertIn("OptionalQdrantUnavailable", adapter)
        self.assertNotIn("import qdrant_client", adapter)
        spec = importlib.util.spec_from_file_location("m33_optional_qdrant", MISSION / "optional_qdrant.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with self.assertRaises(module.OptionalQdrantUnavailable):
            module.open_optional_client()

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

    def test_fixture_is_offline_versioned_and_copied_from_m28(self):
        provenance = VECTORS["provenance"]
        self.assertEqual(provenance["family"], "v06-teaching-embed")
        self.assertEqual(provenance["model"], "v06-teaching-meanpool")
        self.assertEqual(provenance["version"], "v06.1")
        self.assertEqual(provenance["metric"], "cosine")
        self.assertEqual(provenance["normalization"], "l2")
        self.assertEqual(provenance["pooling"], "mean")
        self.assertEqual(provenance["dimensions"], 12)
        self.assertFalse(provenance["downloaded"])
        self.assertFalse(provenance["network_required"])
        self.assertTrue(provenance["not_sentence_transformers"])
        self.assertEqual(provenance["copied_from"], "datasets/M28/embeddings.json")
        self.assertFalse(CORPUS["downloaded"])
        self.assertFalse(QUERIES["downloaded"])
        self.assertFalse(TRANSFER["downloaded"])
        self.assertEqual(TRANSFER["provenance"]["dimensions"], 3)
        mismatch_prov = INCOMPATIBLE["provenance"]
        self.assertEqual(mismatch_prov["version"], "v06.2")
        self.assertEqual(mismatch_prov["model"], "v06-teaching-meanpool-alt")
        self.assertEqual(mismatch_prov["normalization"], "none")
        query_ids = {row["id"] for row in QUERIES["queries"]}
        self.assertIn("q-password", query_ids)
        self.assertIn("q-negation", query_ids)
        chunk_ids = {
            chunk["chunk_id"]
            for document in CORPUS["documents"]
            for chunk in document["chunks"]
        }
        self.assertIn("doc-account-access::c0", chunk_ids)
        self.assertEqual(len(chunk_ids), 14)
        paraphrase = next(row for row in QUERIES["queries"] if row["id"] == "q-paraphrase")
        corpus_texts = {
            chunk["text"]
            for document in CORPUS["documents"]
            for chunk in document["chunks"]
        }
        self.assertNotIn(paraphrase["text"], corpus_texts)
        labels_note = QUERIES["note"].lower()
        self.assertIn("not cosine", labels_note)


@unittest.skipUnless(RUNTIME_DEPS, "install requirements/m33.txt to run NumPy-dependent M33 tests")
class M33RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = CORE.load_canonical_corpus()
        cls.index = CORE.load_canonical_index()
        cls.queries = CORE.load_query_map()
        cls.expected = CORE.load_expected_payload()
        cls.incompatible = CORE.load_incompatible_index()
        cls.encoder = CORE.load_encoder()
        cls.transfer_corpus, cls.transfer_index, cls.transfer_queries, cls.transfer_payload = (
            CORE.load_transfer_bundle()
        )

    def test_index_identity_and_spans_recover_text(self):
        self.assertEqual(self.index.metadata.backend, "exact-in-memory")
        self.assertEqual(self.index.metadata.embedding.version, "v06.1")
        self.assertFalse(self.index.metadata.downloaded)
        self.assertEqual(self.index.metadata.chunk_count, 14)
        self.assertEqual(self.index.metadata.document_count, 7)
        self.assertEqual(self.index.metadata.source_hash, self.expected["source_hash"])
        for document in self.corpus.documents:
            for chunk in document.chunks:
                self.assertEqual(document.text[chunk.span_start : chunk.span_end], chunk.text)
                self.assertAlmostEqual(CORE.l2_norm(self.index.get(chunk.chunk_id).vector), 1.0, places=9)

    def test_encoder_matches_frozen_chunk_vectors(self):
        for chunk in self.corpus.chunks():
            rebuilt = self.encoder.encode(chunk.text)
            stored = self.index.get(chunk.chunk_id).vector
            for left, right in zip(rebuilt, stored, strict=True):
                self.assertAlmostEqual(left, right, places=8)

    def test_useful_whole_and_paraphrase_keep_account_neighbors(self):
        password = CORE.search_labeled("q-password", top_k=5)
        paraphrase = CORE.search_labeled("q-paraphrase", top_k=5)
        self.assertEqual(list(password.ids()[:3]), self.expected["q-password_top3"])
        self.assertEqual(list(paraphrase.ids()[:3]), self.expected["q-paraphrase_top3"])
        account = {"doc-account-access::c0", "doc-account-access::c1", "doc-account-access::c2"}
        self.assertTrue(set(password.ids()[:3]) <= account)
        self.assertNotIn("doc-device-printer::c0", password.ids()[:3])
        evidence = password.hits[0].as_evidence()
        self.assertEqual(evidence["document_id"], "doc-account-access")
        self.assertEqual(evidence["span"]["start"], 0)
        self.assertIn("I forgot my password", evidence["text"])
        self.assertEqual(evidence["index_id"], password.index_id)
        self.assertEqual(evidence["source_hash"], password.source_hash)
        self.assertEqual(evidence["model"], password.embedding.model)
        self.assertEqual(evidence["version"], password.embedding.version)
        self.assertEqual(evidence["metric"], password.metric)
        self.assertEqual(evidence["normalization"], password.normalization)
        query = CORE.encode_query(self.queries["q-paraphrase"].text, query_id="q-paraphrase")
        for record in self.index.records:
            self.assertLess(CORE.cosine_similarity(query.vector, record.vector), 0.999, record.chunk.chunk_id)

    def test_independent_brute_force_matches_service_ranking(self):
        labeled = self.queries["q-password"]
        encoded = CORE.encode_query(labeled.text, query_id=labeled.query_id)
        service = CORE.search(
            self.index,
            encoded,
            top_k=8,
            live_corpus=self.corpus,
        )
        oracle = independent_brute_force(encoded.vector, self.index.records, top_k=8)
        self.assertEqual(service.ids(), tuple(row[0] for row in oracle))
        for hit, row in zip(service.hits, oracle, strict=True):
            self.assertAlmostEqual(hit.score, row[1], places=9)
        filtered = CORE.search(
            self.index,
            encoded,
            top_k=5,
            filters={"topic": "device"},
            live_corpus=self.corpus,
        )
        oracle_filtered = independent_brute_force(
            encoded.vector,
            self.index.records,
            top_k=5,
            filters={"topic": "device"},
        )
        self.assertEqual(filtered.ids(), tuple(row[0] for row in oracle_filtered))
        self.assertEqual(list(filtered.ids()[:2]), self.expected["filter_topic_device_top2"])

    def test_top_k_changes_window_not_scored_candidates(self):
        encoded = CORE.encode_query(self.queries["q-password"].text, query_id="q-password")
        k1 = CORE.search(self.index, encoded, top_k=1, live_corpus=self.corpus)
        k8 = CORE.search(self.index, encoded, top_k=8, live_corpus=self.corpus)
        self.assertEqual(k1.scored_candidates, k8.scored_candidates)
        self.assertEqual(k1.scored_candidates, 14)
        self.assertEqual(len(k1.hits), 1)
        self.assertEqual(len(k8.hits), 8)
        self.assertEqual(k1.ids()[0], k8.ids()[0])
        eval1 = CORE.evaluate_labeled(k1, self.queries["q-password"], k=1)
        eval8 = CORE.evaluate_labeled(k8, self.queries["q-password"], k=8)
        self.assertTrue(eval1["hit_at_k"])
        self.assertGreater(eval8["recall_at_k"], eval1["recall_at_k"] - 1e-12)
        self.assertEqual(eval8["recall_at_k"], 1.0)

    def test_metadata_filter_changes_eligibility_not_labels(self):
        labeled = self.queries["q-password"]
        encoded = CORE.encode_query(labeled.text, query_id=labeled.query_id)
        open_search = CORE.search(self.index, encoded, top_k=5, live_corpus=self.corpus)
        device = CORE.search(
            self.index,
            encoded,
            top_k=5,
            filters={"topic": "device"},
            live_corpus=self.corpus,
        )
        account = CORE.search(
            self.index,
            encoded,
            top_k=5,
            filters={"topic": "account"},
            live_corpus=self.corpus,
        )
        self.assertEqual(device.scored_candidates, 2)
        self.assertTrue(all(hit.metadata_dict()["topic"] == "device" for hit in device.hits))
        self.assertEqual(set(account.ids()), set(labeled.relevant_chunk_ids))
        self.assertEqual(labeled.relevant_chunk_ids, self.queries["q-password"].relevant_chunk_ids)
        device_eval = CORE.evaluate_labeled(device, labeled, k=2)
        self.assertFalse(device_eval["hit_at_k"])
        self.assertEqual(device_eval["recall_at_k"], 0.0)
        self.assertEqual(set(device_eval["trap_hits"]), set(labeled.traps))
        with self.assertRaises(CORE.FilterSchemaError):
            CORE.search(self.index, encoded, filters={"channel": "email"}, live_corpus=self.corpus)

    def test_labeled_hard_cases_keep_frozen_labels(self):
        negation = CORE.search_labeled("q-negation", top_k=3)
        numeric = CORE.search_labeled("q-numeric", top_k=3)
        entity = CORE.search_labeled("q-entity", top_k=3)
        domain = CORE.search_labeled("q-domain", top_k=3)
        self.assertEqual(list(negation.ids()[:2]), self.expected["q-negation_top2"])
        self.assertGreater(negation.hits[1].score, 0.85)
        neg_eval = CORE.evaluate_labeled(negation, self.queries["q-negation"], k=2)
        self.assertNotIn(self.queries["q-negation"].hard_neighbor, self.queries["q-negation"].relevant_chunk_ids)
        self.assertEqual(negation.hits[1].chunk_id, self.queries["q-negation"].hard_neighbor)
        self.assertLess(neg_eval["precision_at_k"], 1.0)
        self.assertTrue(neg_eval["scores_are_not_labels"])
        self.assertEqual(list(numeric.ids()[:2]), self.expected["q-numeric_top2"])
        self.assertGreater(numeric.hits[1].score, 0.85)
        self.assertEqual(list(entity.ids()[:2]), self.expected["q-entity_top2"])
        self.assertEqual(domain.top_id, "doc-weather::c0")
        self.assertNotIn("doc-legal::c0", domain.ids()[:2])

    def test_ties_break_by_chunk_id(self):
        vector = self.index.get("doc-weather::c0").vector
        twins = (
            CORE.IndexRecord(
                chunk=CORE.Chunk(
                    chunk_id="z-twin",
                    document_id="doc-tie",
                    text="rain later",
                    span_start=0,
                    span_end=10,
                    m28_id="",
                    metadata=(("topic", "weather"), ("source", "x"), ("locale", "en")),
                ),
                vector=vector,
            ),
            CORE.IndexRecord(
                chunk=CORE.Chunk(
                    chunk_id="a-twin",
                    document_id="doc-tie",
                    text="rain earlier",
                    span_start=0,
                    span_end=12,
                    m28_id="",
                    metadata=(("topic", "weather"), ("source", "x"), ("locale", "en")),
                ),
                vector=vector,
            ),
        )
        space = CORE.ExactIndex(records=twins, metadata=self.index.metadata)
        ranked = CORE.search(
            space,
            CORE.QueryEmbedding(
                query_id="tie",
                text="rain",
                vector=vector,
                provenance=self.index.metadata.embedding,
            ),
            top_k=2,
            live_corpus=self.corpus,
            enforce_freshness=False,
        )
        self.assertEqual(ranked.ids(), ("a-twin", "z-twin"))
        self.assertAlmostEqual(ranked.hits[0].score, ranked.hits[1].score)
        with self.assertRaises(ValueError):
            CORE.search(self.index, "rain", top_k=0, live_corpus=self.corpus)

    def test_stale_index_serves_old_text_then_fails_closed_then_rebuilds(self):
        labeled = self.queries["q-password"]
        encoded = CORE.encode_query(labeled.text, query_id=labeled.query_id)
        live = CORE.replace_chunk_text(
            self.corpus,
            "doc-account-access::c0",
            "Please reset the printer firmware.",
        )
        self.assertNotEqual(CORE.source_hash(live), self.index.metadata.source_hash)
        stale = CORE.search_unchecked(
            self.index,
            encoded,
            top_k=3,
            live_corpus=live,
        )
        self.assertFalse(stale.enforced_freshness)
        self.assertEqual(stale.top_id, "doc-account-access::c0")
        self.assertEqual(stale.hits[0].text, "I forgot my password and cannot sign in.")
        self.assertNotEqual(stale.hits[0].text, live.get_chunk("doc-account-access::c0").text)
        with self.assertRaises(CORE.IndexStaleError) as raised:
            CORE.search(self.index, encoded, top_k=3, live_corpus=live, enforce_freshness=True)
        self.assertEqual(raised.exception.index_source_hash, self.index.metadata.source_hash)
        rebuilt = CORE.rebuild_index(self.index, live)
        self.assertEqual(rebuilt.metadata.source_hash, CORE.source_hash(live))
        self.assertNotEqual(rebuilt.metadata.source_hash, self.index.metadata.source_hash)
        repaired = CORE.search(rebuilt, encoded, top_k=3, live_corpus=live)
        self.assertTrue(repaired.enforced_freshness)
        self.assertNotEqual(repaired.top_id, "doc-account-access::c0")
        served = {hit.chunk_id: hit.text for hit in repaired.hits}
        if "doc-account-access::c0" in served:
            self.assertEqual(served["doc-account-access::c0"], "Please reset the printer firmware.")
        still_broken = CORE.search_unchecked(self.index, encoded, top_k=1, live_corpus=live)
        self.assertEqual(still_broken.hits[0].text, "I forgot my password and cannot sign in.")

    def test_incompatible_index_is_plausible_then_rejected(self):
        encoded = CORE.encode_query(self.queries["q-password"].text, query_id="q-password")
        mixed = CORE.search_unchecked(
            self.incompatible,
            encoded,
            top_k=5,
            enforce_provenance=False,
            live_corpus=self.corpus,
        )
        self.assertFalse(mixed.enforced_provenance)
        self.assertEqual(list(mixed.ids()[:3]), self.expected["mixed_password_top3"])
        self.assertTrue(any(item_id.startswith("doc-device-printer") for item_id in mixed.ids()[:2]))
        with self.assertRaises(CORE.IndexIncompatibleError) as raised:
            CORE.search(
                self.incompatible,
                encoded,
                top_k=5,
                live_corpus=self.corpus,
                enforce_provenance=True,
            )
        self.assertIn("version", raised.exception.mismatches)
        self.assertIn("model", raised.exception.mismatches)
        self.assertIn("normalization", raised.exception.mismatches)
        same = CORE.search_labeled("q-password")
        self.assertEqual(same.top_id, "doc-account-access::c0")
        self.assertTrue(same.enforced_provenance)

    def test_transfer_fixture_is_hand_computable_and_stale_probe_differs(self):
        query_row = self.transfer_payload["query"]
        encoded = CORE.QueryEmbedding(
            query_id=str(query_row["id"]),
            text=str(query_row["text"]),
            vector=CORE.as_vector(query_row["vector"], name="transfer query", dimensions=3),
            provenance=self.transfer_index.metadata.embedding,
        )
        ranked = CORE.search(
            self.transfer_index,
            encoded,
            top_k=4,
            live_corpus=self.transfer_corpus,
        )
        self.assertEqual(list(ranked.ids()), self.transfer_payload["expected"]["ranking"])
        self.assertAlmostEqual(ranked.hits[0].score, 1.0)
        self.assertAlmostEqual(ranked.hits[1].score, 0.96)
        labeled = self.transfer_queries[0]
        metrics = CORE.evaluate_labeled(ranked, labeled, k=2)
        self.assertEqual(metrics["recall_at_k"], 0.5)
        self.assertTrue(metrics["hit_at_k"])
        self.assertEqual(ranked.hits[1].chunk_id, "t-not-down")
        self.assertNotIn("t-not-down", labeled.relevant_chunk_ids)
        live = CORE.replace_chunk_text(
            self.transfer_corpus,
            "t-offline",
            self.transfer_payload["stale_probe"]["live_text"],
        )
        stale = CORE.search_unchecked(self.transfer_index, encoded, top_k=1, live_corpus=live)
        self.assertEqual(stale.hits[0].text, "the server is unreachable")
        with self.assertRaises(CORE.IndexStaleError):
            CORE.search(self.transfer_index, encoded, top_k=1, live_corpus=live)

    def test_latency_is_recorded_and_filter_reduces_candidates(self):
        encoded = CORE.encode_query(self.queries["q-password"].text, query_id="q-password")
        open_search = CORE.search(self.index, encoded, top_k=1, live_corpus=self.corpus)
        filtered = CORE.search(
            self.index,
            encoded,
            top_k=1,
            filters={"topic": "legal"},
            live_corpus=self.corpus,
        )
        self.assertGreaterEqual(open_search.latency_ms, 0.0)
        self.assertEqual(open_search.scored_candidates, 14)
        self.assertEqual(filtered.scored_candidates, 1)
        self.assertEqual(filtered.top_id, "doc-legal::c0")


if __name__ == "__main__":
    unittest.main()
