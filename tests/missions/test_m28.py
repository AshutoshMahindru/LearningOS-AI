from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
MISSION = ROOT / "missions" / "M28"
NOTEBOOK = ROOT / "labs" / "M28_embeddings.ipynb"
REQUIREMENTS = ROOT / "requirements" / "m28.txt"
DATASETS = ROOT / "datasets" / "M28"


def load_core():
    spec = importlib.util.spec_from_file_location("m28_embedding_core", MISSION / "embedding_core.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load M28 embedding core")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_core()
RUNTIME_DEPS = importlib.util.find_spec("numpy") is not None
CATALOG = json.loads((DATASETS / "catalog.json").read_text(encoding="utf-8"))
EMBEDDINGS = json.loads((DATASETS / "embeddings.json").read_text(encoding="utf-8"))
MISMATCH = json.loads((DATASETS / "mismatch.json").read_text(encoding="utf-8"))
TOKEN_TABLE = json.loads((DATASETS / "token_table.json").read_text(encoding="utf-8"))
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


class M28StaticContractTests(unittest.TestCase):
    def test_required_executable_artifacts_exist(self):
        for path in (
            MISSION / "embedding_core.py",
            NOTEBOOK,
            REQUIREMENTS,
            ROOT / "tests" / "test_m28.py",
            DATASETS / "catalog.json",
            DATASETS / "embeddings.json",
            DATASETS / "mismatch.json",
            DATASETS / "token_table.json",
            DATASETS / "transfer.json",
            DATASETS / "README.md",
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
            "embedding_core.py",
        }
        actual = {path.name for path in MISSION.iterdir() if path.is_file()}
        self.assertTrue(required <= actual, required - actual)
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        for name in required:
            self.assertIn(f"missions/M28/{name}", manifest)

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
        self.assertIn("M28", source)
        self.assertIn("embedding_core.py", source)
        self.assertIn("sys.path", source)
        self.assertIn("from missions.M28.embedding_core import", source)

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
            "VectorIndex",
            "FastAPI",
            "flask",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_notebook_enforces_prediction_before_every_action(self):
        cells = notebook_cells()
        positions = {str(cell["id"]): index for index, cell in enumerate(cells)}
        pairs = (
            ("predict-rank", "run-rank"),
            ("predict-pairwise", "run-pairwise"),
            ("predict-paraphrase", "run-paraphrase"),
            ("predict-lexical", "run-lexical"),
            ("predict-hard", "run-hard"),
            ("predict-domain", "run-domain"),
            ("predict-normalization", "run-normalization"),
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
            "M15 → M27 → M28",
            "cosine",
            "provenance",
            "L2",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, markdown)

    def test_notebook_prints_required_embedding_evidence(self):
        code = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        for token in (
            "load_canonical_space",
            "load_mismatch_space",
            "load_encoder",
            "retrieve_query",
            "retrieve_unchecked",
            "rank_neighbors",
            "cosine_similarity",
            "inner_product",
            "lexical_overlap",
            "compare_lexical_and_semantic",
            "pairwise_cosine",
            "compatibility_report",
            "ProvenanceError",
            'pooling="sum"',
            'normalization="none"',
            "encode_report",
            "retrieval_report",
            'enforce_provenance=True',
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
        self.assertNotIn("torch", joined)
        self.assertNotIn("sentence-transformers", joined)
        self.assertNotIn("transformers", joined)
        self.assertNotIn("faiss", joined)

    def test_embedding_core_top_level_imports_are_stdlib(self):
        source = (MISSION / "embedding_core.py").read_text(encoding="utf-8")
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
        allowed = {
            "__future__",
            "collections",
            "collections.abc",
            "dataclasses",
            "functools",
            "pathlib",
            "hashlib",
            "json",
            "re",
        }
        for name in top_level_imports:
            with self.subTest(name=name):
                self.assertIn(name.split(".")[0] if name != "collections.abc" else name, allowed | {"collections"})
        self.assertNotIn("numpy", top_level_imports)
        self.assertNotIn("torch", top_level_imports)
        self.assertNotIn("sentence_transformers", top_level_imports)
        self.assertNotIn("requests", top_level_imports)
        self.assertIn("_require_numpy", source)

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

    def test_fixture_is_offline_and_versioned(self):
        provenance = EMBEDDINGS["provenance"]
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
        self.assertTrue(provenance["not_model_hub"])
        mismatch_prov = MISMATCH["provenance"]
        self.assertEqual(mismatch_prov["version"], "v06.2")
        self.assertEqual(mismatch_prov["model"], "v06-teaching-meanpool-alt")
        self.assertEqual(mismatch_prov["normalization"], "none")
        self.assertFalse(mismatch_prov["downloaded"])
        self.assertFalse(TOKEN_TABLE["provenance"]["downloaded"])
        self.assertFalse(CATALOG["provenance"]["downloaded"])
        self.assertFalse(TRANSFER["provenance"]["downloaded"])
        self.assertEqual(TRANSFER["provenance"]["dimensions"], 4)
        self.assertIn("d-password-forgot", {row["id"] for row in CATALOG["corpus"]})
        self.assertIn("q-paraphrase", {row["id"] for row in CATALOG["queries"]})
        paraphrase_text = next(row["text"] for row in CATALOG["queries"] if row["id"] == "q-paraphrase")
        corpus_texts = {row["text"] for row in CATALOG["corpus"]}
        self.assertNotIn(paraphrase_text, corpus_texts)
        self.assertEqual(len(TOKEN_TABLE["token_semantics"]), len(set(TOKEN_TABLE["token_semantics"])))


@unittest.skipUnless(RUNTIME_DEPS, "install requirements/m28.txt to run NumPy-dependent M28 tests")
class M28RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.canonical = CORE.load_canonical_space()
        cls.mismatch = CORE.load_mismatch_space()
        cls.encoder = CORE.load_encoder()
        cls.transfer = CORE.load_transfer_space()
        cls.catalog = CORE.load_catalog()

    def test_canonical_vectors_are_unit_width_twelve_and_offline(self):
        self.assertEqual(self.canonical.provenance.dimensions, 12)
        self.assertFalse(self.canonical.provenance.downloaded)
        self.assertFalse(self.canonical.provenance.network_required)
        self.assertEqual(len(self.canonical.documents()), 14)
        for item in self.canonical.items:
            self.assertEqual(len(item.vector), 12)
            self.assertAlmostEqual(CORE.l2_norm(item.vector), 1.0, places=9)

    def test_encoder_matches_frozen_vectors(self):
        for item in self.canonical.items:
            rebuilt = self.encoder.encode(item.text)
            for left, right in zip(item.vector, rebuilt, strict=True):
                self.assertAlmostEqual(left, right, places=8)

    def test_useful_whole_and_paraphrase_keep_account_neighbors(self):
        password = CORE.retrieve_query("q-password")
        paraphrase = CORE.retrieve_query("q-paraphrase")
        expected = self.catalog["expected"]
        self.assertEqual(list(password.ids()[:3]), expected["q-password_top3"])
        self.assertEqual(list(paraphrase.ids()[:3]), expected["q-paraphrase_top3"])
        account = {"d-password-forgot", "d-login-reset", "d-cannot-signin"}
        self.assertTrue(set(password.ids()[:3]) <= account)
        self.assertTrue(set(paraphrase.ids()[:3]) <= account)
        self.assertNotIn("d-printer-reset", password.ids()[:3])
        self.assertNotIn("d-printer-reset", paraphrase.ids()[:3])
        self.assertGreater(password.results[0].score, password.results[3].score)
        query = self.canonical.get("q-paraphrase")
        for doc in self.canonical.documents():
            self.assertLess(CORE.cosine_similarity(query.vector, doc.vector), 0.999, doc.id)

    def test_lexical_and_semantic_rankings_disagree_on_printer_login(self):
        query = self.canonical.get("q-printer")
        rows = CORE.compare_lexical_and_semantic(
            query.text,
            self.canonical,
            query_vector=query.vector,
            query_id=query.id,
            query_provenance=self.canonical.provenance,
        )
        by_id = {row["id"]: row for row in rows}
        self.assertEqual(by_id["d-printer-reset"]["semantic_rank"], 1)
        self.assertEqual(list(CORE.retrieve_query("q-printer").ids()[:2]), self.catalog["expected"]["q-printer_top2"])
        login = by_id["d-login-reset"]
        printer = by_id["d-printer-reset"]
        self.assertGreaterEqual(login["lexical_overlap"], 0.5)
        self.assertLess(printer["semantic_rank"], login["semantic_rank"])
        self.assertTrue(login["disagrees"])
        low_query = self.canonical.get("q-low-overlap")
        login_text = self.canonical.get("d-login-reset").text
        self.assertLess(CORE.lexical_overlap(low_query.text, login_text), 0.25)
        self.assertGreater(CORE.cosine_similarity(low_query.vector, self.canonical.get("d-login-reset").vector), 0.7)
        self.assertEqual(list(CORE.retrieve_query("q-low-overlap").ids()[:3]), self.catalog["expected"]["q-low-overlap_top3"])

    def test_hard_cases_are_close_without_being_oracles(self):
        negation = CORE.retrieve_query("q-negation")
        numeric = CORE.retrieve_query("q-numeric")
        entity = CORE.retrieve_query("q-entity")
        domain = CORE.retrieve_query("q-domain")
        self.assertEqual(list(negation.ids()[:2]), self.catalog["expected"]["q-negation_top2"])
        self.assertGreater(negation.results[1].score, 0.85)
        self.assertEqual(list(numeric.ids()[:2]), self.catalog["expected"]["q-numeric_top2"])
        self.assertGreater(numeric.results[1].score, 0.85)
        self.assertEqual(list(entity.ids()[:2]), self.catalog["expected"]["q-entity_top2"])
        self.assertLess(entity.margin(), 0.12)
        self.assertEqual(domain.top_id, "d-rain")
        self.assertNotIn("d-legal", domain.ids()[:2])

    def test_pairwise_cosine_is_symmetric_with_unit_diagonal(self):
        ids = self.catalog["pairwise_subset"]
        items = [self.canonical.get(item_id) for item_id in ids]
        matrix, ordered = CORE.pairwise_cosine(items)
        self.assertEqual(ordered, tuple(ids))
        self.assertEqual(matrix.shape, (len(ids), len(ids)))
        for i in range(len(ids)):
            self.assertAlmostEqual(float(matrix[i, i]), 1.0, places=6)
            for j in range(len(ids)):
                self.assertAlmostEqual(float(matrix[i, j]), float(matrix[j, i]), places=9)
        approve = self.canonical.get("d-approve-refund")
        deny = self.canonical.get("d-deny-refund")
        self.assertGreater(CORE.cosine_similarity(approve.vector, deny.vector), 0.85)

    def test_declared_cosine_matches_raw_mean_cosine_and_disagrees_with_sum_dot(self):
        query = self.canonical.get("q-printer")
        raw_query = self.encoder.encode(query.text, pooling="mean", normalization="none")
        cosine_rank = CORE.retrieve_query("q-printer")
        ip_rows = []
        for doc in self.canonical.documents():
            raw = self.encoder.encode(doc.text, pooling="mean", normalization="none")
            summed = self.encoder.encode(doc.text, pooling="sum", normalization="none")
            self.assertAlmostEqual(
                CORE.cosine_similarity(query.vector, doc.vector),
                CORE.cosine_similarity(raw_query, raw),
                places=9,
            )
            ip_rows.append((doc.id, CORE.inner_product(query.vector, summed)))
        ip_rows.sort(key=lambda row: (-row[1], row[0]))
        self.assertEqual(cosine_rank.top_id, "d-printer-reset")
        self.assertEqual(ip_rows[0][0], "d-printer-queue")
        self.assertNotEqual(cosine_rank.top_id, ip_rows[0][0])

    def test_l2_dot_equals_cosine_on_unit_store(self):
        left = self.canonical.get("d-password-forgot").vector
        right = self.canonical.get("d-login-reset").vector
        self.assertAlmostEqual(CORE.cosine_similarity(left, right), CORE.inner_product(left, right), places=9)
        declared = CORE.operational_score(left, right, metric="cosine", normalization="l2")
        self.assertAlmostEqual(declared, CORE.cosine_similarity(left, right), places=9)
        raw_left = self.encoder.encode("reset my password", pooling="mean", normalization="none")
        raw_right = self.encoder.encode("Please reset the printer.", pooling="mean", normalization="none")
        l2_dot = CORE.operational_score(raw_left, raw_right, metric="dot", normalization="l2")
        raw_dot = CORE.operational_score(raw_left, raw_right, metric="dot", normalization="none")
        self.assertAlmostEqual(l2_dot, CORE.cosine_similarity(raw_left, raw_right), places=9)
        self.assertNotAlmostEqual(raw_dot, l2_dot, places=6)

    def test_silent_mix_is_plausible_and_wrong_then_rejected(self):
        query = self.canonical.get("q-password")
        mixed = CORE.retrieve_unchecked(
            query.vector,
            self.mismatch,
            query_text=query.text,
            query_id=query.id,
            query_provenance=self.canonical.provenance,
            top_k=5,
        )
        self.assertFalse(mixed.enforced)
        self.assertEqual(list(mixed.ids()[:3]), self.catalog["expected"]["mixed_password_top3"])
        self.assertTrue(any(item_id.startswith("d-printer") for item_id in mixed.ids()[:2]))
        self.assertTrue(all(-2.0 <= item.score <= 2.0 for item in mixed.results))
        report = CORE.compatibility_report(self.canonical.provenance, self.mismatch.provenance)
        self.assertFalse(report["compatible"])
        self.assertIn("model", report["mismatches"])
        self.assertIn("version", report["mismatches"])
        self.assertIn("normalization", report["mismatches"])
        with self.assertRaises(CORE.ProvenanceError) as raised:
            CORE.rank_neighbors(
                query.vector,
                self.mismatch,
                query_text=query.text,
                query_id=query.id,
                query_provenance=self.canonical.provenance,
                enforce_provenance=True,
            )
        self.assertIn("version", raised.exception.mismatches)
        self.assertIn("normalization", raised.exception.mismatches)
        same = CORE.retrieve_query("q-password")
        self.assertEqual(same.top_id, "d-password-forgot")
        self.assertTrue(same.enforced)

    def test_ties_break_by_id_and_empty_text_is_rejected(self):
        vector = self.canonical.get("d-rain").vector
        twins = (
            CORE.EmbeddedItem(id="z-twin", text="rain later", vector=vector, role="document"),
            CORE.EmbeddedItem(id="a-twin", text="rain earlier", vector=vector, role="document"),
        )
        space = CORE.VectorSpace(provenance=self.canonical.provenance, items=twins)
        ranked = CORE.rank_neighbors(
            vector,
            space,
            query_text="rain",
            query_provenance=self.canonical.provenance,
        )
        self.assertEqual(ranked.ids(), ("a-twin", "z-twin"))
        self.assertAlmostEqual(ranked.results[0].score, ranked.results[1].score)
        with self.assertRaises(ValueError):
            self.encoder.encode("the the the")
        with self.assertRaises(ValueError):
            CORE.cosine_similarity((0.0, 0.0), (1.0, 0.0))
        with self.assertRaises(ValueError):
            CORE.rank_neighbors(vector, space, query_provenance=self.canonical.provenance, top_k=0)

    def test_transfer_fixture_is_hand_computable_and_mismatch_probe_differs(self):
        query = self.transfer.get("t-query")
        ranked = CORE.rank_neighbors(
            query.vector,
            self.transfer,
            query_text=query.text,
            query_id=query.id,
            query_provenance=self.transfer.provenance,
        )
        self.assertEqual(ranked.top_id, "t-host")
        self.assertGreater(ranked.results[0].score, ranked.results[1].score)
        paint = self.transfer.get("t-paint")
        self.assertGreater(
            CORE.lexical_overlap(query.text, paint.text),
            CORE.lexical_overlap(query.text, self.transfer.get("t-host").text),
        )
        probe = TRANSFER["mismatch_probe"]
        probe_prov = CORE.provenance_from_mapping(probe["provenance"])
        report = CORE.compatibility_report(self.transfer.provenance, probe_prov)
        self.assertFalse(report["compatible"])
        self.assertIn("version", report["mismatches"])
        self.assertIn("model", report["mismatches"])
        self.assertAlmostEqual(CORE.cosine_similarity(query.vector, self.transfer.get("t-host").vector), 0.96)
        self.assertAlmostEqual(CORE.cosine_similarity(query.vector, self.transfer.get("t-not-down").vector), 0.8)

    def test_spaces_collapse_in_encoder_tokens(self):
        compact = self.encoder.encode("reset my password")
        spaced = self.encoder.encode("reset   my   password")
        self.assertEqual(compact, spaced)
        report = self.encoder.encode_report("reset my password")
        self.assertEqual(report["shape"], (12,))
        self.assertAlmostEqual(report["norm"], 1.0, places=9)
        self.assertFalse(report["downloaded"])
        self.assertEqual(report["version"], "v06.1")


if __name__ == "__main__":
    unittest.main()
