from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
MISSION = ROOT / "missions" / "M27"
NOTEBOOK = ROOT / "labs" / "M27_text_to_tokens.ipynb"
REQUIREMENTS = ROOT / "requirements" / "m27.txt"
DATASETS = ROOT / "datasets" / "M27"


def load_core():
    spec = importlib.util.spec_from_file_location("m27_tokenization_core", MISSION / "tokenization_core.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load M27 tokenization core")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_core()
TEXTS = json.loads((DATASETS / "texts.json").read_text(encoding="utf-8"))
TOKENIZER_SPEC = json.loads((DATASETS / "teaching_tokenizer.json").read_text(encoding="utf-8"))


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


class M27StaticContractTests(unittest.TestCase):
    def test_required_executable_artifacts_exist(self):
        for path in (
            MISSION / "tokenization_core.py",
            NOTEBOOK,
            REQUIREMENTS,
            ROOT / "tests" / "test_m27.py",
            DATASETS / "teaching_tokenizer.json",
            DATASETS / "texts.json",
            DATASETS / "README.md",
        ):
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
            "tiktoken",
            "transformers",
            "autotokenizer",
            "huggingface",
            "embedding",
            "softmax",
            "attention_mask",
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
        self.assertIn("M27", source)
        self.assertIn("tokenization_core.py", source)
        self.assertIn("sys.path", source)
        self.assertIn("from missions.M27.tokenization_core import", source)

    def test_future_mission_boundary_stays_closed_in_code(self):
        source = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        for forbidden in ("embedding", "transformer", "tiktoken", "softmax", "attention_mask"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_notebook_enforces_prediction_before_every_action(self):
        cells = notebook_cells()
        positions = {str(cell["id"]): index for index, cell in enumerate(cells)}
        pairs = (
            ("predict-surface", "run-surface"),
            ("predict-rare", "run-rare"),
            ("predict-ids", "run-ids"),
            ("predict-specials", "run-specials"),
            ("predict-padding", "run-padding"),
            ("predict-truncation", "run-truncation"),
            ("predict-comparison", "run-comparison"),
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
            "M03 → M27 → M28",
            "token budget",
            "[BOS]",
            "[PAD]",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, markdown)

    def test_notebook_prints_required_token_evidence(self):
        code = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        for token in (
            "load_tokenizer",
            "load_texts",
            "normalize_text",
            "compare_schemes",
            "encode_batch",
            "pack_for_context",
            "trace_bpe_word",
            "encoding_report",
            'budget_unit="words"',
            'budget_unit="characters"',
            'budget_unit="tokens"',
            'scheme="word"',
            'scheme="bpe"',
            "padding_mask",
            "TokenBudgetError",
        ):
            with self.subTest(token=token):
                self.assertIn(token, code)

    def test_requirements_are_bounded_and_cover_runtime(self):
        requirements = REQUIREMENTS.read_text(encoding="utf-8").lower().splitlines()
        for package in ("matplotlib", "nbclient", "nbformat", "pytest", "jupyter"):
            with self.subTest(package=package):
                matching = [line for line in requirements if line.startswith(package)]
                self.assertEqual(len(matching), 1)
                self.assertIn(">=", matching[0])
                self.assertIn("<", matching[0])
        self.assertFalse(any(line.startswith("torch") for line in requirements))
        self.assertFalse(any("tiktoken" in line for line in requirements))
        self.assertFalse(any("transformers" in line for line in requirements))

    def test_tokenization_core_uses_only_stdlib_imports(self):
        source = (MISSION / "tokenization_core.py").read_text(encoding="utf-8")
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
        allowed = {"__future__", "dataclasses", "functools", "pathlib", "json", "re"}
        for name in top_level_imports:
            with self.subTest(name=name):
                self.assertIn(name.split(".")[0], allowed)
        self.assertNotIn("numpy", top_level_imports)
        self.assertNotIn("torch", top_level_imports)
        self.assertNotIn("tiktoken", top_level_imports)
        self.assertNotIn("transformers", top_level_imports)
        self.assertNotIn("requests", top_level_imports)

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
        self.assertEqual(TOKENIZER_SPEC["family"], "v06-teaching-tokenizer")
        self.assertEqual(TOKENIZER_SPEC["version"], "v06.1")
        self.assertFalse(TOKENIZER_SPEC["provenance"]["downloaded"])
        self.assertFalse(TOKENIZER_SPEC["provenance"]["network_required"])
        self.assertTrue(TOKENIZER_SPEC["provenance"]["not_a_huggingface_model"])
        self.assertTrue(TOKENIZER_SPEC["provenance"]["not_tiktoken"])
        self.assertEqual(TOKENIZER_SPEC["special_tokens"], ["[PAD]", "[UNK]", "[BOS]", "[EOS]"])
        self.assertGreaterEqual(len(TOKENIZER_SPEC["bpe"]["merges"]), 40)
        self.assertEqual(TEXTS["canonical_sentence"], "the cat sat on the mat")
        self.assertIn("approve_refund", TEXTS["controlled_failure"]["text"])


class M27RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.word = CORE.load_tokenizer("word")
        cls.bpe = CORE.load_tokenizer("bpe")
        cls.texts = CORE.load_texts()

    def test_special_token_ids_are_stable(self):
        for tokenizer in (self.word, self.bpe):
            self.assertEqual(tokenizer.token_to_id("[PAD]"), 0)
            self.assertEqual(tokenizer.token_to_id("[UNK]"), 1)
            self.assertEqual(tokenizer.token_to_id("[BOS]"), 2)
            self.assertEqual(tokenizer.token_to_id("[EOS]"), 3)
            self.assertFalse(tokenizer.identity.downloaded)
            self.assertEqual(tokenizer.version, "v06.1")

    def test_canonical_sentence_round_trips_on_both_schemes(self):
        text = self.texts["canonical_sentence"]
        for tokenizer, expected_key in (
            (self.word, "canonical_word_tokens"),
            (self.bpe, "canonical_bpe_tokens"),
        ):
            encoded = tokenizer.encode(text)
            self.assertEqual(encoded.tokens, tuple(self.texts["expected"][expected_key]))
            self.assertEqual(tokenizer.decode(encoded.ids), text)
            self.assertEqual(encoded.length, 8)
            self.assertEqual(encoded.tokens[0], "[BOS]")
            self.assertEqual(encoded.tokens[-1], "[EOS]")

    def test_surface_variation_normalization_and_punctuation(self):
        variants = self.texts["surface_variants"]
        base = self.bpe.encode(variants["base"])
        whitespace = self.bpe.encode(variants["whitespace"])
        casing = self.bpe.encode(variants["casing"])
        punct = self.bpe.encode(variants["punctuation"])
        self.assertEqual(base.ids, whitespace.ids)
        self.assertEqual(base.ids, casing.ids)
        self.assertEqual(CORE.normalize_text(variants["base"]), CORE.normalize_text(variants["casing"]))
        self.assertNotEqual(base.ids, punct.ids)
        self.assertIn("!", punct.tokens)
        self.assertIn(".", base.tokens)
        self.assertEqual(base.length, punct.length)

    def test_rare_string_unks_on_word_and_fragments_on_bpe(self):
        rare = self.texts["rare_strings"]["identifier"]
        word_enc = self.word.encode(rare)
        bpe_enc = self.bpe.encode(rare)
        self.assertIn("[UNK]", word_enc.tokens)
        self.assertNotIn("[UNK]", bpe_enc.tokens)
        self.assertGreater(bpe_enc.length, word_enc.length)
        self.assertGreater(bpe_enc.length, 8)
        self.assertIn("x", bpe_enc.tokens)
        comparison = CORE.compare_schemes(rare)
        self.assertEqual(comparison["word_length"], word_enc.length)
        self.assertEqual(comparison["bpe_length"], bpe_enc.length)

    def test_special_tokens_consume_context_budget(self):
        text = "the cat sat"
        with_specials = self.bpe.encode(text, add_special_tokens=True)
        without = self.bpe.encode(text, add_special_tokens=False)
        self.assertEqual(with_specials.length, without.length + 2)
        self.assertEqual(self.bpe.token_count(text), with_specials.length)
        self.assertEqual(without.tokens, ("▁the", "▁cat", "▁sat"))

    def test_padding_batch_preserves_order_and_mask(self):
        batch = self.bpe.encode_batch(self.texts["padding_batch"], max_length=10, padding=True, truncation=True)
        self.assertEqual(batch.max_length, 10)
        self.assertEqual(len(batch.encodings), 3)
        for row_ids, row_mask, source in zip(
            batch.input_ids, batch.padding_mask, self.texts["padding_batch"]
        ):
            self.assertEqual(len(row_ids), 10)
            self.assertEqual(len(row_mask), 10)
            unpadded = self.bpe.encode(source, add_special_tokens=True)
            prefix = row_ids[: unpadded.length]
            self.assertEqual(prefix, unpadded.ids)
            self.assertEqual(row_mask[: unpadded.length], tuple(1 for _ in unpadded.ids))
            self.assertTrue(all(value in (0, 1) for value in row_mask))
            if unpadded.length < 10:
                self.assertEqual(row_ids[unpadded.length :], (0,) * (10 - unpadded.length))
                self.assertEqual(row_mask[unpadded.length :], (0,) * (10 - unpadded.length))

    def test_truncation_drops_the_right_suffix(self):
        text = self.texts["truncation_text"]
        full = self.bpe.encode(text, add_special_tokens=True)
        truncated = self.bpe.encode(text, add_special_tokens=True, max_length=8, truncation=True)
        self.assertGreater(full.length, 8)
        self.assertEqual(truncated.length, 8)
        self.assertTrue(truncated.truncated)
        self.assertEqual(truncated.tokens[0], "[BOS]")
        self.assertEqual(truncated.tokens[-1], "[EOS]")
        decoded = self.bpe.decode(truncated.ids)
        self.assertTrue(decoded.startswith("please inspect ticket 4412"))
        self.assertNotIn("invoice", decoded)
        self.assertIn("invoice", self.bpe.decode_pieces(truncated.dropped_tokens))

    def test_word_and_bpe_agree_on_in_vocab_sentence_and_disagree_on_url(self):
        same = CORE.compare_schemes(self.texts["canonical_sentence"])
        self.assertEqual(same["word_length"], same["bpe_length"])
        url = CORE.compare_schemes(self.texts["rare_strings"]["url"])
        self.assertGreater(url["bpe_length"], url["word_length"])
        self.assertGreater(url["length_delta"], 0)

    def test_word_budget_silently_drops_critical_suffix(self):
        failure = self.texts["controlled_failure"]
        packed = CORE.pack_for_context(
            failure["text"],
            self.bpe,
            max_tokens=failure["max_tokens"],
            budget_unit="words",
        )
        self.assertTrue(packed.heuristic_fit)
        self.assertLessEqual(packed.original_word_count, failure["max_tokens"])
        self.assertGreater(packed.original_token_count, failure["max_tokens"])
        self.assertTrue(packed.silent)
        self.assertTrue(packed.truncated)
        self.assertFalse(packed.contains(failure["critical_suffix"], self.bpe))
        self.assertIn("refund", packed.dropped_text)
        self.assertNotIn("approve_refund", packed.decode(self.bpe))

    def test_character_budget_is_the_same_class_of_defect(self):
        failure = self.texts["controlled_failure"]
        packed = CORE.pack_for_context(
            failure["text"],
            self.bpe,
            max_tokens=failure["max_tokens"],
            budget_unit="characters",
        )
        self.assertTrue(packed.heuristic_fit)
        self.assertTrue(packed.silent)
        self.assertFalse(packed.contains(failure["critical_suffix"], self.bpe))

    def test_token_budget_repair_is_honest_and_can_keep_the_suffix(self):
        failure = self.texts["controlled_failure"]
        honest = CORE.pack_for_context(
            failure["text"],
            self.bpe,
            max_tokens=failure["max_tokens"],
            budget_unit="tokens",
        )
        self.assertFalse(honest.heuristic_fit)
        self.assertFalse(honest.silent)
        self.assertTrue(honest.truncated)
        self.assertIn("refund", honest.dropped_text)
        with self.assertRaises(CORE.TokenBudgetError) as raised:
            CORE.pack_for_context(
                failure["text"],
                self.bpe,
                max_tokens=failure["max_tokens"],
                budget_unit="tokens",
                on_overflow="raise",
            )
        self.assertEqual(raised.exception.needed, honest.original_token_count)
        repaired = CORE.pack_for_context(
            failure["text"],
            self.bpe,
            max_tokens=honest.original_token_count,
            budget_unit="tokens",
        )
        self.assertFalse(repaired.truncated)
        self.assertTrue(repaired.contains(failure["critical_suffix"], self.bpe))
        self.assertEqual(repaired.decode(self.bpe), CORE.normalize_text(failure["text"]))

    def test_bpe_trace_for_sat_is_hand_checkable(self):
        snapshots = self.bpe.trace_bpe_word("sat")
        self.assertEqual(snapshots[0], ("▁", "s", "a", "t"))
        self.assertEqual(snapshots[-1], ("▁sat",))
        self.assertIn(("▁", "s", "at"), snapshots)

    def test_unknown_scheme_and_short_max_length_are_rejected(self):
        with self.assertRaises(ValueError):
            CORE.load_tokenizer("sentencepiece")
        with self.assertRaises(ValueError):
            self.bpe.encode("the cat", add_special_tokens=True, max_length=1, truncation=True)
        with self.assertRaises(ValueError):
            CORE.pack_for_context("the cat", self.bpe, max_tokens=12, budget_unit="bytes")


if __name__ == "__main__":
    unittest.main()
