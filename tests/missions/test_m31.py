from __future__ import annotations

import ast
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
MISSION = ROOT / "missions" / "M31"
NOTEBOOK = ROOT / "labs" / "M31_llm_training.ipynb"
REQUIREMENTS = ROOT / "requirements" / "m31.txt"
DATASETS = ROOT / "datasets" / "M31"


def load_core():
    spec = importlib.util.spec_from_file_location(
        "m31_llm_training_core", MISSION / "llm_training_core.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load M31 llm training core")
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


def independent_shift(tokens: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return tokens[:-1], tokens[1:]


def independent_n_targets(n_tokens: int, context_length: int) -> int:
    return max(min(int(n_tokens), int(context_length)) - 1, 0)


def independent_nll(logits: tuple[float, ...], target: int) -> float:
    peak = max(logits)
    shifted = [math.exp(value - peak) for value in logits]
    total = sum(shifted)
    return -math.log(shifted[target] / total)


class M31StaticContractTests(unittest.TestCase):
    def test_required_executable_artifacts_exist(self):
        for path in (
            MISSION / "llm_training_core.py",
            NOTEBOOK,
            REQUIREMENTS,
            ROOT / "tests" / "test_m31.py",
            DATASETS / "corpus.json",
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
            "llm_training_core.py",
        }
        actual = {path.name for path in MISSION.iterdir() if path.is_file()}
        self.assertTrue(required <= actual, required - actual)
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        for name in required:
            self.assertIn(f"missions/M31/{name}", manifest)

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
            "from_pretrained",
            "huggingface",
            "sentence_transformers",
            "autotokenizer",
            "automodel",
            "tiktoken",
            "faiss",
            "qdrant",
            "fastapi",
            "temperature",
            "top_k",
            "top_p",
            "greedy_decode",
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
        self.assertIn("M31", source)
        self.assertIn("llm_training_core.py", source)
        self.assertIn("sys.path", source)
        self.assertIn("from missions.M31.llm_training_core import", source)
        self.assertIn("from missions.M27.tokenization_core import", source)
        self.assertIn("from missions.M30.transformer_block import", source)

    def test_future_mission_boundary_stays_closed_in_code(self):
        source = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        for forbidden in (
            "temperature",
            "top_k",
            "top_p",
            "from_pretrained",
            "FastAPI",
            "VectorIndex",
            "greedy_decode",
            "retrieve_and_generate",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_notebook_enforces_prediction_before_every_action(self):
        cells = notebook_cells()
        positions = {str(cell["id"]): index for index, cell in enumerate(cells)}
        pairs = (
            ("predict-whole", "run-whole"),
            ("predict-pairs", "run-pairs"),
            ("predict-loss", "run-loss"),
            ("predict-train", "run-train"),
            ("predict-heldout", "run-heldout"),
            ("predict-shift", "run-shift"),
            ("predict-context", "run-context"),
            ("predict-contamination", "run-contamination"),
            ("predict-stages", "run-stages"),
            ("predict-code-reading", "run-code-reading"),
            ("predict-failure-shift", "run-failure-shift"),
            ("predict-failure-leak", "run-failure-leak"),
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
        self.assertGreaterEqual(markdown.count("Predict before running"), 13)
        for phrase in (
            "predict → act → observe → explain",
            "timestamp",
            "Controlled failure",
            "UNFILLED BY LEARNER",
            "M30 → M31",
            "tokens[:-1]",
            "training-time",
            "not a production",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, markdown)

        predict_failure = cell_source(cells[positions["predict-failure-shift"]])
        predict_leak = cell_source(cells[positions["predict-failure-leak"]])
        self.assertNotIn("window[:-1]", predict_failure)
        self.assertNotIn('defect="target_shift_wrong"', predict_failure)
        self.assertNotIn("e02", predict_leak)

    def test_notebook_prints_required_training_evidence(self):
        code = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        for token in (
            "shift_tokens",
            "run_causal_pipeline",
            "pipeline_with_defect",
            "repair_run",
            "lineage_with_leak",
            "lineage_report",
            "context_length_effect",
            "softmax_nll",
            "classify_intervention",
            "observability_report",
            "first_divergence",
            "frozen_block_logits",
            "STAGE_DEFINITIONS",
            "SHIFT_TOKENS",
            'defect="target_shift_wrong"',
            'defect="held_out_leak"',
            "SYSTEM_MAP",
            "TRAINING_TIME_BOUNDARY",
            "SCALE_LIMIT",
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
        self.assertNotIn("transformers", joined)
        self.assertNotIn("sentence-transformers", joined)
        self.assertNotIn("faiss", joined)

    def test_training_core_top_level_imports_are_stdlib(self):
        source = (MISSION / "llm_training_core.py").read_text(encoding="utf-8")
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
            "dataclasses",
            "hashlib",
            "importlib.util",
            "json",
            "math",
            "pathlib",
            "sys",
            "typing",
        }
        for name in top_level_imports:
            with self.subTest(name=name):
                self.assertIn(name, allowed)
        self.assertNotIn("numpy", top_level_imports)
        self.assertNotIn("torch", top_level_imports)
        self.assertNotIn("requests", top_level_imports)
        self.assertIn("_require_numpy", source)
        self.assertIn("_load_m27", source)
        self.assertIn("_load_m30", source)
        self.assertIn("deferred to M32", source)

    def test_core_composes_predecessors_and_does_not_open_decoding(self):
        source = (MISSION / "llm_training_core.py").read_text(encoding="utf-8")
        self.assertIn("load_tokenizer", source)
        self.assertIn("transformer_block", source)
        self.assertNotIn("def softmax(", source)
        self.assertNotIn("from_pretrained", source)
        self.assertNotIn("top_k", source)
        self.assertNotIn("top_p", source)
        self.assertIn("token selection is M32", source)

    def test_independent_next_token_shift_arithmetic_is_locked(self):
        tokens = CORE.SHIFT_TOKENS
        inputs, targets = independent_shift(tokens)
        self.assertEqual(tokens, (10, 20, 30, 40))
        self.assertEqual(inputs, CORE.SHIFT_INPUTS)
        self.assertEqual(targets, CORE.SHIFT_TARGETS)
        self.assertEqual(inputs, (10, 20, 30))
        self.assertEqual(targets, (20, 30, 40))
        self.assertNotEqual(inputs, targets)
        self.assertEqual(tokens[:-1], CORE.WRONG_SHIFT_TARGETS)
        pairs = CORE.shift_tokens(tokens)
        self.assertEqual(pairs.inputs, inputs)
        self.assertEqual(pairs.targets, targets)
        self.assertTrue(CORE.pairs_are_causal(tokens, pairs.inputs, pairs.targets))
        wrong = CORE.shift_tokens(tokens, alignment="target_shift_wrong")
        self.assertEqual(wrong.targets, tokens[:-1])
        self.assertFalse(CORE.pairs_are_causal(tokens, wrong.inputs, wrong.targets))
        self.assertEqual(wrong.n_targets, 3)

    def test_independent_context_length_counts_prediction_targets(self):
        sequence = CORE.CONTEXT_SEQUENCE
        self.assertEqual(sequence, (1, 2, 3, 4, 5))
        self.assertEqual(independent_n_targets(5, 5), 4)
        self.assertEqual(independent_n_targets(5, 3), 2)
        self.assertEqual(CORE.n_prediction_targets(5, 5), 4)
        self.assertEqual(CORE.n_prediction_targets(5, 3), 2)
        self.assertEqual(CORE.apply_context_length(sequence, 3), (1, 2, 3))
        rows = CORE.context_length_effect(sequence, (5, 3, 2))
        self.assertEqual(rows[0]["n_targets"], 4)
        self.assertEqual(rows[1]["n_targets"], 2)
        self.assertEqual(rows[2]["n_targets"], 1)
        with self.assertRaises(ValueError):
            CORE.n_prediction_targets(5, 1)

    def test_independent_softmax_nll_matches_hand_arithmetic(self):
        expected_two = independent_nll(CORE.NLL_LOGITS, CORE.NLL_TARGET)
        self.assertAlmostEqual(expected_two, math.log(2.0), places=12)
        self.assertAlmostEqual(CORE.softmax_nll(CORE.NLL_LOGITS, CORE.NLL_TARGET), expected_two, places=12)
        expected_three = independent_nll(CORE.NLL_THREE_LOGITS, CORE.NLL_THREE_TARGET)
        self.assertAlmostEqual(expected_three, math.log(3.0), places=12)
        self.assertAlmostEqual(
            CORE.softmax_nll(CORE.NLL_THREE_LOGITS, CORE.NLL_THREE_TARGET),
            expected_three,
            places=12,
        )

    def test_authored_lineage_overlap_is_the_contamination_definition(self):
        overlap = set(CORE.TRANSFER_TRAIN_IDS) & set(CORE.TRANSFER_EVAL_IDS)
        self.assertEqual(overlap, {"e02"})
        clean_train = tuple(doc_id for doc_id in CORE.TRANSFER_TRAIN_IDS if doc_id != "e02")
        self.assertFalse(set(clean_train) & set(CORE.TRANSFER_EVAL_IDS))

    def test_stage_catalog_keeps_inference_out_of_training(self):
        self.assertEqual(CORE.classify_intervention("next_token_on_unlabeled_corpus"), "pretraining")
        self.assertEqual(CORE.classify_intervention("instruction_supervised_pairs"), "adaptation")
        self.assertEqual(CORE.classify_intervention("preference_ranking_or_rlhf"), "post_training")
        self.assertEqual(CORE.classify_intervention("score_protected_held_out"), "evaluation")
        self.assertEqual(CORE.classify_intervention("generate_from_frozen_checkpoint"), "inference")
        self.assertEqual(CORE.classify_intervention("change_prompt_without_weight_update"), "inference")
        catalog = CORE.stage_catalog()
        self.assertIn("frozen weights", catalog["inference"])
        self.assertIn("unlabeled", catalog["pretraining"])
        with self.assertRaises(ValueError):
            CORE.classify_intervention("temperature_knob")

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

    def test_no_ai_gate_uses_fresh_numbers_without_answers(self):
        no_ai = (MISSION / "no_ai_gate.md").read_text(encoding="utf-8")
        self.assertIn("tokens = (4, 8, 15, 16, 23)", no_ai)
        self.assertIn("targets = (4, 8, 15, 16)", no_ai)
        self.assertIn('train_ids = ("d01", "d02", "e02")', no_ai)
        self.assertEqual(CORE.TRANSFER_TOKENS, (4, 8, 15, 16, 23))
        self.assertEqual(CORE.TRANSFER_WRONG_TARGETS, (4, 8, 15, 16))
        self.assertNotIn("(8, 15, 16, 23)", no_ai)
        self.assertNotIn("1.098612", no_ai)
        self.assertNotIn(str(math.log(3.0))[:8], no_ai)

    def test_status_separates_implementation_from_learner_completion(self):
        status = (MISSION / "status.yaml").read_text(encoding="utf-8")
        self.assertIn("implementation_status: implemented", status)
        self.assertIn("learner_evidence_status: intentionally_unpopulated", status)
        self.assertNotIn("learner_evidence_status: complete", status)

    def test_corpus_fixture_matches_python_constants(self):
        payload = json.loads((DATASETS / "corpus.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["dataset_version"], CORE.DATASET_VERSION)
        self.assertEqual(payload["leak_doc_id"], CORE.LEAK_DOC_ID)
        self.assertFalse(payload["provenance"]["downloaded"])
        self.assertFalse(payload["provenance"]["network_required"])
        rows = tuple((row["id"], row["split"], row["text"]) for row in payload["documents"])
        self.assertEqual(rows, CORE.TEACHING_DOCUMENTS)


@unittest.skipUnless(RUNTIME_DEPS, "install requirements/m31.txt to run NumPy-dependent M31 tests")
class M31RuntimeTests(unittest.TestCase):
    def test_m27_encoding_then_shift_is_causal_on_d01(self):
        tokenizer = CORE.load_teaching_tokenizer()
        docs = CORE.load_teaching_corpus()
        d01 = [doc for doc in docs if doc.doc_id == "d01"][0]
        encoding = CORE.encode_text(d01.text, tokenizer)
        self.assertEqual(encoding.tokens[0], CORE.BOS_TOKEN)
        self.assertEqual(encoding.tokens[-1], CORE.EOS_TOKEN)
        pairs = CORE.shift_tokens(encoding.ids, token_pieces=encoding.tokens)
        self.assertEqual(pairs.inputs, encoding.ids[:-1])
        self.assertEqual(pairs.targets, encoding.ids[1:])
        self.assertTrue(pairs.causal)
        self.assertIn("the", encoding.tokens)

    def test_healthy_pipeline_drops_train_loss_and_keeps_eval_harder(self):
        healthy = CORE.run_causal_pipeline()
        self.assertEqual(healthy.defect, "none")
        self.assertEqual(healthy.alignment, "correct")
        self.assertTrue(all(window.causal for window in healthy.windows))
        self.assertFalse(healthy.used_lineage.contaminated)
        self.assertEqual(healthy.used_lineage.eval_ids, healthy.authored_lineage.eval_ids)
        self.assertLess(healthy.final_train_objective_loss, healthy.train_objective_losses[0])
        self.assertGreater(healthy.final_eval_true_loss, healthy.final_train_objective_loss)
        self.assertGreater(healthy.unseen_pair_nll, healthy.final_train_objective_loss)
        self.assertEqual(healthy.checkpoint.adaptation_stage, "pretrained")
        self.assertTrue(healthy.checkpoint.inference_ready)
        self.assertFalse(healthy.checkpoint.training_time)
        self.assertIn("not evidence of a production LLM", healthy.checkpoint.scale_limit)
        init = math.log(healthy.vocab_size)
        self.assertAlmostEqual(healthy.train_objective_losses[0], init, delta=1e-3)

    def test_context_length_changes_target_count_on_fixed_corpus(self):
        tokenizer = CORE.load_teaching_tokenizer()
        docs = CORE.load_teaching_corpus()
        d01 = [doc for doc in docs if doc.doc_id == "d01"][0]
        full = CORE.encode_document(d01, tokenizer, context_length=CORE.TEACHING_CONTEXT_LENGTH)
        short = CORE.encode_document(d01, tokenizer, context_length=CORE.SHORT_CONTEXT_LENGTH)
        self.assertGreater(full.n_targets, short.n_targets)
        self.assertEqual(short.n_targets, CORE.SHORT_CONTEXT_LENGTH - 1)
        self.assertEqual(short.window, full.tokens[: CORE.SHORT_CONTEXT_LENGTH])
        self.assertTrue(short.causal)

    def test_output_projection_shape_wraps_frozen_m30_block(self):
        m30 = CORE._load_m30()
        np = CORE._require_numpy()
        unembed = np.eye(m30.D_MODEL, 6)
        logits, trace = CORE.frozen_block_logits(m30.X_CASH_CONTEXT, unembed)
        self.assertEqual(tuple(int(dim) for dim in logits.shape), (1, 3, 6))
        self.assertEqual(trace.declared_convention, "pre_norm")
        self.assertEqual(trace.defect, "none")

    def test_shift_defect_first_diverges_at_targets(self):
        healthy = CORE.run_causal_pipeline()
        broken = CORE.pipeline_with_defect(defect="target_shift_wrong")
        self.assertEqual(broken.defect, "target_shift_wrong")
        self.assertEqual(CORE.first_divergence(healthy, broken), "targets")
        self.assertEqual(broken.used_lineage.train_ids, healthy.used_lineage.train_ids)
        self.assertTrue(all(window.inputs == window.window[:-1] for window in broken.windows))
        self.assertTrue(all(window.targets == window.window[:-1] for window in broken.windows))
        self.assertFalse(any(window.causal for window in broken.windows))
        self.assertLess(broken.final_train_objective_loss, broken.train_objective_losses[0])
        true_nll = CORE.true_next_token_nll(broken)
        self.assertGreater(true_nll, broken.final_train_objective_loss)

    def test_leak_defect_first_diverges_at_train_doc_ids(self):
        healthy = CORE.run_causal_pipeline()
        leaked = CORE.pipeline_with_defect(defect="held_out_leak")
        self.assertEqual(leaked.defect, "held_out_leak")
        self.assertEqual(CORE.first_divergence(healthy, leaked), "train_doc_ids")
        self.assertEqual(leaked.used_lineage.eval_ids, healthy.authored_lineage.eval_ids)
        self.assertIn(CORE.LEAK_DOC_ID, leaked.used_lineage.train_ids)
        self.assertTrue(leaked.used_lineage.contaminated)
        report = CORE.lineage_report(leaked.used_lineage)
        self.assertEqual(set(report["overlap"]), {CORE.LEAK_DOC_ID})
        self.assertFalse(report["eval_valid"])
        self.assertTrue(all(window.causal for window in leaked.windows))
        self.assertLess(leaked.unseen_pair_nll, healthy.unseen_pair_nll)
        self.assertLess(leaked.final_eval_true_loss, healthy.final_eval_true_loss)

    def test_repair_uses_broken_trace_objects_and_regression_still_fails(self):
        broken_shift = CORE.pipeline_with_defect(defect="target_shift_wrong")
        repaired_shift = CORE.repair_run(broken_shift)
        self.assertEqual(repaired_shift.defect, "none")
        self.assertEqual(repaired_shift.seed, broken_shift.seed)
        self.assertEqual(repaired_shift.steps, broken_shift.steps)
        self.assertEqual(repaired_shift.documents, broken_shift.documents)
        self.assertTrue(all(window.causal for window in repaired_shift.windows))
        self.assertEqual(CORE.first_divergence(broken_shift, repaired_shift), "targets")
        self.assertFalse(any(window.causal for window in broken_shift.windows))
        broken_leak = CORE.pipeline_with_defect(defect="held_out_leak")
        repaired_leak = CORE.repair_run(broken_leak)
        self.assertFalse(repaired_leak.used_lineage.contaminated)
        self.assertEqual(CORE.first_divergence(broken_leak, repaired_leak), "train_doc_ids")
        self.assertIn(CORE.LEAK_DOC_ID, broken_leak.used_lineage.train_ids)
        source = CORE.repair_run.__doc__ or ""
        self.assertIn("defective trace", source)

    def test_observability_report_states_limits_and_m32_handoff(self):
        trace = CORE.run_causal_pipeline()
        report = CORE.observability_report(trace)
        self.assertEqual(report["version"], CORE.TRAINING_VERSION)
        self.assertEqual(report["alignment"], "correct")
        self.assertIn("token selection is M32", report["training_time_boundary"])
        self.assertIn("does not receive a decoder from M31", report["handoff"])
        self.assertIn("not evidence of a production LLM", report["scale_limit"])

    def test_unsupported_defect_and_leak_of_train_id_are_rejected(self):
        with self.assertRaises(ValueError):
            CORE.pipeline_with_defect(defect="none")
        with self.assertRaises(ValueError):
            CORE.run_causal_pipeline(defect="temperature")
        lineage = CORE.authored_lineage()
        with self.assertRaises(ValueError):
            CORE.lineage_with_leak(lineage, leak_doc_id="d01")

    def test_transfer_fixture_is_hand_computable_without_spoiling_the_gate(self):
        inputs, targets = independent_shift(CORE.TRANSFER_TOKENS)
        self.assertEqual(targets, (8, 15, 16, 23))
        self.assertEqual(independent_n_targets(len(CORE.TRANSFER_TOKENS), 3), 2)
        nll = independent_nll(CORE.TRANSFER_NLL_LOGITS, CORE.TRANSFER_NLL_TARGET)
        self.assertAlmostEqual(nll, math.log(3.0), places=12)
        no_ai = (MISSION / "no_ai_gate.md").read_text(encoding="utf-8")
        self.assertNotIn("(8, 15, 16, 23)", no_ai)
        self.assertNotIn(f"{nll:.6f}", no_ai)


if __name__ == "__main__":
    unittest.main()
