from __future__ import annotations

import ast
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
MISSION = ROOT / "missions" / "M29"
NOTEBOOK = ROOT / "labs" / "M29_attention_context.ipynb"
REQUIREMENTS = ROOT / "requirements" / "m29.txt"


def load_core():
    spec = importlib.util.spec_from_file_location("m29_attention_core", MISSION / "attention_core.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load M29 attention core")
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


def independent_softmax(values: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    maximum = max(values)
    shifted = tuple(math.exp(value - maximum) for value in values)
    total = sum(shifted)
    return tuple(value / total for value in shifted)


def independent_dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


class M29StaticContractTests(unittest.TestCase):
    def test_required_executable_artifacts_exist(self):
        for path in (
            MISSION / "attention_core.py",
            NOTEBOOK,
            REQUIREMENTS,
            ROOT / "tests" / "test_m29.py",
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
            "attention_core.py",
        }
        actual = {path.name for path in MISSION.iterdir() if path.is_file()}
        self.assertTrue(required <= actual, required - actual)
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        for name in required:
            self.assertIn(f"missions/M29/{name}", manifest)

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
            "multi_head",
            "n_head",
            "num_heads",
            "layer_norm",
            "layernorm",
            "rmsnorm",
            "feed_forward",
            "transformer_block",
            "residual_stream",
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
        self.assertIn("M29", source)
        self.assertIn("attention_core.py", source)
        self.assertIn("sys.path", source)
        self.assertIn("from missions.M29.attention_core import", source)

    def test_future_mission_boundary_stays_closed_in_code(self):
        source = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        for forbidden in (
            "multi_head",
            "num_heads",
            "n_head",
            "layer_norm",
            "LayerNorm",
            "feed_forward",
            "transformer_block",
            "from_pretrained",
            "FastAPI",
            "VectorIndex",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_notebook_enforces_prediction_before_every_action(self):
        cells = notebook_cells()
        positions = {str(cell["id"]): index for index, cell in enumerate(cells)}
        pairs = (
            ("predict-whole", "run-whole"),
            ("predict-qkv", "run-qkv"),
            ("predict-scores", "run-scores"),
            ("predict-causal", "run-causal"),
            ("predict-padding", "run-padding"),
            ("predict-scale", "run-scale"),
            ("predict-qk-perturb", "run-qk-perturb"),
            ("predict-value", "run-value"),
            ("predict-failure-axis", "run-failure-axis"),
            ("predict-failure-mask", "run-failure-mask"),
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
        self.assertGreaterEqual(markdown.count("Predict before running"), 11)
        for phrase in (
            "predict → act → observe → explain",
            "timestamp",
            "Controlled failure",
            "UNFILLED BY LEARNER",
            "M16 → M28 → M29",
            "softmax over keys",
            "not a causal explanation",
            "1/sqrt",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, markdown)

    def test_notebook_prints_required_attention_evidence(self):
        code = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        for token in (
            "self_attention",
            "scaled_dot_product_attention",
            "project_qkv",
            "dot_product_scores",
            "scale_scores",
            "causal_additive_mask",
            "padding_additive_mask",
            "softmax_over_keys",
            "aggregate_values",
            "attention_with_defect",
            "repair_attention",
            "weight_invariants",
            "observability_report",
            "X_CASH_CONTEXT",
            "X_WATER_CONTEXT",
            "HAND_Q",
            'defect="softmax_over_queries"',
            'defect="mask_after_softmax"',
            "scale=\"none\"",
            "teaching_batch",
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

    def test_attention_core_top_level_imports_are_stdlib(self):
        source = (MISSION / "attention_core.py").read_text(encoding="utf-8")
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
            "math",
            "typing",
        }
        for name in top_level_imports:
            with self.subTest(name=name):
                self.assertIn(name.split(".")[0], allowed)
        self.assertNotIn("numpy", top_level_imports)
        self.assertNotIn("torch", top_level_imports)
        self.assertNotIn("requests", top_level_imports)
        self.assertIn("_require_numpy", source)
        self.assertIn("multi-head", source.lower())
        self.assertIn("deferred to M30", source)

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
        self.assertIn("q = (1.0, 2.0)", no_ai)
        self.assertIn("k0 = (1.0, 0.0)", no_ai)
        self.assertIn("v2 = (0.0, 0.0, 1.0)", no_ai)
        self.assertIn("1/sqrt(d_k)", no_ai)
        self.assertNotIn("0.731058", no_ai)
        self.assertNotIn("0.575975", no_ai)
        self.assertEqual(CORE.TRANSFER_Q, (1.0, 2.0))
        self.assertEqual(CORE.TRANSFER_K[2], (1.0, 1.0))
        self.assertEqual(CORE.TRANSFER_QUERY_INDEX, 1)

    def test_core_defers_block_mechanisms_in_source(self):
        source = (MISSION / "attention_core.py").read_text(encoding="utf-8")
        self.assertNotIn("num_heads", source)
        self.assertNotIn("layer_norm", source)
        self.assertNotIn("feed_forward", source)
        self.assertNotIn("from_pretrained", source)


@unittest.skipUnless(RUNTIME_DEPS, "install requirements/m29.txt to run NumPy-dependent M29 tests")
class M29RuntimeTests(unittest.TestCase):
    def test_hand_two_key_attention_matches_independent_arithmetic(self):
        expected_w0 = math.e / (math.e + 1.0)
        expected_w1 = 1.0 / (math.e + 1.0)
        expected_out = (expected_w0, 2.0 * expected_w1)
        independent = independent_softmax(CORE.HAND_UNSCALED_SCORES)
        self.assertAlmostEqual(independent[0], expected_w0, places=12)
        self.assertAlmostEqual(independent[1], expected_w1, places=12)
        self.assertAlmostEqual(CORE.HAND_UNSCALED_WEIGHTS[0], expected_w0, places=12)
        self.assertAlmostEqual(CORE.HAND_UNSCALED_OUTPUT[0], expected_out[0], places=12)
        self.assertAlmostEqual(CORE.HAND_UNSCALED_OUTPUT[1], expected_out[1], places=12)

        trace = CORE.scaled_dot_product_attention(
            CORE.HAND_Q, CORE.HAND_K, CORE.HAND_V, scale="none"
        )
        self.assertEqual(trace.shapes["q"], (1, 1, 2))
        self.assertEqual(trace.shapes["k"], (1, 2, 2))
        self.assertEqual(trace.shapes["v"], (1, 2, 2))
        self.assertEqual(trace.shapes["scores"], (1, 1, 2))
        self.assertEqual(trace.shapes["output"], (1, 1, 2))
        self.assertAlmostEqual(float(trace.raw_scores[0, 0, 0]), 1.0, places=12)
        self.assertAlmostEqual(float(trace.raw_scores[0, 0, 1]), 0.0, places=12)
        self.assertAlmostEqual(float(trace.weights[0, 0, 0]), expected_w0, places=9)
        self.assertAlmostEqual(float(trace.weights[0, 0, 1]), expected_w1, places=9)
        self.assertAlmostEqual(float(trace.output[0, 0, 0]), expected_out[0], places=9)
        self.assertAlmostEqual(float(trace.output[0, 0, 1]), expected_out[1], places=9)
        self.assertTrue(trace.invariants()["rows_sum_to_one"])

    def test_scaled_two_key_softmax_uses_inverse_sqrt_d_k(self):
        scale = 1.0 / math.sqrt(2.0)
        expected = independent_softmax((1.0 * scale, 0.0))
        trace = CORE.scaled_dot_product_attention(
            CORE.HAND_Q, CORE.HAND_K, CORE.HAND_V, scale="dk"
        )
        self.assertAlmostEqual(trace.scale, scale, places=12)
        self.assertAlmostEqual(float(trace.scaled_scores[0, 0, 0]), scale, places=12)
        self.assertAlmostEqual(float(trace.weights[0, 0, 0]), expected[0], places=9)
        self.assertAlmostEqual(float(trace.weights[0, 0, 1]), expected[1], places=9)
        self.assertLess(float(trace.weights[0, 0, 0]), math.e / (math.e + 1.0))

    def test_cash_context_bank_is_uniform_and_water_shifts_mass(self):
        cash = CORE.self_attention(CORE.X_CASH_CONTEXT, tokens=CORE.CASH_TOKENS)
        water = CORE.self_attention(CORE.X_WATER_CONTEXT, tokens=CORE.WATER_TOKENS)
        bank = CORE.BANK_INDEX
        for weight in cash.weights[0, bank]:
            self.assertAlmostEqual(float(weight), 1.0 / 3.0, places=9)
        self.assertAlmostEqual(float(cash.output[0, bank, 0]), 1.0, places=9)
        self.assertAlmostEqual(float(cash.output[0, bank, 1]), 1.0, places=9)

        water_scores = (
            independent_dot(CORE.TOKEN_BANK, CORE.TOKEN_RIVER),
            independent_dot(CORE.TOKEN_BANK, CORE.TOKEN_BANK),
            independent_dot(CORE.TOKEN_BANK, CORE.TOKEN_WATER),
        )
        self.assertEqual(water_scores, (2.0, 2.0, 3.0))
        scale = 1.0 / math.sqrt(2.0)
        expected_water = independent_softmax(tuple(score * scale for score in water_scores))
        for left, right in zip(water.weights[0, bank], expected_water, strict=True):
            self.assertAlmostEqual(float(left), right, places=9)
        self.assertGreater(float(water.weights[0, bank, 2]), float(cash.weights[0, bank, 2]))
        self.assertGreater(float(water.output[0, bank, 0]), float(cash.output[0, bank, 0]))
        self.assertLess(float(water.output[0, bank, 1]), float(cash.output[0, bank, 1]))
        self.assertTrue(cash.invariants()["rows_sum_to_one"])
        self.assertTrue(water.invariants()["rows_sum_to_one"])

    def test_teaching_batch_makes_context_the_named_change(self):
        batch = CORE.self_attention(CORE.teaching_batch())
        self.assertEqual(batch.shapes["q"], (2, 3, 2))
        self.assertEqual(batch.shapes["output"], (2, 3, 2))
        cash = CORE.self_attention(CORE.X_CASH_CONTEXT)
        water = CORE.self_attention(CORE.X_WATER_CONTEXT)
        np = CORE._require_numpy()
        self.assertTrue(np.allclose(batch.weights[0], cash.weights[0]))
        self.assertTrue(np.allclose(batch.weights[1], water.weights[0]))
        self.assertFalse(np.allclose(batch.output[0, CORE.BANK_INDEX], batch.output[1, CORE.BANK_INDEX]))

    def test_qkv_projections_follow_row_batch_matmul(self):
        np = CORE._require_numpy()
        x = np.asarray(CORE.X_CASH_CONTEXT, dtype=float)
        q, k, v = CORE.project_qkv(x, CORE.IDENTITY_2, CORE.IDENTITY_2, CORE.IDENTITY_2)
        self.assertTrue(np.allclose(q[0], x))
        self.assertTrue(np.allclose(k[0], x))
        self.assertTrue(np.allclose(v[0], x))
        q2, k2, v2 = CORE.project_qkv(x, CORE.TEACHING_W_Q, CORE.TEACHING_W_K, CORE.TEACHING_W_V)
        self.assertEqual(q2.shape, (1, 3, 2))
        self.assertFalse(np.allclose(q2, q))
        self.assertFalse(np.allclose(k2, k))
        self.assertTrue(np.allclose(v2, v))
        expected_q_bank = tuple(
            independent_dot(CORE.TOKEN_BANK, tuple(row[i] for row in CORE.TEACHING_W_Q))
            for i in range(2)
        )
        self.assertAlmostEqual(float(q2[0, CORE.BANK_INDEX, 0]), expected_q_bank[0], places=12)
        self.assertAlmostEqual(float(q2[0, CORE.BANK_INDEX, 1]), expected_q_bank[1], places=12)

    def test_causal_mask_zeros_future_mass_and_renormalizes(self):
        mask = CORE.causal_additive_mask(3)
        cash = CORE.self_attention(CORE.X_CASH_CONTEXT, mask=mask, tokens=CORE.CASH_TOKENS)
        unmasked = CORE.self_attention(CORE.X_CASH_CONTEXT)
        self.assertAlmostEqual(float(cash.weights[0, 0, 0]), 1.0, places=9)
        self.assertAlmostEqual(float(cash.weights[0, 1, 0]), 0.5, places=9)
        self.assertAlmostEqual(float(cash.weights[0, 1, 1]), 0.5, places=9)
        self.assertAlmostEqual(float(cash.weights[0, 1, 2]), 0.0, places=12)
        self.assertAlmostEqual(float(cash.output[0, 1, 0]), 1.5, places=9)
        self.assertAlmostEqual(float(cash.output[0, 1, 1]), 0.5, places=9)
        self.assertEqual(float(unmasked.weights[0, 1, 2]), float(unmasked.weights[0, 1, 0]))
        invariants = cash.invariants()
        self.assertTrue(invariants["rows_sum_to_one"])
        self.assertTrue(invariants["future_mass_zero"])
        self.assertAlmostEqual(invariants["masked_key_mass"], 0.0, places=12)
        np = CORE._require_numpy()
        self.assertTrue(np.allclose(cash.raw_scores, unmasked.raw_scores))
        self.assertTrue(np.allclose(cash.scaled_scores, unmasked.scaled_scores))

    def test_scale_changes_concentration_with_q_and_k_fixed(self):
        q, k, v = CORE.scale_fixture_qkv()
        unscaled = CORE.scaled_dot_product_attention(q, k, v, scale="none")
        scaled = CORE.scaled_dot_product_attention(q, k, v, scale="dk")
        self.assertAlmostEqual(float(unscaled.raw_scores[0, 0, 0]), 8.0, places=12)
        self.assertAlmostEqual(float(unscaled.raw_scores[0, 0, 1]), 0.0, places=12)
        self.assertGreater(float(unscaled.weights[0, 0, 0]), float(scaled.weights[0, 0, 0]))
        self.assertLess(
            CORE.np_ravel(unscaled.invariants()["entropy"])[0],
            CORE.np_ravel(scaled.invariants()["entropy"])[0],
        )
        np = CORE._require_numpy()
        self.assertTrue(np.allclose(unscaled.q, scaled.q))
        self.assertTrue(np.allclose(unscaled.k, scaled.k))
        self.assertAlmostEqual(scaled.scale, 1.0 / math.sqrt(8.0), places=12)

    def test_query_perturbation_moves_scores_with_values_fixed(self):
        cash = CORE.self_attention(CORE.X_CASH_CONTEXT)
        perturbed_q = CORE.replace_position(cash.q, CORE.BANK_INDEX, (1.5, 0.5))
        moved = CORE.scaled_dot_product_attention(perturbed_q, cash.k, cash.v)
        expected_raw = (
            independent_dot((1.5, 0.5), CORE.TOKEN_RIVER),
            independent_dot((1.5, 0.5), CORE.TOKEN_BANK),
            independent_dot((1.5, 0.5), CORE.TOKEN_CASH),
        )
        self.assertEqual(expected_raw, (3.0, 2.0, 1.0))
        for left, right in zip(moved.raw_scores[0, CORE.BANK_INDEX], expected_raw, strict=True):
            self.assertAlmostEqual(float(left), right, places=12)
        self.assertGreater(float(moved.weights[0, CORE.BANK_INDEX, 0]), float(cash.weights[0, CORE.BANK_INDEX, 0]))
        self.assertLess(float(moved.weights[0, CORE.BANK_INDEX, 2]), float(cash.weights[0, CORE.BANK_INDEX, 2]))
        np = CORE._require_numpy()
        self.assertTrue(np.allclose(moved.v, cash.v))

    def test_value_only_change_keeps_weights_and_moves_output(self):
        cash = CORE.self_attention(CORE.X_CASH_CONTEXT)
        new_v = CORE.replace_position(cash.v, CORE.CONTEXT_INDEX, (0.0, 4.0))
        changed = CORE.scaled_dot_product_attention(cash.q, cash.k, new_v)
        np = CORE._require_numpy()
        self.assertTrue(np.allclose(changed.weights, cash.weights))
        self.assertTrue(np.allclose(changed.raw_scores, cash.raw_scores))
        expected = (
            (1.0 / 3.0) * 2.0 + (1.0 / 3.0) * 1.0 + (1.0 / 3.0) * 0.0,
            (1.0 / 3.0) * 0.0 + (1.0 / 3.0) * 1.0 + (1.0 / 3.0) * 4.0,
        )
        self.assertAlmostEqual(float(changed.output[0, CORE.BANK_INDEX, 0]), expected[0], places=9)
        self.assertAlmostEqual(float(changed.output[0, CORE.BANK_INDEX, 1]), expected[1], places=9)
        self.assertAlmostEqual(expected[1], 5.0 / 3.0, places=12)
        self.assertFalse(np.allclose(changed.output[0, CORE.BANK_INDEX], cash.output[0, CORE.BANK_INDEX]))

    def test_softmax_over_queries_breaks_row_sums_and_repair_uses_broken_scores(self):
        cash = CORE.self_attention(CORE.X_CASH_CONTEXT)
        broken = CORE.attention_with_defect(
            cash.q,
            cash.k,
            cash.v,
            defect="softmax_over_queries",
        )
        self.assertEqual(broken.defect, "softmax_over_queries")
        self.assertEqual(broken.softmax_axis, CORE.QUERY_AXIS)
        invariants = broken.invariants()
        self.assertFalse(invariants["rows_sum_to_one"])
        np = CORE._require_numpy()
        col_sums = broken.weights.sum(axis=CORE.QUERY_AXIS)
        self.assertTrue(np.allclose(col_sums, 1.0))
        self.assertTrue(np.allclose(broken.scaled_scores, cash.scaled_scores))
        self.assertFalse(np.allclose(broken.weights, cash.weights))

        from_broken_scores = CORE.softmax_over_keys(broken.masked_scores)
        repaired = CORE.repair_attention(broken)
        self.assertEqual(repaired.defect, "none")
        self.assertEqual(repaired.softmax_axis, CORE.KEY_AXIS)
        self.assertTrue(np.allclose(from_broken_scores, repaired.weights))
        self.assertTrue(np.allclose(repaired.weights, cash.weights))
        self.assertTrue(np.allclose(repaired.q, broken.q))
        self.assertTrue(np.allclose(repaired.k, broken.k))
        self.assertTrue(np.allclose(repaired.v, broken.v))
        self.assertTrue(repaired.invariants()["rows_sum_to_one"])
        still_broken = CORE.attention_with_defect(
            broken.q, broken.k, broken.v, defect="softmax_over_queries"
        )
        self.assertFalse(still_broken.invariants()["rows_sum_to_one"])

    def test_mask_after_softmax_zeros_future_without_renormalizing(self):
        mask = CORE.causal_additive_mask(3)
        cash = CORE.self_attention(CORE.X_CASH_CONTEXT)
        broken = CORE.attention_with_defect(
            cash.q,
            cash.k,
            cash.v,
            mask=mask,
            defect="mask_after_softmax",
        )
        self.assertEqual(broken.mask_timing, "after_softmax")
        np = CORE._require_numpy()
        self.assertFalse(np.allclose(broken.masked_scores, broken.scaled_scores))
        self.assertTrue(
            np.allclose(
                broken.masked_scores,
                CORE.apply_additive_mask(broken.scaled_scores, broken.mask),
            )
        )
        invariants = broken.invariants()
        self.assertTrue(invariants["future_mass_zero"])
        self.assertFalse(invariants["rows_sum_to_one"])
        self.assertAlmostEqual(float(broken.weights[0, 0].sum()), float(cash.weights[0, 0, 0]), places=9)
        self.assertAlmostEqual(float(broken.weights[0, 1].sum()), 2.0 / 3.0, places=9)

        repaired_weights = CORE.softmax_over_keys(
            CORE.apply_additive_mask(broken.scaled_scores, broken.mask)
        )
        repaired = CORE.repair_attention(broken)
        np = CORE._require_numpy()
        self.assertTrue(np.allclose(repaired.weights, repaired_weights))
        self.assertTrue(repaired.invariants()["rows_sum_to_one"])
        self.assertTrue(repaired.invariants()["future_mass_zero"])
        self.assertAlmostEqual(float(repaired.weights[0, 1, 0]), 0.5, places=9)

    def test_padding_mask_blocks_a_key_and_fully_masked_rows_raise(self):
        cash = CORE.self_attention(CORE.X_CASH_CONTEXT)
        pad = CORE.padding_additive_mask((True, True, False))
        masked = CORE.scaled_dot_product_attention(cash.q, cash.k, cash.v, mask=pad)
        self.assertAlmostEqual(float(masked.weights[0, 1, 2]), 0.0, places=12)
        self.assertTrue(masked.invariants()["rows_sum_to_one"])
        with self.assertRaises(ValueError):
            CORE.scaled_dot_product_attention(
                cash.q, cash.k, cash.v, mask=CORE.padding_additive_mask((False, False, False))
            )

    def test_shape_mismatches_are_rejected(self):
        with self.assertRaises(ValueError):
            CORE.dot_product_scores(((1.0, 0.0),), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
        with self.assertRaises(ValueError):
            CORE.scaled_dot_product_attention(
                CORE.HAND_Q, CORE.HAND_K, ((1.0, 0.0, 0.0),)
            )
        with self.assertRaises(ValueError):
            CORE.project_sequence(CORE.X_CASH_CONTEXT, ((1.0, 0.0),))
        with self.assertRaises(ValueError):
            CORE.attention_with_defect(CORE.HAND_Q, CORE.HAND_K, CORE.HAND_V, defect="n_head")

    def test_transfer_fixture_is_hand_computable_without_spoiling_the_gate(self):
        raw = tuple(independent_dot(CORE.TRANSFER_Q, key) for key in CORE.TRANSFER_K)
        self.assertEqual(raw, (1.0, 2.0, 3.0))
        scale = 1.0 / math.sqrt(2.0)
        scaled = tuple(score * scale for score in raw)
        unmasked = independent_softmax(scaled)
        causal_scores = (scaled[0], scaled[1], CORE.MASK_FILL)
        causal = independent_softmax(causal_scores)
        self.assertAlmostEqual(causal[2], 0.0, places=12)
        self.assertAlmostEqual(causal[0] + causal[1], 1.0, places=9)
        output = tuple(
            causal[0] * CORE.TRANSFER_V[0][axis]
            + causal[1] * CORE.TRANSFER_V[1][axis]
            + causal[2] * CORE.TRANSFER_V[2][axis]
            for axis in range(3)
        )
        mask = CORE.causal_additive_mask(1, 3)
        # Position-1 causal pattern: allow keys 0 and 1.
        mask[0, 2] = CORE.MASK_FILL
        mask[0, 0] = 0.0
        mask[0, 1] = 0.0
        trace = CORE.scaled_dot_product_attention(
            CORE.TRANSFER_Q, CORE.TRANSFER_K, CORE.TRANSFER_V, mask=mask
        )
        for left, right in zip(trace.weights[0, 0], causal, strict=True):
            self.assertAlmostEqual(float(left), right, places=9)
        for left, right in zip(trace.output[0, 0], output, strict=True):
            self.assertAlmostEqual(float(left), right, places=9)
        self.assertNotIn("0.575975", (MISSION / "no_ai_gate.md").read_text(encoding="utf-8"))

    def test_observability_report_states_the_interpretation_limit(self):
        trace = CORE.self_attention(CORE.X_CASH_CONTEXT, tokens=CORE.CASH_TOKENS)
        report = CORE.observability_report(trace)
        self.assertEqual(report["softmax_axis"], CORE.KEY_AXIS)
        self.assertEqual(report["defect"], "none")
        self.assertTrue(report["rows_sum_to_one"])
        self.assertIn("not a causal explanation of intent", report["interpretation_limit"])
        self.assertEqual(report["checkpoints"], CORE.TRACE_CHECKPOINTS)
        self.assertEqual(report["shapes"]["output"], (1, 3, 2))


if __name__ == "__main__":
    unittest.main()
