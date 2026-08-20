from __future__ import annotations

import ast
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
MISSION = ROOT / "missions" / "M30"
NOTEBOOK = ROOT / "labs" / "M30_transformer_block.ipynb"
REQUIREMENTS = ROOT / "requirements" / "m30.txt"


def load_core():
    spec = importlib.util.spec_from_file_location(
        "m30_transformer_block", MISSION / "transformer_block.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load M30 transformer block")
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


def independent_add(
    left: tuple[float, ...], right: tuple[float, ...]
) -> tuple[float, ...]:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def independent_layer_norm(
    values: tuple[float, ...], eps: float = CORE.LN_EPS
) -> tuple[float, ...]:
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / len(values)
    scale = math.sqrt(var + float(eps))
    return tuple((value - mean) / scale for value in values)


def independent_relu(values: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(value if value > 0.0 else 0.0 for value in values)


class M30StaticContractTests(unittest.TestCase):
    def test_required_executable_artifacts_exist(self):
        for path in (
            MISSION / "transformer_block.py",
            NOTEBOOK,
            REQUIREMENTS,
            ROOT / "tests" / "test_m30.py",
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
            "transformer_block.py",
        }
        actual = {path.name for path in MISSION.iterdir() if path.is_file()}
        self.assertTrue(required <= actual, required - actual)
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        for name in required:
            self.assertIn(f"missions/M30/{name}", manifest)

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
            "cross_entropy",
            "next_token",
            "temperature",
            "top_k",
            "top_p",
            "contamination",
            "fine_tune",
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
        self.assertIn("M30", source)
        self.assertIn("transformer_block.py", source)
        self.assertIn("sys.path", source)
        self.assertIn("from missions.M30.transformer_block import", source)
        self.assertIn("from missions.M29.attention_core import", source)

    def test_future_mission_boundary_stays_closed_in_code(self):
        source = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        for forbidden in (
            "cross_entropy",
            "next_token",
            "from_pretrained",
            "temperature",
            "top_k",
            "contamination",
            "FastAPI",
            "VectorIndex",
            "fine_tune",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_notebook_enforces_prediction_before_every_action(self):
        cells = notebook_cells()
        positions = {str(cell["id"]): index for index, cell in enumerate(cells)}
        pairs = (
            ("predict-whole", "run-whole"),
            ("predict-m29-sublayer", "run-m29-sublayer"),
            ("predict-heads", "run-heads"),
            ("predict-wo", "run-wo"),
            ("predict-residual", "run-residual"),
            ("predict-ablation", "run-ablation"),
            ("predict-norm", "run-norm"),
            ("predict-ffn", "run-ffn"),
            ("predict-parity", "run-parity"),
            ("predict-code-reading", "run-code-reading"),
            ("predict-failure-residual", "run-failure-residual"),
            ("predict-failure-norm", "run-failure-norm"),
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
            "M29 → M30",
            "pre-norm",
            "parallel learned projections",
            "stream + sublayer",
            "not a universal",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, markdown)

    def test_notebook_prints_required_block_evidence(self):
        code = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        for token in (
            "transformer_block",
            "multi_head_attention",
            "split_heads",
            "merge_heads",
            "layer_norm",
            "residual_add",
            "feed_forward",
            "block_with_defect",
            "repair_block",
            "ablate_residual",
            "first_divergence",
            "checkpoint_parity",
            "independent_pre_norm_compose",
            "golden_pre_norm_trace",
            "observability_report",
            "X_CASH_CONTEXT",
            "X_WATER_CONTEXT",
            'convention="post_norm"',
            'defect="residual_wrong_branch"',
            'defect="norm_wrong_boundary"',
            'activation="identity"',
            "scaled_dot_product_attention",
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

    def test_block_core_top_level_imports_are_stdlib(self):
        source = (MISSION / "transformer_block.py").read_text(encoding="utf-8")
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
            "importlib.util",
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
        self.assertIn("_load_m29", source)
        self.assertIn("scaled_dot_product_attention", source)
        self.assertIn("deferred to M31", source)

    def test_core_composes_m29_and_does_not_redefine_softmax(self):
        source = (MISSION / "transformer_block.py").read_text(encoding="utf-8")
        self.assertIn("scaled_dot_product_attention", source)
        self.assertNotIn("def softmax", source)
        self.assertNotIn("def _softmax", source)
        self.assertNotIn("cross_entropy", source)
        self.assertNotIn("from_pretrained", source)
        self.assertIn("No loss, no optimizer, no next-token pairs", source)

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
        self.assertIn("x = (2.0, -1.0, 0.5)", no_ai)
        self.assertIn("(-0.5, 1.5, 0.0)", no_ai)
        self.assertIn("v = (0.0, 4.0)", no_ai)
        self.assertIn("d_model=8", no_ai)
        self.assertNotIn("1.5, 0.5, 0.5", no_ai)
        self.assertNotIn("5.199624", no_ai)
        self.assertEqual(CORE.TRANSFER_X, (2.0, -1.0, 0.5))
        self.assertEqual(CORE.TRANSFER_LN, (0.0, 4.0))
        self.assertEqual(CORE.TRANSFER_SHAPES["n_heads"], 4)

    def test_status_does_not_claim_repository_executable(self):
        status = (MISSION / "status.yaml").read_text(encoding="utf-8")
        self.assertIn("does not mark M30 repository-executable", status)
        self.assertIn("intentionally_unpopulated", status)


@unittest.skipUnless(RUNTIME_DEPS, "install requirements/m30.txt to run NumPy-dependent M30 tests")
class M30RuntimeTests(unittest.TestCase):
    def test_independent_residual_arithmetic_is_elementwise_add(self):
        expected = independent_add(CORE.RESIDUAL_STREAM, CORE.RESIDUAL_SUBLAYER)
        self.assertEqual(expected, CORE.RESIDUAL_SUM)
        self.assertEqual(expected, (1.0, 1.0, 1.0))
        added = CORE.residual_add(CORE.RESIDUAL_STREAM, CORE.RESIDUAL_SUBLAYER)
        for left, right in zip(CORE.nested_tuples(added), expected, strict=True):
            self.assertAlmostEqual(left, right, places=12)
        with self.assertRaises(ValueError):
            CORE.residual_add((1.0, 0.0), (1.0, 0.0, 0.0))

    def test_independent_layer_norm_matches_eps_aware_hand_arithmetic(self):
        expected = independent_layer_norm(CORE.LN_VECTOR, eps=CORE.LN_EPS)
        scale = math.sqrt(CORE.LN_VAR + CORE.LN_EPS)
        self.assertAlmostEqual(expected[0], 1.0 / scale, places=12)
        self.assertAlmostEqual(expected[1], -1.0 / scale, places=12)
        computed = CORE.layer_norm(CORE.LN_VECTOR)
        for left, right in zip(CORE.nested_tuples(computed), expected, strict=True):
            self.assertAlmostEqual(left, right, places=9)
        limit = CORE.LN_NORMALIZED
        self.assertAlmostEqual(limit[0], 1.0, places=12)
        self.assertLess(abs(float(computed.reshape(-1)[0]) - 1.0), 2e-5)

    def test_independent_relu_ffn_is_position_wise(self):
        hidden = independent_relu(CORE.FFN_X)
        self.assertEqual(hidden, CORE.FFN_HIDDEN)
        _pre, computed_hidden, computed_out = CORE.feed_forward(
            CORE.FFN_X, CORE.FFN_W1, CORE.FFN_B1, CORE.FFN_W2, CORE.FFN_B2
        )
        self.assertAlmostEqual(float(computed_hidden.reshape(-1)[0]), 1.0, places=12)
        self.assertAlmostEqual(float(computed_hidden.reshape(-1)[1]), 0.0, places=12)
        self.assertAlmostEqual(float(computed_out.reshape(-1)[0]), CORE.FFN_OUT[0], places=12)
        self.assertAlmostEqual(float(computed_out.reshape(-1)[1]), CORE.FFN_OUT[1], places=12)

    def test_split_merge_roundtrip_and_shape_contract(self):
        np = CORE._require_numpy()
        batch, seq, d_model, n_heads = 2, 5, 8, 4
        tensor = np.arange(batch * seq * d_model, dtype=float).reshape(batch, seq, d_model)
        heads = CORE.split_heads(tensor, n_heads)
        self.assertEqual(heads.shape, (batch, seq, n_heads, d_model // n_heads))
        merged = CORE.merge_heads(heads)
        self.assertTrue(np.allclose(merged, tensor))
        with self.assertRaises(ValueError):
            CORE.split_heads(tensor, 3)

    def test_raw_identity_head0_matches_m29_cash_attention(self):
        m29 = CORE._load_m29()
        m29_cash = m29.self_attention(m29.X_CASH_CONTEXT)
        mha = CORE.multi_head_attention(CORE.X_CASH_CONTEXT, CORE.teaching_params())
        np = CORE._require_numpy()
        self.assertEqual(mha.shapes["q_heads"], (1, 3, 2, 2))
        self.assertEqual(mha.shapes["attn_concat"], (1, 3, 4))
        self.assertTrue(np.allclose(mha.head_weights[0, 0], m29_cash.weights[0]))
        self.assertTrue(np.allclose(mha.head_outputs[0, :, 0, :], m29_cash.output[0]))
        self.assertTrue(np.allclose(mha.head_weights[0, 0, CORE.BANK_INDEX], 1.0 / 3.0))
        self.assertTrue(np.allclose(mha.attn_concat, mha.attn_projected))
        self.assertIn("scaled_dot_product_attention", CORE.multi_head_attention.__doc__ or "")

    def test_head1_is_a_parallel_projection_not_a_copy_of_head0(self):
        mha = CORE.multi_head_attention(CORE.X_CASH_CONTEXT, CORE.teaching_params())
        np = CORE._require_numpy()
        self.assertFalse(np.allclose(mha.head_weights[0, 0], mha.head_weights[0, 1]))
        self.assertFalse(np.allclose(mha.q_heads[0, :, 0, :], mha.q_heads[0, :, 1, :]))
        merged = CORE.merge_heads(mha.head_outputs)
        self.assertTrue(np.allclose(merged, mha.attn_concat))
        slice0 = mha.head_slice(0)
        self.assertEqual(slice0["output"].shape, (1, 3, 2))

    def test_pre_norm_block_shapes_and_independent_residual_add(self):
        trace = CORE.transformer_block(CORE.X_CASH_CONTEXT, CORE.teaching_params())
        np = CORE._require_numpy()
        self.assertEqual(trace.shapes["x"], (1, 3, 4))
        self.assertEqual(trace.shapes["q_heads"], (1, 3, 2, 2))
        self.assertEqual(trace.shapes["ffn_hidden"], (1, 3, 8))
        self.assertEqual(trace.shapes["output"], (1, 3, 4))
        self.assertEqual(trace.declared_convention, "pre_norm")
        self.assertEqual(trace.defect, "none")
        independent_attn = np.asarray(trace.x) + np.asarray(trace.attn_projected)
        independent_ffn = np.asarray(trace.attn_residual) + np.asarray(trace.ffn_projected)
        self.assertTrue(np.allclose(independent_attn, trace.attn_add))
        self.assertTrue(np.allclose(independent_attn, trace.attn_residual))
        self.assertTrue(np.allclose(independent_ffn, trace.ffn_add))
        self.assertTrue(np.allclose(independent_ffn, trace.output))
        self.assertFalse(np.allclose(trace.attn_add, trace.attn_norm + trace.attn_projected))

    def test_independent_compose_and_frozen_golden_match_the_block(self):
        params = CORE.teaching_params()
        trace = CORE.golden_pre_norm_trace()
        composed = CORE.independent_pre_norm_compose(CORE.X_CASH_CONTEXT, params)
        np = CORE._require_numpy()
        for name in (
            "attn_norm",
            "attn_projected",
            "attn_add",
            "attn_residual",
            "ffn_hidden",
            "ffn_projected",
            "output",
        ):
            with self.subTest(name=name):
                self.assertTrue(np.allclose(trace.checkpoint(name), composed[name]))
        self.assertTrue(np.allclose(trace.attn_residual[0], CORE.GOLDEN_CASH_ATTN_RESIDUAL, atol=1e-9))
        self.assertTrue(np.allclose(trace.output[0], CORE.GOLDEN_CASH_OUTPUT, atol=1e-9))
        self.assertIsNone(CORE.first_divergence(trace, CORE.reference_pre_norm_block(CORE.X_CASH_CONTEXT)))

    def test_context_change_moves_block_output_with_params_fixed(self):
        params = CORE.teaching_params()
        cash = CORE.transformer_block(CORE.X_CASH_CONTEXT, params, tokens=CORE.CASH_TOKENS)
        water = CORE.transformer_block(CORE.X_WATER_CONTEXT, params, tokens=CORE.WATER_TOKENS)
        batch = CORE.transformer_block(CORE.teaching_batch(), params)
        np = CORE._require_numpy()
        self.assertTrue(np.allclose(cash.x[0, CORE.BANK_INDEX], water.x[0, CORE.BANK_INDEX]))
        self.assertFalse(np.allclose(cash.output[0, CORE.BANK_INDEX], water.output[0, CORE.BANK_INDEX]))
        self.assertEqual(batch.shapes["output"], (2, 3, 4))
        self.assertTrue(np.allclose(batch.output[0], cash.output[0]))
        self.assertTrue(np.allclose(batch.output[1], water.output[0]))

    def test_output_projection_is_the_named_change_after_merge(self):
        identity = CORE.transformer_block(CORE.X_CASH_CONTEXT, CORE.teaching_params())
        half = CORE.transformer_block(
            CORE.X_CASH_CONTEXT,
            CORE.params_with_output_projection(CORE.teaching_params(), CORE.TEACHING_W_O_HALF),
        )
        np = CORE._require_numpy()
        self.assertTrue(np.allclose(identity.attn_concat, half.attn_concat))
        self.assertTrue(np.allclose(half.attn_projected, 0.5 * half.attn_concat))
        self.assertEqual(CORE.first_divergence(identity, half), "attn_projected")

    def test_residual_ablation_drops_one_add_with_params_fixed(self):
        params = CORE.teaching_params()
        healthy = CORE.transformer_block(CORE.X_CASH_CONTEXT, params)
        skipped = CORE.ablate_residual(CORE.X_CASH_CONTEXT, params, which="attn")
        np = CORE._require_numpy()
        self.assertEqual(skipped.skip_residual, "attn")
        self.assertTrue(np.allclose(skipped.attn_projected, healthy.attn_projected))
        self.assertTrue(np.allclose(skipped.attn_add, skipped.attn_projected))
        self.assertEqual(CORE.first_divergence(healthy, skipped), "attn_add")
        skipped_ffn = CORE.ablate_residual(CORE.X_CASH_CONTEXT, params, which="ffn")
        self.assertEqual(CORE.first_divergence(healthy, skipped_ffn), "ffn_add")
        self.assertTrue(np.allclose(skipped_ffn.attn_residual, healthy.attn_residual))

    def test_labeled_post_norm_differs_and_is_not_a_defect(self):
        params = CORE.teaching_params()
        pre = CORE.transformer_block(CORE.X_CASH_CONTEXT, params, convention="pre_norm")
        post = CORE.transformer_block(CORE.X_CASH_CONTEXT, params, convention="post_norm")
        np = CORE._require_numpy()
        self.assertEqual(post.declared_convention, "post_norm")
        self.assertEqual(post.defect, "none")
        self.assertEqual(CORE.first_divergence(pre, post), "attn_norm")
        self.assertTrue(np.allclose(post.attn_norm, post.x))
        self.assertFalse(np.allclose(post.attn_residual, post.attn_add))
        self.assertFalse(np.allclose(pre.output, post.output))

    def test_ffn_activation_change_keeps_attention_residual_fixed(self):
        relu = CORE.transformer_block(CORE.X_CASH_CONTEXT, CORE.teaching_params())
        identity = CORE.transformer_block(
            CORE.X_CASH_CONTEXT,
            CORE.params_with_activation(CORE.teaching_params(), "identity"),
        )
        np = CORE._require_numpy()
        self.assertTrue(np.allclose(relu.attn_residual, identity.attn_residual))
        self.assertEqual(CORE.first_divergence(relu, identity), "ffn_hidden")
        self.assertFalse(np.allclose(relu.ffn_hidden, identity.ffn_hidden))

    def test_residual_wrong_branch_first_diverges_at_attn_add(self):
        params = CORE.teaching_params()
        golden = CORE.transformer_block(CORE.X_CASH_CONTEXT, params)
        broken = CORE.block_with_defect(
            CORE.X_CASH_CONTEXT, params, defect="residual_wrong_branch"
        )
        np = CORE._require_numpy()
        self.assertEqual(broken.defect, "residual_wrong_branch")
        self.assertEqual(CORE.first_divergence(golden, broken), "attn_add")
        self.assertTrue(np.allclose(broken.attn_projected, golden.attn_projected))
        self.assertTrue(np.allclose(broken.attn_norm, golden.attn_norm))
        self.assertTrue(np.allclose(broken.attn_add, broken.attn_norm + broken.attn_projected))
        self.assertFalse(np.allclose(broken.attn_add, broken.x + broken.attn_projected))

    def test_norm_wrong_boundary_first_diverges_at_attn_norm(self):
        params = CORE.teaching_params()
        golden = CORE.transformer_block(CORE.X_CASH_CONTEXT, params)
        broken = CORE.block_with_defect(
            CORE.X_CASH_CONTEXT, params, defect="norm_wrong_boundary"
        )
        np = CORE._require_numpy()
        self.assertEqual(broken.declared_convention, "pre_norm")
        self.assertEqual(CORE.first_divergence(golden, broken), "attn_norm")
        self.assertTrue(np.allclose(broken.attn_norm, broken.x))
        self.assertFalse(np.allclose(broken.attn_norm, golden.attn_norm))
        self.assertTrue(np.allclose(broken.attn_add, broken.x + broken.attn_projected))

    def test_repair_uses_broken_trace_objects_and_regression_still_fails(self):
        params = CORE.teaching_params()
        golden = CORE.transformer_block(CORE.X_CASH_CONTEXT, params)
        broken = CORE.block_with_defect(
            CORE.X_CASH_CONTEXT, params, defect="residual_wrong_branch"
        )
        repaired = CORE.repair_block(broken)
        np = CORE._require_numpy()
        self.assertEqual(repaired.defect, "none")
        self.assertIsNone(CORE.first_divergence(golden, repaired))
        self.assertTrue(np.allclose(repaired.x, broken.x))
        self.assertTrue(np.allclose(repaired.params.w_q, broken.params.w_q))
        still = CORE.block_with_defect(
            broken.x, broken.params, defect="residual_wrong_branch"
        )
        self.assertEqual(CORE.first_divergence(golden, still), "attn_add")
        broken_norm = CORE.block_with_defect(
            CORE.X_CASH_CONTEXT, params, defect="norm_wrong_boundary"
        )
        repaired_norm = CORE.repair_block(broken_norm)
        self.assertIsNone(CORE.first_divergence(golden, repaired_norm))
        source = CORE.repair_block.__doc__ or ""
        self.assertIn("defective trace", source)

    def test_causal_mask_reaches_m29_and_zeros_future_head_mass(self):
        m29 = CORE._load_m29()
        mask = m29.causal_additive_mask(3)
        trace = CORE.transformer_block(
            CORE.X_CASH_CONTEXT, CORE.teaching_params(), mask=mask
        )
        np = CORE._require_numpy()
        future = trace.head_weights[0, :, 0, 1] + trace.head_weights[0, :, 0, 2]
        future = future + trace.head_weights[0, :, 1, 2]
        self.assertTrue(np.allclose(future, 0.0))
        self.assertAlmostEqual(float(trace.head_weights[0, 0, 0, 0]), 1.0, places=9)

    def test_layer_norm_is_per_position_over_features(self):
        np = CORE._require_numpy()
        x = np.asarray(CORE.X_CASH_CONTEXT, dtype=float)
        normalized = CORE.layer_norm(x, CORE.TEACHING_LN_GAMMA, CORE.TEACHING_LN_BETA)
        means = normalized.mean(axis=-1)
        self.assertTrue(np.allclose(means, 0.0, atol=1e-7))
        bank_independent = independent_layer_norm(tuple(float(v) for v in x[CORE.BANK_INDEX]))
        for left, right in zip(normalized[CORE.BANK_INDEX], bank_independent, strict=True):
            self.assertAlmostEqual(float(left), right, places=9)

    def test_observability_report_states_limits_and_m31_handoff(self):
        trace = CORE.transformer_block(
            CORE.X_CASH_CONTEXT, CORE.teaching_params(), tokens=CORE.CASH_TOKENS
        )
        report = CORE.observability_report(trace)
        self.assertEqual(report["declared_convention"], "pre_norm")
        self.assertEqual(report["version"], CORE.BLOCK_VERSION)
        self.assertEqual(report["checkpoints"], CORE.TRACE_CHECKPOINTS)
        self.assertIn("parallel learned projections", report["head_interpretation_limit"])
        self.assertIn("stream + sublayer", report["residual_interpretation_limit"])
        self.assertIn("architecture convention", report["norm_interpretation_limit"])
        self.assertIn("does not receive a training loop", report["handoff"])

    def test_unsupported_defect_and_mixed_skip_are_rejected(self):
        with self.assertRaises(ValueError):
            CORE.block_with_defect(CORE.X_CASH_CONTEXT, defect="n_head")
        with self.assertRaises(ValueError):
            CORE.transformer_block(
                CORE.X_CASH_CONTEXT,
                defect="residual_wrong_branch",
                skip_residual="attn",
            )
        with self.assertRaises(ValueError):
            CORE.transformer_block(CORE.X_CASH_CONTEXT, convention="rmsnorm")
        with self.assertRaises(ValueError):
            CORE.transformer_block(
                CORE.X_CASH_CONTEXT,
                convention="post_norm",
                defect="residual_wrong_branch",
            )
        with self.assertRaises(ValueError):
            CORE.transformer_block(
                CORE.X_CASH_CONTEXT,
                convention="post_norm",
                defect="norm_wrong_boundary",
            )

    def test_transfer_fixture_is_hand_computable_without_spoiling_the_gate(self):
        residual = independent_add(CORE.TRANSFER_X, CORE.TRANSFER_SUBLAYER)
        self.assertEqual(residual, (1.5, 0.5, 0.5))
        ln = independent_layer_norm(CORE.TRANSFER_LN, eps=0.0)
        self.assertAlmostEqual(ln[0], -1.0, places=12)
        self.assertAlmostEqual(ln[1], 1.0, places=12)
        shapes = CORE.TRANSFER_SHAPES
        self.assertEqual(
            (shapes["batch"], shapes["seq"], shapes["n_heads"], shapes["d_head"]),
            (2, 5, 4, 2),
        )
        self.assertEqual(shapes["batch"] * shapes["seq"] * shapes["d_model"], 2 * 5 * 8)
        no_ai = (MISSION / "no_ai_gate.md").read_text(encoding="utf-8")
        self.assertNotIn("1.5, 0.5, 0.5", no_ai)
        self.assertNotIn("(-1.0, 1.0)", no_ai)


if __name__ == "__main__":
    unittest.main()
