from __future__ import annotations

import ast
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MISSION = ROOT / "missions" / "M32"
NOTEBOOK = ROOT / "labs" / "M32_inference_adaptation.ipynb"
REQUIREMENTS = ROOT / "requirements" / "m32.txt"


def load_core():
    spec = importlib.util.spec_from_file_location(
        "m32_inference_adaptation", MISSION / "inference_adaptation.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load M32 inference adaptation core")
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


def independent_greedy(logits) -> int:
    values = tuple(float(value) for value in logits)
    best = max(values)
    for index, value in enumerate(values):
        if value == best:
            return index
    raise AssertionError("argmax failed")


def independent_softmax(logits, temperature: float = 1.0) -> list[float]:
    scaled = [float(value) / float(temperature) for value in logits]
    peak = max(scaled)
    shifted = [math.exp(value - peak) for value in scaled]
    total = sum(shifted)
    return [mass / total for mass in shifted]


def independent_entropy(probs) -> float:
    return -sum(mass * math.log(mass) for mass in probs if mass > 0.0)


class M32StaticContractTests(unittest.TestCase):
    def test_required_executable_artifacts_exist(self):
        for path in (
            MISSION / "inference_adaptation.py",
            NOTEBOOK,
            REQUIREMENTS,
            ROOT / "tests" / "test_m32.py",
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
            "inference_adaptation.py",
        }
        actual = {path.name for path in MISSION.iterdir() if path.is_file()}
        self.assertTrue(required <= actual, required - actual)
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        for name in required:
            self.assertIn(f"missions/M32/{name}", manifest)

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
            "vectorindex",
            "retrieve_and_generate",
            "execute_tool",
            "context_pack",
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
        self.assertIn("M32", source)
        self.assertIn("inference_adaptation.py", source)
        self.assertIn("sys.path", source)
        self.assertIn("from missions.M32.inference_adaptation import", source)
        self.assertIn("from missions.M31.llm_training_core import", source)

    def test_future_mission_boundary_stays_closed_in_code(self):
        source = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        for forbidden in (
            "from_pretrained",
            "FastAPI",
            "VectorIndex",
            "retrieve_and_generate",
            "execute_tool",
            "context_pack",
            "cite_sources",
            "abstain",
            "QdrantClient",
            "sentence_transformers",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_notebook_enforces_prediction_before_every_action(self):
        cells = notebook_cells()
        positions = {str(cell["id"]): index for index, cell in enumerate(cells)}
        pairs = (
            ("predict-whole", "run-whole"),
            ("predict-greedy", "run-greedy"),
            ("predict-temperature", "run-temperature"),
            ("predict-filters", "run-filters"),
            ("predict-seed", "run-seed"),
            ("predict-config", "run-config"),
            ("predict-stop", "run-stop"),
            ("predict-prompt", "run-prompt"),
            ("predict-adaptation", "run-adaptation"),
            ("predict-code-reading", "run-code-reading"),
            ("predict-failure-uncontrolled", "run-failure-uncontrolled"),
            ("predict-failure-adaptation", "run-failure-adaptation"),
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
            "M31 → M32",
            "training-time",
            "not a production",
            "InferenceConfig",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, markdown)

        predict_uncontrolled = cell_source(cells[positions["predict-failure-uncontrolled"]])
        predict_adapt = cell_source(cells[positions["predict-failure-adaptation"]])
        self.assertNotIn('defect="uncontrolled_settings"', predict_uncontrolled)
        self.assertNotIn("1.7", predict_uncontrolled)
        self.assertNotIn("3203", predict_uncontrolled)
        self.assertNotIn('defect="wrong_adaptation"', predict_adapt)
        self.assertNotIn("retrieval", predict_adapt.lower())

        predict_code = cell_source(cells[positions["predict-code-reading"]])
        predict_bullets = predict_code.split("Predict:", 1)[-1]
        self.assertIn("prepare_distribution", predict_bullets)
        self.assertIn("repair_run", predict_bullets)
        self.assertIn("run_inference", predict_bullets)
        self.assertNotIn("optional_live_complete", predict_bullets)
        self.assertNotIn("compared as a model change", predict_code)

    def test_notebook_prints_required_inference_evidence(self):
        code = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        for token in (
            "greedy_token",
            "softmax_probs",
            "apply_top_k",
            "apply_top_p",
            "sample_once",
            "run_inference",
            "pipeline_with_defect",
            "repair_run",
            "first_divergence",
            "decide_scenario",
            "observability_report",
            "config_as_evidence",
            "GREEDY_LOGITS",
            "TEMP_LOGITS",
            "FILTER_LOGITS",
            "SYSTEM_MAP",
            "ADAPTATION_HIERARCHY",
            "SCALE_LIMIT",
            'defect="uncontrolled_settings"',
            'defect="wrong_adaptation"',
            "optional_live_complete",
            "compare_outputs_naive",
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
        self.assertNotIn("openai", joined)
        self.assertNotIn("qdrant", joined)

    def test_inference_core_top_level_imports_are_stdlib(self):
        source = (MISSION / "inference_adaptation.py").read_text(encoding="utf-8")
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
            "random",
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
        self.assertIn("_load_m31", source)
        self.assertIn("deferred", source.lower())

    def test_core_consumes_m31_and_does_not_open_search_or_tools(self):
        source = (MISSION / "inference_adaptation.py").read_text(encoding="utf-8")
        self.assertIn("attach_m31_checkpoint", source)
        self.assertIn("StageAwareCheckpoint", source)
        self.assertIn("token selection", source.lower())
        self.assertNotIn("from_pretrained", source)
        self.assertNotIn("VectorIndex", source)
        self.assertNotIn("QdrantClient", source)
        self.assertNotIn("def execute_tool", source)
        self.assertNotIn("retrieve_and_generate", source)
        self.assertIn("decision only", source)
        self.assertIn("LiveAdapterUnavailable", source)

    def test_independent_greedy_arithmetic_is_locked(self):
        self.assertEqual(CORE.GREEDY_LOGITS, (1.0, 3.0, 2.0, 0.0))
        self.assertEqual(independent_greedy(CORE.GREEDY_LOGITS), 1)
        self.assertEqual(CORE.greedy_token(CORE.GREEDY_LOGITS), 1)
        self.assertEqual(CORE.GREEDY_INDEX, 1)
        self.assertEqual(independent_greedy(CORE.TIE_LOGITS), 0)
        self.assertEqual(CORE.greedy_token(CORE.TIE_LOGITS), 0)
        scaled = tuple(value / 2.0 for value in CORE.GREEDY_LOGITS)
        self.assertEqual(CORE.greedy_token(scaled), CORE.greedy_token(CORE.GREEDY_LOGITS))

    def test_independent_softmax_temperature_arithmetic_is_locked(self):
        p1 = independent_softmax(CORE.TEMP_LOGITS, 1.0)
        self.assertAlmostEqual(p1[0], 0.75, places=12)
        self.assertAlmostEqual(p1[1], 0.25, places=12)
        core_p1 = CORE.softmax_probs(CORE.TEMP_LOGITS, 1.0)
        self.assertAlmostEqual(core_p1[0], 0.75, places=12)
        self.assertAlmostEqual(core_p1[1], 0.25, places=12)

        p05 = independent_softmax(CORE.TEMP_LOGITS, 0.5)
        self.assertAlmostEqual(p05[0], 0.9, places=12)
        self.assertAlmostEqual(CORE.softmax_probs(CORE.TEMP_LOGITS, 0.5)[0], 0.9, places=12)

        p2 = independent_softmax(CORE.TEMP_LOGITS, 2.0)
        expected = math.sqrt(3.0) / (math.sqrt(3.0) + 1.0)
        self.assertAlmostEqual(p2[0], expected, places=12)
        self.assertAlmostEqual(CORE.softmax_probs(CORE.TEMP_LOGITS, 2.0)[0], expected, places=12)

        self.assertLess(independent_entropy(p05), independent_entropy(p1))
        self.assertLess(independent_entropy(p1), independent_entropy(p2))
        self.assertLess(CORE.entropy(p05), CORE.entropy(tuple(p1)))
        self.assertEqual(CORE.greedy_token(CORE.TEMP_LOGITS), 0)
        with self.assertRaises(ValueError):
            CORE.softmax_probs(CORE.TEMP_LOGITS, 0.0)

    def test_independent_top_k_and_top_p_arithmetic_is_locked(self):
        probs = independent_softmax(CORE.FILTER_LOGITS, 1.0)
        self.assertEqual(CORE.FILTER_COUNTS, (10, 6, 3, 1))
        self.assertAlmostEqual(probs[0], 0.5, places=12)
        self.assertAlmostEqual(probs[1], 0.3, places=12)
        self.assertAlmostEqual(probs[2], 0.15, places=12)
        self.assertAlmostEqual(probs[3], 0.05, places=12)
        self.assertAlmostEqual(CORE.softmax_probs(CORE.FILTER_LOGITS)[0], 0.5, places=12)

        _, topk = CORE.prepare_distribution(CORE.FILTER_LOGITS, top_k=2)
        self.assertAlmostEqual(topk[0], 0.625, places=12)
        self.assertAlmostEqual(topk[1], 0.375, places=12)
        self.assertEqual(topk[2], 0.0)
        self.assertEqual(topk[3], 0.0)

        _, keep_one = CORE.prepare_distribution(CORE.FILTER_LOGITS, top_p=0.50)
        self.assertAlmostEqual(keep_one[0], 1.0, places=12)
        self.assertEqual(keep_one[1], 0.0)

        _, keep_three = CORE.prepare_distribution(CORE.FILTER_LOGITS, top_p=0.81)
        self.assertGreater(keep_three[2], 0.0)
        self.assertEqual(keep_three[3], 0.0)
        self.assertNotEqual(topk[2], keep_three[2])
        masked = CORE.apply_top_k(CORE.FILTER_LOGITS, 2)
        self.assertTrue(math.isinf(masked[2]) and masked[2] < 0.0)

    def test_seed_replay_is_deterministic_and_can_differ(self):
        once = CORE.sample_once(CORE.FILTER_LOGITS, seed=CORE.SEED)
        replay = CORE.sample_once(CORE.FILTER_LOGITS, seed=CORE.SEED)
        other = CORE.sample_once(CORE.FILTER_LOGITS, seed=CORE.SEED_OTHER)
        self.assertEqual(once.token_id, replay.token_id)
        self.assertEqual(once.token_id, 1)
        self.assertEqual(other.token_id, 2)
        self.assertNotEqual(once.token_id, other.token_id)

    def test_prompt_stop_and_config_are_one_named_change_each(self):
        greedy_a = CORE.run_inference(CORE.PROMPT_A, CORE.make_config(prompt_ids=CORE.PROMPT_A, do_sample=False))
        greedy_b = CORE.run_inference(CORE.PROMPT_B, CORE.make_config(prompt_ids=CORE.PROMPT_B, do_sample=False))
        short = CORE.run_inference(
            CORE.PROMPT_A,
            CORE.make_config(prompt_ids=CORE.PROMPT_A, do_sample=False, max_tokens=2),
        )
        self.assertEqual(greedy_a.generated_ids, (1, 2, 3))
        self.assertEqual(greedy_a.stop_reason, "stop_token")
        self.assertEqual(greedy_b.generated_ids, (2, 3))
        self.assertEqual(short.generated_ids, (1, 2))
        self.assertEqual(short.stop_reason, "max_tokens")
        self.assertEqual(CORE.first_divergence(greedy_a, greedy_b), "prompt_ids")
        self.assertEqual(CORE.first_divergence(greedy_a, short), "max_tokens")
        self.assertEqual(greedy_a.config.temperature, greedy_b.config.temperature)
        self.assertEqual(greedy_a.config.seed, short.config.seed)
        evidence = CORE.config_as_evidence(greedy_a.config)
        self.assertEqual(evidence["fingerprint"], CORE.config_fingerprint(greedy_a.config))
        self.assertFalse(evidence["training_time"])
        self.assertIn("checkpoint_id", evidence)

    def test_adaptation_rubric_distinguishes_freshness_from_weights(self):
        self.assertEqual(CORE.decide_scenario("email_tone").chosen_route, "prompt")
        self.assertEqual(CORE.decide_scenario("vendor_policy").chosen_route, "retrieval")
        self.assertEqual(CORE.decide_scenario("invoice_vat").chosen_route, "tools")
        self.assertEqual(CORE.decide_scenario("always_refuse_competitor_praise").chosen_route, "parameters")
        self.assertEqual(CORE.decide_scenario("stale_site_hours").chosen_route, "retrieval")
        self.assertEqual(CORE.decide_adaptation(CORE.TRANSFER_MENU_SIGNALS), "retrieval")
        self.assertEqual(CORE.ADAPTATION_ROUTES, ("prompt", "retrieval", "tools", "parameters"))
        with self.assertRaises(ValueError):
            CORE.decide_adaptation({"freshness": True, "unknown_flag": True})

    def test_m31_checkpoint_is_consumed_not_retrained(self):
        from missions.M31.llm_training_core import StageAwareCheckpoint

        checkpoint = CORE.attach_m31_checkpoint()
        self.assertIsInstance(checkpoint, StageAwareCheckpoint)
        self.assertEqual(type(checkpoint).__module__, "missions.M31.llm_training_core")
        self.assertFalse(checkpoint.training_time)
        self.assertTrue(checkpoint.inference_ready)
        self.assertEqual(checkpoint.version, "v07-teaching-lm-1")
        self.assertEqual(checkpoint.adaptation_stage, "pretrained")
        self.assertIn("token selection is M32", checkpoint.audit["training_time_boundary"])
        self.assertIn("token selection is M32", CORE.training_time_boundary())
        with self.assertRaises(ValueError):
            CORE.InferenceConfig(
                checkpoint_id="x",
                prompt_ids=CORE.PROMPT_A,
                training_time=True,
            )

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
        self.assertIn("log(6)", no_ai)
        self.assertIn("log(3)", no_ai)
        self.assertIn("cafeteria menu", no_ai)
        self.assertEqual(CORE.TRANSFER_LOGITS[0], math.log(6.0))
        self.assertEqual(CORE.TRANSFER_GREEDY_INDEX, 0)
        self.assertNotIn("(0.6, 0.3, 0.1)", no_ai)
        self.assertNotIn("0.666", no_ai)
        self.assertNotIn("the answer is retrieval", no_ai.lower())

    def test_status_does_not_claim_repository_executable(self):
        status = (MISSION / "status.yaml").read_text(encoding="utf-8")
        self.assertIn("does not mark M32 repository-executable", status)
        self.assertIn("intentionally_unpopulated", status)

    def test_live_adapter_is_optional_and_fail_closed(self):
        cfg = CORE.make_config()
        with self.assertRaises(CORE.LiveAdapterUnavailable):
            CORE.optional_live_complete("hello", cfg)
        source = (MISSION / "inference_adaptation.py").read_text(encoding="utf-8")
        self.assertNotIn("openai", source.lower())
        self.assertIn("not required", CORE.optional_live_complete.__doc__.lower())

    def test_uncontrolled_defect_first_diverges_at_temperature(self):
        healthy = CORE.run_inference(CORE.PROMPT_A, CORE.make_config(prompt_ids=CORE.PROMPT_A, do_sample=True))
        broken = CORE.pipeline_with_defect(defect="uncontrolled_settings")
        self.assertEqual(broken.defect, "uncontrolled_settings")
        self.assertEqual(broken.claim, "model_changed")
        self.assertEqual(broken.mismatched_fields, ("temperature", "seed"))
        self.assertEqual(CORE.first_divergence(broken.left, broken.right), "temperature")
        self.assertEqual(broken.left.checkpoint_id, broken.right.checkpoint_id)
        self.assertEqual(broken.left.prompt_ids, broken.right.prompt_ids)
        self.assertEqual(broken.audit["naive_compare"], "model_changed")
        self.assertEqual(broken.audit["controlled_compare"], "uncontrolled_settings")
        self.assertEqual(healthy.checkpoint_id, broken.left.checkpoint_id)

    def test_wrong_adaptation_defect_proposes_parameters_for_freshness(self):
        broken = CORE.pipeline_with_defect(defect="wrong_adaptation")
        self.assertEqual(broken.defect, "wrong_adaptation")
        self.assertEqual(broken.decision.chosen_route, "parameters")
        self.assertEqual(broken.decision.case_id, "stale_site_hours")
        self.assertTrue(broken.decision.signals["freshness"])
        self.assertEqual(broken.audit["rubric_would_choose"], "retrieval")

    def test_repair_uses_broken_trace_objects_and_regression_still_fails(self):
        broken_settings = CORE.pipeline_with_defect(defect="uncontrolled_settings")
        repaired_settings = CORE.repair_run(broken_settings)
        self.assertEqual(repaired_settings.defect, "none")
        self.assertEqual(repaired_settings.left.config, broken_settings.reference_config)
        self.assertEqual(repaired_settings.left.generated_ids, repaired_settings.right.generated_ids)
        self.assertEqual(repaired_settings.left.config.seed, broken_settings.reference_config.seed)
        self.assertEqual(CORE.first_divergence(broken_settings.left, broken_settings.right), "temperature")
        self.assertNotEqual(broken_settings.left.config.seed, broken_settings.right.config.seed)

        broken_adapt = CORE.pipeline_with_defect(defect="wrong_adaptation")
        repaired_adapt = CORE.repair_run(broken_adapt)
        self.assertEqual(repaired_adapt.decision.chosen_route, "retrieval")
        self.assertEqual(repaired_adapt.decision.signals, broken_adapt.decision.signals)
        self.assertEqual(broken_adapt.decision.chosen_route, "parameters")
        source = CORE.repair_run.__doc__ or ""
        self.assertIn("defective", source)

    def test_observability_report_states_limits_and_later_handoff(self):
        trace = CORE.run_inference(CORE.PROMPT_A, CORE.make_config(prompt_ids=CORE.PROMPT_A))
        report = CORE.observability_report(trace)
        self.assertEqual(report["version"], CORE.INFERENCE_VERSION)
        self.assertFalse(report["training_time"])
        self.assertIn("do not receive a search index", report["handoff"])
        self.assertIn("not a production decoder", report["scale_limit"].lower() or CORE.SCALE_LIMIT.lower())
        self.assertIn("token selection is M32", report["training_time_boundary"])

    def test_unsupported_defect_and_live_none_repair_are_rejected(self):
        with self.assertRaises(ValueError):
            CORE.pipeline_with_defect(defect="none")
        with self.assertRaises(ValueError):
            CORE.pipeline_with_defect(defect="temperature")
        healthy = CORE.run_inference(CORE.PROMPT_A, CORE.make_config())
        with self.assertRaises(ValueError):
            CORE.repair_run(
                CORE.FailureTrace(defect="none", claim="ok", left=healthy, right=healthy)
            )

    def test_transfer_fixture_is_hand_computable_without_spoiling_the_gate(self):
        probs = independent_softmax(CORE.TRANSFER_LOGITS, 1.0)
        self.assertAlmostEqual(probs[0], 0.6, places=12)
        self.assertAlmostEqual(probs[1], 0.3, places=12)
        self.assertAlmostEqual(probs[2], 0.1, places=12)
        self.assertEqual(independent_greedy(CORE.TRANSFER_LOGITS), 0)
        self.assertAlmostEqual(CORE.softmax_probs(CORE.TRANSFER_LOGITS)[0], 0.6, places=12)
        no_ai = (MISSION / "no_ai_gate.md").read_text(encoding="utf-8")
        self.assertNotIn("(0.6, 0.3, 0.1)", no_ai)
        self.assertNotIn(f"{probs[0]:.6f}", no_ai)


@unittest.skipUnless(RUNTIME_DEPS, "install requirements/m32.txt to run NumPy-dependent M32 tests")
class M32RuntimeTests(unittest.TestCase):
    def test_numpy_softmax_matches_independent_temperature_arithmetic(self):
        for temperature in CORE.TEMPERATURES:
            with self.subTest(temperature=temperature):
                independent = independent_softmax(CORE.TEMP_LOGITS, temperature)
                numpy_probs = CORE.numpy_softmax(CORE.TEMP_LOGITS, temperature)
                self.assertEqual(len(numpy_probs), 2)
                self.assertAlmostEqual(float(numpy_probs[0]), independent[0], places=12)
                self.assertAlmostEqual(float(numpy_probs[1]), independent[1], places=12)

    def test_numpy_filter_softmax_matches_counts(self):
        numpy_probs = CORE.numpy_softmax(CORE.FILTER_LOGITS, 1.0)
        self.assertAlmostEqual(float(numpy_probs[0]), 0.5, places=12)
        self.assertAlmostEqual(float(numpy_probs[1]), 0.3, places=12)


if __name__ == "__main__":
    unittest.main()
