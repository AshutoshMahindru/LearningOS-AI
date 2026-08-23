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
MISSION = ROOT / "missions" / "M37"
NOTEBOOK = ROOT / "labs" / "M37_tool_calling.ipynb"
REQUIREMENTS = ROOT / "requirements" / "m37.txt"


def load_core():
    spec = importlib.util.spec_from_file_location(
        "m37_tool_runtime", MISSION / "tool_runtime.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load M37 tool runtime")
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


def independent_vat(amount: float, rate: float) -> tuple[float, float]:
    tax = float(amount) * float(rate)
    return tax, float(amount) + tax


def independent_issues(kind_set: set[str], result) -> set[str]:
    return {issue.kind for issue in result.issues} & kind_set


class M37StaticContractTests(unittest.TestCase):
    def test_required_executable_artifacts_exist(self):
        for path in (
            MISSION / "tool_runtime.py",
            MISSION / "optional_live_llm.py",
            NOTEBOOK,
            REQUIREMENTS,
            ROOT / "tests" / "test_m37.py",
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
            "tool_runtime.py",
            "optional_live_llm.py",
        }
        actual = {path.name for path in MISSION.iterdir() if path.is_file()}
        self.assertTrue(required <= actual, required - actual)
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        for name in required:
            self.assertIn(f"missions/M37/{name}", manifest)

    def test_yaml_is_well_formed(self):
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
        self.assertGreaterEqual(notebook.get("nbformat_minor", 0), 5)
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
            "from anthropic",
            "import anthropic",
            "from langgraph",
            "import langgraph",
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
            "stategraph",
            "memorysaver",
            "retrieve_and_generate",
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
        self.assertIn("M37", source)
        self.assertIn("tool_runtime.py", source)
        self.assertIn("sys.path", source)
        self.assertIn("from missions.M32.inference_adaptation import", source)
        self.assertIn("from missions.M37.tool_runtime import", source)

    def test_future_mission_boundary_stays_closed_in_code(self):
        source = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        for forbidden in (
            "from_pretrained",
            "FastAPI",
            "retrieve_and_generate",
            "QdrantClient",
            "StateGraph",
            "MemorySaver",
            "sentence_transformers",
            "optional_live_llm",
            "softmax_probs",
            "pack_context",
            "chunk_overlap",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_notebook_enforces_prediction_before_every_action(self):
        cells = notebook_cells()
        positions = {str(cell["id"]): index for index, cell in enumerate(cells)}
        pairs = (
            ("predict-whole", "run-whole"),
            ("predict-schema", "run-schema"),
            ("predict-fixtures", "run-fixtures"),
            ("predict-validation", "run-validation"),
            ("predict-results", "run-results"),
            ("predict-selection", "run-selection"),
            ("predict-idempotency", "run-idempotency"),
            ("predict-retry", "run-retry"),
            ("predict-tool-error", "run-tool-error"),
            ("predict-code-reading", "run-code-reading"),
            ("predict-failure", "run-failure"),
            ("predict-failure-duplicate", "run-failure-duplicate"),
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
            "M32 → M37",
            "not a production",
            "InferenceConfig",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, markdown)

        predict_malformed = cell_source(cells[positions["predict-failure"]])
        predict_dup = cell_source(cells[positions["predict-failure-duplicate"]])
        predict_repair = cell_source(cells[positions["predict-failure-repair"]])
        self.assertNotIn('defect="malformed_reaches_side_effect"', predict_malformed)
        self.assertNotIn("sixteen", predict_malformed)
        self.assertNotIn('defect="duplicate_side_effect"', predict_dup)
        self.assertNotIn("skip_idempotency", predict_dup)
        self.assertNotIn('defect="malformed_reaches_side_effect"', predict_repair)
        self.assertNotIn('defect="duplicate_side_effect"', predict_repair)

        predict_code = cell_source(cells[positions["predict-code-reading"]])
        predict_bullets = predict_code.split("Predict:", 1)[-1]
        self.assertIn("run_tool_call", predict_bullets)
        self.assertIn("validate_arguments", predict_bullets)
        self.assertIn("execute_tool", predict_bullets)
        self.assertIn("repair_run", predict_bullets)
        self.assertNotIn("optional_live_propose", predict_bullets)
        self.assertNotIn("execution_reached is False", predict_code)

        repair_src = cell_source(cells[positions["run-failure-repair"]])
        self.assertIn("repair_run", repair_src)
        self.assertIn("broken_malformed", repair_src)
        self.assertNotIn('defect="none"', repair_src)

        code_reading = cell_source(cells[positions["run-code-reading"]])
        self.assertIn("inspect.getsource(run_tool_call)", code_reading)
        self.assertNotIn('"idempotency" in', code_reading)
        self.assertNotIn("'idempotency' in", code_reading)
        self.assertIn("effect_count", code_reading)
        self.assertIn("retry_budget_remaining", code_reading)
        self.assertIn("error_type", code_reading)

    def test_notebook_prints_required_tool_evidence(self):
        code = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        for token in (
            "run_tool_call",
            "validate_arguments",
            "validate_proposal",
            "execute_tool",
            "evaluate_selection",
            "pipeline_with_defect",
            "repair_run",
            "observability_report",
            "SYSTEM_MAP",
            "SCALE_LIMIT",
            "VAT_AMOUNT",
            "VAT_RATE",
            "INVALID_FIXTURES",
            "sticky_invalid_repairer",
            "vat_fill_repairer",
            'defect="malformed_reaches_side_effect"',
            'defect="duplicate_side_effect"',
            "optional_live_propose",
            "weights_updated",
            "execution_reached",
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
        for forbidden in (
            "torch",
            "transformers",
            "sentence-transformers",
            "faiss",
            "openai",
            "anthropic",
            "langgraph",
            "qdrant",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, joined)

    def test_tool_runtime_top_level_imports_are_stdlib(self):
        source = (MISSION / "tool_runtime.py").read_text(encoding="utf-8")
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
            "pathlib",
            "re",
            "sys",
            "typing",
        }
        for name in top_level_imports:
            with self.subTest(name=name):
                self.assertIn(name, allowed)
        self.assertNotIn("numpy", top_level_imports)
        self.assertNotIn("torch", top_level_imports)
        self.assertNotIn("requests", top_level_imports)
        self.assertNotIn("openai", top_level_imports)
        self.assertNotIn("anthropic", top_level_imports)
        self.assertNotIn("langgraph", top_level_imports)
        self.assertIn("_require_numpy", source)
        self.assertIn("_load_m32", source)
        self.assertIn("deferred", source.lower())
        self.assertNotIn("optional_live_llm", source)

    def test_core_consumes_m32_and_does_not_open_rag_or_graphs(self):
        source = (MISSION / "tool_runtime.py").read_text(encoding="utf-8")
        self.assertIn("attach_inference_evidence", source)
        self.assertIn("InferenceConfig", source)
        self.assertIn("weights_updated", source)
        self.assertNotIn("from_pretrained", source)
        self.assertNotIn("QdrantClient", source)
        self.assertNotIn("retrieve_and_generate", source)
        self.assertNotIn("StateGraph", source)
        self.assertNotIn("MemorySaver", source)
        self.assertNotIn("def softmax_probs", source)
        self.assertIn("LiveAdapterUnavailable", source)
        self.assertIn("state machine", source.lower())
        self.assertIn("weights_updated", source)

    def test_independent_vat_arithmetic_is_locked(self):
        self.assertEqual(CORE.VAT_AMOUNT, 80.0)
        self.assertEqual(CORE.VAT_RATE, 0.25)
        tax, total = independent_vat(CORE.VAT_AMOUNT, CORE.VAT_RATE)
        self.assertEqual(tax, 20.0)
        self.assertEqual(total, 100.0)
        self.assertEqual(CORE.VAT_TAX, 20.0)
        self.assertEqual(CORE.VAT_TOTAL, 100.0)
        trace = CORE.run_tool_call("vat_on_80_at_025")
        self.assertTrue(trace.execution_reached)
        self.assertEqual(trace.result.output["tax"], tax)
        self.assertEqual(trace.result.output["total"], total)
        self.assertFalse(trace.weights_updated)
        self.assertFalse(trace.inference["training_time"])
        self.assertEqual(trace.stages, ("selection", "validation", "execution", "result"))

    def test_strict_schema_rejects_invalid_and_accepts_valid(self):
        registry = CORE.default_registry()
        spec = registry.get("compute_vat")
        valid = CORE.validate_arguments(
            spec, {"amount": CORE.VAT_AMOUNT, "rate": CORE.VAT_RATE}
        )
        self.assertTrue(valid.ok)
        self.assertEqual(valid.normalized["amount"], 80.0)

        missing = CORE.validate_proposal(
            CORE.parse_proposal(CORE.INVALID_FIXTURES["missing_rate"]), registry
        )
        wrong = CORE.validate_proposal(
            CORE.parse_proposal(CORE.INVALID_FIXTURES["wrong_type"]), registry
        )
        extra = CORE.validate_proposal(
            CORE.parse_proposal(CORE.INVALID_FIXTURES["extra_field"]), registry
        )
        unsafe = CORE.validate_proposal(
            CORE.parse_proposal(CORE.INVALID_FIXTURES["unsafe_rate"]), registry
        )
        boolean = CORE.validate_proposal(
            CORE.parse_proposal(CORE.INVALID_FIXTURES["bool_amount"]), registry
        )
        self.assertFalse(missing.ok)
        self.assertIn("missing", independent_issues({"missing"}, missing))
        self.assertIn("wrong_type", independent_issues({"wrong_type"}, wrong))
        self.assertIn("extra", independent_issues({"extra"}, extra))
        self.assertIn("constraint", independent_issues({"constraint"}, unsafe))
        self.assertIn("wrong_type", independent_issues({"wrong_type"}, boolean))
        self.assertTrue(missing.repairable)
        self.assertFalse(wrong.repairable)

    def test_invalid_calls_never_reach_execution(self):
        session = CORE.RuntimeSession()
        for name in ("missing_rate", "wrong_type", "extra_field", "unsafe_rate", "bool_amount"):
            with self.subTest(name=name):
                before = session.execution_count
                trace = CORE.run_tool_call(
                    CORE.INVALID_FIXTURES[name], session=session
                )
                self.assertFalse(trace.execution_reached, name)
                self.assertEqual(trace.result.error_kind, "schema")
                self.assertEqual(trace.result.error_type, "SchemaError")
                self.assertEqual(session.execution_count, before)
                self.assertNotIn("execution", trace.stages)

        parse_trace = CORE.run_tool_call(CORE.INVALID_FIXTURES["malformed_json"])
        self.assertFalse(parse_trace.execution_reached)
        self.assertEqual(parse_trace.error_kind, "parse")

        unknown = CORE.run_tool_call(CORE.INVALID_FIXTURES["unknown_tool"])
        self.assertFalse(unknown.execution_reached)
        self.assertEqual(unknown.validation.issues[0].kind, "unknown_tool")

    def test_structured_success_and_error_results_serialize(self):
        ok = CORE.run_tool_call("vat_on_80_at_025")
        payload = json.loads(CORE.result_as_json(ok.result))
        self.assertEqual(payload["status"], "success")
        self.assertIsNone(payload["error_kind"])
        self.assertEqual(payload["output"]["tax"], 20.0)

        bad = CORE.run_tool_call(CORE.INVALID_FIXTURES["missing_rate"])
        bad_payload = json.loads(CORE.result_as_json(bad.result))
        self.assertEqual(bad_payload["status"], "schema_error")
        self.assertEqual(bad_payload["error_kind"], "schema")
        self.assertIsNone(bad_payload["output"])

    def test_selection_is_nontrivial_on_a_fixed_case_set(self):
        report = CORE.evaluate_selection()
        self.assertEqual(report["n"], 3)
        self.assertEqual(report["n_correct"], 3)
        by_id = {row["case_id"]: row for row in report["rows"]}
        self.assertEqual(by_id["vat_on_80_at_025"]["selected"], "compute_vat")
        self.assertEqual(by_id["price_of_sku_7"]["selected"], "lookup_catalog_price")
        self.assertIsNone(by_id["write_a_haiku"]["selected"])
        haiku = CORE.run_tool_call("write_a_haiku")
        self.assertEqual(haiku.result.status, "no_tool")
        self.assertFalse(haiku.execution_reached)
        sku = CORE.run_tool_call("price_of_sku_7")
        self.assertEqual(sku.result.output["price"], 42.0)

    def test_side_effect_replay_is_idempotent_and_needs_approval(self):
        session = CORE.RuntimeSession()
        denied = CORE.run_tool_call("post_vat_to_ledger", session=session, approved=False)
        self.assertEqual(denied.result.status, "permission_denied")
        self.assertFalse(denied.execution_reached)
        self.assertEqual(session.ledger.effect_count, 0)

        first = CORE.run_tool_call("post_vat_to_ledger", session=session, approved=True)
        self.assertTrue(first.execution_reached)
        self.assertFalse(first.replayed)
        self.assertEqual(session.ledger.effect_count, 1)
        replay = CORE.run_tool_call("post_vat_to_ledger", session=session, approved=True)
        self.assertTrue(replay.replayed)
        self.assertEqual(session.ledger.effect_count, 1)
        self.assertEqual(session.execution_count, 1)
        self.assertEqual(replay.result.output["entry_id"], first.result.output["entry_id"])

    def test_bounded_repair_retry_exhausts_and_can_succeed(self):
        session = CORE.RuntimeSession()
        exhausted = CORE.run_tool_call(
            CORE.INVALID_FIXTURES["missing_rate"],
            session=session,
            max_attempts=CORE.MAX_ATTEMPTS,
            repairer=CORE.sticky_invalid_repairer,
        )
        self.assertEqual(exhausted.result.status, "retry_exhausted")
        self.assertEqual(exhausted.result.error_kind, "retry_exhausted")
        self.assertEqual(len(exhausted.attempts), CORE.MAX_ATTEMPTS)
        self.assertEqual(exhausted.retry_budget_remaining, 0)
        self.assertFalse(exhausted.execution_reached)
        self.assertEqual(session.execution_count, 0)

        repaired = CORE.run_tool_call(
            CORE.INVALID_FIXTURES["missing_rate"],
            max_attempts=CORE.MAX_ATTEMPTS,
            repairer=CORE.vat_fill_repairer,
        )
        self.assertEqual(repaired.result.status, "success")
        self.assertEqual(repaired.result.output["tax"], 20.0)
        self.assertEqual(len(repaired.attempts), 2)
        self.assertTrue(repaired.execution_reached)

    def test_schema_failure_is_distinct_from_tool_failure(self):
        schema = CORE.run_tool_call(CORE.INVALID_FIXTURES["missing_rate"])
        tool = CORE.run_tool_call(CORE.INVALID_FIXTURES["unknown_sku"])
        self.assertEqual(schema.result.error_kind, "schema")
        self.assertFalse(schema.execution_reached)
        self.assertEqual(tool.result.error_kind, "tool")
        self.assertEqual(tool.result.error_type, "ToolError")
        self.assertTrue(tool.execution_reached)
        self.assertTrue(tool.validation.ok)
        pattern = CORE.run_tool_call(CORE.INVALID_FIXTURES["bad_sku_pattern"])
        self.assertEqual(pattern.result.error_kind, "schema")
        self.assertFalse(pattern.execution_reached)

    def test_m32_checkpoint_identity_is_package_module(self):
        from missions.M32.inference_adaptation import InferenceConfig

        cfg = CORE.make_tool_config()
        self.assertIsInstance(cfg, InferenceConfig)
        self.assertEqual(type(cfg).__module__, "missions.M32.inference_adaptation")
        self.assertFalse(cfg.training_time)
        evidence = CORE.attach_inference_evidence(cfg)
        self.assertFalse(evidence["weights_updated"])
        self.assertFalse(evidence["training_time"])
        self.assertEqual(evidence["fingerprint"], CORE.attach_inference_evidence(cfg)["fingerprint"])

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
        self.assertIn("40", no_ai)
        self.assertIn("0.125", no_ai)
        self.assertIn("compute_duty", no_ai)
        self.assertEqual(CORE.TRANSFER_DUTY_VALUE, 40.0)
        self.assertEqual(CORE.TRANSFER_DUTY_RATE, 0.125)
        duty, total = independent_vat(CORE.TRANSFER_DUTY_VALUE, CORE.TRANSFER_DUTY_RATE)
        self.assertEqual(duty, 5.0)
        self.assertEqual(total, 45.0)
        self.assertNotIn("5.0", no_ai)
        self.assertNotIn("45.0", no_ai)
        self.assertNotIn("the answer is schema", no_ai.lower())
        self.assertNotIn("P1 is valid", no_ai)

    def test_status_does_not_claim_repository_executable(self):
        status = (MISSION / "status.yaml").read_text(encoding="utf-8")
        self.assertIn("does not mark M37 repository-executable", status)
        self.assertIn("intentionally_unpopulated", status)

    def test_live_adapter_is_optional_and_fail_closed(self):
        with self.assertRaises(CORE.LiveAdapterUnavailable):
            CORE.optional_live_propose("compute VAT")
        source = (MISSION / "tool_runtime.py").read_text(encoding="utf-8")
        self.assertNotIn("openai", source.lower())
        self.assertNotIn("import anthropic", source)
        self.assertNotIn("import langgraph", source)
        self.assertIn("not required", CORE.optional_live_propose.__doc__.lower())

    def test_optional_live_llm_module_is_unavailable_and_unimported(self):
        adapter = (MISSION / "optional_live_llm.py").read_text(encoding="utf-8")
        self.assertIn("OptionalLiveLLMUnavailable", adapter)
        self.assertNotIn("import openai", adapter)
        spec = importlib.util.spec_from_file_location(
            "m37_optional_live_llm", MISSION / "optional_live_llm.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with self.assertRaises(module.OptionalLiveLLMUnavailable):
            module.open_optional_live_proposer()
        notebook_code = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        self.assertNotIn("optional_live_llm", notebook_code)
        self.assertNotIn("optional_live_llm", (MISSION / "tool_runtime.py").read_text(encoding="utf-8"))

    def test_malformed_defect_reaches_side_effect_and_repair_refuses(self):
        broken = CORE.pipeline_with_defect(defect="malformed_reaches_side_effect")
        self.assertEqual(broken.defect, "malformed_reaches_side_effect")
        self.assertTrue(broken.validation_bypassed)
        self.assertTrue(broken.execution_reached)
        self.assertEqual(broken.effect_count, 1)
        self.assertEqual(broken.audit["amount_type"], "str")
        self.assertFalse(broken.audit["healthy_validation_ok"])
        repaired = CORE.repair_run(broken)
        self.assertEqual(repaired.defect, "none")
        self.assertFalse(repaired.execution_reached)
        self.assertEqual(repaired.effect_count, 0)
        self.assertEqual(repaired.result_trace.result.error_kind, "schema")
        self.assertEqual(broken.effect_count, 1)
        self.assertTrue(broken.validation_bypassed)
        source = CORE.repair_run.__doc__ or ""
        self.assertIn("defective", source)

    def test_duplicate_defect_posts_twice_and_repair_posts_once(self):
        broken = CORE.pipeline_with_defect(defect="duplicate_side_effect")
        self.assertEqual(broken.defect, "duplicate_side_effect")
        self.assertFalse(broken.idempotency_consulted)
        self.assertEqual(broken.effect_count, 2)
        self.assertNotEqual(broken.audit["first_entry_id"], broken.audit["second_entry_id"])
        repaired = CORE.repair_run(broken)
        self.assertEqual(repaired.effect_count, 1)
        self.assertTrue(repaired.audit["second_replayed"])
        self.assertEqual(broken.effect_count, 2)
        self.assertFalse(broken.idempotency_consulted)

    def test_unsupported_defect_and_healthy_repair_are_rejected(self):
        with self.assertRaises(ValueError):
            CORE.pipeline_with_defect(defect="none")
        with self.assertRaises(ValueError):
            CORE.pipeline_with_defect(defect="temperature")
        healthy = CORE.run_tool_call("vat_on_80_at_025")
        with self.assertRaises(ValueError):
            CORE.repair_run(
                CORE.FailureTrace(
                    defect="none",
                    claim="ok",
                    proposal=healthy.proposal,
                    result_trace=healthy,
                    initial_entries=(),
                    effect_count=0,
                    execution_reached=True,
                    validation_bypassed=False,
                    idempotency_consulted=True,
                    session_execution_count=1,
                )
            )

    def test_observability_report_states_limits_and_later_handoff(self):
        trace = CORE.run_tool_call("vat_on_80_at_025")
        report = CORE.observability_report(trace)
        self.assertEqual(report["version"], CORE.RUNTIME_VERSION)
        self.assertFalse(report["weights_updated"])
        self.assertFalse(report["training_time"])
        self.assertIn("state machine", report["handoff"].lower())
        self.assertIn("not a production", report["scale_limit"].lower())
        contract = CORE.handoff_contract()
        self.assertEqual(contract["trace_stages"], CORE.TRACE_STAGES)
        self.assertIn("M38", contract["handoff"])

    def test_manifest_and_content_name_sources_without_sdks(self):
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        content = (MISSION / "content.yaml").read_text(encoding="utf-8")
        readme = (MISSION / "README.md").read_text(encoding="utf-8")
        self.assertIn("anthropic-agents", manifest)
        self.assertIn("langgraph-docs", manifest)
        self.assertIn("tool-using-agents", manifest)
        self.assertIn("anthropic-agents", content)
        self.assertIn("langgraph-docs", readme)
        self.assertIn("M32", manifest)

    def test_transfer_fixture_is_hand_computable_without_spoiling_the_gate(self):
        duty, total = independent_vat(CORE.TRANSFER_DUTY_VALUE, CORE.TRANSFER_DUTY_RATE)
        self.assertEqual(duty, 5.0)
        self.assertEqual(total, 45.0)
        no_ai = (MISSION / "no_ai_gate.md").read_text(encoding="utf-8")
        self.assertNotIn("5.0", no_ai)
        self.assertNotIn(f"{duty:.6f}", no_ai)


@unittest.skipUnless(RUNTIME_DEPS, "install requirements/m37.txt to run NumPy-dependent M37 tests")
class M37RuntimeTests(unittest.TestCase):
    def test_numpy_vat_table_matches_independent_arithmetic(self):
        tax, total = CORE.numpy_vat_table((CORE.VAT_AMOUNT, 40.0), CORE.VAT_RATE)
        independent_tax, independent_total = independent_vat(CORE.VAT_AMOUNT, CORE.VAT_RATE)
        self.assertEqual(len(tax), 2)
        self.assertAlmostEqual(float(tax[0]), independent_tax, places=12)
        self.assertAlmostEqual(float(total[0]), independent_total, places=12)
        transfer_tax, transfer_total = independent_vat(40.0, CORE.VAT_RATE)
        self.assertAlmostEqual(float(tax[1]), transfer_tax, places=12)
        self.assertAlmostEqual(float(total[1]), transfer_total, places=12)


if __name__ == "__main__":
    unittest.main()
