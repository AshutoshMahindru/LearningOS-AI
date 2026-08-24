from __future__ import annotations

import ast
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MISSION = ROOT / "missions" / "M40"
NOTEBOOK = ROOT / "labs" / "M40_ai_system_evals.ipynb"
REQUIREMENTS = ROOT / "requirements" / "m40.txt"
DATASETS = ROOT / "datasets" / "M40"


def load_core():
    packaged_name = "missions.M40.evaluation_harness"
    existing = sys.modules.get(packaged_name)
    if existing is not None:
        return existing
    try:
        return importlib.import_module(packaged_name)
    except ImportError:
        spec = importlib.util.spec_from_file_location(
            packaged_name, MISSION / "evaluation_harness.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load M40 evaluation harness")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


CORE = load_core()
RUNTIME_DEPS = importlib.util.find_spec("numpy") is not None
PACK = json.loads((DATASETS / "eval_pack.json").read_text(encoding="utf-8"))
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


class M40StaticContractTests(unittest.TestCase):
    def test_required_executable_artifacts_exist(self):
        for path in (
            MISSION / "evaluation_harness.py",
            MISSION / "optional_llm_judge.py",
            NOTEBOOK,
            REQUIREMENTS,
            ROOT / "tests" / "test_m40.py",
            DATASETS / "eval_pack.json",
            DATASETS / "contaminated_pack.json",
            DATASETS / "expected.json",
            DATASETS / "rubric_labels.json",
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
            "evaluation_harness.py",
            "optional_llm_judge.py",
        }
        actual = {path.name for path in MISSION.iterdir() if path.is_file()}
        self.assertTrue(required <= actual, required - actual)
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        for name in required:
            self.assertIn(f"missions/M40/{name}", manifest)

    def test_json_fixtures_parse_and_yaml_is_well_formed(self):
        for name in (
            "eval_pack.json",
            "contaminated_pack.json",
            "expected.json",
            "rubric_labels.json",
            "transfer.json",
        ):
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
            "braintrust",
            "promptfoo",
            "from missions.m41",
            "architecture_blueprint",
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
        self.assertIn("M40", source)
        self.assertIn("evaluation_harness.py", source)
        self.assertIn("sys.path", source)
        self.assertIn("from missions.M34.rag_pipeline import", source)
        self.assertIn("from missions.M39.robust_agent import", source)
        self.assertIn("from missions.M40.evaluation_harness import", source)

    def test_future_mission_boundary_stays_closed_in_code(self):
        source = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        for forbidden in (
            "from_pretrained",
            "FastAPI",
            "QdrantClient",
            "architecture_blueprint",
            "from missions.M41",
            "sentence_transformers",
            "chunk_overlap",
            "softmax_probs",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_notebook_enforces_prediction_before_every_action(self):
        cells = notebook_cells()
        positions = {str(cell["id"]): index for index, cell in enumerate(cells)}
        pairs = (
            ("predict-whole", "run-whole"),
            ("predict-graders", "run-graders"),
            ("predict-rubric", "run-rubric"),
            ("predict-slices", "run-slices"),
            ("predict-proxies", "run-proxies"),
            ("predict-regression", "run-regression"),
            ("predict-ablation", "run-ablation"),
            ("predict-code-reading", "run-code-reading"),
            ("predict-failure", "run-failure"),
            ("predict-failure-repair", "run-failure-repair"),
            ("predict-contamination", "run-contamination"),
            ("predict-contamination-repair", "run-contamination-repair"),
        )
        for prediction, action in pairs:
            with self.subTest(prediction=prediction, action=action):
                self.assertLess(positions[prediction], positions[action])
                self.assertEqual(cells[positions[prediction]].get("cell_type"), "markdown")
                self.assertIn("Predict before running", cell_source(cells[positions[prediction]]))

        markdown = "\n".join(
            cell_source(cell) for cell in cells if cell.get("cell_type") == "markdown"
        )
        self.assertGreaterEqual(markdown.count("Predict before running"), 12)
        for phrase in (
            "predict → act → observe → explain",
            "timestamp",
            "Controlled failure",
            "UNFILLED BY LEARNER",
            "M34 → M40",
            "M39 → M40",
            "not a production",
            "V11 does not close",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, markdown)

        for pred_id in (
            "predict-failure",
            "predict-failure-repair",
            "predict-contamination",
            "predict-contamination-repair",
        ):
            text = cell_source(cells[positions[pred_id]])
            self.assertNotIn('defect="hidden_critical"', text)
            self.assertNotIn('defect="contaminated_pack"', text)
            self.assertNotIn('defect="unsupported_citation"', text)

        predict_code = cell_source(cells[positions["predict-code-reading"]])
        predict_bullets = predict_code.split("Predict:", 1)[-1]
        self.assertIn("load_eval_pack", predict_bullets)
        self.assertIn("decide_release_gate", predict_bullets)
        self.assertIn("ablate_trace", predict_bullets)
        self.assertIn("optional_llm_judge", predict_bullets)
        self.assertNotIn("fail_reasons =", predict_code)
        self.assertNotIn("EvalVersionError", predict_code)

        repair_src = cell_source(cells[positions["run-failure-repair"]])
        contam_repair_src = cell_source(cells[positions["run-contamination-repair"]])
        self.assertIn("repair_run", repair_src)
        self.assertIn("broken_hidden", repair_src)
        self.assertNotIn("broken_contam", repair_src)
        self.assertNotIn('defect="none"', repair_src)
        self.assertLess(positions["run-failure-repair"], positions["predict-contamination"])
        self.assertIn("repair_run", contam_repair_src)
        self.assertIn("broken_contam", contam_repair_src)
        self.assertNotIn("broken_hidden", contam_repair_src)

        code_reading = cell_source(cells[positions["run-code-reading"]])
        self.assertIn("inspect.getsource(load_eval_pack)", code_reading)
        self.assertIn("inspect.getsource(invoke_case)", code_reading)
        self.assertIn("inspect.getsource(grade_tool_schema)", code_reading)
        self.assertIn("inspect.getsource(grade_citation_support)", code_reading)
        self.assertIn("inspect.getsource(grade_state_termination)", code_reading)
        self.assertIn("inspect.getsource(grade_idempotency)", code_reading)
        self.assertIn("inspect.getsource(decide_release_gate)", code_reading)
        self.assertIn("inspect.getsource(ablate_trace)", code_reading)
        self.assertNotIn('"fail" in', code_reading)
        self.assertNotIn("'fail' in", code_reading)
        self.assertIn("case_ids", code_reading)
        self.assertIn("fail_reasons", code_reading)
        self.assertIn("slice_fail_rates", code_reading)

    def test_map_cells_do_not_print_next_prediction_answers(self):
        cells = notebook_cells()
        positions = {str(cell["id"]): index for index, cell in enumerate(cells)}
        inspect_src = cell_source(cells[positions["inspect-input"]])
        self.assertIn("SCENARIO_TAXONOMY", inspect_src)
        self.assertIn("DETERMINISTIC_GRADERS", inspect_src)
        self.assertIn("EVAL_VERSION", inspect_src)
        self.assertNotIn("run_suite(", inspect_src)
        self.assertNotIn("decide_release_gate(", inspect_src)
        self.assertNotIn("inject_regression(", inspect_src)
        self.assertNotIn("pipeline_with_defect(", inspect_src)
        self.assertNotIn("task_success_rate", inspect_src)
        self.assertNotIn("HIDDEN_CRITICAL_CASE", inspect_src)
        self.assertNotIn("unsupported_claim", inspect_src)
        self.assertNotIn("double_post", inspect_src)

        self.assertGreater(positions["predict-whole"], positions["inspect-input"])
        self.assertGreater(positions["run-whole"], positions["predict-whole"])

        predict_whole = cell_source(cells[positions["predict-whole"]])
        self.assertNotIn("baseline_gate.passed", predict_whole)
        self.assertNotIn("n_critical_fail == 0", predict_whole)

        predict_slices = cell_source(cells[positions["predict-slices"]])
        self.assertNotIn("0.083", predict_slices)
        self.assertNotIn("critical_fail_rate:0.083", predict_slices)

    def test_notebook_prints_required_workflow_evidence(self):
        code = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        for token in (
            "load_eval_pack",
            "invoke_case",
            "grade_tool_schema",
            "grade_citation_support",
            "grade_state_termination",
            "grade_idempotency",
            "run_suite",
            "decide_release_gate",
            "calibrate_rubric",
            "inject_regression",
            "ablate_trace",
            "pipeline_with_defect",
            "repair_run",
            "optional_llm_judge",
            "SYSTEM_MAP",
            "SCALE_LIMIT",
            "EVAL_VERSION",
            "CANONICAL_POLICY",
            "AGGREGATE_ONLY_POLICY",
            'defect="hidden_critical"',
            'defect="contaminated_pack"',
            'defect="unsupported_citation"',
            "answer_labeled",
            "run_robust_task",
            "verify_support",
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
            "braintrust",
            "promptfoo",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, joined)

    def test_core_top_level_imports_are_stdlib(self):
        source = (MISSION / "evaluation_harness.py").read_text(encoding="utf-8")
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
            "importlib",
            "importlib.util",
            "json",
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
        self.assertNotIn("openai", top_level_imports)
        self.assertNotIn("anthropic", top_level_imports)
        self.assertNotIn("langgraph", top_level_imports)
        self.assertIn("_require_numpy", source)
        self.assertIn("_load_m34", source)
        self.assertIn("_load_m39", source)
        self.assertIn("deferred", source.lower())

    def test_core_consumes_m34_and_m39_as_packages(self):
        source = (MISSION / "evaluation_harness.py").read_text(encoding="utf-8")
        self.assertIn("missions.M34.rag_pipeline", source)
        self.assertIn("missions.M39.robust_agent", source)
        self.assertIn("sys.modules.get", source)
        self.assertIn("answer_labeled", source)
        self.assertIn("run_robust_task", source)
        self.assertIn("verify_support", source)
        self.assertNotIn("from_pretrained", source)
        self.assertNotIn("QdrantClient", source)
        self.assertNotIn("from langgraph", source)
        self.assertNotIn("import langgraph", source)
        self.assertNotIn("import openai", source)
        self.assertIn("OptionalLLMJudgeUnavailable", source)
        self.assertIn("anthropic-evals", source)
        self.assertIn("anthropic-agents", source)
        self.assertIn("ai-system-evals", source)
        self.assertIn("optional_llm_judge", source)
        self.assertNotIn("optional_llm_judge import", source)

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
        self.assertIn("Atlas of Rivers", no_ai)
        self.assertIn("QH 441", no_ai)
        self.assertIn("BIN-12", no_ai)
        self.assertIn("8802", no_ai)
        self.assertIn("3", no_ai)
        self.assertEqual(TRANSFER["library"]["title"], "Atlas of Rivers")
        self.assertEqual(TRANSFER["library"]["call_number"], "QH 441")
        self.assertEqual(TRANSFER["warehouse"]["bin"], "BIN-12")
        self.assertEqual(TRANSFER["warehouse"]["occupancy"], 3)
        self.assertEqual(TRANSFER["warehouse"]["refund_order"], 8802)
        self.assertNotIn("the answer is floor 4", no_ai.lower())
        self.assertNotIn("gate must fail", no_ai.lower())
        self.assertNotIn("critical_fail_rate is", no_ai.lower())

    def test_status_does_not_claim_repository_executable(self):
        status = (MISSION / "status.yaml").read_text(encoding="utf-8")
        self.assertIn("does not mark M40 repository-executable", status)
        self.assertIn("intentionally_unpopulated", status)
        self.assertIn("M01-M36", status)
        self.assertIn("v11_closed: false", status)

    def test_fixture_is_offline_versioned_and_split(self):
        self.assertEqual(PACK["eval_version"], "m40.eval.v1")
        self.assertFalse(PACK["downloaded"])
        self.assertFalse(PACK["network_required"])
        self.assertFalse(PACK["contaminated"])
        self.assertTrue(PACK["held_out_untuned"])
        ids = [row["id"] for row in PACK["cases"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 12)
        self.assertIn("rag-grounded-reset", ids)
        self.assertIn("tool-idempotency-replay", ids)
        splits = {row["id"]: row["split"] for row in PACK["cases"]}
        self.assertEqual(splits["rag-holdout-email"], "holdout")
        self.assertEqual(splits["tool-idempotency-replay"], "holdout")
        families = {row["family"] for row in PACK["cases"]}
        self.assertEqual(families, {"rag", "agent", "tool"})
        self.assertEqual(EXPECTED["eval_version"], "m40.eval.v1")
        self.assertEqual(EXPECTED["n"], 12)
        self.assertTrue(EXPECTED["baseline_gate_passed"])
        self.assertFalse(EXPECTED["regression"]["canonical_gate_passed"])
        note = PACK["note"].lower()
        self.assertIn("not", note)
        self.assertIn("holdout", note)

    def test_contaminated_pack_is_marked_and_smaller(self):
        dirty = json.loads((DATASETS / "contaminated_pack.json").read_text(encoding="utf-8"))
        self.assertEqual(dirty["eval_version"], "m40.eval.tuned-dev")
        self.assertTrue(dirty["contaminated"])
        self.assertTrue(dirty["tuned_against"])
        self.assertFalse(dirty["held_out_untuned"])
        dirty_ids = [row["id"] for row in dirty["cases"]]
        self.assertNotIn("rag-holdout-email", dirty_ids)
        self.assertNotIn("tool-idempotency-replay", dirty_ids)
        self.assertLess(len(dirty_ids), 12)

    def test_optional_judge_module_is_unavailable_and_unimported(self):
        adapter = (MISSION / "optional_llm_judge.py").read_text(encoding="utf-8")
        self.assertIn("OptionalLLMJudgeUnavailable", adapter)
        self.assertNotIn("import openai", adapter)
        spec = importlib.util.spec_from_file_location(
            "m40_optional_llm_judge", MISSION / "optional_llm_judge.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with self.assertRaises(module.OptionalLLMJudgeUnavailable):
            module.open_optional_llm_judge()
        harness_src = (MISSION / "evaluation_harness.py").read_text(encoding="utf-8")
        self.assertNotIn("optional_llm_judge import", harness_src)

    def test_live_judge_adapter_is_optional_and_fail_closed(self):
        with self.assertRaises(CORE.OptionalLLMJudgeUnavailable):
            CORE.optional_llm_judge("trace")
        with self.assertRaises(CORE.RubricCannotGateInvariants):
            CORE.decide_release_gate(
                CORE.SuiteReport(
                    eval_version=CORE.EVAL_VERSION,
                    harness_version=CORE.HARNESS_VERSION,
                    n=1,
                    n_task_success=1,
                    task_success_rate=1.0,
                    n_invariant_pass=1,
                    invariant_pass_rate=1.0,
                    n_critical_fail=0,
                    critical_fail_rate=0.0,
                    slice_fail_rates={},
                    family_success={},
                    component_rates={},
                    mean_cost_proxy=0.0,
                    mean_step_count=0.0,
                    pack_contaminated=False,
                    held_out_untuned=True,
                    tuned_against=False,
                    defect="none",
                    case_ids=("x",),
                    rows=(),
                    pack_hash="x",
                ),
                rubric_only=True,
            )

    def test_load_eval_pack_rejects_contaminated_canonical_request(self):
        pack = CORE.load_eval_pack(require_canonical=True)
        self.assertEqual(pack.eval_version, "m40.eval.v1")
        self.assertEqual(pack.case_ids, tuple(EXPECTED["case_ids"]))
        with self.assertRaises(CORE.EvalVersionError):
            CORE.load_eval_pack(DATASETS / "contaminated_pack.json", require_canonical=True)
        dirty = CORE.load_eval_pack(DATASETS / "contaminated_pack.json")
        self.assertTrue(dirty.contaminated)
        self.assertEqual(dirty.eval_version, CORE.CONTAMINATED_VERSION)

    def test_package_identity_matches_m34_and_m39(self):
        m34 = CORE._load_m34()
        m39 = CORE._load_m39()
        self.assertEqual(m34.__name__, "missions.M34.rag_pipeline")
        self.assertEqual(m39.__name__, "missions.M39.robust_agent")
        self.assertIs(m34, sys.modules["missions.M34.rag_pipeline"])
        self.assertIs(m39, sys.modules["missions.M39.robust_agent"])

    def test_tool_schema_grader_localizes_extra_field_without_execution(self):
        pack = CORE.load_eval_pack(require_canonical=True)
        case = pack.get("tool-schema-invalid")
        trace = CORE.invoke_case(case)
        grades = CORE.grade_case(case, trace)
        self.assertEqual(len(grades), 1)
        grade = grades[0]
        self.assertEqual(grade.grader, "tool_schema")
        self.assertTrue(grade.passed)
        self.assertIsNone(grade.localized_failure)
        self.assertFalse(trace.execution_reached)
        self.assertFalse(trace.schema_ok)
        self.assertIn("extra:currency", grade.evidence["issues"])
        self.assertTrue(CORE.outcome_success(case, trace))

    def test_idempotency_grader_localizes_double_post(self):
        pack = CORE.load_eval_pack(require_canonical=True)
        case = pack.get("tool-idempotency-replay")
        healthy = CORE.invoke_case(case)
        healthy_grade = CORE.grade_idempotency(case, healthy)
        self.assertTrue(healthy_grade.passed)
        self.assertEqual(healthy.effect_count, 1)
        self.assertTrue(healthy.replayed)
        broken = CORE.invoke_case(case, defect="duplicate_side_effect")
        broken_grade = CORE.grade_idempotency(case, broken)
        self.assertFalse(broken_grade.passed)
        self.assertEqual(broken_grade.localized_failure, "double_post:effect_count=2")
        self.assertEqual(broken.effect_count, 2)
        self.assertEqual(healthy.effect_count, 1)

    def test_state_termination_localizes_complete_versus_degraded(self):
        pack = CORE.load_eval_pack(require_canonical=True)
        degraded_case = pack.get("agent-degraded-fallback")
        degraded = CORE.invoke_case(degraded_case)
        grade = CORE.grade_state_termination(degraded_case, degraded)
        self.assertTrue(grade.passed)
        self.assertEqual(degraded.terminal, "degraded")
        self.assertTrue(degraded.degraded)
        purchase = pack.get("agent-purchase-complete")
        mismatched = CORE.grade_state_termination(degraded_case, CORE.invoke_case(purchase))
        self.assertFalse(mismatched.passed)
        self.assertEqual(mismatched.localized_failure, "terminal:complete!=degraded")

    def test_ablation_blocks_named_diagnoses(self):
        obs = {
            field: None
            for field in CORE.OBSERVABILITY_FIELDS
        }
        obs["used_memory_ids"] = ["mem-sku7-price-stale"]
        obs["citation_ids"] = ["doc-account-access::c0"]
        memory = CORE.ablate_trace(obs, "used_memory_ids")
        self.assertEqual(memory.removed, "used_memory_ids")
        self.assertEqual(memory.blocked_diagnoses, ("stale_memory_posted_as_complete",))
        self.assertNotIn("used_memory_ids", memory.remaining)
        citation = CORE.ablate_trace(obs, "citation_ids")
        self.assertEqual(citation.blocked_diagnoses, ("unsupported_citation_localization",))

    def test_manifest_and_content_name_sources_without_sdks(self):
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        content = (MISSION / "content.yaml").read_text(encoding="utf-8")
        readme = (MISSION / "README.md").read_text(encoding="utf-8")
        self.assertIn("anthropic-evals", manifest)
        self.assertIn("anthropic-agents", manifest)
        self.assertIn("ai-system-evals", manifest)
        self.assertIn("anthropic-evals", content)
        self.assertIn("anthropic-agents", readme)
        self.assertIn("M34", manifest)
        self.assertIn("M39", manifest)
        self.assertIn("phase_end: true", manifest)

    def test_handoff_contract_keeps_v11_open(self):
        contract = CORE.handoff_contract()
        self.assertEqual(contract["eval_version"], CORE.EVAL_VERSION)
        self.assertEqual(contract["sources"], ["anthropic-evals", "anthropic-agents"])
        self.assertFalse(contract["v11_closed"])
        self.assertIn("M41", contract["handoff"])
        self.assertEqual(contract["architecture_diagram"], "deferred to M41")
        self.assertIn("not a production", contract["scale_limit"].lower())

    def test_unsupported_governance_defect_and_healthy_repair_are_rejected(self):
        with self.assertRaises(ValueError):
            CORE.pipeline_with_defect(defect="none")
        with self.assertRaises(ValueError):
            CORE.pipeline_with_defect(defect="temperature")
        pack = CORE.load_eval_pack(require_canonical=True)
        healthy_rows = []
        for case_id in ("tool-schema-invalid", "tool-idempotency-replay", "agent-no-match"):
            case = pack.get(case_id)
            trace = CORE.invoke_case(case)
            grades = CORE.grade_case(case, trace)
            healthy_rows.append(
                CORE.CaseReport(
                    case_id=case.case_id,
                    family=case.family,
                    scenario=case.scenario,
                    split=case.split,
                    severity=case.severity,
                    task_success=True,
                    invariant_pass=all(g.passed for g in grades),
                    critical_fail=False,
                    grades=grades,
                    trace=trace,
                    defect="none",
                )
            )
        report = CORE.aggregate_report(healthy_rows, pack)
        with self.assertRaises(ValueError):
            CORE.repair_run(
                CORE.FailureTrace(
                    defect="none",
                    claim="ok",
                    report=report,
                    decision=CORE.decide_release_gate(report),
                    pack_version=pack.eval_version,
                    pack_path=pack.path,
                    system_defect="none",
                    policy_name=CORE.CANONICAL_POLICY.name,
                    initial_case_ids=pack.case_ids,
                )
            )


@unittest.skipUnless(RUNTIME_DEPS, "install requirements/m40.txt to run NumPy-dependent M40 tests")
class M40RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack = CORE.load_eval_pack(require_canonical=True)
        cls.baseline = CORE.run_suite(cls.pack)
        cls.expected = CORE.load_expected_payload()

    def test_baseline_matches_frozen_expected(self):
        self.assertEqual(list(self.baseline.case_ids), self.expected["case_ids"])
        self.assertEqual(self.baseline.n, 12)
        self.assertEqual(self.baseline.n_task_success, 12)
        self.assertEqual(self.baseline.n_critical_fail, 0)
        self.assertEqual(self.baseline.eval_version, "m40.eval.v1")
        gate = CORE.decide_release_gate(self.baseline, CORE.CANONICAL_POLICY)
        self.assertTrue(gate.passed)
        self.assertEqual(gate.fail_reasons, ())
        reset = next(row for row in self.baseline.rows if row.case_id == "rag-grounded-reset")
        self.assertEqual(list(reset.trace.citation_ids), ["doc-account-access::c1"])
        self.assertTrue(reset.trace.support_ok)
        self.assertTrue(reset.invariant_pass)

    def test_citation_grader_localizes_unsupported_claim(self):
        case = self.pack.get("rag-grounded-reset")
        broken = CORE.invoke_case(case, defect="unsupported_citation")
        grade = CORE.grade_citation_support(case, broken)
        self.assertFalse(grade.passed)
        self.assertEqual(grade.localized_failure, "unsupported_claim:doc-account-access::c0")
        self.assertEqual(broken.status, "answered")
        self.assertTrue(CORE.outcome_success(case, broken))
        self.assertFalse(broken.support_ok)
        m34 = CORE._load_m34()
        support = m34.verify_support(broken.rag.answer, broken.rag.pack)
        self.assertFalse(support.ok)

    def test_regression_injection_fails_canonical_gate_on_unchanged_pack(self):
        injected = CORE.inject_regression(self.pack, defect="unsupported_citation")
        self.assertEqual(injected.eval_version, self.pack.eval_version)
        self.assertEqual(list(injected.case_ids), list(self.pack.case_ids))
        self.assertEqual(injected.n_task_success, 12)
        self.assertEqual(injected.n_critical_fail, 1)
        self.assertGreater(injected.slice_fail_rates["citation_support"], 0.0)
        gate = CORE.decide_release_gate(injected, CORE.CANONICAL_POLICY)
        self.assertFalse(gate.passed)
        self.assertIn("critical_fail_rate:0.083>0.0", gate.fail_reasons)
        self.assertIn("slice:citation_support:0.250>0.0", gate.fail_reasons)
        aggregate = CORE.decide_release_gate(injected, CORE.AGGREGATE_ONLY_POLICY)
        self.assertTrue(aggregate.passed)

    def test_rubric_calibration_disagrees_on_fluency_and_keeps_invariants(self):
        traces = {row.case_id: row.trace for row in self.baseline.rows}
        report = CORE.calibrate_rubric(traces_by_source=traces)
        self.assertEqual(report["n"], 4)
        self.assertEqual(report["n_disagree"], 1)
        self.assertEqual(report["disagreement_rate"], 0.25)
        self.assertTrue(report["deterministic_required_for_invariants"])
        self.assertFalse(report["llm_judge_required"])
        fluency = next(row for row in report["rows"] if row["dimension"] == "answer_fluency")
        self.assertEqual(fluency["gold"], 1)
        self.assertEqual(fluency["predicted"], 2)
        self.assertTrue(fluency["disagree"])

    def test_hidden_critical_repair_uses_same_report(self):
        broken = CORE.pipeline_with_defect(defect="hidden_critical")
        self.assertEqual(broken.defect, "hidden_critical")
        self.assertTrue(broken.decision.passed)
        self.assertEqual(broken.decision.policy_name, "aggregate_only")
        self.assertEqual(broken.report.n_task_success, 12)
        self.assertEqual(broken.report.n_critical_fail, 1)
        self.assertEqual(list(broken.report.case_ids), list(self.pack.case_ids))
        repaired = CORE.repair_run(broken)
        self.assertEqual(repaired.defect, "none")
        self.assertFalse(repaired.decision.passed)
        self.assertIs(repaired.report, broken.report)
        self.assertTrue(broken.decision.passed)
        self.assertIn("slice:citation_support:0.250>0.0", repaired.decision.fail_reasons)
        localized = next(
            row for row in broken.report.rows if row.case_id == CORE.HIDDEN_CRITICAL_CASE
        )
        self.assertEqual(
            localized.grades[0].localized_failure,
            "unsupported_claim:doc-account-access::c0",
        )

    def test_contaminated_repair_reloads_clean_versioned_set(self):
        broken = CORE.pipeline_with_defect(defect="contaminated_pack")
        self.assertEqual(broken.pack_version, "m40.eval.tuned-dev")
        self.assertTrue(broken.report.pack_contaminated)
        self.assertLess(broken.report.n, 12)
        self.assertNotIn("tool-idempotency-replay", broken.report.case_ids)
        self.assertTrue(broken.decision.passed)
        repaired = CORE.repair_run(broken)
        self.assertEqual(repaired.pack_version, "m40.eval.v1")
        self.assertFalse(repaired.report.pack_contaminated)
        self.assertEqual(repaired.report.n, 12)
        self.assertTrue(repaired.decision.passed)
        self.assertEqual(broken.pack_version, "m40.eval.tuned-dev")
        self.assertLess(broken.report.n, 12)

    def test_numpy_family_success_matches_independent_list(self):
        labels, values = CORE.numpy_slice_matrix(self.baseline)
        independent = [
            sum(1 for row in self.baseline.rows if row.family == name and row.task_success)
            / max(1, sum(1 for row in self.baseline.rows if row.family == name))
            for name in labels
        ]
        self.assertEqual(list(values), independent)
        self.assertEqual(int(round(float(values[labels.index("rag")]))), 1)


if __name__ == "__main__":
    unittest.main()
