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
MISSION = ROOT / "missions" / "M38"
NOTEBOOK = ROOT / "labs" / "M38_stateful_agent.ipynb"
REQUIREMENTS = ROOT / "requirements" / "m38.txt"


def load_core():
    packaged_name = "missions.M38.agent_workflow"
    existing = sys.modules.get(packaged_name)
    if existing is not None:
        return existing
    try:
        return importlib.import_module(packaged_name)
    except ImportError:
        spec = importlib.util.spec_from_file_location(
            packaged_name, MISSION / "agent_workflow.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load M38 agent workflow")
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


def independent_sku7_price() -> float:
    return 42.0


class M38StaticContractTests(unittest.TestCase):
    def test_required_executable_artifacts_exist(self):
        for path in (
            MISSION / "agent_workflow.py",
            MISSION / "optional_langgraph.py",
            NOTEBOOK,
            REQUIREMENTS,
            ROOT / "tests" / "test_m38.py",
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
            "agent_workflow.py",
            "optional_langgraph.py",
        }
        actual = {path.name for path in MISSION.iterdir() if path.is_file()}
        self.assertTrue(required <= actual, required - actual)
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        for name in required:
            self.assertIn(f"missions/M38/{name}", manifest)

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
            "memorysaver",
            "retrieve_and_generate",
            "from missions.m38.optional_langgraph",
            "compile_optional_graph",
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
        self.assertIn("M38", source)
        self.assertIn("agent_workflow.py", source)
        self.assertIn("sys.path", source)
        self.assertIn("from missions.M37.tool_runtime import", source)
        self.assertIn("from missions.M38.agent_workflow import", source)

    def test_future_mission_boundary_stays_closed_in_code(self):
        source = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        for forbidden in (
            "from_pretrained",
            "FastAPI",
            "retrieve_and_generate",
            "QdrantClient",
            "MemorySaver",
            "sentence_transformers",
            "from missions.M38.optional_langgraph",
            "compile_optional_graph",
            "softmax_probs",
            "pack_context",
            "chunk_overlap",
            "fallback_ladder",
            "memory_store",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_notebook_enforces_prediction_before_every_action(self):
        cells = notebook_cells()
        positions = {str(cell["id"]): index for index, cell in enumerate(cells)}
        pairs = (
            ("predict-whole", "run-whole"),
            ("predict-resume", "run-resume"),
            ("predict-loop", "run-loop"),
            ("predict-approval", "run-approval"),
            ("predict-invalid", "run-invalid"),
            ("predict-replay", "run-replay"),
            ("predict-code-reading", "run-code-reading"),
            ("predict-failure", "run-failure"),
            ("predict-failure-replayed", "run-failure-replayed"),
            ("predict-failure-repair", "run-failure-repair"),
            ("predict-replayed-repair", "run-replayed-repair"),
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
            "M37 → M38",
            "not a production",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, markdown)

        predict_loop = cell_source(cells[positions["predict-failure"]])
        predict_replayed = cell_source(cells[positions["predict-failure-replayed"]])
        predict_repair = cell_source(cells[positions["predict-failure-repair"]])
        self.assertNotIn('defect="infinite_loop"', predict_loop)
        self.assertNotIn('defect="replayed_side_effect"', predict_loop)
        self.assertNotIn('defect="infinite_loop"', predict_replayed)
        self.assertNotIn('defect="replayed_side_effect"', predict_replayed)
        self.assertNotIn('defect="infinite_loop"', predict_repair)
        self.assertNotIn('defect="replayed_side_effect"', predict_repair)

        predict_code = cell_source(cells[positions["predict-code-reading"]])
        predict_bullets = predict_code.split("Predict:", 1)[-1]
        self.assertIn("apply_transition", predict_bullets)
        self.assertIn("run_workflow", predict_bullets)
        self.assertIn("checkpoint", predict_bullets)
        self.assertIn("resume", predict_bullets)
        self.assertIn("repair_run", predict_bullets)
        self.assertNotIn("optional_langgraph_compile", predict_bullets)
        self.assertNotIn("execution_reached is False", predict_code)

        repair_src = cell_source(cells[positions["run-failure-repair"]])
        replayed_repair_src = cell_source(cells[positions["run-replayed-repair"]])
        self.assertIn("repair_run", repair_src)
        self.assertIn("broken_loop", repair_src)
        self.assertNotIn("broken_replayed", repair_src)
        self.assertNotIn("repaired_replayed", repair_src)
        self.assertNotIn('defect="none"', repair_src)
        self.assertLess(positions["run-failure-repair"], positions["predict-replayed-repair"])
        self.assertIn("repair_run", replayed_repair_src)
        self.assertIn("broken_replayed", replayed_repair_src)
        self.assertNotIn("broken_loop", replayed_repair_src)

        code_reading = cell_source(cells[positions["run-code-reading"]])
        self.assertIn("inspect.getsource(run_workflow)", code_reading)
        self.assertNotIn('"idempotency" in', code_reading)
        self.assertNotIn("'idempotency' in", code_reading)
        self.assertIn("effect_count", code_reading)
        self.assertIn("last_tool_result", code_reading)
        self.assertIn("model_turn", code_reading)

    def test_notebook_prints_required_workflow_evidence(self):
        code = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        for token in (
            "run_workflow",
            "apply_transition",
            "checkpoint",
            "resume",
            "replay_trace",
            "pipeline_with_defect",
            "repair_run",
            "observability_report",
            "SYSTEM_MAP",
            "SCALE_LIMIT",
            "CATALOG_SKU",
            "CATALOG_PRICE",
            "MAX_STEPS",
            'defect="infinite_loop"',
            'defect="replayed_side_effect"',
            "optional_langgraph_compile",
            "weights_updated",
            "effect_count",
            "last_tool_result",
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

    def test_workflow_top_level_imports_are_stdlib(self):
        source = (MISSION / "agent_workflow.py").read_text(encoding="utf-8")
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
        self.assertIn("_load_m37", source)
        self.assertIn("deferred", source.lower())
        self.assertNotIn("optional_langgraph.py", source)

    def test_core_consumes_m37_and_does_not_open_memory_or_rag(self):
        source = (MISSION / "agent_workflow.py").read_text(encoding="utf-8")
        self.assertIn("_load_m37", source)
        self.assertIn("run_tool_call", source)
        self.assertIn("last_tool_result", source)
        self.assertIn("loop_exhausted", source)
        self.assertNotIn("from_pretrained", source)
        self.assertNotIn("QdrantClient", source)
        self.assertNotIn("retrieve_and_generate", source)
        self.assertNotIn("from langgraph", source)
        self.assertNotIn("import langgraph", source)
        self.assertNotIn("def softmax_probs", source)
        self.assertNotIn("fallback_ladder", source)
        self.assertIn("OptionalLangGraphUnavailable", source)
        self.assertIn("state schema", source.lower())
        self.assertIn("weights_updated", source)

    def test_independent_catalog_price_is_locked(self):
        self.assertEqual(CORE.CATALOG_PRICE, 42.0)
        self.assertEqual(CORE.CATALOG_SKU, "SKU-7")
        self.assertEqual(independent_sku7_price(), 42.0)
        from missions.M37.tool_runtime import CATALOG

        self.assertEqual(CATALOG["SKU-7"]["price"], independent_sku7_price())
        result = CORE.run_workflow("purchase_sku7", approval="granted")
        self.assertEqual(result.state.terminal, "complete")
        self.assertEqual(result.state.node, "complete")
        self.assertEqual(result.effect_count, 1)
        self.assertEqual(result.state.last_tool_result["amount"], independent_sku7_price())
        self.assertEqual(result.session.executions, ["lookup_catalog_price", "post_ledger_entry"])
        self.assertFalse(result.state.inference["weights_updated"])
        self.assertFalse(result.state.inference["training_time"])

    def test_m37_package_identity_matches(self):
        m37 = CORE._load_m37()
        self.assertEqual(m37.__name__, "missions.M37.tool_runtime")
        session = CORE.make_session()
        self.assertEqual(type(session).__module__, "missions.M37.tool_runtime")
        self.assertIs(m37, sys.modules["missions.M37.tool_runtime"])
        result = CORE.run_workflow("purchase_sku7", approval="granted")
        self.assertEqual(type(result.session).__module__, "missions.M37.tool_runtime")
        self.assertEqual(type(result.session.ledger).__module__, "missions.M37.tool_runtime")

    def test_resume_does_not_duplicate_side_effects(self):
        first = CORE.run_workflow(
            "purchase_sku7",
            approval="granted",
            interrupt_when="after_lookup",
        )
        self.assertTrue(first.interrupted)
        self.assertEqual(first.state.node, "decide")
        self.assertEqual(first.effect_count, 0)
        self.assertEqual(first.state.last_tool_result["price"], independent_sku7_price())
        self.assertEqual(first.session.executions, ["lookup_catalog_price"])
        payload = CORE.checkpoint(first.state, first.session)
        json.dumps(payload)
        self.assertEqual(payload["last_tool_result"]["price"], independent_sku7_price())
        self.assertEqual(payload["node"], "decide")
        second = CORE.resume(payload, approval="granted")
        self.assertEqual(second.state.terminal, "complete")
        self.assertEqual(second.effect_count, 1)
        self.assertEqual(second.session.executions, ["lookup_catalog_price", "post_ledger_entry"])
        self.assertEqual(second.state.last_tool_result["amount"], independent_sku7_price())
        self.assertIn(CORE.LEDGER_KEY, second.state.completed_effect_keys)

    def test_loop_bound_terminates_unresolved_fixture(self):
        result = CORE.run_workflow(
            "unresolved_lookup",
            max_steps=3,
            approval="granted",
        )
        self.assertEqual(result.state.terminal, "loop_exhausted")
        self.assertEqual(result.state.node, "loop_exhausted")
        self.assertEqual(result.state.model_turn, 3)
        self.assertEqual(result.effect_count, 0)
        self.assertFalse(result.aborted_ceiling)
        self.assertEqual(result.session.executions, ["lookup_catalog_price"] * 3)

    def test_approval_gates_ledger_execution(self):
        denied = CORE.run_workflow("purchase_sku7", approval="denied")
        self.assertEqual(denied.state.terminal, "denied")
        self.assertEqual(denied.effect_count, 0)
        self.assertEqual(denied.session.executions, ["lookup_catalog_price"])
        pending = CORE.run_workflow("purchase_sku7", approval=None)
        self.assertTrue(pending.interrupted)
        self.assertEqual(pending.state.node, "approve")
        self.assertEqual(pending.effect_count, 0)
        granted = CORE.resume(CORE.checkpoint(pending.state, pending.session), approval="granted")
        self.assertEqual(granted.state.terminal, "complete")
        self.assertEqual(granted.effect_count, 1)

    def test_invalid_transition_is_rejected_without_mutation(self):
        state = CORE.initial_state()
        before = state.as_dict()
        with self.assertRaises(CORE.InvalidTransition) as ctx:
            CORE.apply_transition(state, "execute")
        self.assertEqual(ctx.exception.src, "start")
        self.assertEqual(ctx.exception.dest, "execute")
        self.assertEqual(state.node, "start")
        self.assertEqual(state.as_dict(), before)
        with self.assertRaises(CORE.InvalidTransition):
            CORE.grant_approval(state)

    def test_replay_matches_recorded_happy_path(self):
        original = CORE.run_workflow("purchase_sku7", approval="granted")
        replayed = CORE.replay_trace(original.proposals, approval="granted")
        self.assertEqual(replayed.state.terminal, original.state.terminal)
        self.assertEqual(replayed.effect_count, original.effect_count)
        self.assertEqual(replayed.session.executions, original.session.executions)
        fixture = CORE.replay_trace()
        self.assertEqual(fixture.state.terminal, "complete")
        self.assertEqual(fixture.effect_count, 1)

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
        self.assertIn("BIN-4", no_ai)
        self.assertIn("9", no_ai)
        self.assertIn("reserve_stock", no_ai)
        self.assertEqual(CORE.TRANSFER_BIN, "BIN-4")
        self.assertEqual(CORE.TRANSFER_QTY, 9)
        self.assertEqual(CORE.TRANSFER_MAX_STEPS, 5)
        self.assertNotIn("the answer is last_tool_result", no_ai.lower())
        self.assertNotIn("terminal is loop_exhausted", no_ai.lower())
        self.assertNotIn("effect_count must be 0 after deny", no_ai.lower())

    def test_status_does_not_claim_repository_executable(self):
        status = (MISSION / "status.yaml").read_text(encoding="utf-8")
        self.assertIn("does not mark M38 repository-executable", status)
        self.assertIn("intentionally_unpopulated", status)

    def test_live_and_langgraph_adapters_are_optional_and_fail_closed(self):
        with self.assertRaises(CORE.LiveAdapterUnavailable):
            CORE.optional_live_propose("lookup SKU-7")
        with self.assertRaises(CORE.OptionalLangGraphUnavailable):
            CORE.optional_langgraph_compile()
        source = (MISSION / "agent_workflow.py").read_text(encoding="utf-8")
        self.assertNotIn("openai", source.lower())
        self.assertNotIn("import anthropic", source)
        self.assertNotIn("import langgraph", source)
        self.assertIn("not required", CORE.optional_langgraph_compile.__doc__.lower())

    def test_optional_langgraph_module_is_unavailable_and_unimported(self):
        adapter = (MISSION / "optional_langgraph.py").read_text(encoding="utf-8")
        self.assertIn("OptionalLangGraphUnavailable", adapter)
        self.assertNotIn("import langgraph", adapter)
        spec = importlib.util.spec_from_file_location(
            "m38_optional_langgraph", MISSION / "optional_langgraph.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with self.assertRaises(module.OptionalLangGraphUnavailable):
            module.compile_optional_graph()
        notebook_code = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        self.assertNotIn("from missions.M38.optional_langgraph", notebook_code)
        self.assertNotIn("compile_optional_graph", notebook_code)
        self.assertNotIn(
            "optional_langgraph.py",
            (MISSION / "agent_workflow.py").read_text(encoding="utf-8"),
        )

    def test_infinite_loop_defect_exceeds_bound_and_repair_terminates(self):
        broken = CORE.pipeline_with_defect(defect="infinite_loop")
        self.assertEqual(broken.defect, "infinite_loop")
        self.assertFalse(broken.loop_bound_enforced)
        self.assertNotEqual(broken.terminal, "loop_exhausted")
        self.assertGreater(broken.model_turn, broken.audit["max_steps"])
        self.assertTrue(broken.audit["aborted_ceiling"])
        repaired = CORE.repair_run(broken)
        self.assertEqual(repaired.defect, "none")
        self.assertEqual(repaired.terminal, "loop_exhausted")
        self.assertEqual(repaired.model_turn, broken.audit["max_steps"])
        self.assertTrue(repaired.loop_bound_enforced)
        self.assertNotEqual(broken.terminal, "loop_exhausted")
        self.assertGreater(broken.model_turn, broken.audit["max_steps"])
        source = CORE.repair_run.__doc__ or ""
        self.assertIn("defective", source.lower())

    def test_replayed_defect_posts_twice_and_repair_posts_once(self):
        broken = CORE.pipeline_with_defect(defect="replayed_side_effect")
        self.assertEqual(broken.defect, "replayed_side_effect")
        self.assertFalse(broken.idempotency_consulted)
        self.assertEqual(broken.effect_count, 2)
        self.assertNotEqual(broken.audit["first_entry_id"], broken.audit["second_entry_id"])
        repaired = CORE.repair_run(broken)
        self.assertEqual(repaired.effect_count, 1)
        self.assertEqual(repaired.terminal, "complete")
        self.assertTrue(repaired.idempotency_consulted)
        self.assertEqual(broken.effect_count, 2)
        self.assertFalse(broken.idempotency_consulted)

    def test_unsupported_defect_and_healthy_repair_are_rejected(self):
        with self.assertRaises(ValueError):
            CORE.pipeline_with_defect(defect="none")
        with self.assertRaises(ValueError):
            CORE.pipeline_with_defect(defect="temperature")
        healthy = CORE.run_workflow("purchase_sku7", approval="granted")
        with self.assertRaises(ValueError):
            CORE.repair_run(
                CORE.FailureTrace(
                    defect="none",
                    claim="ok",
                    state=healthy.state,
                    initial_checkpoint=CORE.checkpoint(healthy.state, healthy.session),
                    effect_count=healthy.effect_count,
                    model_turn=healthy.state.model_turn,
                    terminal=healthy.state.terminal,
                    node=healthy.state.node,
                    last_tool_result=healthy.state.last_tool_result,
                    loop_bound_enforced=True,
                    idempotency_consulted=True,
                    last_tool_result_preserved=True,
                )
            )

    def test_observability_report_states_limits_and_later_handoff(self):
        result = CORE.run_workflow("purchase_sku7", approval="granted")
        report = CORE.observability_report(result)
        self.assertEqual(report["version"], CORE.WORKFLOW_VERSION)
        self.assertEqual(report["node"], "complete")
        self.assertEqual(report["effect_count"], 1)
        self.assertFalse(report["weights_updated"])
        self.assertFalse(report["training_time"])
        self.assertIn("M39", report["handoff"])
        self.assertIn("not a production", report["scale_limit"].lower())
        contract = CORE.handoff_contract()
        self.assertEqual(contract["state_fields"], CORE.STATE_FIELDS)
        self.assertIn("M39", contract["handoff"])

    def test_manifest_and_content_name_sources_without_sdks(self):
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        content = (MISSION / "content.yaml").read_text(encoding="utf-8")
        readme = (MISSION / "README.md").read_text(encoding="utf-8")
        self.assertIn("anthropic-agents", manifest)
        self.assertIn("langgraph-docs", manifest)
        self.assertIn("tool-using-agents", manifest)
        self.assertIn("anthropic-agents", content)
        self.assertIn("langgraph-docs", readme)
        self.assertIn("M37", manifest)

    def test_state_schema_is_explicit_before_use(self):
        for field_name in (
            "node",
            "last_tool_result",
            "completed_effect_keys",
            "model_turn",
            "max_steps",
            "pending_action",
            "ledger_entries",
        ):
            with self.subTest(field_name=field_name):
                self.assertIn(field_name, CORE.STATE_FIELDS)
        graph = CORE.graph_public()
        self.assertIn("loop_exhausted", graph["terminals"])
        self.assertEqual(graph["edges"]["start"], ["decide"])
        self.assertNotIn("execute", graph["edges"]["start"])


@unittest.skipUnless(RUNTIME_DEPS, "install requirements/m38.txt to run NumPy-dependent M38 tests")
class M38RuntimeTests(unittest.TestCase):
    def test_numpy_node_counts_match_independent_history(self):
        result = CORE.run_workflow("purchase_sku7", approval="granted")
        nodes, counts = CORE.numpy_node_counts(result.state.history)
        independent = [sum(1 for item in result.state.history if item.src == node) for node in nodes]
        self.assertEqual(list(counts), independent)
        self.assertGreater(int(counts[nodes.index("execute")]), 0)
        self.assertEqual(int(counts[nodes.index("start")]), 1)


if __name__ == "__main__":
    unittest.main()
