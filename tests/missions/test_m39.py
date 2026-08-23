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
MISSION = ROOT / "missions" / "M39"
NOTEBOOK = ROOT / "labs" / "M39_memory_routing_fallbacks.ipynb"
REQUIREMENTS = ROOT / "requirements" / "m39.txt"


def load_core():
    packaged_name = "missions.M39.robust_agent"
    existing = sys.modules.get(packaged_name)
    if existing is not None:
        return existing
    try:
        return importlib.import_module(packaged_name)
    except ImportError:
        spec = importlib.util.spec_from_file_location(
            packaged_name, MISSION / "robust_agent.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load M39 robust agent")
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


class M39StaticContractTests(unittest.TestCase):
    def test_required_executable_artifacts_exist(self):
        for path in (
            MISSION / "robust_agent.py",
            NOTEBOOK,
            REQUIREMENTS,
            ROOT / "tests" / "test_m39.py",
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
            "robust_agent.py",
        }
        actual = {path.name for path in MISSION.iterdir() if path.is_file()}
        self.assertTrue(required <= actual, required - actual)
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        for name in required:
            self.assertIn(f"missions/M39/{name}", manifest)

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
            "from missions.m34",
            "from missions.m36",
            "from missions.m32",
            "from missions.m40",
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
        self.assertIn("M39", source)
        self.assertIn("robust_agent.py", source)
        self.assertIn("sys.path", source)
        self.assertIn("from missions.M38.agent_workflow import", source)
        self.assertIn("from missions.M39.robust_agent import", source)

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
            "softmax_probs",
            "pack_context",
            "chunk_overlap",
            "eval_harness(",
            "from missions.M40",
            "from missions.M36",
            "from missions.M34",
            "from missions.M32",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_notebook_enforces_prediction_before_every_action(self):
        cells = notebook_cells()
        positions = {str(cell["id"]): index for index, cell in enumerate(cells)}
        pairs = (
            ("predict-whole", "run-whole"),
            ("predict-relevance", "run-relevance"),
            ("predict-stale", "run-stale"),
            ("predict-routes", "run-routes"),
            ("predict-primary-failure", "run-primary-failure"),
            ("predict-loop", "run-loop"),
            ("predict-code-reading", "run-code-reading"),
            ("predict-failure", "run-failure"),
            ("predict-failure-oscillation", "run-failure-oscillation"),
            ("predict-failure-repair", "run-failure-repair"),
            ("predict-oscillation-repair", "run-oscillation-repair"),
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
            "M38 → M39",
            "not a production",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, markdown)

        predict_stale = cell_source(cells[positions["predict-failure"]])
        predict_osc = cell_source(cells[positions["predict-failure-oscillation"]])
        predict_repair = cell_source(cells[positions["predict-failure-repair"]])
        self.assertNotIn('defect="stale_memory_trusted"', predict_stale)
        self.assertNotIn('defect="fallback_oscillation"', predict_stale)
        self.assertNotIn('defect="stale_memory_trusted"', predict_osc)
        self.assertNotIn('defect="fallback_oscillation"', predict_osc)
        self.assertNotIn('defect="stale_memory_trusted"', predict_repair)
        self.assertNotIn('defect="fallback_oscillation"', predict_repair)

        predict_code = cell_source(cells[positions["predict-code-reading"]])
        predict_bullets = predict_code.split("Predict:", 1)[-1]
        self.assertIn("retrieve_memory", predict_bullets)
        self.assertIn("select_route", predict_bullets)
        self.assertIn("run_fallback_ladder", predict_bullets)
        self.assertIn("run_robust_task", predict_bullets)
        self.assertIn("repair_run", predict_bullets)
        self.assertNotIn("degraded is False", predict_code)
        self.assertNotIn("attempts == 1", predict_code)

        repair_src = cell_source(cells[positions["run-failure-repair"]])
        osc_repair_src = cell_source(cells[positions["run-oscillation-repair"]])
        self.assertIn("repair_run", repair_src)
        self.assertIn("broken_stale", repair_src)
        self.assertNotIn("broken_osc", repair_src)
        self.assertNotIn("repaired_osc", repair_src)
        self.assertNotIn('defect="none"', repair_src)
        self.assertLess(positions["run-failure-repair"], positions["predict-oscillation-repair"])
        self.assertIn("repair_run", osc_repair_src)
        self.assertIn("broken_osc", osc_repair_src)
        self.assertNotIn("broken_stale", osc_repair_src)

        code_reading = cell_source(cells[positions["run-code-reading"]])
        self.assertIn("inspect.getsource(retrieve_memory)", code_reading)
        self.assertIn("inspect.getsource(select_route)", code_reading)
        self.assertIn("inspect.getsource(run_fallback_ladder)", code_reading)
        self.assertIn("inspect.getsource(run_robust_task)", code_reading)
        self.assertIn("inspect.getsource(repair_run)", code_reading)
        self.assertNotIn('"degraded" in', code_reading)
        self.assertNotIn("'degraded' in", code_reading)
        self.assertIn("retrieved_ids", code_reading)
        self.assertIn(".route", code_reading)
        self.assertIn(".attempts", code_reading)
        self.assertIn(".degraded", code_reading)

    def test_map_cells_do_not_print_next_prediction_answers(self):
        cells = notebook_cells()
        positions = {str(cell["id"]): index for index, cell in enumerate(cells)}
        inspect_src = cell_source(cells[positions["inspect-input"]])
        self.assertIn("FIELD_CLASSIFICATION", inspect_src)
        self.assertIn("WORKING_EPHEMERAL", inspect_src)
        self.assertNotIn("retrieve_memory(", inspect_src)
        self.assertNotIn("select_route(", inspect_src)
        self.assertNotIn("run_robust_task(", inspect_src)
        self.assertNotIn("ROUTE_CASES", inspect_src)
        self.assertNotIn("Compose a haiku", inspect_src)
        self.assertNotIn("mem-bin4-qty", inspect_src)
        self.assertNotIn("mem-sku7-price-stale", inspect_src)
        self.assertNotIn("99.0", inspect_src)
        self.assertNotIn("STALE_PRICE", inspect_src)
        self.assertNotIn("included_ids", inspect_src)

        relevance_src = cell_source(cells[positions["run-relevance"]])
        self.assertGreater(positions["predict-relevance"], positions["inspect-input"])
        self.assertGreater(positions["run-relevance"], positions["predict-relevance"])
        self.assertIn("demo_store_relevant_and_irrelevant", relevance_src)

        stale_predict = cell_source(cells[positions["predict-stale"]])
        self.assertNotIn("reason': 'expired'", stale_predict)
        self.assertNotIn('reason": "expired"', stale_predict)

        route_predict = cell_source(cells[positions["predict-routes"]])
        self.assertIn(CORE.NO_MATCH_TASK, route_predict)
        # Allowed route names may appear; the case→route mapping must not.
        self.assertNotIn('DEFAULT_TASK, "catalog_purchase"', route_predict)
        self.assertNotIn("→ catalog_purchase", route_predict)
        self.assertNotIn("-> catalog_purchase", route_predict)
        self.assertNotIn("maps to catalog_purchase", route_predict)

    def test_notebook_prints_required_workflow_evidence(self):
        code = "\n".join(
            cell_source(cell) for cell in notebook_cells() if cell.get("cell_type") == "code"
        )
        for token in (
            "run_robust_task",
            "retrieve_memory",
            "select_route",
            "run_fallback_ladder",
            "pipeline_with_defect",
            "repair_run",
            "observability_report",
            "SYSTEM_MAP",
            "SCALE_LIMIT",
            "FIELD_CLASSIFICATION",
            "CATALOG_SKU",
            "CATALOG_PRICE",
            "MAX_ATTEMPTS",
            "CIRCUIT_THRESHOLD",
            'defect="stale_memory_trusted"',
            'defect="fallback_oscillation"',
            "optional_langgraph_store",
            "retrieved_ids",
            "degraded",
            "circuit_open",
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

    def test_core_top_level_imports_are_stdlib(self):
        source = (MISSION / "robust_agent.py").read_text(encoding="utf-8")
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
        self.assertIn("_load_m38", source)
        self.assertIn("deferred", source.lower())

    def test_core_consumes_m38_and_does_not_open_eval_or_rag(self):
        source = (MISSION / "robust_agent.py").read_text(encoding="utf-8")
        self.assertIn("_load_m38", source)
        self.assertIn("missions.M38.agent_workflow", source)
        self.assertIn("run_workflow", source)
        self.assertIn("retrieve_memory", source)
        self.assertIn("select_route", source)
        self.assertIn("run_fallback_ladder", source)
        self.assertIn("circuit", source.lower())
        self.assertIn("degraded", source.lower())
        self.assertNotIn("from_pretrained", source)
        self.assertNotIn("QdrantClient", source)
        self.assertNotIn("retrieve_and_generate", source)
        self.assertNotIn("from langgraph", source)
        self.assertNotIn("import langgraph", source)
        self.assertNotIn("def softmax_probs", source)
        self.assertIn("OptionalLangGraphUnavailable", source)
        self.assertIn("LiveAdapterUnavailable", source)
        self.assertIn("FIELD_CLASSIFICATION", source)
        self.assertIn("working_ephemeral", source)
        self.assertIn("weights_updated", source)

    def test_independent_catalog_price_is_locked(self):
        self.assertEqual(CORE.CATALOG_PRICE, 42.0)
        self.assertEqual(CORE.CATALOG_SKU, "SKU-7")
        self.assertEqual(independent_sku7_price(), 42.0)
        from missions.M37.tool_runtime import CATALOG
        from missions.M38.agent_workflow import CATALOG_PRICE as M38_PRICE

        self.assertEqual(CATALOG["SKU-7"]["price"], independent_sku7_price())
        self.assertEqual(M38_PRICE, independent_sku7_price())
        store = CORE.MemoryStore().put(CORE.catalog_price_entry())
        result = CORE.run_robust_task(CORE.DEFAULT_TASK, store=store)
        self.assertEqual(result.route, "catalog_purchase")
        self.assertEqual(result.terminal, "complete")
        self.assertFalse(result.degraded)
        self.assertEqual(result.attempts, 1)
        self.assertEqual(result.effect_count, 1)
        self.assertEqual(result.posted_amount, independent_sku7_price())
        self.assertEqual(result.workflow.state.last_tool_result["amount"], independent_sku7_price())
        self.assertEqual(
            result.workflow.session.executions,
            ["lookup_catalog_price", "post_ledger_entry"],
        )
        self.assertFalse(result.workflow.state.inference["weights_updated"])
        self.assertFalse(result.workflow.state.inference["training_time"])

    def test_m38_package_identity_matches(self):
        m38 = CORE._load_m38()
        self.assertEqual(m38.__name__, "missions.M38.agent_workflow")
        self.assertIs(m38, sys.modules["missions.M38.agent_workflow"])
        store = CORE.MemoryStore().put(CORE.catalog_price_entry())
        result = CORE.run_robust_task(CORE.DEFAULT_TASK, store=store)
        self.assertEqual(type(result.workflow.state).__module__, "missions.M38.agent_workflow")
        self.assertEqual(type(result.workflow.session).__module__, "missions.M37.tool_runtime")
        self.assertEqual(type(result.workflow.session.ledger).__module__, "missions.M37.tool_runtime")

    def test_field_classification_is_not_hidden(self):
        for name in (
            "node",
            "step",
            "pending_action",
            "last_tool_result",
            "approval",
            "history",
        ):
            with self.subTest(name=name):
                self.assertIn(name, CORE.WORKING_EPHEMERAL)
        self.assertIn("catalog_price_fact", CORE.PERSISTENT_CANDIDATES)
        self.assertIn("working_ephemeral", CORE.FIELD_CLASSIFICATION)
        self.assertIn("last_tool_result", CORE.FIELD_CLASSIFICATION["note"])

    def test_memory_relevance_excludes_irrelevant_ids(self):
        result = CORE.run_robust_task(
            CORE.DEFAULT_TASK,
            store=CORE.demo_store_relevant_and_irrelevant(),
        )
        self.assertEqual(result.retrieved_ids, (CORE.MEM_SKU7_PRICE,))
        excluded = {row["id"]: row["reason"] for row in result.excluded}
        self.assertEqual(excluded[CORE.MEM_BIN4_QTY], "scope_mismatch")
        self.assertEqual(excluded[CORE.MEM_SKU9_PRICE], "sku_mismatch")
        self.assertNotIn(CORE.MEM_BIN4_QTY, result.retrieved_ids)
        self.assertNotIn(CORE.MEM_SKU9_PRICE, result.retrieved_ids)
        self.assertEqual(result.used_memory_ids, ())
        self.assertEqual(result.posted_amount, independent_sku7_price())

    def test_stale_and_superseded_memory_are_flagged(self):
        query = CORE.memory_query_from_task(CORE.DEFAULT_TASK, route="catalog_purchase")
        aged = CORE.retrieve_memory(CORE.demo_store_stale_and_fresh(), query, now=CORE.DEFAULT_NOW)
        self.assertEqual(aged.included_ids, (CORE.MEM_SKU7_FRESH,))
        excluded = {item.entry_id: reason for item, reason in aged.excluded}
        self.assertEqual(excluded[CORE.MEM_SKU7_STALE], "expired")
        self.assertNotIn(CORE.MEM_SKU7_STALE, aged.included_ids)

        superseded = CORE.retrieve_memory(CORE.demo_store_superseded(), query, now=CORE.DEFAULT_NOW)
        self.assertEqual(superseded.included_ids, (CORE.MEM_SKU7_FRESH,))
        super_ex = {item.entry_id: reason for item, reason in superseded.excluded}
        self.assertEqual(super_ex[CORE.MEM_SKU7_SUPERSEDED], "superseded")

        healthy = CORE.run_robust_task(
            CORE.DEFAULT_TASK,
            store=CORE.demo_store_stale_and_fresh(),
            now=CORE.DEFAULT_NOW,
        )
        self.assertNotIn(CORE.MEM_SKU7_STALE, healthy.retrieved_ids)
        self.assertEqual(healthy.posted_amount, independent_sku7_price())
        self.assertEqual(healthy.used_memory_ids, ())
        self.assertFalse(healthy.degraded)

    def test_route_selection_on_frozen_case_set(self):
        store = CORE.MemoryStore().put(CORE.catalog_price_entry())
        observed = []
        for task, expected in CORE.ROUTE_CASES:
            decision = CORE.select_route(task)
            result = CORE.run_robust_task(task, store=store)
            observed.append((task, decision.route, result.terminal, result.workflow is not None))
            with self.subTest(task=task):
                self.assertEqual(decision.route, expected)
                self.assertEqual(result.route, expected)
        self.assertEqual([row[1] for row in observed], ["catalog_purchase", "catalog_lookup", "no_match"])
        purchase, lookup, refused = [
            CORE.run_robust_task(task, store=store) for task, _ in CORE.ROUTE_CASES
        ]
        self.assertEqual(purchase.terminal, "complete")
        self.assertEqual(purchase.effect_count, 1)
        self.assertEqual(lookup.terminal, "complete")
        self.assertEqual(lookup.effect_count, 0)
        self.assertTrue(lookup.workflow.interrupted)
        self.assertEqual(refused.terminal, "no_match")
        self.assertIsNone(refused.workflow)
        self.assertEqual(refused.attempts, 0)
        self.assertFalse(refused.degraded)

    def test_primary_failure_uses_bounded_fallback_and_degraded_contract(self):
        store = CORE.MemoryStore().put(CORE.catalog_price_entry())
        result = CORE.run_robust_task(
            CORE.DEFAULT_TASK,
            store=store,
            inject="primary_failure",
        )
        self.assertEqual(result.route, "catalog_purchase")
        self.assertEqual(result.terminal, "degraded")
        self.assertTrue(result.degraded)
        self.assertNotEqual(result.terminal, "complete")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.fallbacks_used, ("lookup_only",))
        self.assertFalse(result.circuit_open)
        self.assertEqual(result.effect_count, 0)
        self.assertEqual(result.last_tool_result["price"], independent_sku7_price())
        self.assertEqual(result.claim, "lookup_without_post")
        self.assertLessEqual(result.attempts, CORE.MAX_ATTEMPTS)

    def test_fallback_loop_hits_hard_circuit_bound(self):
        store = CORE.MemoryStore().put(CORE.catalog_price_entry())
        result = CORE.run_robust_task(
            CORE.DEFAULT_TASK,
            store=store,
            inject="all_failures",
        )
        self.assertEqual(result.terminal, "circuit_open")
        self.assertTrue(result.circuit_open)
        self.assertFalse(result.degraded)
        self.assertEqual(result.attempts, CORE.CIRCUIT_THRESHOLD)
        self.assertLessEqual(result.attempts, CORE.MAX_ATTEMPTS)
        self.assertFalse(result.aborted_ceiling)
        self.assertEqual([event["rung"] for event in result.trace], ["primary", "lookup_only"])
        self.assertEqual(result.effect_count, 0)

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
        self.assertIn("SKU-21", no_ai)
        self.assertIn("13.0", no_ai)
        self.assertIn("27.0", no_ai)
        self.assertIn("BIN-8", no_ai)
        self.assertIn("4401", no_ai)
        self.assertEqual(CORE.TRANSFER_SKU, "SKU-21")
        self.assertEqual(CORE.TRANSFER_STALE_PRICE, 13.0)
        self.assertEqual(CORE.TRANSFER_CURRENT_PRICE, 27.0)
        self.assertEqual(CORE.TRANSFER_IRRELEVANT_BIN, "BIN-8")
        self.assertEqual(CORE.TRANSFER_NOW, 40)
        self.assertEqual(CORE.TRANSFER_EXPIRES_AT, 8)
        self.assertNotIn("the answer is catalog_lookup", no_ai.lower())
        self.assertNotIn("terminal is circuit_open", no_ai.lower())
        self.assertNotIn("posted_amount must be 13.0", no_ai.lower())

    def test_status_does_not_claim_repository_executable(self):
        status = (MISSION / "status.yaml").read_text(encoding="utf-8")
        self.assertIn("does not mark M39 repository-executable", status)
        self.assertIn("intentionally_unpopulated", status)
        self.assertIn("M01-M35", status)
        self.assertIn("M37 is still missing from executable inventory", status)

    def test_live_and_langgraph_adapters_are_optional_and_fail_closed(self):
        with self.assertRaises(CORE.LiveAdapterUnavailable):
            CORE.optional_live_retrieve("SKU-7")
        with self.assertRaises(CORE.OptionalLangGraphUnavailable):
            CORE.optional_langgraph_store()
        source = (MISSION / "robust_agent.py").read_text(encoding="utf-8")
        self.assertNotIn("openai", source.lower())
        self.assertNotIn("import anthropic", source)
        self.assertNotIn("import langgraph", source)
        self.assertIn("not required", CORE.optional_langgraph_store.__doc__.lower())

    def test_stale_memory_defect_posts_stale_amount_and_repair_uses_catalog(self):
        broken = CORE.pipeline_with_defect(defect="stale_memory_trusted")
        self.assertEqual(broken.defect, "stale_memory_trusted")
        self.assertEqual(broken.terminal, "complete")
        self.assertFalse(broken.degraded)
        self.assertEqual(broken.posted_amount, CORE.STALE_PRICE)
        self.assertEqual(broken.used_memory_ids, (CORE.MEM_SKU7_STALE,))
        self.assertIn(CORE.MEM_SKU7_STALE, broken.result.retrieved_ids)
        self.assertEqual(broken.audit["executions"], ["post_ledger_entry"])
        self.assertNotEqual(broken.posted_amount, independent_sku7_price())
        repaired = CORE.repair_run(broken)
        self.assertEqual(repaired.defect, "none")
        self.assertEqual(repaired.terminal, "complete")
        self.assertFalse(repaired.degraded)
        self.assertEqual(repaired.posted_amount, independent_sku7_price())
        self.assertEqual(repaired.used_memory_ids, ())
        self.assertNotIn(CORE.MEM_SKU7_STALE, repaired.result.retrieved_ids)
        self.assertEqual(broken.posted_amount, CORE.STALE_PRICE)
        self.assertEqual(broken.terminal, "complete")
        source = CORE.repair_run.__doc__ or ""
        self.assertIn("defective", source.lower())

    def test_oscillation_defect_exceeds_bound_and_repair_opens_circuit(self):
        broken = CORE.pipeline_with_defect(defect="fallback_oscillation")
        self.assertEqual(broken.defect, "fallback_oscillation")
        self.assertFalse(broken.circuit_open)
        self.assertGreater(broken.attempts, CORE.MAX_ATTEMPTS)
        self.assertTrue(broken.result.aborted_ceiling)
        self.assertNotEqual(broken.terminal, "circuit_open")
        rungs = broken.audit["rungs"]
        self.assertGreater(len(rungs), CORE.MAX_ATTEMPTS)
        self.assertEqual(rungs[0], "primary")
        self.assertEqual(rungs[1], "lookup_only")
        repaired = CORE.repair_run(broken)
        self.assertEqual(repaired.defect, "none")
        self.assertEqual(repaired.terminal, "circuit_open")
        self.assertTrue(repaired.circuit_open)
        self.assertEqual(repaired.attempts, CORE.CIRCUIT_THRESHOLD)
        self.assertLessEqual(repaired.attempts, CORE.MAX_ATTEMPTS)
        self.assertFalse(repaired.result.aborted_ceiling)
        self.assertGreater(broken.attempts, CORE.MAX_ATTEMPTS)
        self.assertFalse(broken.circuit_open)

    def test_unsupported_defect_and_healthy_repair_are_rejected(self):
        with self.assertRaises(ValueError):
            CORE.pipeline_with_defect(defect="none")
        with self.assertRaises(ValueError):
            CORE.pipeline_with_defect(defect="temperature")
        healthy = CORE.run_robust_task(
            CORE.DEFAULT_TASK,
            store=CORE.MemoryStore().put(CORE.catalog_price_entry()),
        )
        with self.assertRaises(ValueError):
            CORE.repair_run(
                CORE.FailureTrace(
                    defect="none",
                    claim="ok",
                    result=healthy,
                    initial_store=tuple(CORE.MemoryStore().put(CORE.catalog_price_entry()).as_dicts()),
                    initial_task=CORE.DEFAULT_TASK,
                    initial_now=CORE.DEFAULT_NOW,
                    inject=None,
                )
            )

    def test_observability_report_states_limits_and_later_handoff(self):
        result = CORE.run_robust_task(
            CORE.DEFAULT_TASK,
            store=CORE.MemoryStore().put(CORE.catalog_price_entry()),
        )
        report = CORE.observability_report(result)
        self.assertEqual(report["version"], CORE.ROBUST_VERSION)
        self.assertEqual(report["route"], "catalog_purchase")
        self.assertEqual(report["attempts"], 1)
        self.assertFalse(report["degraded"])
        self.assertFalse(report["weights_updated"])
        self.assertIn("M40", report["handoff"])
        self.assertIn("not a production", report["scale_limit"].lower())
        contract = CORE.handoff_contract()
        self.assertEqual(contract["trace_fields"], list(CORE.TRACE_FIELDS))
        self.assertIn("M40", contract["handoff"])
        self.assertEqual(contract["eval_harness"], "deferred to M40")
        self.assertIn("retrieved_ids", contract["trace_fields"])

    def test_manifest_and_content_name_sources_without_sdks(self):
        manifest = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        content = (MISSION / "content.yaml").read_text(encoding="utf-8")
        readme = (MISSION / "README.md").read_text(encoding="utf-8")
        self.assertIn("anthropic-agents", manifest)
        self.assertIn("langgraph-docs", manifest)
        self.assertIn("tool-using-agents", manifest)
        self.assertIn("anthropic-agents", content)
        self.assertIn("langgraph-docs", readme)
        self.assertIn("M38", manifest)
        self.assertIn("memory", manifest)
        self.assertIn("routing", manifest)
        self.assertIn("fallbacks", manifest)

    def test_write_memory_attaches_provenance(self):
        store = CORE.write_memory(
            CORE.MemoryStore(),
            entry_id="mem-test",
            key="sku:SKU-7:price",
            value=42.0,
            scope="catalog",
            now=10,
            source="unit-test",
            sku="SKU-7",
            ttl=5,
        )
        entry = store.get("mem-test")
        self.assertEqual(entry.provenance.source, "unit-test")
        self.assertEqual(entry.provenance.written_at, 10)
        self.assertEqual(entry.expires_at, 15)
        self.assertEqual(CORE.lifecycle(entry, 14), "active")
        self.assertEqual(CORE.lifecycle(entry, 15), "expired")


@unittest.skipUnless(RUNTIME_DEPS, "install requirements/m39.txt to run NumPy-dependent M39 tests")
class M39RuntimeTests(unittest.TestCase):
    def test_numpy_terminal_counts_match_independent_list(self):
        store = CORE.MemoryStore().put(CORE.catalog_price_entry())
        results = [
            CORE.run_robust_task(CORE.DEFAULT_TASK, store=store),
            CORE.run_robust_task(CORE.DEFAULT_TASK, store=store, inject="primary_failure"),
            CORE.run_robust_task(CORE.DEFAULT_TASK, store=store, inject="all_failures"),
            CORE.run_robust_task(CORE.NO_MATCH_TASK, store=store),
        ]
        labels, counts = CORE.numpy_terminal_counts(results)
        independent = [sum(1 for item in results if item.terminal == name) for name in labels]
        self.assertEqual(list(counts), independent)
        self.assertEqual(int(counts[labels.index("complete")]), 1)
        self.assertEqual(int(counts[labels.index("degraded")]), 1)
        self.assertEqual(int(counts[labels.index("circuit_open")]), 1)
        self.assertEqual(int(counts[labels.index("no_match")]), 1)


if __name__ == "__main__":
    unittest.main()
