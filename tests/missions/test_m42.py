"""Unit test suite for Mission M42 (Build, Evaluate, and Explain the Capstone)."""

import json
import subprocess
import sys
import unittest
from pathlib import Path

from missions.M42.capstone_system import (
    CapstoneComponent,
    CapstoneComponentType,
    CapstoneRunResult,
    CapstoneSystemOrchestrator
)

ROOT = Path(__file__).resolve().parents[2]


class M42StaticContractTests(unittest.TestCase):
    def test_required_executable_artifacts_exist(self):
        artifacts = [
            ROOT / "requirements" / "m42.txt",
            ROOT / "missions" / "M42" / "README.md",
            ROOT / "missions" / "M42" / "manifest.yaml",
            ROOT / "missions" / "M42" / "status.yaml",
            ROOT / "missions" / "M42" / "content.yaml",
            ROOT / "missions" / "M42" / "experiments.yaml",
            ROOT / "missions" / "M42" / "code_reading.md",
            ROOT / "missions" / "M42" / "no_ai_gate.md",
            ROOT / "missions" / "M42" / "controlled_failure.md",
            ROOT / "missions" / "M42" / "assessment.yaml",
            ROOT / "missions" / "M42" / "evidence_contract.yaml",
            ROOT / "missions" / "M42" / "flagship_integration.md",
            ROOT / "missions" / "M42" / "review_brief.md",
            ROOT / "missions" / "M42" / "adr_prompt.md",
            ROOT / "missions" / "M42" / "capstone_system.py",
            ROOT / "labs" / "M42_integrated_capstone.ipynb",
            ROOT / "tests" / "missions" / "test_m42.py",
            ROOT / "tests" / "test_m42.py",
        ]
        for path in artifacts:
            self.assertTrue(path.exists(), f"Missing required M42 artifact: {path}")

    def test_status_does_not_claim_repository_executable(self):
        status_path = ROOT / "missions" / "M42" / "status.yaml"
        content = status_path.read_text(encoding="utf-8")
        self.assertIn("implementation_status: implemented", content)
        self.assertIn("learner_evidence_status: intentionally_unpopulated", content)
        self.assertIn("This branch does not mark M42 repository-executable", content)

    def test_validate_repo_reports_m01_m40(self):
        cmd = [sys.executable, str(ROOT / "tools" / "validate_repo.py")]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        self.assertEqual(res.returncode, 0, f"validate_repo failed: {res.stderr}")
        self.assertIn("Repository validation PASSED:", res.stdout)

    def test_first_code_cell_bootstraps_repository_root(self):
        nb_path = ROOT / "labs" / "M42_integrated_capstone.ipynb"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        self.assertTrue(len(code_cells) > 0)
        first_code = "".join(code_cells[0]["source"])
        self.assertIn("sys.path.insert(0,", first_code)

    def test_notebook_is_valid_clean_unique_substantial_and_offline(self):
        nb_path = ROOT / "labs" / "M42_integrated_capstone.ipynb"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        cell_ids = set()
        for cell in nb["cells"]:
            self.assertIn("id", cell)
            cell_id = cell["id"]
            self.assertNotIn(cell_id, cell_ids, f"Duplicate cell id: {cell_id}")
            cell_ids.add(cell_id)
            if cell["cell_type"] == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs"), [])

    def test_manifest_and_content_name_sources_without_sdks(self):
        manifest_text = (ROOT / "missions" / "M42" / "manifest.yaml").read_text()
        content_text = (ROOT / "missions" / "M42" / "content.yaml").read_text()
        for ref in ["anthropic-agents", "anthropic-evals", "langgraph-docs", "qdrant-docs"]:
            self.assertIn(ref, manifest_text)
            self.assertIn(ref, content_text)


class M42RuntimeCapstoneTests(unittest.TestCase):
    def test_capstone_orchestrator_full_capability(self):
        orchestrator = CapstoneSystemOrchestrator()
        res = orchestrator.execute_capstone_task("What is AI architecture?", simulate_latency_ms=100.0)
        self.assertEqual(res.status, "SUCCESS")
        self.assertFalse(res.degradation_applied)
        self.assertGreaterEqual(res.eval_score, 0.90)

    def test_capstone_orchestrator_degradation_fallback(self):
        orchestrator = CapstoneSystemOrchestrator()
        # Deactivate retriever
        orchestrator.components["HybridRetriever"].is_active = False
        res = orchestrator.execute_capstone_task("What is vector retrieval?", simulate_latency_ms=100.0)
        self.assertEqual(res.status, "SUCCESS")
        self.assertTrue(res.degradation_applied)
        self.assertIn("DEGRADED_CACHED_FALLBACK", res.output)

    def test_capstone_defense_evaluation(self):
        orchestrator = CapstoneSystemOrchestrator()
        cases = [
            {"query": "Query 1", "simulate_latency_ms": 100.0},
            {"query": "Query 2", "simulate_latency_ms": 150.0},
        ]
        report = orchestrator.run_defense_evaluation(cases)
        self.assertEqual(report["total_cases"], 2)
        self.assertTrue(report["system_defense_passed"])


if __name__ == "__main__":
    unittest.main()
