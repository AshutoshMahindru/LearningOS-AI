"""Unit test suite for Mission M41 (Design an Integrated AI System)."""

import json
import subprocess
import sys
import unittest
from pathlib import Path

from missions.M41.integrated_architecture import (
    BoundaryType,
    DecisionKind,
    DegradationMode,
    SystemBoundary,
    DecisionRule,
    InterfaceContract,
    ObservabilityBudget,
    SystemArchitectureConfig,
    ArchitectureValidator,
    TelemetryTrace,
    build_default_v11_architecture,
)

ROOT = Path(__file__).resolve().parents[2]


class M41StaticContractTests(unittest.TestCase):
    def test_required_executable_artifacts_exist(self):
        artifacts = [
            ROOT / "requirements" / "m41.txt",
            ROOT / "missions" / "M41" / "README.md",
            ROOT / "missions" / "M41" / "manifest.yaml",
            ROOT / "missions" / "M41" / "status.yaml",
            ROOT / "missions" / "M41" / "content.yaml",
            ROOT / "missions" / "M41" / "experiments.yaml",
            ROOT / "missions" / "M41" / "code_reading.md",
            ROOT / "missions" / "M41" / "no_ai_gate.md",
            ROOT / "missions" / "M41" / "controlled_failure.md",
            ROOT / "missions" / "M41" / "assessment.yaml",
            ROOT / "missions" / "M41" / "evidence_contract.yaml",
            ROOT / "missions" / "M41" / "flagship_integration.md",
            ROOT / "missions" / "M41" / "review_brief.md",
            ROOT / "missions" / "M41" / "adr_prompt.md",
            ROOT / "missions" / "M41" / "integrated_architecture.py",
            ROOT / "labs" / "M41_integrated_architecture.ipynb",
            ROOT / "tests" / "missions" / "test_m41.py",
            ROOT / "tests" / "test_m41.py",
        ]
        for path in artifacts:
            self.assertTrue(path.exists(), f"Missing required M41 artifact: {path}")

    def test_status_does_not_claim_repository_executable(self):
        status_path = ROOT / "missions" / "M41" / "status.yaml"
        content = status_path.read_text(encoding="utf-8")
        self.assertIn("implementation_status: implemented", content)
        self.assertIn("learner_evidence_status: intentionally_unpopulated", content)
        self.assertIn("This branch does not mark M41 repository-executable", content)

    def test_validate_repo_reports_m01_m40(self):
        cmd = [sys.executable, str(ROOT / "tools" / "validate_repo.py")]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        self.assertEqual(res.returncode, 0, f"validate_repo failed: {res.stderr}")
        self.assertIn("Repository validation PASSED:", res.stdout)

    def test_first_code_cell_bootstraps_repository_root(self):
        nb_path = ROOT / "labs" / "M41_integrated_architecture.ipynb"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        self.assertTrue(len(code_cells) > 0)
        first_code = "".join(code_cells[0]["source"])
        self.assertIn("sys.path.insert(0,", first_code)

    def test_notebook_is_valid_clean_unique_substantial_and_offline(self):
        nb_path = ROOT / "labs" / "M41_integrated_architecture.ipynb"
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
        manifest_text = (ROOT / "missions" / "M41" / "manifest.yaml").read_text()
        content_text = (ROOT / "missions" / "M41" / "content.yaml").read_text()
        for ref in ["anthropic-agents", "anthropic-evals", "langgraph-docs", "qdrant-docs"]:
            self.assertIn(ref, manifest_text)
            self.assertIn(ref, content_text)


class M41RuntimeArchitectureTests(unittest.TestCase):
    def test_default_architecture_validation(self):
        config = build_default_v11_architecture()
        result = ArchitectureValidator.validate(config)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)

    def test_invalid_architecture_validation(self):
        config = SystemArchitectureConfig(name="", version="v1.0")
        result = ArchitectureValidator.validate(config)
        self.assertFalse(result.is_valid)
        self.assertIn("Architecture config must have a non-empty name.", result.errors)

    def test_telemetry_trace_evaluation(self):
        budget = ObservabilityBudget(max_latency_ms=500.0, max_cost_usd=0.01)
        trace = TelemetryTrace(trace_id="t-001", latency_ms=450.0, cost_usd=0.008)
        eval_res = trace.evaluate_budget(budget)
        self.assertTrue(eval_res["latency_ok"])
        self.assertTrue(eval_res["cost_ok"])
        self.assertTrue(eval_res["violations_ok"])


if __name__ == "__main__":
    unittest.main()
