from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
MISSION = ROOT / "missions" / "M01"
NOTEBOOK = ROOT / "labs" / "M01_ai_ml_landscape.ipynb"


class M01MissionPackageTests(unittest.TestCase):
    def test_required_artifacts_exist(self) -> None:
        required = [
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
            "status.yaml",
        ]
        for name in required:
            self.assertTrue((MISSION / name).is_file(), name)
        self.assertTrue(NOTEBOOK.is_file())
        self.assertTrue((ROOT / "datasets" / "M01" / "README.md").is_file())
        self.assertTrue((ROOT / "requirements" / "m01.txt").is_file())

    def test_manifest_declares_m01_p0_v00_and_no_external_api(self) -> None:
        text = (MISSION / "manifest.yaml").read_text(encoding="utf-8")
        for expected in [
            "id: M01",
            "phase: P0",
            "flagship: V00",
            "cpu_only: true",
            "requires_secrets: false",
            "requires_paid_api: false",
            "network_required: false",
        ]:
            self.assertIn(expected, text)

    def test_evidence_contract_contains_no_prefilled_learner_evidence(self) -> None:
        text = (MISSION / "evidence_contract.yaml").read_text(encoding="utf-8")
        self.assertNotIn("learner_evidence:", text)
        self.assertNotIn("learner_response:", text)
        self.assertIn("required_evidence:", text)

    def test_notebook_is_runnable_shape_and_has_no_network_code(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(len(notebook["cells"]), 10)
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        forbidden = [
            "import requests",
            "import httpx",
            "from openai",
            "import openai",
            "api_key",
            "os.environ",
            "socket.",
        ]
        for token in forbidden:
            self.assertNotIn(token, code)
        for required_symbol in [
            "train_classifier",
            "predict",
            "embed",
            "retrieve",
            "run_application",
            "evaluate",
        ]:
            self.assertIn(required_symbol, code)


if __name__ == "__main__":
    unittest.main()
