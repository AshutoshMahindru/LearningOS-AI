import json
import tempfile
import unittest
from pathlib import Path

from learning_os.evidence_engine import EvidenceEngine
from learning_os.gate_engine import GateEngine
from learning_os.mission_loader import MissionRepository
from learning_os.mission_runner import MissionRunner
from learning_os.storage import StateStore


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "data").mkdir()
        source = Path(__file__).resolve().parents[1] / "data" / "missions.json"
        (self.root / "data" / "missions.json").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        store = StateStore(self.root)
        self.evidence = EvidenceEngine(store)
        self.runner = MissionRunner(MissionRepository(self.root), store, GateEngine(self.evidence))

    def tearDown(self):
        self.tmp.cleanup()

    def test_catalog_has_42_missions(self):
        self.assertEqual(len(self.runner.missions.all()), 42)

    def test_gate_requires_apprenticeship_evidence(self):
        self.runner.start("M01")
        self.evidence.add("M01", "artifact", "system map", no_ai=True, transfer=True, explanation=True)
        self.assertEqual(self.runner.gate("M01")["status"], "PASS")

    def test_next_advances_after_pass(self):
        self.runner.start("M01")
        self.evidence.add("M01", "artifact", "system map", no_ai=True, transfer=True, explanation=True)
        self.runner.gate("M01")
        self.assertIn("M02", self.runner.next_action())


if __name__ == "__main__":
    unittest.main()
