import shutil
import tempfile
import unittest
from pathlib import Path

from learning_os.dashboard import DashboardService
from learning_os.closed_loop import LearningLoop


ROOT = Path(__file__).resolve().parents[1]


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        shutil.copytree(ROOT / "data", self.root / "data")
        shutil.copytree(ROOT / "web", self.root / "web")
        (self.root / "tracking").mkdir()
        payloads = {
            "learner_state.json": '{"learner_id":"default","current_mission":null,"mission_status":{},"blockers":[]}',
            "learner_model.json": '{"learner_id":"default","current_mission":null,"autonomy_level":"A1","competencies":{},"misconceptions":[],"confidence":{},"retention_due":[],"open_side_quests":[],"evidence_summary":{},"gate_history":{},"autonomy_history":[]}',
            "evidence.json": "[]", "sessions.json": "[]", "competencies.json": "{}", "retention_events.json": "[]", "side_quests.json": "[]", "autonomy_events.json": "[]"
        }
        for name, content in payloads.items():
            (self.root / "tracking" / name).write_text(content, encoding="utf-8")
        self.loop = LearningLoop(self.root)
        self.dashboard = DashboardService(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_empty_dashboard_starts_at_m01(self):
        snapshot = self.dashboard.snapshot()
        self.assertEqual(snapshot["runtime"]["next_action"]["action"], "START")
        self.assertEqual(snapshot["runtime"]["next_action"]["target"], "M01")
        self.assertEqual(snapshot["progress"]["total_missions"], 42)
        self.assertEqual(snapshot["progress"]["passed_count"], 0)
        self.assertEqual(snapshot["labs"]["repository_executable_count"], 35)

    def test_dashboard_reflects_passed_mission_and_evidence(self):
        self.loop.start("M01")
        self.loop.record_evidence("M01", "artifact", "AI system map", ["system mapping"], True, True, True)
        self.loop.gate("M01")
        snapshot = DashboardService(self.root).snapshot()
        self.assertEqual(snapshot["progress"]["passed_count"], 1)
        self.assertEqual(snapshot["runtime"]["next_action"]["action"], "ADVANCE")
        self.assertEqual(snapshot["runtime"]["next_action"]["target"], "M02")
        self.assertEqual(snapshot["recent_evidence"][0]["summary"], "AI system map")
        self.assertEqual(snapshot["learner_model"]["competencies"][0]["name"], "system mapping")

    def test_dashboard_html_is_read_only_runtime_surface(self):
        html = (self.root / "web" / "dashboard.html").read_text(encoding="utf-8")
        self.assertIn("/api/dashboard", html)
        self.assertIn("Read-only dashboard", html)


if __name__ == "__main__":
    unittest.main()
