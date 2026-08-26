import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from learning_os.app import AppService
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
        self.app = AppService(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_empty_dashboard_starts_at_m01(self):
        snapshot = self.app.snapshot()
        self.assertEqual(snapshot["runtime"]["next_action"]["action"], "START")
        self.assertEqual(snapshot["runtime"]["next_action"]["target"], "M01")
        self.assertEqual(snapshot["runtime"]["selected_mission"], "M01")
        self.assertEqual(snapshot["progress"]["total_missions"], 42)
        self.assertEqual(snapshot["progress"]["passed_count"], 0)
        self.assertEqual(snapshot["labs"]["repository_executable_count"], 42)
        self.assertEqual(len(snapshot["missions"]), 42)
        self.assertEqual(snapshot["player"]["current_step"], "whole")
        self.assertEqual(len(snapshot["player"]["steps"]), 9)
        self.assertIn("evidence_intelligence", snapshot)

    def test_dashboard_reflects_passed_mission_and_evidence(self):
        self.loop.start("M01")
        self.loop.record_evidence("M01", "artifact", "AI system map", ["system mapping"], True, True, True)
        self.loop.gate("M01")
        snapshot = AppService(self.root).snapshot()
        self.assertEqual(snapshot["progress"]["passed_count"], 1)
        self.assertEqual(snapshot["runtime"]["next_action"]["action"], "ADVANCE")
        self.assertEqual(snapshot["runtime"]["next_action"]["target"], "M02")
        self.assertEqual(snapshot["recent_evidence"][0]["summary"], "AI system map")
        self.assertEqual(snapshot["workspace"]["evidence"][0]["summary"], "AI system map")
        self.assertEqual(snapshot["learner_model"]["competencies"][0]["name"], "system mapping")
        self.assertTrue(all(item["complete"] for item in snapshot["workspace"]["gate_checklist"]))
        self.assertEqual(snapshot["evidence_intelligence"]["gate_status"], "PASS")

    def test_app_service_uses_same_closed_loop_state(self):
        started = self.app.start("m01")
        self.assertEqual(started["snapshot"]["runtime"]["current_mission"], "M01")
        result = self.app.record_evidence({
            "mission_id": "M01",
            "type": "artifact",
            "summary": "Independent system map and boundary explanation",
            "competencies": ["system mapping"],
            "no_ai": True,
            "transfer": True,
            "explanation": True,
        })
        self.assertEqual(result["gate"]["status"], "PASS")
        gated = self.app.run_gate("M01")
        self.assertEqual(gated["gate"]["status"], "PASS")
        self.assertEqual(gated["snapshot"]["progress"]["passed_count"], 1)

    def test_guided_player_enforces_evidence_and_locks_tutor_at_no_ai_step(self):
        self.app.start("M01")
        for step in ["whole", "map", "interrogate"]:
            result = self.app.complete_player_step({"mission_id": "M01", "step_id": step, "response": "working note"})
        self.assertEqual(result["player"]["current_step"], "experiment")
        with self.assertRaisesRegex(ValueError, "Record artifact"):
            self.app.complete_player_step({"mission_id": "M01", "step_id": "experiment"})

        self.app.record_evidence({"mission_id": "M01", "type": "lab", "summary": "Ran the canonical experiment and inspected state changes"})
        self.app.complete_player_step({"mission_id": "M01", "step_id": "experiment"})
        self.app.complete_player_step({"mission_id": "M01", "step_id": "break"})
        self.app.record_evidence({"mission_id": "M01", "type": "note", "summary": "Explained the observed mechanism and its boundary", "explanation": True})
        self.app.complete_player_step({"mission_id": "M01", "step_id": "explain"})
        self.assertEqual(self.app.snapshot("M01")["player"]["current_step"], "no_ai")
        self.assertTrue(self.app.snapshot("M01")["tutor"]["locked"])
        with self.assertRaisesRegex(ValueError, "locked"):
            self.app.ask_tutor({"mission_id": "M01", "message": "Give me the answer"})

        self.app.record_evidence({"mission_id": "M01", "type": "artifact", "summary": "Rebuilt the system map from memory without assistance", "no_ai": True})
        self.app.complete_player_step({"mission_id": "M01", "step_id": "no_ai"})
        self.app.record_evidence({"mission_id": "M01", "type": "note", "summary": "Mapped a fresh unseen AI architecture and stated uncertainty", "transfer": True})
        self.app.complete_player_step({"mission_id": "M01", "step_id": "transfer"})
        self.app.run_gate("M01")
        completed = self.app.complete_player_step({"mission_id": "M01", "step_id": "gate"})
        self.assertEqual(completed["player"]["current_step"], "complete")

    def test_tutor_has_offline_socratic_fallback(self):
        self.app.start("M01")
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            answer = self.app.ask_tutor({"mission_id": "M01", "message": "I am stuck and confused"})
        self.assertEqual(answer["turn"]["provider"], "local-socratic")
        self.assertIn("input", answer["turn"]["assistant"].lower())

    def test_lab_inspection_is_restricted_to_canonical_mission_notebook(self):
        (self.root / "labs").mkdir()
        source = next((ROOT / "labs").glob("M01_*.ipynb"))
        shutil.copy2(source, self.root / "labs" / source.name)
        app = AppService(self.root)
        info = app.labs.inspect("M01")
        self.assertTrue(info["path"].startswith("labs/M01_"))
        self.assertGreater(info["cells"], 0)

    def test_app_rejects_empty_evidence_summary(self):
        self.app.start("M01")
        with self.assertRaisesRegex(ValueError, "summary"):
            self.app.record_evidence({"mission_id": "M01", "summary": "   "})

    def test_dashboard_html_is_guided_app_surface(self):
        html = (self.root / "web" / "dashboard.html").read_text(encoding="utf-8")
        for endpoint in ["/api/dashboard", "/api/start", "/api/evidence", "/api/gate", "/api/player/complete", "/api/tutor", "/api/lab/run"]:
            self.assertIn(endpoint, html)
        for surface in ["Mission progress", "Mission player", "Socratic tutor", "Evidence intelligence", "Canonical mission notebook"]:
            self.assertIn(surface, html)
        self.assertNotIn("Read-only dashboard", html)


if __name__ == "__main__":
    unittest.main()
