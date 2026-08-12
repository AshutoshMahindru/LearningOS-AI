import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from learning_os.autonomy_engine import AutonomyEngine
from learning_os.closed_loop import LearningLoop
from learning_os.storage import StateStore


ROOT = Path(__file__).resolve().parents[1]


class ClosedLoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        shutil.copytree(ROOT / "data", self.root / "data")
        (self.root / "tracking").mkdir()
        for name, content in {
            "learner_state.json": '{"learner_id":"default","current_mission":null,"mission_status":{},"blockers":[]}',
            "learner_model.json": '{"learner_id":"default","current_mission":null,"autonomy_level":"A1","competencies":{},"misconceptions":[],"confidence":{},"retention_due":[],"open_side_quests":[],"evidence_summary":{},"gate_history":{},"autonomy_history":[]}',
            "evidence.json": "[]", "sessions.json": "[]", "competencies.json": "{}", "retention_events.json": "[]", "side_quests.json": "[]", "autonomy_events.json": "[]"
        }.items():
            (self.root / "tracking" / name).write_text(content, encoding="utf-8")
        self.loop = LearningLoop(self.root)

    def tearDown(self): self.tmp.cleanup()

    def test_untouched_mission_continues_instead_of_failing_gate(self):
        self.loop.start("M01")
        self.assertEqual(self.loop.step()["action"], "CONTINUE")

    def test_evidence_updates_learner_model_and_pass_advances(self):
        self.loop.start("M01")
        self.loop.record_evidence("M01", "artifact", "system map", ["system mapping"], True, True, True)
        model = self.loop.learner.get()
        self.assertEqual(model["competencies"]["system mapping"]["level"], 4)
        result = self.loop.gate("M01")
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(self.loop.step()["target"], "M02")
        self.assertTrue(self.loop.retention.events())

    def test_side_quest_overrides_main_mission_until_passed(self):
        self.loop.start("M01")
        quest = self.loop.side_quests.open("M01", "python", "traceback blocker", "M01 experiment cell", 45)
        self.assertEqual(self.loop.step()["action"], "ZOOM_IN")
        self.loop.side_quests.close(quest["id"], "PASS", "Can now trace the failure")
        self.assertEqual(self.loop.step()["action"], "CONTINUE")

    def test_due_retention_routes_before_continuing(self):
        self.loop.start("M01")
        self.loop.retention.schedule("system mapping", "M01", now=datetime.now(timezone.utc), due_in_days=0)
        self.assertEqual(self.loop.step()["action"], "RETENTION")

    def test_autonomy_ratchets_only_with_all_positive_signals(self):
        engine = AutonomyEngine(self.root, StateStore(self.root))
        unchanged = engine.evaluate({"successful_no_ai_gate", "successful_transfer"})
        self.assertEqual(unchanged["to"], "A1")
        raised = engine.evaluate({"successful_no_ai_gate", "successful_transfer", "successful_review"})
        self.assertEqual(raised["to"], "A2")


if __name__ == "__main__": unittest.main()
