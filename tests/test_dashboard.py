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
        (self.root / "labs").mkdir()
        source = next((ROOT / "labs").glob("M01_*.ipynb"))
        shutil.copy2(source, self.root / "labs" / source.name)
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

    def _mark_m01_contract_ready(self):
        self.app.m01._save({
            "whole_run": {"status": "PASS", "events": ["system ran"]},
            "whole_observation": "I observed the whole system and its major boundaries.",
            "initial_map": "data -> training -> model state -> inference -> application",
            "questions": "What changes during inference and what remains fixed?",
            "experiments": {eid: {"prediction": "a committed prediction", "result": {"status": "PASS"}, "reflection": "This observation explains the relevant system boundary clearly."} for eid in ["E1", "E2", "E3", "E4", "E5"]},
            "controlled_failure": "I repaired the confused architecture using observed behavior.",
            "explanation": "I can explain training inference retrieval tools memory and evaluation.",
            "no_ai_submission": "I reconstructed the architecture independently from memory.",
            "transfer_submission": "I mapped an unseen architecture and stated uncertainty explicitly.",
        })

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
        self.assertEqual(snapshot["m01_experience"]["mission_id"], "M01")
        self.assertEqual(len(snapshot["m01_experience"]["experiments"]), 5)
        self.assertEqual(snapshot["reference_workspace_url"], "/m01")

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

    def test_app_service_uses_same_closed_loop_state_and_m01_contract(self):
        started = self.app.start("m01")
        self.assertEqual(started["snapshot"]["runtime"]["current_mission"], "M01")
        result = self.app.record_evidence({
            "mission_id": "M01", "type": "artifact", "summary": "Independent system map and boundary explanation",
            "competencies": ["system mapping"], "no_ai": True, "transfer": True, "explanation": True,
        })
        self.assertEqual(result["gate"]["status"], "PASS")
        with self.assertRaisesRegex(ValueError, "evidence contract"):
            self.app.run_gate("M01")
        self._mark_m01_contract_ready()
        gated = self.app.run_gate("M01")
        self.assertEqual(gated["gate"]["status"], "PASS")
        self.assertEqual(gated["snapshot"]["progress"]["passed_count"], 1)

    def test_m01_experiments_require_map_interrogation_prediction_and_order(self):
        view = self.app.m01_view()
        self.assertEqual(view["experiments"][0]["status"], "ready")
        self.assertEqual(view["experiments"][1]["status"], "locked")
        with self.assertRaisesRegex(ValueError, "prediction"):
            self.app.m01_run_experiment({"experiment_id": "E1"})
        with self.assertRaisesRegex(ValueError, "Map and Interrogate"):
            self.app.m01_prediction({"experiment_id": "E1", "prediction": "Inference will not mutate learned state"})
        self.app.m01_save_stage({"stage": "map", "content": "data -> training -> model state -> inference -> application"})
        self.app.m01_save_stage({"stage": "interrogate", "content": "What state changes and what evidence would falsify my explanation?"})
        view = self.app.m01_prediction({"experiment_id": "E1", "prediction": "Inference will not mutate learned state"})
        self.assertEqual(view["experiments"][0]["status"], "predicted")
        with self.assertRaisesRegex(ValueError, "Complete E1"):
            self.app.m01_prediction({"experiment_id": "E2", "prediction": "Retraining will change model state"})

    def test_m01_batch_lab_runner_is_disabled(self):
        with self.assertRaisesRegex(ValueError, "guided experiment runner"):
            self.app.run_lab({"mission_id": "M01"})

    def test_m01_stage_work_generates_evidence(self):
        self.app.m01_save_stage({"stage": "map", "content": "data -> training -> model state -> inference -> application"})
        records = self.loop.evidence.for_mission("M01")
        self.assertTrue(any(str(item.get("summary", "")).startswith("m01:map:") for item in records))
        self.assertTrue(self.app.m01_view()["gate"]["checks"][0]["complete"])

    def test_m01_no_ai_requires_prior_work_and_locks_tutor(self):
        with self.assertRaisesRegex(ValueError, "Complete E1-E5"):
            self.app.m01_save_stage({"stage": "no_ai_begin"})
        self._mark_m01_contract_ready()
        state = self.app.m01._state()
        state["no_ai_submission"] = ""
        self.app.m01._save(state)
        view = self.app.m01_save_stage({"stage": "no_ai_begin"})
        self.assertEqual(view["no_ai_submission"], "__ACTIVE__")
        with self.assertRaisesRegex(ValueError, "locked"):
            self.app.ask_tutor({"mission_id": "M01", "message": "Help me"})

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
        self._mark_m01_contract_ready()
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
        info = self.app.labs.inspect("M01")
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

    def test_m01_html_exposes_prediction_gated_reference_flow(self):
        html = (self.root / "web" / "m01.html").read_text(encoding="utf-8")
        for endpoint in ["/api/m01", "/api/m01/whole/run", "/api/m01/prediction", "/api/m01/experiment/run", "/api/m01/reflection", "/api/m01/stage"]:
            self.assertIn(endpoint, html)
        for phrase in ["Run whole system once", "Prediction-gated experiments", "No-AI reconstruction", "M01 evidence contract"]:
            self.assertIn(phrase, html)


if __name__ == "__main__":
    unittest.main()
