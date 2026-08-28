from __future__ import annotations

from pathlib import Path
from typing import Any

from .closed_loop import LearningLoop
from .dashboard import DashboardService
from .lab_runner import LabRunner
from .m01_experience import M01Experience
from .mission_player import MissionPlayer
from .tutor import TutorEngine


class AppService:
    """Mutation boundary for the local Learning OS web application.

    The app deliberately reuses the same closed-loop runtime as the CLI. This
    keeps learner state, evidence, gates, retention and side quests canonical
    rather than creating a second web-only state model.
    """

    EVIDENCE_TYPES = {"artifact", "lab", "build", "review", "design", "note"}

    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root)
        self.loop = LearningLoop(self.root)
        self.dashboard = DashboardService(self.root)
        self.player = MissionPlayer(self.root, self.loop)
        self.tutor = TutorEngine(self.root, self.loop)
        self.labs = LabRunner(self.root, self.loop)
        self.m01 = M01Experience(self.root, self.loop)

    def snapshot(self, mission_id: str | None = None) -> dict[str, Any]:
        snapshot = self.dashboard.snapshot(mission_id)
        mid = snapshot["runtime"].get("selected_mission")
        if mid:
            snapshot["player"] = self.player.view(mid)
            snapshot["tutor"] = {
                "locked": snapshot["player"]["current_step"] == "no_ai",
                "history": self.tutor.history(mid),
                "remote_configured": bool(__import__("os").getenv("OPENAI_API_KEY", "").strip()),
            }
            try:
                snapshot["lab_runner"] = {
                    "notebook": self.labs.inspect(mid),
                    "recent_runs": self.labs.recent(mid),
                }
            except ValueError as exc:
                snapshot["lab_runner"] = {"notebook": None, "recent_runs": [], "error": str(exc)}
            snapshot["evidence_intelligence"] = self.evidence_intelligence(mid, snapshot)
            if mid == "M01":
                snapshot["m01_experience"] = self.m01.view()
        return snapshot

    def _mission_id(self, value: Any = None) -> str:
        raw = str(value or "").strip()
        if not raw:
            current = self.loop.runner.status().get("mission") or {}
            raw = str(current.get("id") or "")
        if not raw:
            raise ValueError("mission_id is required because no mission is active")
        mission = self.loop.missions.get(raw)
        return mission["id"]

    @staticmethod
    def _bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def start(self, mission_id: Any) -> dict[str, Any]:
        mid = self._mission_id(mission_id)
        mission = self.loop.start(mid)
        return {"mission": mission, "snapshot": self.snapshot(mid)}

    def record_evidence(self, payload: dict[str, Any]) -> dict[str, Any]:
        mid = self._mission_id(payload.get("mission_id"))
        summary = str(payload.get("summary") or "").strip()
        if not summary:
            raise ValueError("Evidence summary is required")

        evidence_type = str(payload.get("type") or "artifact").strip().lower()
        if evidence_type not in self.EVIDENCE_TYPES:
            allowed = ", ".join(sorted(self.EVIDENCE_TYPES))
            raise ValueError(f"Unsupported evidence type {evidence_type!r}; choose one of: {allowed}")

        raw_competencies = payload.get("competencies") or []
        if isinstance(raw_competencies, str):
            raw_competencies = [raw_competencies]
        competencies = [str(item).strip() for item in raw_competencies if str(item).strip()]

        record = self.loop.record_evidence(
            mid,
            evidence_type,
            summary,
            competencies,
            self._bool(payload.get("no_ai")),
            self._bool(payload.get("transfer")),
            self._bool(payload.get("explanation")),
        )
        gate = self.loop.gates.evaluate(mid)
        return {
            "evidence": record,
            "gate": {"status": gate.status, "reasons": gate.reasons},
            "snapshot": self.snapshot(mid),
        }

    def run_gate(self, mission_id: Any = None) -> dict[str, Any]:
        mid = self._mission_id(mission_id)
        if mid == "M01" and not self.m01.view()["gate"]["ready"]:
            missing = [item["label"] for item in self.m01.view()["gate"]["checks"] if not item["complete"]]
            raise ValueError("M01 evidence contract is not ready: " + ", ".join(missing))
        result = self.loop.gate(mid)
        return {"gate": result, "snapshot": self.snapshot(mid)}

    def complete_player_step(self, payload: dict[str, Any]) -> dict[str, Any]:
        mid = self._mission_id(payload.get("mission_id"))
        player = self.player.complete(mid, str(payload.get("step_id") or ""), str(payload.get("response") or ""))
        return {"player": player, "snapshot": self.snapshot(mid)}

    def reset_player(self, mission_id: Any) -> dict[str, Any]:
        mid = self._mission_id(mission_id)
        return {"player": self.player.reset(mid), "snapshot": self.snapshot(mid)}

    def ask_tutor(self, payload: dict[str, Any]) -> dict[str, Any]:
        mid = self._mission_id(payload.get("mission_id"))
        if mid == "M01" and self.m01.view().get("no_ai_submission") == "__ACTIVE__":
            raise ValueError("Tutor assistance is locked during the M01 no-AI gate")
        result = self.tutor.ask(mid, str(payload.get("message") or ""))
        result["snapshot"] = self.snapshot(mid)
        return result

    def run_lab(self, payload: dict[str, Any]) -> dict[str, Any]:
        mid = self._mission_id(payload.get("mission_id"))
        if mid == "M01":
            raise ValueError("M01 uses the guided experiment runner. Open /m01 and complete prediction-gated E1-E5 instead of batch-running the notebook.")
        result = self.labs.run(mid, int(payload.get("timeout_seconds") or 240))
        return {"run": result, "snapshot": self.snapshot(mid)}

    def m01_view(self) -> dict[str, Any]:
        return self.m01.view()

    def m01_save_stage(self, payload: dict[str, Any]) -> dict[str, Any]:
        stage = str(payload.get("stage") or "").strip().lower()
        if stage == "no_ai_begin":
            state = self.m01._state()
            state["no_ai_submission"] = "__ACTIVE__"
            self.m01._save(state)
            return self.m01.view()
        if stage == "no_ai_cancel":
            state = self.m01._state()
            if state.get("no_ai_submission") == "__ACTIVE__":
                state["no_ai_submission"] = ""
                self.m01._save(state)
            return self.m01.view()
        return self.m01.save_stage(stage, str(payload.get("content") or ""))

    def m01_prediction(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.m01.save_prediction(str(payload.get("experiment_id") or ""), str(payload.get("prediction") or ""))

    def m01_run_experiment(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.m01.run_experiment(str(payload.get("experiment_id") or ""), int(payload.get("timeout_seconds") or 180))

    def m01_reflection(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.m01.save_reflection(str(payload.get("experiment_id") or ""), str(payload.get("reflection") or ""))

    def evidence_intelligence(self, mission_id: str, snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
        mid = self.loop.missions.get(mission_id)["id"]
        records = self.loop.evidence.for_mission(mid)
        gate = self.loop.gates.evaluate(mid)
        if mid == "M01":
            contract = self.m01.view()["gate"]
            missing_contract = [item for item in contract["checks"] if not item["complete"]]
            return {
                "gate_status": gate.status,
                "missing": [item["id"] for item in missing_contract],
                "actions": [f"Complete M01 requirement: {item['label']}." for item in missing_contract] or ["M01 evidence contract is complete. Run the formal gate."],
                "weak_evidence_count": 0,
                "evidence_count": len(records),
            }
        checklist = (snapshot or self.dashboard.snapshot(mid))["workspace"]["gate_checklist"]
        missing = [item for item in checklist if not item["complete"]]
        actions: list[str] = []
        mapping = {
            "artifact": "Produce and record an independent artifact, lab, build or review result.",
            "explanation": "Explain the observed mechanism in your own words and mark that evidence as explanation.",
            "transfer": "Apply the competency to an unseen case and record transfer evidence.",
            "no_ai": "Complete the mission's no-AI task from memory and record it explicitly as no-AI evidence.",
        }
        actions.extend(mapping[item["id"]] for item in missing if item["id"] in mapping)
        if not actions and gate.status != "PASS":
            actions.extend(str(reason) for reason in gate.reasons)
        if not actions and gate.status == "PASS":
            actions.append("Evidence is gate-ready. Run the formal gate if it has not yet been recorded.")
        weak = [record for record in records if len(str(record.get("summary") or "").split()) < 6]
        return {
            "gate_status": gate.status,
            "missing": [item["id"] for item in missing],
            "actions": actions,
            "weak_evidence_count": len(weak),
            "evidence_count": len(records),
        }

    def complete_retention(self, event_id: Any, passed: Any) -> dict[str, Any]:
        event = str(event_id or "").strip()
        if not event:
            raise ValueError("event_id is required")
        result = self.loop.retention.complete(event, self._bool(passed))
        return {"retention": result, "snapshot": self.snapshot()}

    def open_side_quest(self, payload: dict[str, Any]) -> dict[str, Any]:
        mid = self._mission_id(payload.get("mission_id"))
        target = str(payload.get("target") or "").strip()
        reason = str(payload.get("reason") or "").strip()
        return_target = str(payload.get("return_target") or "").strip()
        if not target or not reason or not return_target:
            raise ValueError("target, reason and return_target are required")
        minutes = int(payload.get("minutes") or 60)
        quest = self.loop.side_quests.open(mid, target, reason, return_target, minutes)
        return {"side_quest": quest, "snapshot": self.snapshot(mid)}

    def close_side_quest(self, payload: dict[str, Any]) -> dict[str, Any]:
        quest_id = str(payload.get("id") or "").strip()
        assessment = str(payload.get("assessment") or "").strip().upper()
        outcome = str(payload.get("outcome") or "").strip()
        if not quest_id or not assessment:
            raise ValueError("id and assessment are required")
        quest = self.loop.side_quests.close(quest_id, assessment, outcome)
        return {"side_quest": quest, "snapshot": self.snapshot(quest["mission_id"])}
