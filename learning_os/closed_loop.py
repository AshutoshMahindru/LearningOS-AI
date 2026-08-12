from __future__ import annotations

from pathlib import Path
from typing import Any

from .autonomy_engine import AutonomyEngine
from .decision_engine import DecisionEngine
from .evidence_engine import EvidenceEngine
from .gate_engine import GateEngine
from .learner_model import LearnerModelEngine
from .mission_loader import MissionRepository
from .mission_runner import MissionRunner
from .retention_engine import RetentionEngine
from .side_quest_engine import SideQuestEngine
from .storage import StateStore


class LearningLoop:
    """Single service boundary for the closed learning loop."""

    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root)
        self.store = StateStore(root)
        self.missions = MissionRepository(root)
        self.evidence = EvidenceEngine(self.store)
        self.gates = GateEngine(self.evidence)
        self.runner = MissionRunner(self.missions, self.store, self.gates)
        self.learner = LearnerModelEngine(self.store)
        self.retention = RetentionEngine(self.store)
        self.autonomy = AutonomyEngine(root, self.store)
        self.side_quests = SideQuestEngine(self.store)
        self.decisions = DecisionEngine(root, self.store, self.evidence, self.gates)

    def start(self, mission_id: str) -> dict[str, Any]:
        mission = self.runner.start(mission_id)
        self.learner.set_current_mission(mission["id"])
        return mission

    def record_evidence(self, mission_id: str, evidence_type: str, summary: str, competencies: list[str], no_ai: bool, transfer: bool, explanation: bool) -> dict[str, Any]:
        record = self.evidence.add(mission_id, evidence_type, summary, competencies, no_ai, transfer, explanation)
        self.learner.ingest_evidence(record)
        return record

    def gate(self, mission_id: str) -> dict[str, Any]:
        mission = self.missions.get(mission_id)
        result = self.runner.gate(mission_id)
        history = self.learner.record_gate(mission["id"], result["status"], result["reasons"])
        records = self.evidence.for_mission(mission["id"])
        signals: set[str] = set()
        if result["status"] == "PASS":
            if any(record.get("no_ai") for record in records): signals.add("successful_no_ai_gate")
            if any(record.get("transfer") for record in records): signals.add("successful_transfer")
            if any(record.get("type") in {"review", "design"} for record in records): signals.add("successful_review")
            for competency in mission.get("competencies", []):
                self.retention.schedule(competency, mission["id"])
        elif result["status"] == "FAIL" and history["failures"] >= 2:
            signals.add("repeated_gate_failure")
        autonomy_event = self.autonomy.evaluate(signals) if signals else None
        return {**result, "autonomy_event": autonomy_event}

    def step(self, mission_id: str | None = None) -> dict[str, Any]:
        return self.decisions.step(mission_id)
