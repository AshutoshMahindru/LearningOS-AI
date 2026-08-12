from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .gate_engine import GateEngine
from .mission_loader import MissionRepository
from .storage import StateStore


class MissionRunner:
    def __init__(self, missions: MissionRepository, store: StateStore, gates: GateEngine) -> None:
        self.missions = missions
        self.store = store
        self.gates = gates

    def start(self, mission_id: str) -> dict[str, Any]:
        mission = self.missions.get(mission_id)
        learner = self.store.learner()
        learner["current_mission"] = mission["id"]
        learner.setdefault("mission_status", {})[mission["id"]] = "IN_PROGRESS"
        self.store.save_learner(learner)
        sessions = self.store.sessions()
        session = {
            "mission_id": mission["id"],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "runtime": ["whole", "map", "interrogate", "manipulate", "zoom-if-blocked", "decompose", "rebuild", "break", "explain", "no-ai-transfer", "gate"],
            "evidence_required": ["artifact", "explanation", "transfer", "no-ai"],
        }
        sessions.append(session)
        self.store.save_sessions(sessions)
        return mission

    def status(self) -> dict[str, Any]:
        learner = self.store.learner()
        current = learner.get("current_mission")
        return {"learner": learner, "mission": self.missions.get(current) if current else None}

    def next_action(self) -> str:
        learner = self.store.learner()
        blockers = learner.get("blockers", [])
        if blockers:
            return f"ZOOM_IN: resolve blocker: {blockers[-1]}"
        current = learner.get("current_mission")
        if not current:
            return "START: choose M01 or another mission."
        result = self.gates.evaluate(current)
        if result.status == "PASS":
            nxt = self.missions.next_after(current)
            return f"ADVANCE: {nxt['id']} - {nxt['title']}" if nxt else "COMPLETE: capstone passed."
        return f"CONTINUE: {current}; gate={result.status}; repair: {'; '.join(result.reasons)}"

    def gate(self, mission_id: str) -> dict[str, Any]:
        result = self.gates.evaluate(mission_id)
        learner = self.store.learner()
        if result.status == "PASS":
            learner.setdefault("mission_status", {})[mission_id.upper()] = "PASSED"
            self.store.save_learner(learner)
        return {"mission_id": mission_id.upper(), "status": result.status, "reasons": result.reasons}
