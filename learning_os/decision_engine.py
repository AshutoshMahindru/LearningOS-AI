from __future__ import annotations

from pathlib import Path
from typing import Any

from .adaptive_router import AdaptiveRouter
from .evidence_engine import EvidenceEngine
from .gate_engine import GateEngine
from .mission_context import MissionContextAssembler
from .mission_loader import MissionRepository
from .retention_engine import RetentionEngine
from .side_quest_engine import SideQuestEngine
from .storage import StateStore


class DecisionEngine:
    """Chooses one next pedagogical action from actual learner evidence and state."""

    def __init__(self, root: str | Path, store: StateStore, evidence: EvidenceEngine, gates: GateEngine) -> None:
        self.root = Path(root)
        self.store = store
        self.evidence = evidence
        self.gates = gates
        self.missions = MissionRepository(root)
        self.context = MissionContextAssembler(root, gates)
        self.retention = RetentionEngine(store)
        self.side_quests = SideQuestEngine(store)
        self.router = AdaptiveRouter()

    def step(self, mission_id: str | None = None) -> dict[str, Any]:
        learner = self.store.learner()
        mid = (mission_id or learner.get("current_mission") or "").upper()
        if not mid:
            return {"action": "START", "reason": "No active mission.", "target": "M01"}
        mission = self.missions.get(mid)

        active_quests = self.side_quests.active(mid)
        if active_quests:
            quest = active_quests[-1]
            return {
                "action": "ZOOM_IN",
                "reason": "An open side quest must be resolved and returned from.",
                "target": quest["target"],
                "side_quest": quest,
            }

        context = self.context.build(mid)
        records = self.evidence.for_mission(mid)
        effective_gate = context["gate"]["status"] if records else None
        due = self.retention.due()
        decision = self.router.decide(
            mission,
            blockers=learner.get("blockers", []),
            gate_status=effective_gate,
            retention_due=bool(due),
            unmet_prerequisites=context["prerequisites"]["unmet"],
        )
        target = decision.target
        if decision.action == "ADVANCE":
            nxt = self.missions.next_after(mid)
            target = nxt["id"] if nxt else None
        elif decision.action == "RETENTION" and due:
            target = due[0]["id"]
        return {
            "action": decision.action,
            "reason": decision.reason,
            "target": target,
            "mission_id": mid,
            "gate_status": effective_gate,
            "evidence_count": len(records),
            "unmet_prerequisites": context["prerequisites"]["unmet"],
            "retention_due": [event["id"] for event in due],
            "lab": context["lab"],
        }
