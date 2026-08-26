from __future__ import annotations

from pathlib import Path
from typing import Any

from .closed_loop import LearningLoop
from .dashboard import DashboardService


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

    def snapshot(self, mission_id: str | None = None) -> dict[str, Any]:
        return self.dashboard.snapshot(mission_id)

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
        result = self.loop.gate(mid)
        return {"gate": result, "snapshot": self.snapshot(mid)}

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
