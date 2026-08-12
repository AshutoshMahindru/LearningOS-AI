from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .closed_loop import LearningLoop
from .mission_context import MissionContextAssembler


class DashboardService:
    """Read-only projection of the closed-loop runtime for human interfaces."""

    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root)
        self.loop = LearningLoop(self.root)

    def snapshot(self, mission_id: str | None = None) -> dict[str, Any]:
        raw = self.loop.runner.status()
        current = mission_id or (raw.get("mission") or {}).get("id")
        learner = self.loop.learner.get()
        model_competencies = learner.get("competencies", {})
        competency_rows = [
            {
                "name": name,
                "level": item.get("level", 0),
                "confidence": item.get("confidence", 0.0),
                "evidence_count": len(item.get("evidence", [])),
            }
            for name, item in model_competencies.items()
        ]
        competency_rows.sort(key=lambda row: (-row["level"], row["name"].lower()))

        evidence = list(reversed(self.loop.store.evidence()))[:20]
        due = self.loop.retention.due()
        active_side_quests = self.loop.side_quests.active(current) if current else self.loop.side_quests.active()
        decision = self.loop.step(current)
        autonomy = self.loop.autonomy.status()
        mission_status = raw.get("learner", {}).get("mission_status", {})
        passed = sorted(mid for mid, status in mission_status.items() if status == "PASSED")

        context = None
        if current:
            context = MissionContextAssembler(self.root, self.loop.gates).build(current)

        lab_payload = json.loads((self.root / "data" / "lab_status.json").read_text(encoding="utf-8"))
        return {
            "runtime": {
                "status": "closed-loop",
                "current_mission": current,
                "next_action": decision,
            },
            "progress": {
                "passed_missions": passed,
                "passed_count": len(passed),
                "total_missions": len(self.loop.missions.all()),
                "completion_ratio": round(len(passed) / max(1, len(self.loop.missions.all())), 4),
            },
            "mission": context,
            "learner_model": {
                "autonomy_level": learner.get("autonomy_level", "A1"),
                "competencies": competency_rows,
                "misconceptions": learner.get("misconceptions", []),
                "gate_history": learner.get("gate_history", {}),
            },
            "autonomy": autonomy,
            "retention_due": due,
            "open_side_quests": active_side_quests,
            "recent_evidence": evidence,
            "labs": {
                "repository_executable": lab_payload.get("repository_executable", []),
                "source_package_executable": lab_payload.get("source_package_executable", []),
                "repository_executable_count": len(lab_payload.get("repository_executable", [])),
                "source_package_executable_count": len(lab_payload.get("source_package_executable", [])),
                "note": lab_payload.get("note", ""),
            },
        }
