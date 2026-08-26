from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .closed_loop import LearningLoop
from .mission_context import MissionContextAssembler


class DashboardService:
    """Read projection of the closed-loop runtime for learner interfaces."""

    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root)
        self.loop = LearningLoop(self.root)

    @staticmethod
    def _gate_checklist(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "id": "artifact",
                "label": "Independent deliverable",
                "complete": any(record.get("type") in {"artifact", "lab", "build", "review"} for record in records),
            },
            {
                "id": "explanation",
                "label": "Can explain what happened",
                "complete": any(record.get("explanation") for record in records),
            },
            {
                "id": "transfer",
                "label": "Unseen transfer demonstrated",
                "complete": any(record.get("transfer") for record in records),
            },
            {
                "id": "no_ai",
                "label": "No-AI competence demonstrated",
                "complete": any(record.get("no_ai") for record in records),
            },
        ]

    def _workspace_artifacts(self, mission_id: str) -> list[dict[str, str]]:
        artifacts: list[dict[str, str]] = []
        labs = self.root / "labs"
        if labs.exists():
            for path in sorted(labs.glob(f"{mission_id}_*.ipynb")):
                artifacts.append({"kind": "notebook", "label": path.name, "path": path.relative_to(self.root).as_posix()})

        mission_dir = self.root / "missions" / mission_id
        for name, label in (
            ("README.md", "Mission brief"),
            ("code_reading.md", "Code reading"),
            ("controlled_failure.md", "Controlled failure"),
            ("no_ai_gate.md", "No-AI gate"),
            ("flagship_integration.md", "Flagship integration"),
            ("review_brief.md", "Engineering review"),
            ("adr_prompt.md", "ADR prompt"),
        ):
            path = mission_dir / name
            if path.exists():
                artifacts.append({"kind": "document", "label": label, "path": path.relative_to(self.root).as_posix()})
        return artifacts

    def snapshot(self, mission_id: str | None = None) -> dict[str, Any]:
        raw = self.loop.runner.status()
        active = (raw.get("mission") or {}).get("id")
        missions = self.loop.missions.all()
        selected = (mission_id or active or (missions[0]["id"] if missions else None))
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

        all_evidence = self.loop.store.evidence()
        recent_evidence = list(reversed(all_evidence))[:20]
        selected_evidence = list(reversed(self.loop.evidence.for_mission(selected))) if selected else []
        due = self.loop.retention.due()
        open_side_quests = self.loop.side_quests.active()
        mission_side_quests = self.loop.side_quests.active(selected) if selected else []
        decision = self.loop.step(active)
        autonomy = self.loop.autonomy.status()
        mission_status = raw.get("learner", {}).get("mission_status", {})
        passed = sorted(mid for mid, status in mission_status.items() if status == "PASSED")

        context = None
        if selected:
            context = MissionContextAssembler(self.root, self.loop.gates).build(selected)

        mission_rows: list[dict[str, Any]] = []
        phase_progress: dict[str, dict[str, int]] = {}
        for mission in missions:
            mid = mission["id"]
            status = mission_status.get(mid, "NOT_STARTED")
            if mid == active and status != "PASSED":
                status = "CURRENT"
            row = dict(mission)
            row["status"] = status
            row["selected"] = mid == selected
            row["recommended"] = decision.get("target") == mid
            mission_rows.append(row)
            phase = mission.get("phase", "")
            bucket = phase_progress.setdefault(phase, {"passed": 0, "total": 0})
            bucket["total"] += 1
            if mission_status.get(mid) == "PASSED":
                bucket["passed"] += 1

        lab_payload = json.loads((self.root / "data" / "lab_status.json").read_text(encoding="utf-8"))
        return {
            "runtime": {
                "status": "closed-loop",
                "current_mission": active,
                "selected_mission": selected,
                "next_action": decision,
            },
            "progress": {
                "passed_missions": passed,
                "passed_count": len(passed),
                "total_missions": len(missions),
                "completion_ratio": round(len(passed) / max(1, len(missions)), 4),
                "by_phase": phase_progress,
            },
            "missions": mission_rows,
            "mission": context,
            "workspace": {
                "artifacts": self._workspace_artifacts(selected) if selected else [],
                "evidence": selected_evidence,
                "gate_checklist": self._gate_checklist(selected_evidence),
                "side_quests": mission_side_quests,
            },
            "learner_model": {
                "autonomy_level": learner.get("autonomy_level", "A1"),
                "competencies": competency_rows,
                "misconceptions": learner.get("misconceptions", []),
                "gate_history": learner.get("gate_history", {}),
            },
            "autonomy": autonomy,
            "retention_due": due,
            "open_side_quests": open_side_quests,
            "recent_evidence": recent_evidence,
            "labs": {
                "repository_executable": lab_payload.get("repository_executable", []),
                "source_package_executable": lab_payload.get("source_package_executable", []),
                "repository_executable_count": len(lab_payload.get("repository_executable", [])),
                "source_package_executable_count": len(lab_payload.get("source_package_executable", [])),
                "note": lab_payload.get("note", ""),
            },
        }
