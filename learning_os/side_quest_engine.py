from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .storage import StateStore


class SideQuestEngine:
    def __init__(self, store: StateStore) -> None:
        self.store = store

    def all(self) -> list[dict[str, Any]]:
        return self.store.read("side_quests.json", [])

    def open(
        self,
        mission_id: str,
        target: str,
        reason: str,
        return_target: str,
        expected_minutes: int = 60,
    ) -> dict[str, Any]:
        if expected_minutes > 120:
            raise ValueError("Side quests are bounded to 120 minutes by default.")
        quests = self.all()
        quest = {
            "id": f"SQ-{len(quests)+1:05d}",
            "mission_id": mission_id.upper(),
            "target": target,
            "reason": reason,
            "return_target": return_target,
            "expected_minutes": expected_minutes,
            "status": "OPEN",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        quests.append(quest)
        self.store.write("side_quests.json", quests)
        self._sync_model(quests)
        return quest

    def active(self, mission_id: str | None = None) -> list[dict[str, Any]]:
        quests = [quest for quest in self.all() if quest["status"] == "OPEN"]
        if mission_id:
            quests = [quest for quest in quests if quest["mission_id"] == mission_id.upper()]
        return quests

    def close(self, quest_id: str, micro_assessment: str, outcome: str) -> dict[str, Any]:
        if micro_assessment not in {"PASS", "PARTIAL", "FAIL"}:
            raise ValueError("micro_assessment must be PASS, PARTIAL, or FAIL")
        quests = self.all()
        quest = next((item for item in quests if item["id"] == quest_id), None)
        if quest is None:
            raise KeyError(f"Unknown side quest: {quest_id}")
        quest["micro_assessment"] = micro_assessment
        quest["outcome"] = outcome
        if micro_assessment == "PASS":
            quest["status"] = "CLOSED"
            quest["closed_at"] = datetime.now(timezone.utc).isoformat()
        self.store.write("side_quests.json", quests)
        self._sync_model(quests)
        return quest

    def _sync_model(self, quests: list[dict[str, Any]]) -> None:
        model = self.store.read("learner_model.json", {})
        if model:
            model["open_side_quests"] = [quest["id"] for quest in quests if quest["status"] == "OPEN"]
            self.store.write("learner_model.json", model)
