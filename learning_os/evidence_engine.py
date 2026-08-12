from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .storage import StateStore


class EvidenceEngine:
    def __init__(self, store: StateStore) -> None:
        self.store = store

    def add(
        self,
        mission_id: str,
        evidence_type: str,
        summary: str,
        competencies: list[str] | None = None,
        no_ai: bool = False,
        transfer: bool = False,
        explanation: bool = False,
    ) -> dict[str, Any]:
        record = {
            "id": f"EV-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            "mission_id": mission_id.upper(),
            "type": evidence_type,
            "summary": summary,
            "competencies": competencies or [],
            "no_ai": no_ai,
            "transfer": transfer,
            "explanation": explanation,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        records = self.store.evidence()
        records.append(record)
        self.store.save_evidence(records)
        return record

    def for_mission(self, mission_id: str) -> list[dict[str, Any]]:
        mid = mission_id.upper()
        return [e for e in self.store.evidence() if e.get("mission_id") == mid]
