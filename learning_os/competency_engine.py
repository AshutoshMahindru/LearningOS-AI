from __future__ import annotations

from typing import Any

from .storage import StateStore


class CompetencyEngine:
    """Evidence-backed L0-L5 competency state."""

    def __init__(self, store: StateStore) -> None:
        self.store = store

    def state(self) -> dict[str, Any]:
        return self.store.competencies()

    def record(self, competency: str, level: int, evidence_id: str, confidence: float = 0.5) -> None:
        if not 0 <= level <= 5:
            raise ValueError("Competency level must be between 0 and 5")
        state = self.store.competencies()
        item = state.setdefault(competency, {"level": 0, "confidence": 0.0, "evidence": []})
        item["level"] = max(item["level"], level)
        item["confidence"] = max(item["confidence"], min(max(confidence, 0.0), 1.0))
        if evidence_id not in item["evidence"]:
            item["evidence"].append(evidence_id)
        self.store.save_competencies(state)
