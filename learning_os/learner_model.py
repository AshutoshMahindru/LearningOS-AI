from __future__ import annotations

from typing import Any

from .storage import StateStore


class LearnerModelEngine:
    """Maintains an evidence-derived learner model separate from raw activity logs."""

    def __init__(self, store: StateStore) -> None:
        self.store = store

    def get(self) -> dict[str, Any]:
        return self.store.read(
            "learner_model.json",
            {
                "learner_id": "default",
                "current_mission": None,
                "autonomy_level": "A1",
                "competencies": {},
                "misconceptions": [],
                "confidence": {},
                "retention_due": [],
                "open_side_quests": [],
                "evidence_summary": {},
                "gate_history": {},
                "autonomy_history": [],
            },
        )

    def save(self, model: dict[str, Any]) -> None:
        self.store.write("learner_model.json", model)

    def set_current_mission(self, mission_id: str) -> None:
        model = self.get()
        model["current_mission"] = mission_id.upper()
        self.save(model)

    @staticmethod
    def _evidence_level(record: dict[str, Any]) -> int:
        level = 1
        if record.get("explanation"):
            level = max(level, 2)
        if record.get("type") in {"artifact", "lab", "build", "review", "design"}:
            level = max(level, 3)
        if record.get("no_ai") and record.get("transfer") and record.get("explanation"):
            level = max(level, 4)
        if record.get("type") in {"review", "design"} and level >= 4:
            level = 5
        return level

    @staticmethod
    def _confidence(record: dict[str, Any]) -> float:
        signals = sum(bool(record.get(key)) for key in ("no_ai", "transfer", "explanation"))
        return min(1.0, 0.35 + 0.15 * signals)

    def ingest_evidence(self, record: dict[str, Any]) -> dict[str, Any]:
        model = self.get()
        mission_id = record["mission_id"]
        summary = model.setdefault("evidence_summary", {}).setdefault(
            mission_id,
            {"records": 0, "no_ai": 0, "transfer": 0, "explanation": 0},
        )
        summary["records"] += 1
        for key in ("no_ai", "transfer", "explanation"):
            if record.get(key):
                summary[key] += 1

        level = self._evidence_level(record)
        confidence = self._confidence(record)
        for competency in record.get("competencies", []):
            item = model.setdefault("competencies", {}).setdefault(
                competency,
                {"level": 0, "confidence": 0.0, "evidence": []},
            )
            item["level"] = max(item["level"], level)
            item["confidence"] = max(item["confidence"], confidence)
            if record["id"] not in item["evidence"]:
                item["evidence"].append(record["id"])
            model.setdefault("confidence", {})[competency] = item["confidence"]

        self.save(model)
        return model

    def record_gate(self, mission_id: str, status: str, reasons: list[str]) -> dict[str, Any]:
        model = self.get()
        history = model.setdefault("gate_history", {}).setdefault(
            mission_id.upper(), {"attempts": 0, "passes": 0, "failures": 0, "last_status": None, "last_reasons": []}
        )
        history["attempts"] += 1
        history["last_status"] = status
        history["last_reasons"] = reasons
        if status == "PASS":
            history["passes"] += 1
        elif status == "FAIL":
            history["failures"] += 1
        self.save(model)
        return history

    def add_misconception(self, concept: str, note: str) -> None:
        model = self.get()
        item = {"concept": concept, "note": note, "status": "OPEN"}
        if item not in model.setdefault("misconceptions", []):
            model["misconceptions"].append(item)
        self.save(model)
