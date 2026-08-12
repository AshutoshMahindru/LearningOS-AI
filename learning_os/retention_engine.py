from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from .storage import StateStore


INTERVALS_DAYS = (7, 21, 90)


class RetentionEngine:
    def __init__(self, store: StateStore) -> None:
        self.store = store

    def events(self) -> list[dict[str, Any]]:
        return self.store.read("retention_events.json", [])

    def _save(self, events: list[dict[str, Any]]) -> None:
        self.store.write("retention_events.json", events)
        self.refresh_model_due()

    def schedule(
        self,
        competency: str,
        mission_id: str,
        stage: int = 0,
        now: datetime | None = None,
        due_in_days: int | None = None,
    ) -> dict[str, Any]:
        now = now or datetime.now(timezone.utc)
        events = self.events()
        for event in events:
            if event["competency"] == competency and event["mission_id"] == mission_id.upper() and event["status"] in {"SCHEDULED", "DUE"}:
                return event
        interval = due_in_days if due_in_days is not None else INTERVALS_DAYS[min(stage, len(INTERVALS_DAYS) - 1)]
        event = {
            "id": f"RT-{len(events)+1:05d}",
            "competency": competency,
            "mission_id": mission_id.upper(),
            "stage": stage,
            "scheduled_at": now.isoformat(),
            "due_at": (now + timedelta(days=interval)).isoformat(),
            "status": "SCHEDULED",
        }
        events.append(event)
        self._save(events)
        return event

    def due(self, now: datetime | None = None) -> list[dict[str, Any]]:
        now = now or datetime.now(timezone.utc)
        events = self.events()
        changed = False
        due_events: list[dict[str, Any]] = []
        for event in events:
            if event["status"] in {"SCHEDULED", "DUE"} and datetime.fromisoformat(event["due_at"]) <= now:
                if event["status"] != "DUE":
                    event["status"] = "DUE"
                    changed = True
                due_events.append(event)
        if changed:
            self._save(events)
        else:
            self.refresh_model_due(due_events)
        return due_events

    def complete(self, event_id: str, passed: bool, now: datetime | None = None) -> dict[str, Any]:
        now = now or datetime.now(timezone.utc)
        events = self.events()
        target = next((event for event in events if event["id"] == event_id), None)
        if target is None:
            raise KeyError(f"Unknown retention event: {event_id}")
        target["status"] = "PASSED" if passed else "FAILED"
        target["completed_at"] = now.isoformat()
        self.store.write("retention_events.json", events)
        if passed and target["stage"] + 1 < len(INTERVALS_DAYS):
            self.schedule(target["competency"], target["mission_id"], stage=target["stage"] + 1, now=now)
        elif not passed:
            self.schedule(target["competency"], target["mission_id"], stage=target["stage"], now=now, due_in_days=1)
        self.refresh_model_due()
        return target

    def refresh_model_due(self, due_events: list[dict[str, Any]] | None = None) -> None:
        model = self.store.read("learner_model.json", {})
        if not model:
            return
        if due_events is None:
            now = datetime.now(timezone.utc)
            due_events = [
                event for event in self.events()
                if event["status"] in {"SCHEDULED", "DUE"} and datetime.fromisoformat(event["due_at"]) <= now
            ]
        model["retention_due"] = [event["id"] for event in due_events]
        self.store.write("learner_model.json", model)
