from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .storage import StateStore


class AutonomyEngine:
    def __init__(self, root: str | Path, store: StateStore) -> None:
        self.root = Path(root)
        self.store = store
        policy = json.loads((self.root / "data" / "autonomy_policy.json").read_text(encoding="utf-8"))
        self.levels = [item["level"] for item in policy["levels"]]
        self.policy = policy

    def status(self) -> dict[str, Any]:
        model = self.store.read("learner_model.json", {"autonomy_level": "A1"})
        level = model.get("autonomy_level", "A1")
        detail = next(item for item in self.policy["levels"] if item["level"] == level)
        return {"level": level, **detail}

    def evaluate(self, signals: set[str]) -> dict[str, Any]:
        model = self.store.read("learner_model.json", {"autonomy_level": "A1", "autonomy_history": []})
        current = model.get("autonomy_level", "A1")
        index = self.levels.index(current)
        increase = set(self.policy["ratchet_rules"]["increase"])
        decrease = set(self.policy["ratchet_rules"]["decrease"])
        reason = "No ratchet threshold met."
        new_index = index
        if signals & decrease:
            new_index = max(0, index - 1)
            reason = f"Autonomy reduced because of: {sorted(signals & decrease)}"
        elif increase.issubset(signals):
            new_index = min(len(self.levels) - 1, index + 1)
            reason = "Autonomy increased after independent gate, transfer, and review evidence."
        new_level = self.levels[new_index]
        event = {
            "at": datetime.now(timezone.utc).isoformat(),
            "from": current,
            "to": new_level,
            "signals": sorted(signals),
            "reason": reason,
        }
        model["autonomy_level"] = new_level
        model.setdefault("autonomy_history", []).append(event)
        self.store.write("learner_model.json", model)
        events = self.store.read("autonomy_events.json", [])
        events.append(event)
        self.store.write("autonomy_events.json", events)
        return event
