from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MissionRepository:
    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root)
        path = self.root / "data" / "missions.json"
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        self._missions = {m["id"]: m for m in payload["missions"]}

    def get(self, mission_id: str) -> dict[str, Any]:
        key = mission_id.upper()
        if key not in self._missions:
            raise KeyError(f"Unknown mission: {mission_id}")
        return self._missions[key]

    def all(self) -> list[dict[str, Any]]:
        return list(self._missions.values())

    def next_after(self, mission_id: str) -> dict[str, Any] | None:
        missions = self.all()
        for index, mission in enumerate(missions):
            if mission["id"] == mission_id.upper():
                return missions[index + 1] if index + 1 < len(missions) else None
        raise KeyError(f"Unknown mission: {mission_id}")
