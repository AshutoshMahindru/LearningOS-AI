from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class StateStore:
    """Small JSON persistence layer for the local-first Learning OS runtime."""

    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root)
        self.tracking = self.root / "tracking"
        self.tracking.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.tracking / name

    def read(self, name: str, default: Any) -> Any:
        path = self._path(name)
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def write(self, name: str, value: Any) -> None:
        path = self._path(name)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(value, fh, indent=2, ensure_ascii=False)
            fh.write("\n")

    def learner(self) -> dict[str, Any]:
        return self.read("learner_state.json", {"learner_id": "default", "current_mission": None, "mission_status": {}, "blockers": []})

    def save_learner(self, value: dict[str, Any]) -> None:
        self.write("learner_state.json", value)

    def evidence(self) -> list[dict[str, Any]]:
        return self.read("evidence.json", [])

    def save_evidence(self, value: list[dict[str, Any]]) -> None:
        self.write("evidence.json", value)

    def sessions(self) -> list[dict[str, Any]]:
        return self.read("sessions.json", [])

    def save_sessions(self, value: list[dict[str, Any]]) -> None:
        self.write("sessions.json", value)

    def competencies(self) -> dict[str, Any]:
        return self.read("competencies.json", {})

    def save_competencies(self, value: dict[str, Any]) -> None:
        self.write("competencies.json", value)
