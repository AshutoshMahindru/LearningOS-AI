from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class PrerequisiteGraph:
    def __init__(self, root: str | Path = ".") -> None:
        path = Path(root) / "data" / "mission_dependencies.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.dependencies: dict[str, dict[str, list[str]]] = payload["dependencies"]

    def blocking(self, mission_id: str) -> list[str]:
        return list(self.dependencies[mission_id.upper()]["blocking"])

    def helpful(self, mission_id: str) -> list[str]:
        return list(self.dependencies[mission_id.upper()]["helpful"])

    def unmet(self, mission_id: str, mission_status: dict[str, Any]) -> list[str]:
        return [m for m in self.blocking(mission_id) if mission_status.get(m) != "PASSED"]

    def transitive_blocking(self, mission_id: str) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()

        def visit(mid: str) -> None:
            for dep in self.blocking(mid):
                if dep not in seen:
                    seen.add(dep)
                    visit(dep)
                    result.append(dep)

        visit(mission_id.upper())
        return result

    def narrowest_unmet(self, mission_id: str, mission_status: dict[str, Any]) -> str | None:
        direct = self.unmet(mission_id, mission_status)
        if direct:
            return direct[0]
        return None
