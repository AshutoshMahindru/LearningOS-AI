from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LabRegistry:
    def __init__(self, root: str | Path = ".") -> None:
        path = Path(root) / "data" / "lab_status.json"
        self.payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))

    def status(self, mission_id: str) -> dict[str, Any]:
        mid = mission_id.upper()
        return {
            "mission_id": mid,
            "repository_executable": mid in self.payload["repository_executable"],
            "source_artifact_available": mid in self.payload["source_package_executable"],
            "specification_only_in_source_package": mid in self.payload["source_package_specification_only"],
        }
