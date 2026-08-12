from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SourceRegistry:
    def __init__(self, root: str | Path = ".") -> None:
        base = Path(root) / "data"
        self._sources = self._load_index(base / "source_registry.json", "sources")
        self._content = self._load_index(base / "content_registry.json", "content")

    @staticmethod
    def _load_index(path: Path, key: str) -> dict[str, dict[str, Any]]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {item["id"]: item for item in payload[key]}

    def source(self, source_id: str) -> dict[str, Any]:
        return self._sources[source_id]

    def sources(self, source_ids: list[str]) -> list[dict[str, Any]]:
        return [self.source(source_id) for source_id in source_ids]

    def content(self) -> list[dict[str, Any]]:
        return list(self._content.values())

    def content_for_mission(self, mission_id: str) -> list[dict[str, Any]]:
        mid = mission_id.upper()
        return [item for item in self._content.values() if mid in item.get("missions", [])]
