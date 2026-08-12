from __future__ import annotations

from pathlib import Path
from typing import Any

from .retrieval import LocalVectorRetriever, RetrievalBackend
from .source_registry import SourceRegistry


DEPTH_ORDER = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5}


class ContentRouter:
    def __init__(self, root: str | Path = ".", backend: RetrievalBackend | None = None) -> None:
        self.registry = SourceRegistry(root)
        self.backend = backend or LocalVectorRetriever()

    def route(
        self,
        mission: dict[str, Any],
        blocker: str | None = None,
        max_depth: str | None = None,
        top_k: int = 4,
    ) -> list[dict[str, Any]]:
        query = " ".join([
            mission.get("title", ""),
            mission.get("objective", ""),
            " ".join(mission.get("competencies", [])),
            blocker or "",
        ])
        candidates = self.registry.content_for_mission(mission["id"])
        if blocker:
            candidates = self.registry.content()
        if max_depth:
            ceiling = DEPTH_ORDER[max_depth]
            candidates = [item for item in candidates if DEPTH_ORDER.get(item.get("depth", "L5"), 5) <= ceiling]
        hits = self.backend.search(query, candidates, top_k=top_k)
        routed: list[dict[str, Any]] = []
        for hit in hits:
            item = dict(hit.item)
            item["score"] = round(hit.score, 4)
            item["sources"] = self.registry.sources(item.get("source_ids", []))
            routed.append(item)
        return routed
