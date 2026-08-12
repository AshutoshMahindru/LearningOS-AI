from __future__ import annotations

import csv
import re
from collections import deque
from pathlib import Path
from typing import Any


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _split_refs(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split("|") if part.strip()]


def _tokens(value: str) -> set[str]:
    return set(_TOKEN_RE.findall(value.casefold()))


class KnowledgeGraph:
    """Runtime adapter for the canonical 253-node concept dependency graph.

    The canonical authored sources live in data/knowledge_graph.yaml and
    data/knowledge_graph.csv. Runtime code reads CSV using the Python standard
    library so graph access does not add a YAML dependency to the product.
    Relationships in the canonical source are concept names; this adapter
    resolves them to stable node IDs at read time without rewriting source data.
    """

    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root)
        path = self.root / "data" / "knowledge_graph.csv"
        if not path.exists():
            raise FileNotFoundError(f"Canonical knowledge graph missing: {path}")

        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        self._nodes: list[dict[str, Any]] = []
        self._by_id: dict[str, dict[str, Any]] = {}
        self._by_concept: dict[str, dict[str, Any]] = {}

        for row in rows:
            node = {
                "id": row["id"].strip(),
                "domain": row["domain"].strip(),
                "concept": row["concept"].strip(),
                "prerequisite_concepts": _split_refs(row.get("prerequisites")),
                "enable_concepts": _split_refs(row.get("enables")),
                "first_mission": row.get("first_mission", "").strip() or None,
                "minimum_viable_understanding": row.get("minimum_viable_understanding", "").strip(),
                "defer_until_later": row.get("defer_until_later", "").strip(),
            }
            if node["id"] in self._by_id:
                raise ValueError(f"Duplicate knowledge node id: {node['id']}")
            key = node["concept"].casefold()
            if key in self._by_concept:
                raise ValueError(f"Duplicate knowledge concept: {node['concept']}")
            self._nodes.append(node)
            self._by_id[node["id"]] = node
            self._by_concept[key] = node

        # Resolve authored concept-name relationships once, preserving source order.
        for node in self._nodes:
            node["prerequisites"] = [self.resolve(ref)["id"] for ref in node["prerequisite_concepts"]]
            node["enables"] = [self.resolve(ref)["id"] for ref in node["enable_concepts"]]

    def __len__(self) -> int:
        return len(self._nodes)

    def nodes(self) -> list[dict[str, Any]]:
        return [dict(node) for node in self._nodes]

    def resolve(self, ref: str) -> dict[str, Any]:
        candidate = ref.strip()
        if candidate in self._by_id:
            return self._by_id[candidate]
        node = self._by_concept.get(candidate.casefold())
        if node is None:
            raise KeyError(f"Unknown knowledge node or concept: {ref}")
        return node

    def get(self, ref: str) -> dict[str, Any]:
        return dict(self.resolve(ref))

    def nodes_for_mission(self, mission_id: str) -> list[dict[str, Any]]:
        mid = mission_id.upper()
        return [dict(node) for node in self._nodes if node["first_mission"] == mid]

    def prerequisites(self, ref: str, transitive: bool = False) -> list[dict[str, Any]]:
        node = self.resolve(ref)
        if not transitive:
            return [dict(self._by_id[node_id]) for node_id in node["prerequisites"]]
        return self._walk(node["prerequisites"], "prerequisites")

    def enables(self, ref: str, transitive: bool = False) -> list[dict[str, Any]]:
        node = self.resolve(ref)
        if not transitive:
            return [dict(self._by_id[node_id]) for node_id in node["enables"]]
        return self._walk(node["enables"], "enables")

    def _walk(self, seeds: list[str], edge: str) -> list[dict[str, Any]]:
        queue = deque(seeds)
        seen: set[str] = set()
        ordered: list[dict[str, Any]] = []
        while queue:
            node_id = queue.popleft()
            if node_id in seen:
                continue
            seen.add(node_id)
            node = self._by_id[node_id]
            ordered.append(dict(node))
            queue.extend(node[edge])
        return ordered

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Small deterministic concept matcher used for blocker/context expansion."""
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        q = query.casefold().strip()
        scored: list[tuple[float, int, dict[str, Any]]] = []
        for index, node in enumerate(self._nodes):
            concept = node["concept"].casefold()
            text = " ".join(
                [node["concept"], node["domain"], node["minimum_viable_understanding"]]
            )
            node_tokens = _tokens(text)
            overlap = len(query_tokens & node_tokens)
            if not overlap and q not in concept and concept not in q:
                continue
            score = overlap / max(1, len(query_tokens))
            if q == concept:
                score += 2.0
            elif q in concept or concept in q:
                score += 1.0
            scored.append((score, -index, node))
        scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
        return [dict(item[2]) for item in scored[:limit]]

    def context_for(self, query: str, prerequisite_limit: int = 8) -> dict[str, Any] | None:
        matches = self.search(query, limit=1)
        if not matches:
            return None
        node = matches[0]
        prerequisites = self.prerequisites(node["id"], transitive=True)[:prerequisite_limit]
        return {
            "node": node,
            "prerequisites": prerequisites,
        }
