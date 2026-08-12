from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Protocol


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+.-]*")


@dataclass(frozen=True)
class SearchHit:
    item: dict[str, Any]
    score: float


class RetrievalBackend(Protocol):
    def search(self, query: str, items: Iterable[dict[str, Any]], top_k: int = 5) -> list[SearchHit]: ...


class LocalVectorRetriever:
    """Dependency-free bag-of-words vector/cosine fallback.

    It is intentionally simple. An external vector backend can implement the same
    RetrievalBackend protocol later without changing the content-router contract.
    """

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return TOKEN_RE.findall(text.lower())

    @classmethod
    def _vector(cls, text: str) -> Counter[str]:
        return Counter(cls._tokens(text))

    @staticmethod
    def _cosine(a: Counter[str], b: Counter[str]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(value * b.get(key, 0) for key, value in a.items())
        norm_a = math.sqrt(sum(value * value for value in a.values()))
        norm_b = math.sqrt(sum(value * value for value in b.values()))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    @staticmethod
    def _document_text(item: dict[str, Any]) -> str:
        parts = [
            item.get("title", ""),
            item.get("summary", ""),
            " ".join(item.get("topics", [])),
            " ".join(item.get("missions", [])),
            item.get("mode", ""),
            item.get("depth", ""),
        ]
        return " ".join(parts)

    def search(self, query: str, items: Iterable[dict[str, Any]], top_k: int = 5) -> list[SearchHit]:
        qv = self._vector(query)
        hits = [SearchHit(item=item, score=self._cosine(qv, self._vector(self._document_text(item)))) for item in items]
        hits.sort(key=lambda hit: (-hit.score, hit.item.get("id", "")))
        return hits[:top_k]
