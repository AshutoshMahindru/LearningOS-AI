"""Small, deterministic vector instruments for mission M15.

The module deliberately requires callers to choose a metric. It never silently
normalizes because magnitude may be part of the representation's meaning.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math
from typing import Any


Vector = tuple[float, ...]


def as_vector(values: Sequence[float], *, name: str = "vector") -> Vector:
    """Return a finite, nonempty, one-dimensional vector of floats."""

    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of numbers")
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only numbers") from exc
    if not vector:
        raise ValueError(f"{name} must not be empty")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _pair(left: Sequence[float], right: Sequence[float]) -> tuple[Vector, Vector]:
    left_vector = as_vector(left, name="left vector")
    right_vector = as_vector(right, name="right vector")
    if len(left_vector) != len(right_vector):
        raise ValueError("vectors must have the same dimensions")
    return left_vector, right_vector


def add(left: Sequence[float], right: Sequence[float]) -> Vector:
    """Add vectors coordinate by coordinate."""

    left_vector, right_vector = _pair(left, right)
    return tuple(a + b for a, b in zip(left_vector, right_vector, strict=True))


def subtract(left: Sequence[float], right: Sequence[float]) -> Vector:
    """Subtract ``right`` from ``left`` coordinate by coordinate."""

    left_vector, right_vector = _pair(left, right)
    return tuple(a - b for a, b in zip(left_vector, right_vector, strict=True))


def norm(vector: Sequence[float]) -> float:
    """Compute the Euclidean (L2) magnitude of a vector."""

    checked = as_vector(vector)
    return math.sqrt(math.fsum(value * value for value in checked))


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute magnitude-sensitive alignment."""

    left_vector, right_vector = _pair(left, right)
    return math.fsum(
        a * b for a, b in zip(left_vector, right_vector, strict=True)
    )


def normalize(vector: Sequence[float]) -> Vector:
    """Return a unit vector; reject the directionless zero vector."""

    checked = as_vector(vector)
    magnitude = norm(checked)
    if math.isclose(magnitude, 0.0, abs_tol=1e-12):
        raise ValueError("cannot normalize a zero vector")
    return tuple(value / magnitude for value in checked)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Measure directional agreement in [-1, 1] for nonzero vectors."""

    left_vector, right_vector = _pair(left, right)
    denominator = norm(left_vector) * norm(right_vector)
    if math.isclose(denominator, 0.0, abs_tol=1e-12):
        raise ValueError("cosine similarity is undefined for a zero vector")
    score = dot(left_vector, right_vector) / denominator
    return max(-1.0, min(1.0, score))


def euclidean_distance(left: Sequence[float], right: Sequence[float]) -> float:
    """Measure straight-line distance between equal-dimensional vectors."""

    return norm(subtract(left, right))


def compare_vectors(
    query: Sequence[float], candidate: Sequence[float]
) -> dict[str, float]:
    """Emit the core pairwise measurements used by the V04 lab."""

    query_vector, candidate_vector = _pair(query, candidate)
    return {
        "query_norm": norm(query_vector),
        "candidate_norm": norm(candidate_vector),
        "dot": dot(query_vector, candidate_vector),
        "cosine": cosine_similarity(query_vector, candidate_vector),
        "euclidean": euclidean_distance(query_vector, candidate_vector),
    }


def rank_vectors(
    query: Sequence[float],
    candidates: Iterable[Mapping[str, Any]],
    *,
    metric: str,
) -> list[dict[str, float | str]]:
    """Rank named candidate vectors with explicit metric and stable ID ties.

    Cosine similarity and dot product are ordered high-to-low. Euclidean
    distance is ordered low-to-high. Candidate mappings must contain unique
    string ``id`` values and a ``vector`` sequence.
    """

    query_vector = as_vector(query, name="query")
    scorers = {
        "cosine": cosine_similarity,
        "dot": dot,
        "euclidean": euclidean_distance,
    }
    if metric not in scorers:
        raise ValueError("metric must be one of: cosine, dot, euclidean")

    scored: list[dict[str, float | str]] = []
    seen_ids: set[str] = set()
    for candidate in candidates:
        identifier = candidate.get("id")
        values = candidate.get("vector")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("each candidate must have a nonempty string id")
        if identifier in seen_ids:
            raise ValueError(f"duplicate candidate id: {identifier}")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(f"candidate {identifier} must have a vector sequence")
        candidate_vector = as_vector(values, name=f"candidate {identifier}")
        score = scorers[metric](query_vector, candidate_vector)
        scored.append({"id": identifier, "score": score})
        seen_ids.add(identifier)

    if metric in {"cosine", "dot"}:
        return sorted(scored, key=lambda item: (-float(item["score"]), str(item["id"])))
    return sorted(scored, key=lambda item: (float(item["score"]), str(item["id"])))
