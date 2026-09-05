"""Mission-local M13 neighbor query. Not a platform API route.

Worker execute uses KNN_EXECUTE_SOURCE (no imports). This module does not import
pandas or sklearn at module level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent
CASES_CSV = PACKAGE_ROOT / "data" / "knn_scale_cases.csv"

KNN_EXECUTE_SOURCE = """
points = parameters["points"]
query = parameters["query"]
k = int(parameters.get("k") or 3)

def dist(a, b):
    total = 0.0
    for left, right in zip(a, b):
        delta = left - right
        total += delta * delta
    return total ** 0.5

ranked = []
for item in points:
    ranked.append((dist(query, item["xy"]), item["id"], item["label"], item["xy"]))
ranked.sort(key=lambda row: (row[0], row[1]))
neighbors = ranked[:k]
votes = {}
for _distance, _nid, label, _xy in neighbors:
    votes[label] = votes.get(label, 0) + 1
predicted = None
best = -1
for label, count in votes.items():
    if count > best:
        predicted = label
        best = count
payload = {
    "k": k,
    "predicted_class": predicted,
    "neighbor_ids": [item[1] for item in neighbors],
    "distances": [item[0] for item in neighbors],
    "votes": votes,
}
print(payload)
{"type": "metric", "title": "knn_query", "payload": payload}
"""


def cases_csv_text() -> str:
    return CASES_CSV.read_text(encoding="utf-8")


def default_parameters() -> dict[str, Any]:
    return {
        "k": 3,
        "query": [6, 6],
        "points": [
            {"id": "g1", "xy": [1, 8], "label": "guided"},
            {"id": "g2", "xy": [2, 7], "label": "guided"},
            {"id": "g3", "xy": [3, 9], "label": "guided"},
            {"id": "g4", "xy": [7, 3], "label": "guided"},
            {"id": "i1", "xy": [7, 8], "label": "independent"},
            {"id": "i2", "xy": [8, 7], "label": "independent"},
            {"id": "i3", "xy": [9, 9], "label": "independent"},
            {"id": "i4", "xy": [3, 2], "label": "independent"},
        ],
    }


def run_knn_query(parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Hand-sized Euclidean kNN. No pandas."""

    params = parameters or default_parameters()
    points = params["points"]
    query = params["query"]
    k = int(params.get("k") or 3)

    def dist(left: list[float], right: list[float]) -> float:
        return sum((a - b) ** 2 for a, b in zip(left, right)) ** 0.5

    ranked = sorted(
        (
            (dist(query, item["xy"]), item["id"], item["label"], item["xy"])
            for item in points
        ),
        key=lambda row: (row[0], row[1]),
    )
    neighbors = ranked[:k]
    votes: dict[str, int] = {}
    for _distance, _nid, label, _xy in neighbors:
        votes[label] = votes.get(label, 0) + 1
    predicted = max(votes, key=votes.get)
    payload = {
        "k": k,
        "predicted_class": predicted,
        "neighbor_ids": [item[1] for item in neighbors],
        "distances": [item[0] for item in neighbors],
        "votes": votes,
    }
    return [
        {"type": "metric", "title": "knn_query", "payload": payload},
        {
            "type": "table",
            "title": "neighbors",
            "payload": {
                "columns": ["id", "distance", "label"],
                "rows": [[item[1], item[0], item[2]] for item in neighbors],
            },
        },
    ]
