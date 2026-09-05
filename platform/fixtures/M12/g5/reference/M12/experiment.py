"""Mission-local M12 ensemble comparison. Not a platform API route.

Worker execute uses ENSEMBLE_EXECUTE_SOURCE (no imports). This module does not
import pandas or sklearn at module level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent
ENSEMBLE_CSV = PACKAGE_ROOT / "data" / "ensemble_fixture.csv"

ENSEMBLE_EXECUTE_SOURCE = """
points = parameters["points"]
labels = parameters["labels"]
bootstraps = parameters["bootstraps"]
query = parameters["query"]

def stump_predict(threshold, row):
    return 1 if row[0] > threshold else 0

def majority(votes):
    ones = sum(votes)
    return 1 if ones * 2 >= len(votes) else 0

tree_threshold = 0.0
tree_pred = stump_predict(tree_threshold, query)
bag_votes = []
for sample in bootstraps:
    sample_x = [points[i][0] for i in sample]
    sample_y = [labels[i] for i in sample]
    best_t, best_acc = 0.0, -1.0
    for candidate in sample_x:
        correct = 0
        for x, y in zip(sample_x, sample_y):
            pred = 1 if x > candidate else 0
            if pred == y:
                correct += 1
        acc = correct / len(sample_x)
        if acc > best_acc:
            best_acc = acc
            best_t = candidate
    bag_votes.append(stump_predict(best_t, query))
bag_pred = majority(bag_votes)
disagreement = 0
for vote in bag_votes:
    if vote != bag_pred:
        disagreement += 1
payload = {
    "tree_pred": tree_pred,
    "bag_pred": bag_pred,
    "bag_votes": bag_votes,
    "disagreement": disagreement,
    "n_bootstraps": len(bootstraps),
    "more_trees_repair_corrupted_labels": False,
}
print(payload)
{"type": "metric", "title": "ensemble_comparison", "payload": payload}
"""


def ensemble_csv_text() -> str:
    return ENSEMBLE_CSV.read_text(encoding="utf-8")


def default_parameters() -> dict[str, Any]:
    return {
        "points": [[-1.0], [-0.5], [0.2], [0.8], [1.2], [1.6]],
        "labels": [0, 0, 0, 1, 1, 1],
        "bootstraps": [[0, 1, 2, 3], [1, 2, 3, 4], [0, 3, 4, 5], [0, 1, 4, 5]],
        "query": [0.4],
    }


def run_ensemble_comparison(parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Tiny stdlib bagging-versus-stump comparison. No pandas."""

    params = parameters or default_parameters()
    points = params["points"]
    labels = params["labels"]
    bootstraps = params["bootstraps"]
    query = params["query"]

    def stump_predict(threshold: float, row: list[float]) -> int:
        return 1 if row[0] > threshold else 0

    tree_pred = stump_predict(0.0, query)
    bag_votes: list[int] = []
    for sample in bootstraps:
        sample_x = [points[i][0] for i in sample]
        sample_y = [labels[i] for i in sample]
        best_t, best_acc = 0.0, -1.0
        for candidate in sample_x:
            correct = sum(
                (1 if x > candidate else 0) == y for x, y in zip(sample_x, sample_y)
            )
            acc = correct / len(sample_x)
            if acc > best_acc:
                best_acc = acc
                best_t = candidate
        bag_votes.append(stump_predict(best_t, query))
    ones = sum(bag_votes)
    bag_pred = 1 if ones * 2 >= len(bag_votes) else 0
    disagreement = sum(vote != bag_pred for vote in bag_votes)
    payload = {
        "tree_pred": tree_pred,
        "bag_pred": bag_pred,
        "bag_votes": bag_votes,
        "disagreement": disagreement,
        "n_bootstraps": len(bootstraps),
        "more_trees_repair_corrupted_labels": False,
    }
    return [
        {"type": "metric", "title": "ensemble_comparison", "payload": payload},
        {
            "type": "table",
            "title": "bootstrap_votes",
            "payload": {
                "columns": ["bootstrap", "vote"],
                "rows": [[index, vote] for index, vote in enumerate(bag_votes)],
            },
        },
    ]
