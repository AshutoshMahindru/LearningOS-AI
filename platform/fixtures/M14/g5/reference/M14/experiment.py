"""Mission-local M14 scale-trap diagnostic. Not a platform API route.

Worker execute uses SCALE_EXECUTE_SOURCE (no imports). This module does not
import pandas or sklearn at module level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent
SESSIONS_CSV = PACKAGE_ROOT / "data" / "learning_sessions.csv"

SCALE_EXECUTE_SOURCE = """
rows = parameters["rows"]
features = parameters["features"]
pair = parameters["pair"]
left, right = pair

def sq(a, b):
    delta = a - b
    return delta * delta

contrib = {}
for name in features:
    contrib[name] = sq(left[name], right[name])
total = 0.0
for value in contrib.values():
    total += value
shares = {}
for name, value in contrib.items():
    shares[name] = value / total if total else 0.0
dominant = None
best = -1.0
for name, share in shares.items():
    if share > best:
        dominant = name
        best = share
payload = {
    "dominant_feature": dominant,
    "dominant_share": best,
    "squared_contributions": contrib,
    "shares": shares,
    "clusters_are_true_classes": False,
    "n_rows": len(rows),
}
print(payload)
[
    {
        "type": "table",
        "title": "raw_distance_budget",
        "payload": {
            "columns": ["feature", "squared_contribution", "share"],
            "rows": [[name, contrib[name], shares[name]] for name in features],
        },
    },
    {"type": "metric", "title": "scale_trap", "payload": payload},
]
"""


def sessions_csv_text() -> str:
    return SESSIONS_CSV.read_text(encoding="utf-8")


def default_parameters() -> dict[str, Any]:
    return {
        "features": ["practice_ratio", "activity_events"],
        "pair": [
            {"practice_ratio": 0.18, "activity_events": 910},
            {"practice_ratio": 0.22, "activity_events": 4050},
        ],
        "rows": [{"id": "LS001"}, {"id": "LS002"}],
    }


def run_scale_trap(parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Quantify raw squared-distance dominance. No pandas."""

    params = parameters or default_parameters()
    features = params["features"]
    left, right = params["pair"]
    contrib = {name: (left[name] - right[name]) ** 2 for name in features}
    total = sum(contrib.values())
    shares = {name: (value / total if total else 0.0) for name, value in contrib.items()}
    dominant = max(shares, key=shares.get)
    payload = {
        "dominant_feature": dominant,
        "dominant_share": shares[dominant],
        "squared_contributions": contrib,
        "shares": shares,
        "clusters_are_true_classes": False,
        "n_rows": len(params["rows"]),
    }
    return [
        {
            "type": "table",
            "title": "raw_distance_budget",
            "payload": {
                "columns": ["feature", "squared_contribution", "share"],
                "rows": [[name, contrib[name], shares[name]] for name in features],
            },
        },
        {"type": "metric", "title": "scale_trap", "payload": payload},
    ]
