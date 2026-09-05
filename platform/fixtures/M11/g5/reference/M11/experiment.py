"""Mission-local M11 path tracer. Not a platform API route.

Worker execute uses PATH_EXECUTE_SOURCE (no imports) so the frozen sandbox never
imports pandas or sklearn. This module does not import pandas at module level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent
READINESS_CSV = PACKAGE_ROOT / "data" / "learner_readiness.csv"

PATH_EXECUTE_SOURCE = """
query = parameters["query"]
path = []
if query["practice_accuracy"] <= 0.73:
    path.append("practice_accuracy <= 0.73")
    if query["attendance_pct"] <= 86.50:
        path.append("attendance_pct <= 86.50")
        predicted = 0
    else:
        path.append("attendance_pct > 86.50")
        predicted = 1
else:
    path.append("practice_accuracy > 0.73")
    if query["study_hours_week"] <= 4.50:
        path.append("study_hours_week <= 4.50")
        predicted = 0
    else:
        path.append("study_hours_week > 4.50")
        predicted = 1
payload = {
    "path": path,
    "predicted_class": predicted,
    "leaf_is_training_distribution": True,
    "causal_claim_licensed": False,
}
print(payload)
{"type": "trace", "title": "decision_path", "payload": payload}
"""


def readiness_csv_text() -> str:
    return READINESS_CSV.read_text(encoding="utf-8")


def _trace_row(query: dict[str, Any]) -> dict[str, Any]:
    path: list[str] = []
    if query["practice_accuracy"] <= 0.73:
        path.append("practice_accuracy <= 0.73")
        if query["attendance_pct"] <= 86.50:
            path.append("attendance_pct <= 86.50")
            predicted = 0
        else:
            path.append("attendance_pct > 86.50")
            predicted = 1
    else:
        path.append("practice_accuracy > 0.73")
        if query["study_hours_week"] <= 4.50:
            path.append("study_hours_week <= 4.50")
            predicted = 0
        else:
            path.append("study_hours_week > 4.50")
            predicted = 1
    return {
        "path": path,
        "predicted_class": predicted,
        "leaf_is_training_distribution": True,
        "causal_claim_licensed": False,
    }


def run_path_trace(query: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return WP-137 trace/metric blocks for the no-AI textual tree. No pandas."""

    payload = _trace_row(
        query
        or {
            "study_hours_week": 6,
            "practice_accuracy": 0.70,
            "attendance_pct": 91,
        }
    )
    return [
        {"type": "trace", "title": "decision_path", "payload": payload},
        {
            "type": "metric",
            "title": "path_summary",
            "payload": {
                "predicted_class": payload["predicted_class"],
                "n_comparisons": len(payload["path"]),
                "causal_claim_licensed": False,
            },
        },
    ]
