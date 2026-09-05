from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
M11_PACKAGE = REPO_ROOT / "platform" / "fixtures" / "M11"


def _experiment():
    path = M11_PACKAGE / "g5" / "reference" / "M11" / "experiment.py"
    spec = importlib.util.spec_from_file_location("m11_fixture_experiment_quality", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixture_csv_matches_curriculum_dataset() -> None:
    canonical = REPO_ROOT / "datasets" / "M11" / "learner_readiness.csv"
    experiment = _experiment()
    assert experiment.READINESS_CSV.read_text(encoding="utf-8") == canonical.read_text(encoding="utf-8")


def test_path_trace_matches_no_ai_textual_tree() -> None:
    run_path_trace = _experiment().run_path_trace
    blocks = run_path_trace(
        {"study_hours_week": 6, "practice_accuracy": 0.70, "attendance_pct": 91}
    )
    types = [block["type"] for block in blocks]
    assert types == ["trace", "metric"]
    payload = blocks[0]["payload"]
    assert payload["predicted_class"] == 1
    assert payload["path"] == ["practice_accuracy <= 0.73", "attendance_pct > 86.50"]
    assert payload["causal_claim_licensed"] is False
    second = run_path_trace(
        {"study_hours_week": 4, "practice_accuracy": 0.80, "attendance_pct": 95}
    )
    assert second[0]["payload"]["predicted_class"] == 0
    assert second[0]["payload"]["path"] == [
        "practice_accuracy > 0.73",
        "study_hours_week <= 4.50",
    ]
