from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
M14_PACKAGE = REPO_ROOT / "platform" / "fixtures" / "M14"


def _experiment():
    path = M14_PACKAGE / "g5" / "reference" / "M14" / "experiment.py"
    spec = importlib.util.spec_from_file_location("m14_fixture_experiment_quality", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixture_csv_matches_curriculum_dataset() -> None:
    canonical = REPO_ROOT / "datasets" / "M14" / "learning_sessions.csv"
    experiment = _experiment()
    assert experiment.SESSIONS_CSV.read_text(encoding="utf-8") == canonical.read_text(encoding="utf-8")


def test_scale_trap_is_dominated_by_activity_events() -> None:
    blocks = _experiment().run_scale_trap()
    types = [block["type"] for block in blocks]
    assert types == ["table", "metric"]
    metric = blocks[1]["payload"]
    assert metric["dominant_feature"] == "activity_events"
    assert metric["dominant_share"] > 0.9
    assert metric["clusters_are_true_classes"] is False
    table = {row[0]: row[2] for row in blocks[0]["payload"]["rows"]}
    assert table["activity_events"] > table["practice_ratio"]
